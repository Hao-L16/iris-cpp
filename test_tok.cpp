#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "check.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdarg>

static struct ggml_context * C;      // 全局,少传一个参数
static struct ggml_context * CW;
static float GN_EPS = 1e-6f;


// conv2d + bias
static struct ggml_tensor * conv(struct ggml_tensor * x, const char * p,
                                 int stride = 1, int pad = 1) {
    char nm[160];
    snprintf(nm, sizeof(nm), "%s.weight", p);
    struct ggml_tensor * w = ggml_get_tensor(CW, nm);
    snprintf(nm, sizeof(nm), "%s.bias", p);
    struct ggml_tensor * b = ggml_get_tensor(CW, nm);
    if (!w || !b) { fprintf(stderr, "[X] 找不到 %s\n", p); exit(1); }
    x = ggml_conv_2d_direct(C, w, x, stride, stride, pad, pad, 1, 1);
    return ggml_add(C, x, ggml_reshape_4d(C, b, 1, 1, b->ne[0], 1));
}

// GroupNorm(32, eps=1e-6) + per-channel scale/shift
static struct ggml_tensor * gnorm(struct ggml_tensor * x, const char * p) {
    char nm[160];
    snprintf(nm, sizeof(nm), "%s.weight", p);
    struct ggml_tensor * w = ggml_get_tensor(CW, nm);
    snprintf(nm, sizeof(nm), "%s.bias", p);
    struct ggml_tensor * b = ggml_get_tensor(CW, nm);
    x = ggml_group_norm(C, x, 32, GN_EPS);
    x = ggml_mul(C, x, ggml_reshape_4d(C, w, 1, 1, w->ne[0], 1));
    return ggml_add(C, x, ggml_reshape_4d(C, b, 1, 1, b->ne[0], 1));
}

static struct ggml_tensor * tap[24]; static int nt = 0;

// ---- ResnetBlock(ch_mult 全 1,无 nin_shortcut)----
static struct ggml_tensor * resblock(struct ggml_tensor * x, const char * p, bool d) {
    char nm[160];
    snprintf(nm, sizeof(nm), "%s.norm1", p);
    struct ggml_tensor * h = gnorm(x, nm);          if (d) tap[nt++] = h;
    h = ggml_silu(C, h);                            if (d) tap[nt++] = h;
    snprintf(nm, sizeof(nm), "%s.conv1", p);
    h = conv(h, nm);                                if (d) tap[nt++] = h;
    snprintf(nm, sizeof(nm), "%s.norm2", p);
    h = ggml_silu(C, gnorm(h, nm));
    snprintf(nm, sizeof(nm), "%s.conv2", p);
    h = conv(h, nm);                                if (d) tap[nt++] = h;
    return ggml_add(C, x, h);
}

// ---- AttnBlock(单头空间注意力)----
static struct ggml_tensor * attnblock(struct ggml_tensor * x, const char * p, bool d) {
    char nm[160];
    const int Wd = x->ne[0], Hd = x->ne[1], Ch = x->ne[2], HW = Wd*Hd;

    snprintf(nm, sizeof(nm), "%s.norm", p);
    struct ggml_tensor * h = gnorm(x, nm);          if (d) tap[nt++] = h;

    snprintf(nm, sizeof(nm), "%s.q", p); struct ggml_tensor * q = conv(h, nm, 1, 0);
    snprintf(nm, sizeof(nm), "%s.k", p); struct ggml_tensor * k = conv(h, nm, 1, 0);
    snprintf(nm, sizeof(nm), "%s.v", p); struct ggml_tensor * v = conv(h, nm, 1, 0);
    if (d) { tap[nt++] = q; tap[nt++] = k; tap[nt++] = v; }

    // [W,H,C,1] -> [HW,C] -> 转置 -> [C,HW]
    struct ggml_tensor * q2 = ggml_cont(C, ggml_transpose(C, ggml_reshape_2d(C, q, HW, Ch)));
    struct ggml_tensor * k2 = ggml_cont(C, ggml_transpose(C, ggml_reshape_2d(C, k, HW, Ch)));
    struct ggml_tensor * v2 = ggml_reshape_2d(C, v, HW, Ch);     // v 不用转置

    // w[key, query] —— 与 PyTorch (b, hw_q, hw_k) 的字节布局一致
    struct ggml_tensor * w = ggml_mul_mat(C, k2, q2);
    w = ggml_scale(C, w, 1.0f / sqrtf((float)Ch));  if (d) tap[nt++] = w;
    w = ggml_soft_max(C, w);                        if (d) tap[nt++] = w;

    struct ggml_tensor * o = ggml_mul_mat(C, w, v2);            // [HW_q, C]
    o = ggml_reshape_4d(C, o, Wd, Hd, Ch, 1);       if (d) tap[nt++] = o;

    snprintf(nm, sizeof(nm), "%s.proj_out", p);
    return ggml_add(C, x, conv(o, nm, 1, 0));
}

int main() {
    struct gguf_init_params gp = { false, &CW };
    struct gguf_context * gg = gguf_init_from_file("iris-tokenizer-f32.gguf", gp);
    if (!gg) { fprintf(stderr, "[X] gguf 加载失败\n"); return 1; }

    const int ki = gguf_find_key(gg,"group_norm_eps");
    if (ki >= 0) GN_EPS = gguf_get_val_f32(gg,ki);
    printf("[*] GN_EPS = %g\n",GN_EPS);

    struct ggml_init_params ip = { 4096ull*1024*1024, NULL, false };
    C = ggml_init(ip);

    // 输入(dump 里已经做过 mul(2).sub(1))
    std::vector<float> xv(3*64*64);
    FILE * f = fopen("ref/tok_input.bin", "rb");
    if (!f || fread(xv.data(), sizeof(float), xv.size(), f) != xv.size()) {
        fprintf(stderr, "[X] 读不了 ref/tok_input.bin\n"); return 1; }
    fclose(f);
    struct ggml_tensor * x = ggml_new_tensor_4d(C, GGML_TYPE_F32, 64, 64, 3, 1);
    memcpy(x->data, xv.data(), xv.size()*sizeof(float));

    // ---------- Encoder ----------
    struct ggml_tensor * h = conv(x, "encoder.conv_in");
    struct ggml_tensor * t_convin = h;

    struct ggml_tensor * t_L[5]; struct ggml_tensor * t_pad0 = NULL;
    char p[160];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 2; j++) {
            snprintf(p, sizeof(p), "encoder.down.%d.block.%d", i, j);
            h = resblock(h, p, (i==0 && j==0));
            if (i == 2 || i == 3) {
                snprintf(p, sizeof(p), "encoder.down.%d.attn.%d", i, j);
                h = attnblock(h, p, (i==2 && j==0));
            }
        }
        if (i != 4) {                                   // Downsample
            struct ggml_tensor * pd = ggml_pad(C, h, 1, 1, 0, 0);   // 只补右/下
            if (i == 0) t_pad0 = pd;
            snprintf(p, sizeof(p), "encoder.down.%d.downsample.conv", i);
            h = conv(pd, p, 2, 0);
        }
        t_L[i] = h;
    }

    h = resblock(h, "encoder.mid.block_1", false);
    h = attnblock(h, "encoder.mid.attn_1", false);
    h = resblock(h, "encoder.mid.block_2", false);
    struct ggml_tensor * t_mid = h;

    h = conv(ggml_silu(C, gnorm(h, "encoder.norm_out")), "encoder.conv_out");
    struct ggml_tensor * t_z = h;

    // -----------T5:量化---------
    struct ggml_tensor * z = conv(h,"pre_quant_conv",1,0);
    struct ggml_tensor * t_z2 = z;

    // [4,4,512,1] -> [16,512] ->转置 ->[512,16]
    struct ggml_tensor * zf = ggml_cont(C,ggml_transpose(C,ggml_reshape_2d(C,z,16,512)));
    struct ggml_tensor * t_zflat = zf;

    struct ggml_tensor * emb = ggml_get_tensor(CW,"embedding.weight");   //ne=[512,512]
    if (!emb) {fprintf(stderr,"[X] 找不到embedding.weight\n"); return 1;}

    // dist = -2*z·e + ||z||^2 + ||e||^2
    struct ggml_tensor * dots = ggml_mul_mat(C,emb,zf);
    struct ggml_tensor * zsq  = ggml_sum_rows(C,ggml_sqr(C,zf));
    struct ggml_tensor * esq  = ggml_cont(C,ggml_transpose(C,ggml_sum_rows(C,ggml_sqr(C,emb))));

    struct ggml_tensor * dist = ggml_scale(C,dots,-2.0f);
    dist = ggml_add(C,dist,zsq);       //沿vocab广播
    dist = ggml_add(C,dist,esq);       //沿位置广播
    struct ggml_tensor * t_dist = dist;

    //argmin = argmax(-dist);ggml没有argmin
    struct ggml_tensor * toks = ggml_argmax(C,ggml_scale(C,dist,-1.0f));  //I32 [16]

    //z_q = emb[toks],再转回[4,4,512,1]
    struct ggml_tensor * zq = ggml_get_rows(C,emb,toks);
    zq = ggml_reshape_4d(C,ggml_cont(C,ggml_transpose(C,zq)),4,4,512,1);
    struct ggml_tensor * t_zq = zq;

    // --------- T6:decoder -------------
    h = conv(zq,"post_quant_conv",1,0);       struct ggml_tensor * t_decin = h;
    h = conv(h,"decoder.conv_in");            struct ggml_tensor * t_decconv = h;

    h = resblock(h,"decoder.mid.block_1",false);
    h = attnblock(h,"decoder.mid.attn_1",false);
    h = resblock(h,"decoder.mid.block_2",false); struct ggml_tensor * t_decmid = h;

    struct ggml_tensor * t_dL[5];
    struct ggml_tensor * t_us0 = NULL;
    for(int i = 4; i >= 0; i--) {     //decoder每层3个block
        for(int j = 0;j<3;j++) {
            snprintf(p,sizeof(p),"decoder.up.%d.block.%d",i,j);
            h = resblock(h,p,false);
            if(i == 2 || i == 3){
                snprintf(p,sizeof(p),"decoder.up.%d.attn.%d",i,j);
                h=attnblock(h,p,false);
            }
        }
        if(i !=0){
            h = ggml_upscale(C,h,2,GGML_SCALE_MODE_NEAREST);
            if (i == 4) t_us0 = h;
            snprintf(p,sizeof(p),"decoder.up.%d.upsample.conv",i);
            h = conv(h,p);
        }
        t_dL[i] = h;
    }

    h = conv(ggml_silu(C,gnorm(h,"decoder.norm_out")),"decoder.conv_out");
    struct ggml_tensor * t_rec = h;

    struct ggml_cgraph * gf = ggml_new_graph_custom(C,16384,false);
    ggml_build_forward_expand(gf,t_rec);
    for( int i = 0; i<nt; i++) ggml_build_forward_expand(gf,tap[i]);
    ggml_build_forward_expand(gf,t_pad0);
    ggml_build_forward_expand(gf,t_us0);
    ggml_build_forward_expand(gf,toks);
    ggml_graph_compute_with_ctx(C,gf,1);


    printf("[1] conv_in + ResnetBlock:\n");
    check("tok_enc_conv_in",   t_convin);
    check("tok_enc_rb0_norm1", tap[0]);
    check("tok_enc_rb0_sw1",   tap[1]);
    check("tok_enc_rb0_conv1", tap[2]);
    check("tok_enc_rb0_conv2", tap[3]);

    printf("[2] Downsample(非对称 padding):\n");
    check("tok_enc_ds0_pad",   t_pad0);

    printf("[3] AttnBlock:\n");
    check("tok_enc_at0_norm",   tap[4]);
    check("tok_enc_at0_q",      tap[5]);
    check("tok_enc_at0_k",      tap[6]);
    check("tok_enc_at0_v",      tap[7]);
    check("tok_enc_at0_w_raw",  tap[8]);
    check("tok_enc_at0_w_sm",   tap[9]);
    check("tok_enc_at0_attout", tap[10]);

    printf("[4] 各层输出:\n");
    for (int i = 0; i < 5; i++) { snprintf(p,sizeof(p),"tok_enc_L%d",i); check(p, t_L[i]); }
    check("tok_enc_mid", t_mid);
    check("tok_enc_z",   t_z);

    printf("[5] 量化:\n");
    check("tok_z",      t_z2);
    check("tok_z_flat", t_zflat);
    check("tok_dist",   t_dist);
    check("tok_z_q",    t_zq);

    // tokens是int32,不能用check()
    {
        int32_t tr[16];
        FILE * ft = fopen("ref/tok_tokens.bin","rb");
        if (ft && fread(tr,sizeof(int32_t),16,ft) == 16){
            fclose(ft);
            const int32_t * tg = (const int32_t *) toks->data;
            int bad = 0;
            printf(" tokens ggml:");
            for (int i=0;i<16;i++){ printf("%d ",tg[i]);if(tg[i]!=tr[i]) bad++;}
            printf("\n tokens torch: ");
            for (int i=0;i<16;i++) printf("%d ",tr[i]);
            printf("\n ==> %s (%d/16)\n", bad == 0? "PASS" : "FAIL", 16 - bad);
            if(bad) g_fail++;
        } else if(ft) fclose(ft);
    }

    printf("[6] decoder:\n");
    check("tok_dec_in",      t_decin);
    check("tok_dec_conv_in", t_decconv);
    check("tok_dec_mid",     t_decmid);
    check("tok_dec_us0_up",  t_us0);
    for (int i=4; i>=0;i--) {snprintf(p,sizeof(p),"tok_dec_L%d",i); check(p,t_dL[i]);}
    check("tok_rec",t_rec);

    printf("\n==> %s\n", g_fail == 0 ? "ALL PASS" : "FAIL");
    ggml_free(C); ggml_free(CW); gguf_free(gg);
    return g_fail;
}
