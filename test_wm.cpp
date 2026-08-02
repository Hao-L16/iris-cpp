#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "check.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

static const int TPB = 17, NBLK = 2, T = TPB * NBLK;   // 34
static const int E = 256, NH = 4, HS = E / NH;         // 64
static const int OBS_V = 512;
static const float EPS = 1e-5f;

static void print_ne(const char * tag, const struct ggml_tensor * t) {
    printf("    %-22s ne=[%lld %lld %lld %lld]\n", tag,
           (long long)t->ne[0], (long long)t->ne[1],
           (long long)t->ne[2], (long long)t->ne[3]);
}
static struct ggml_tensor * W(struct ggml_context * c, const char * name) {
    struct ggml_tensor * t = ggml_get_tensor(c, name);
    if (!t) { fprintf(stderr, "[X] 找不到 %s\n", name); exit(1); }
    return t;
}
// LayerNorm = norm + scale + shift
static struct ggml_tensor * layer_norm(struct ggml_context * c, struct ggml_tensor * x,
                                       struct ggml_tensor * w, struct ggml_tensor * b) {
    x = ggml_norm(c, x, EPS);
    x = ggml_mul(c, x, w);
    return ggml_add(c, x, b);
}
// y = Wx + b
static struct ggml_tensor * linear(struct ggml_context * c, struct ggml_tensor * w,
                                   struct ggml_tensor * b, struct ggml_tensor * x) {
    return ggml_add(c, ggml_mul_mat(c, w, x), b);
}

int main() {
    // ---------- 1. 权重 ----------
    struct ggml_context * ctx_w = NULL;
    struct gguf_init_params gp = { false, &ctx_w };
    struct gguf_context * gg = gguf_init_from_file("iris-worldmodel-f32.gguf", gp);
    if (!gg) { fprintf(stderr, "[X] gguf 加载失败\n"); return 1; }

    struct ggml_tensor * e_act = W(ctx_w, "embedder.embedding_tables.0.weight"); // [256,4]
    struct ggml_tensor * e_obs = W(ctx_w, "embedder.embedding_tables.1.weight"); // [256,512]
    struct ggml_tensor * pos   = W(ctx_w, "pos_emb.weight");                     // [256,340]

    printf("[1] 权重:\n");
    print_ne("embedding_tables.0", e_act);
    print_ne("embedding_tables.1", e_obs);
    print_ne("pos_emb", pos);

    struct ggml_tensor *ln1w[10],*ln1b[10],*ln2w[10],*ln2b[10];
    struct ggml_tensor *qw[10],*qb[10],*kw[10],*kb[10],*vw[10],*vb[10],*pw[10],*pb[10];
    struct ggml_tensor *f1w[10],*f1b[10],*f2w[10],*f2b[10];

    char n[128];
    #define BW(L,field) (snprintf(n, sizeof(n),"transformer.blocks.%d." field,L),W(ctx_w,n))
    for (int L=0;L < 10; L++) {
        ln1w[L]=BW(L,"ln1.weight");      ln1b[L]=BW(L,"ln1.bias");
        ln2w[L]=BW(L,"ln2.weight");      ln2b[L]=BW(L,"ln2.bias");
        qw[L]=BW(L,"attn.query.weight"); qb[L]=BW(L,"attn.query.bias");
        kw[L]=BW(L,"attn.key.weight");   kb[L]=BW(L,"attn.key.bias");
        vw[L]=BW(L,"attn.value.weight"); vb[L]=BW(L,"attn.value.bias");
        pw[L]=BW(L,"attn.proj.weight");  pb[L]=BW(L,"attn.proj.bias");
        f1w[L]=BW(L,"mlp.0.weight");     f1b[L]=BW(L,"mlp.0.bias");
        f2w[L]=BW(L,"mlp.2.weight");     f2b[L]=BW(L,"mlp.2.bias");
    }
    struct ggml_tensor * lnfw = W(ctx_w,"transformer.ln_f.weight");
    struct ggml_tensor * lnfb = W(ctx_w,"transformer.ln_f.bias");

    struct ggml_tensor * hw[3][2], *hb[3][2];
    const char * hname[3] ={"head_observations", "head_rewards","head_ends"};
    for(int i =0; i<3; i++){
        snprintf(n,sizeof(n),"%s.head_module.0.weight",hname[i]);hw[i][0]=W(ctx_w,n);
        snprintf(n,sizeof(n),"%s.head_module.0.bias",  hname[i]);hb[i][0]=W(ctx_w,n);
        snprintf(n,sizeof(n),"%s.head_module.2.weight",hname[i]);hw[i][1]=W(ctx_w,n);
        snprintf(n,sizeof(n),"%s.head_module.2.bias",  hname[i]);hb[i][1]=W(ctx_w,n);
    }
    // ---------- 2. 输入 ----------
    struct ggml_init_params ip = { 512ull*1024*1024, NULL, false };
    struct ggml_context * ctx0 = ggml_init(ip);

    // token id:obs 位置用原值,动作位置重映射到 512+a(拼表之后的行号)
    std::vector<int32_t> tok(T);
    { FILE * f = fopen("ref/wm_tokens.bin", "rb");
      if (!f) { fprintf(stderr, "[X] 打不开 ref/wm_tokens.bin\n"); return 1; }
      if (fread(tok.data(), sizeof(int32_t), T, f) != (size_t)T) return 1;
      fclose(f); }
    for (int p = 0; p < T; p++) if (p % TPB == TPB-1) tok[p] += OBS_V;

    struct ggml_tensor * ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    memcpy(ids->data, tok.data(), T * sizeof(int32_t));

    // ---------- 3. 建图 ----------
    // A1: 两张表拼成 516 行,一次 get_rows;pos 取前 T 行
    struct ggml_tensor * tbl = ggml_concat(ctx0, e_obs, e_act, 1);      // [256,516]
    struct ggml_tensor * t_embed = ggml_get_rows(ctx0, tbl, ids);       // [256,34]
    struct ggml_tensor * pos_t = ggml_view_2d(ctx0, pos, E, T, pos->nb[1], 0);
    struct ggml_tensor * x = ggml_add(ctx0, t_embed, pos_t);
    struct ggml_tensor * t_seq = x;

    // A2: block 0
    struct ggml_tensor * t_blk[10];
    struct ggml_tensor * tap[16] = {0};

    for (int L=0;L<10;L++){
        const bool d =(L == 0);
        int ti = 0;

        struct ggml_tensor *h = layer_norm(ctx0,x,ln1w[L],ln1b[L]);
        if (d) tap[ti++] = h;

        struct ggml_tensor * q = linear(ctx0,qw[L],qb[L],h); if (d) tap[ti++] = q;
        struct ggml_tensor * k = linear(ctx0,kw[L],kb[L],h); if (d) tap[ti++] = k;
        struct ggml_tensor * v = linear(ctx0,vw[L],vb[L],h); if (d) tap[ti++] = v;

        struct ggml_tensor * Q = ggml_permute(ctx0,ggml_reshape_3d(ctx0,q,HS,NH,T),0,2,1,3);
        struct ggml_tensor * K = ggml_permute(ctx0,ggml_reshape_3d(ctx0,k,HS,NH,T),0,2,1,3);

        struct ggml_tensor * KQ = ggml_scale(ctx0,ggml_mul_mat(ctx0,K,Q),1.0f/sqrtf((float)HS));
        if (d) tap[ti++] = KQ;

        KQ = ggml_soft_max(ctx0,ggml_diag_mask_inf(ctx0,KQ,0));
        if (d) tap[ti++] = KQ;

        struct ggml_tensor * V = ggml_cont(ctx0,ggml_permute(ctx0,ggml_reshape_3d(ctx0,v,HS,NH,T),1,2,0,3));
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0,V,KQ);

        struct ggml_tensor * y = ggml_cont_2d(ctx0,ggml_permute(ctx0,KQV,0,2,1,3),E,T);
        if (d) tap[ti++] = y;

        y = linear(ctx0,pw[L],pb[L],y); if (d) tap[ti++] = y;
        x = ggml_add(ctx0,x,y);         if (d) tap[ti++] = x;

        struct ggml_tensor * m = layer_norm(ctx0,x,ln2w[L],ln2b[L]);    if (d) tap[ti++] = m;
        m = linear(ctx0,f1w[L], f1b[L],m);                              if (d) tap[ti++] = m;
        m = ggml_gelu_erf(ctx0,m);                                      if (d) tap[ti++] = m;
        m = linear(ctx0,f2w[L], f2b[L],m);                              if (d) tap[ti++] = m;

        x = ggml_add(ctx0,x,m);
        t_blk[L] = x;
    }

    // ------- ln_f --------
    x = layer_norm(ctx0,x,lnfw,lnfb);
    struct ggml_tensor * t_lnf = x;

    // ---------三个head（取模切片）--------
    std::vector<int32_t> io,ia;
    for (int p =0; p < T; p++) {
        if (p % TPB != TPB-2) io.push_back(p);       // obs头:跳过 p%17 == 15
        if (p % TPB == TPB-1) ia.push_back(p);       // reward/end头：只在p%17==16
    }
    printf("[*] slice: obs %zu 个，act %zu 个\n", io.size(), ia.size());     //期望 32/2

    struct ggml_tensor * id_o = ggml_new_tensor_1d(ctx0,GGML_TYPE_I32,io.size());
    struct ggml_tensor * id_a = ggml_new_tensor_1d(ctx0,GGML_TYPE_I32,ia.size());
    memcpy(id_o->data,io.data(),io.size()*sizeof(int32_t));
    memcpy(id_a->data,ia.data(),ia.size()*sizeof(int32_t));

    struct ggml_tensor * t_head[3];
    for (int i=0;i<3;i++){
        struct ggml_tensor * s =ggml_get_rows(ctx0,x,i == 0 ? id_o : id_a);
        s = ggml_relu(ctx0,linear(ctx0,hw[i][0],hb[i][0],s));
        t_head[i] = linear(ctx0,hw[i][1],hb[i][1],s);
    }

    // ---------- 建图 ------------
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    for (int i = 0; i<3;i++) ggml_build_forward_expand(gf,t_head[i]);
    ggml_build_forward_expand(gf,tap[4]);     //保险起见显示注册

    // ---------- 4. 执行 ----------
    ggml_graph_compute_with_ctx(ctx0, gf, 1);

    // ---------- 5. 比对 ----------
    printf("[2] 输入 + block 0 子步骤:\n");
    check("wm_embed",t_embed);
    check("wm_seq",  t_seq);

    const char * tn[13] = {"b0_ln1","b0_q","b0_k","b0_v","b0_att_raw","b0_att_sm",
                           "b0_att_out","b0_proj","b0_res1","b0_ln2","b0_fc",
                           "b0_gelu","b0_mlp"};

    for (int i=0;i<13;i++){
        snprintf(n,sizeof(n),"wm_%s",tn[i]);
        check(n,tap[i]);
    }

    printf("[3] 10个block + ln_f:\n");

    for (int L = 0; L<10;L++) {
        snprintf(n,sizeof(n),"wm_blk%d",L);
        check(n,t_blk[L]);
    }
    check("wm_lnf",t_lnf);

    printf("[4] 输出头:\n");
    check("wm_logits_obs",    t_head[0]);
    check("wm_logits_rewards",t_head[1]);
    check("wm_logits_ends",   t_head[2]);
    // --------独立验证-----------
    {
        const int NPOS =32, V=512;
        std::vector<float> ref(NPOS *V);
        FILE * f =fopen("ref/wm_logits_obs.bin","rb");
        if (f && fread(ref.data(),sizeof(float), NPOS*V,f) == (size_t)(NPOS*V)) {
            fclose(f);
            const float * g = (const float *) t_head[0]->data;
            int bad = 0;
            for (int p=0; p<NPOS; p++){
                int ag = 0, ar = 0;
                for (int j = 1; j < V; j++) {
                    if (g[p*V+j] > g[p*V+ag]) ag = j;
                    if (ref[p*V+j]>ref[p*V+ar]) ar = j;
                }
                if (ag!=ar){printf(" pos %2d:ggml->%d torch->%d\n",p,ag,ar); bad++;}
            }
            printf("[5] argmax 一致:%d/%d位置\n",NPOS - bad,NPOS);
        } else if (f) fclose(f);
    }
    printf("\n==> %s\n", g_fail == 0 ? "ALL PASS" : "FAIL");
    ggml_free(ctx0); ggml_free(ctx_w); gguf_free(gg);
    return g_fail;

}
