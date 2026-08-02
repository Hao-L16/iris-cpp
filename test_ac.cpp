// test_ac.cpp
#include "ggml.h"
#include "ggml-cpu.h"     // ggml_graph_compute_with_ctx 在这里,不在 ggml.h
#include "gguf.h"
#include "check.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

static std::vector<float> read_bin(const char * path, size_t n) {
    std::vector<float> v(n);
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[X] 打不开 %s\n", path); exit(1); }
    size_t got = fread(v.data(), sizeof(float), n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "[X] %s: 读到 %zu 个 float,期望 %zu\n", path, got, n); exit(1); }
    return v;
}

static void print_ne(const char * tag, const struct ggml_tensor * t) {
    printf("    %-16s ne=[%lld %lld %lld %lld]  type=%s\n", tag,
           (long long)t->ne[0], (long long)t->ne[1],
           (long long)t->ne[2], (long long)t->ne[3],
           ggml_type_name(t->type));
}

static struct ggml_tensor * W(struct ggml_context * c, const char * name) {
    struct ggml_tensor * t = ggml_get_tensor(c, name);
    if (!t) { fprintf(stderr, "[X] 找不到张量 %s\n", name); exit(1); }
    return t;
}


int main() {
    // ================= 1. 载入 GGUF 权重 =================
    struct ggml_context * ctx_w = NULL;
    struct gguf_init_params gp = { /* no_alloc = */ false, /* ctx = */ &ctx_w };

    struct gguf_context * gg = gguf_init_from_file("iris-actorcritic-f32.gguf", gp);
    if (!gg) { fprintf(stderr, "[X] gguf 加载失败\n"); return 1; }

    struct ggml_tensor * cw[4];
    struct ggml_tensor * cb[4];

    printf("[1] 权重载入:\n");
    for (int i=0; i < 4; i++) {
        char name[64];

        snprintf(name, sizeof(name), "conv%d.weight", i+1);
        cw[i] = W(ctx_w, name);
        print_ne(name, cw[i]);

        snprintf(name, sizeof(name), "conv%d.bias", i+1);
        cb[i] = W(ctx_w, name);
    }

    struct ggml_tensor * w_ih = W(ctx_w, "lstm.weight_ih");
    struct ggml_tensor * w_hh = W(ctx_w, "lstm.weight_hh");
    struct ggml_tensor * b_ih = W(ctx_w, "lstm.bias_ih");
    struct ggml_tensor * b_hh = W(ctx_w, "lstm.bias_hh");
    struct ggml_tensor * w_a  = W(ctx_w, "actor_linear.weight");
    struct ggml_tensor * b_a  = W(ctx_w, "actor_linear.bias");
    struct ggml_tensor * w_c  = W(ctx_w, "critic_linear.weight");
    struct ggml_tensor * b_c  = W(ctx_w, "critic_linear.bias");

    print_ne("lstm.weight_ih", w_ih);
    print_ne("lstm.weight_hh", w_hh);


    // ================= 2. 计算 context =================
    struct ggml_init_params ip = { 256ull * 1024 * 1024, NULL, false };
    struct ggml_context * ctx0 = ggml_init(ip);
    if (!ctx0) { fprintf(stderr, "[X] ggml_init 失败\n"); return 1; }

    // 输入:host 侧先做 x*2-1(数据预处理,不属于被验证的层)
    std::vector<float> xin = read_bin("ref/inp.bin", 3 * 64 * 64);
    for (auto & v : xin) v = v * 2.0f - 1.0f;

    struct ggml_tensor * inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 64, 64, 3, 1);
    memcpy(inp->data, xin.data(), xin.size() * sizeof(float));
    printf("[2] 输入构造:\n");
    print_ne("input", inp);          // 期望 ne=[64 64 3 1]

    //灌入hx/cx
    std::vector<float> hxv = read_bin("ref/hx.bin",512);
    std::vector<float> cxv = read_bin("ref/cx.bin",512);

    struct ggml_tensor * hx = ggml_new_tensor_2d(ctx0,GGML_TYPE_F32,512,1);
    struct ggml_tensor * cx = ggml_new_tensor_2d(ctx0,GGML_TYPE_F32,512,1);
    memcpy(hx->data,hxv.data(),512 * sizeof(float));
    memcpy(cx->data,cxv.data(),512 * sizeof(float));

    // ================= 3. 建图(此时不计算) =================
    struct ggml_tensor * t_conv[4];
    struct ggml_tensor * t_pool[4];
    struct ggml_tensor * t_relu[4];

    struct ggml_tensor * cur =inp;

    for (int i=0; i<4; i++) {
        cur = ggml_conv_2d_direct(ctx0,cw[i],cur, 1,1,1,1,1,1);
        cur = ggml_add(ctx0,cur,ggml_reshape_4d(ctx0,cb[i],1,1,cb[i]->ne[0],1));
        t_conv[i] = cur;

        // 2×2 最大池化
        cur = ggml_pool_2d(ctx0,cur,GGML_OP_POOL_MAX,2,2,2,2,0.0f,0.0f);
        t_pool[i] = cur;

        //Relu
        cur = ggml_relu(ctx0,cur);
        t_relu[i] = cur;
    }

    // ----- flatten：【4，4，64，1]->[1024,1],零成本，只换形状标签 ----
    struct ggml_tensor * t_flat = ggml_reshape_2d(ctx0,cur,1024,1);

    // ----- LSTM 门的原始值 -------
    struct ggml_tensor * gates = ggml_mul_mat(ctx0,w_ih,t_flat);  //[2048,1]
    gates = ggml_add(ctx0,gates, ggml_mul_mat(ctx0,w_hh,hx));
    gates = ggml_add(ctx0,gates,b_ih);
    gates = ggml_add(ctx0,gates,b_hh);
    struct ggml_tensor * t_gates = gates;

    // -----切成四分，pytorch 门序是i,f,g,o----
    const size_t S = 512 * sizeof(float);
    struct ggml_tensor * g_i = ggml_sigmoid(ctx0,ggml_view_1d(ctx0,gates,512,0*S));
    struct ggml_tensor * g_f = ggml_sigmoid(ctx0,ggml_view_1d(ctx0,gates,512,1*S));
    struct ggml_tensor * g_g = ggml_tanh(ctx0,ggml_view_1d(ctx0,gates,512,2*S));
    struct ggml_tensor * g_o = ggml_sigmoid(ctx0,ggml_view_1d(ctx0,gates,512,3*S));

    // ---- c' = f*c + i*g; h' = o*tanh(c')
    struct ggml_tensor * cx2 = ggml_add(ctx0,ggml_mul(ctx0,g_f,cx),ggml_mul(ctx0,g_i,g_g));
    struct ggml_tensor * t_cx_out = cx2;
    struct ggml_tensor * hx2 = ggml_mul(ctx0,g_o,ggml_tanh(ctx0,cx2));
    struct ggml_tensor * t_hx_out = hx2;

    // ----两个输出头----
    struct ggml_tensor * t_logits = ggml_add(ctx0,ggml_mul_mat(ctx0,w_a,hx2),b_a);
    struct ggml_tensor * t_value  = ggml_add(ctx0,ggml_mul_mat(ctx0,w_c,hx2),b_c);

    // -----建图-----
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf,t_logits);
    ggml_build_forward_expand(gf,t_value);

    // ================= 4. 执行(单线程,累加顺序最确定) =================
    ggml_graph_compute_with_ctx(ctx0, gf, 1);


    // ================= 5. 比对 =================
    printf("[3] 逐层比对:\n");
    for (int i = 0; i < 4;i++){
        char name[32];
        snprintf(name, sizeof(name),"conv%d",i+1); check(name, t_conv[i]);
        snprintf(name, sizeof(name),"pool%d",i+1); check(name, t_pool[i]);
        snprintf(name, sizeof(name),"relu%d",i+1); check(name, t_relu[i]);
    }
    check("flat",  t_flat);
    check("gates", t_gates);
    check("cx_out",t_cx_out);
    check("hx_out",t_hx_out);
    check("logits",t_logits);
    check("value", t_value);

    printf("\n==> %s\n", g_fail == 0 ? "ALL PASS" : "FAIL");

    ggml_free(ctx0);
    ggml_free(ctx_w);
    gguf_free(gg);
    return g_fail;

}
