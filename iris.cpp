// iris.cpp — 三个组件的 ggml 实现
//
// 建图代码基本原样搬自 test_ac.cpp / test_wm.cpp / test_tok.cpp,
// 改动只有:输入不再从 ref/*.bin 读、结果拷贝到调用者的数组、删掉 check()。

#include "iris.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

using namespace iris;

// ================================================================
//  文件内全局:当前正在建的图用哪个计算池 / 哪个权重池
//  (每个 public 函数入口处设置,避免所有 helper 多传两个参数)
// ================================================================
static ggml_context * C  = nullptr;   // 计算池
static ggml_context * WC = nullptr;   // 权重池
static float GN_EPS = 1e-6f;

// ---------------- 取权重 ----------------
static ggml_tensor * W(const char * name) {
    ggml_tensor * t = ggml_get_tensor(WC, name);
    if (!t) { fprintf(stderr, "[X] 找不到张量 %s\n", name); exit(1); }
    return t;
}
static ggml_tensor * Wf(const char * fmt, int a) {
    char nm[192]; snprintf(nm, sizeof(nm), fmt, a); return W(nm);
}
static ggml_tensor * Wf2(const char * fmt, int a, int b) {
    char nm[192]; snprintf(nm, sizeof(nm), fmt, a, b); return W(nm);
}

// ---------------- 通用算子封装 ----------------

// y = Wx + b   (线性层,不用转置)
static ggml_tensor * linear(ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    return ggml_add(C, ggml_mul_mat(C, w, x), b);
}

// LayerNorm = norm + scale + shift
static ggml_tensor * layer_norm(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(C, x, eps);
    x = ggml_mul(C, x, w);
    return ggml_add(C, x, b);
}

// conv2d + bias(bias 必须 reshape 成 [1,1,C,1] 才能沿宽高广播)
static ggml_tensor * conv(ggml_tensor * x, const char * p, int stride = 1, int pad = 1) {
    char nm[192];
    snprintf(nm, sizeof(nm), "%s.weight", p);  ggml_tensor * w = W(nm);
    snprintf(nm, sizeof(nm), "%s.bias",   p);  ggml_tensor * b = W(nm);
    x = ggml_conv_2d_direct(C, w, x, stride, stride, pad, pad, 1, 1);
    return ggml_add(C, x, ggml_reshape_4d(C, b, 1, 1, b->ne[0], 1));
}

// GroupNorm(32, eps) + per-channel scale/shift
static ggml_tensor * gnorm(ggml_tensor * x, const char * p) {
    char nm[192];
    snprintf(nm, sizeof(nm), "%s.weight", p);  ggml_tensor * w = W(nm);
    snprintf(nm, sizeof(nm), "%s.bias",   p);  ggml_tensor * b = W(nm);
    x = ggml_group_norm(C, x, 32, GN_EPS);
    x = ggml_mul(C, x, ggml_reshape_4d(C, w, 1, 1, w->ne[0], 1));
    return ggml_add(C, x, ggml_reshape_4d(C, b, 1, 1, b->ne[0], 1));
}

// ---------------- tokenizer 的两个积木 ----------------

// ResnetBlock:ch_mult 全 1,所以 in==out,没有 nin_shortcut
static ggml_tensor * resblock(ggml_tensor * x, const char * p) {
    char nm[192];
    snprintf(nm, sizeof(nm), "%s.norm1", p);  ggml_tensor * h = gnorm(x, nm);
    h = ggml_silu(C, h);
    snprintf(nm, sizeof(nm), "%s.conv1", p);  h = conv(h, nm);
    snprintf(nm, sizeof(nm), "%s.norm2", p);  h = ggml_silu(C, gnorm(h, nm));
    snprintf(nm, sizeof(nm), "%s.conv2", p);  h = conv(h, nm);
    return ggml_add(C, x, h);
}

// AttnBlock:单头空间注意力,q/k/v/proj 全是 1x1 卷积
static ggml_tensor * attnblock(ggml_tensor * x, const char * p) {
    char nm[192];
    const int Wd = (int)x->ne[0], Hd = (int)x->ne[1], Ch = (int)x->ne[2];
    const int HW = Wd * Hd;

    snprintf(nm, sizeof(nm), "%s.norm", p);  ggml_tensor * h = gnorm(x, nm);
    snprintf(nm, sizeof(nm), "%s.q", p);     ggml_tensor * q = conv(h, nm, 1, 0);
    snprintf(nm, sizeof(nm), "%s.k", p);     ggml_tensor * k = conv(h, nm, 1, 0);
    snprintf(nm, sizeof(nm), "%s.v", p);     ggml_tensor * v = conv(h, nm, 1, 0);

    // [W,H,C,1] -> [HW,C] -> 转置 -> [C,HW];v 不用转置
    ggml_tensor * q2 = ggml_cont(C, ggml_transpose(C, ggml_reshape_2d(C, q, HW, Ch)));
    ggml_tensor * k2 = ggml_cont(C, ggml_transpose(C, ggml_reshape_2d(C, k, HW, Ch)));
    ggml_tensor * v2 = ggml_reshape_2d(C, v, HW, Ch);

    ggml_tensor * w = ggml_mul_mat(C, k2, q2);              // [key, query]
    w = ggml_scale(C, w, 1.0f / sqrtf((float)Ch));          // 缩放用全通道数
    w = ggml_soft_max(C, w);

    ggml_tensor * o = ggml_mul_mat(C, w, v2);               // [HW_q, C]
    o = ggml_reshape_4d(C, o, Wd, Hd, Ch, 1);

    snprintf(nm, sizeof(nm), "%s.proj_out", p);
    return ggml_add(C, x, conv(o, nm, 1, 0));
}

// ================================================================
//  加载 / 释放
// ================================================================

static bool load_one(const char * dir, const char * fname,
                     ggml_context ** wctx, gguf_context ** gg) {
    std::string path = std::string(dir) + "/" + fname;
    gguf_init_params gp = { /*no_alloc=*/false, /*ctx=*/wctx };
    *gg = gguf_init_from_file(path.c_str(), gp);
    if (!*gg) { fprintf(stderr, "[X] 加载失败: %s\n", path.c_str()); return false; }
    return true;
}

bool iris_load(iris_model & m, const char * dir) {
    if (!load_one(dir, "iris-tokenizer-f32.gguf",   &m.wctx_tok, &m.gg_tok)) return false;
    if (!load_one(dir, "iris-worldmodel-f32.gguf",  &m.wctx_wm,  &m.gg_wm))  return false;
    if (!load_one(dir, "iris-actorcritic-f32.gguf", &m.wctx_ac,  &m.gg_ac))  return false;

    // GroupNorm 的 eps 从 metadata 读,不硬编码
    const int ki = gguf_find_key(m.gg_tok, "group_norm_eps");
    if (ki >= 0) m.gn_eps = gguf_get_val_f32(m.gg_tok, ki);

    // 计算内存池:一次性 malloc,之后每步复用
    // 数值偏保守;如果 ggml 报 "not enough space" 就调大,
    // 上板前要实测峰值(KV260 只有 4GB)
    m.buf_tok_size = 768ull * 1024 * 1024;
    m.buf_wm_size  = 768ull * 1024 * 1024;
    m.buf_ac_size  = 128ull * 1024 * 1024;

    m.buf_tok = malloc(m.buf_tok_size);
    m.buf_wm  = malloc(m.buf_wm_size);
    m.buf_ac  = malloc(m.buf_ac_size);
    if (!m.buf_tok || !m.buf_wm || !m.buf_ac) {
        fprintf(stderr, "[X] 计算内存池分配失败\n"); return false;
    }

    printf("[iris] 已加载三个组件 (gn_eps=%g, 内存池 %.0f MB)\n",
           m.gn_eps,
           (m.buf_tok_size + m.buf_wm_size + m.buf_ac_size) / 1048576.0);
    return true;
}

void iris_free(iris_model & m) {
    if (m.wctx_tok) ggml_free(m.wctx_tok);
    if (m.wctx_wm)  ggml_free(m.wctx_wm);
    if (m.wctx_ac)  ggml_free(m.wctx_ac);
    if (m.gg_tok)   gguf_free(m.gg_tok);
    if (m.gg_wm)    gguf_free(m.gg_wm);
    if (m.gg_ac)    gguf_free(m.gg_ac);
    free(m.buf_tok); free(m.buf_wm); free(m.buf_ac);
    m = iris_model{};
}

// 用预分配的 buffer 开一个计算池(mem_buffer != NULL 时 ggml 不接管所有权)
static ggml_context * begin(void * buf, size_t size) {
    ggml_init_params ip = { size, buf, false };
    ggml_context * c = ggml_init(ip);
    if (!c) { fprintf(stderr, "[X] ggml_init 失败\n"); exit(1); }
    return c;
}

// ================================================================
//  tokenizer: encode
// ================================================================

void iris_tok_encode(iris_model & m, const float * frame, int32_t * tokens) {
    C = begin(m.buf_tok, m.buf_tok_size);
    WC = m.wctx_tok;
    GN_EPS = m.gn_eps;

    ggml_tensor * x = ggml_new_tensor_4d(C, GGML_TYPE_F32, FRAME_HW, FRAME_HW, FRAME_C, 1);
    memcpy(x->data, frame, FRAME_N * sizeof(float));

    // ---- Encoder ----
    ggml_tensor * h = conv(x, "encoder.conv_in");
    char p[192];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 2; j++) {
            snprintf(p, sizeof(p), "encoder.down.%d.block.%d", i, j);
            h = resblock(h, p);
            if (i == 2 || i == 3) {
                snprintf(p, sizeof(p), "encoder.down.%d.attn.%d", i, j);
                h = attnblock(h, p);
            }
        }
        if (i != 4) {                                        // Downsample
            ggml_tensor * pd = ggml_pad(C, h, 1, 1, 0, 0);   // 只补右/下(非对称)
            snprintf(p, sizeof(p), "encoder.down.%d.downsample.conv", i);
            h = conv(pd, p, 2, 0);
        }
    }
    h = resblock(h, "encoder.mid.block_1");
    h = attnblock(h, "encoder.mid.attn_1");
    h = resblock(h, "encoder.mid.block_2");
    h = conv(ggml_silu(C, gnorm(h, "encoder.norm_out")), "encoder.conv_out");

    // ---- 量化:pre_quant_conv → 转置 → 距离矩阵 → argmin ----
    ggml_tensor * z  = conv(h, "pre_quant_conv", 1, 0);
    ggml_tensor * zf = ggml_cont(C, ggml_transpose(C,
                            ggml_reshape_2d(C, z, N_TOKENS, VOCAB)));   // [512,16]

    ggml_tensor * emb  = W("embedding.weight");                          // [512,512]
    ggml_tensor * dots = ggml_mul_mat(C, emb, zf);                       // [vocab, 16]
    ggml_tensor * zsq  = ggml_sum_rows(C, ggml_sqr(C, zf));              // [1, 16]
    ggml_tensor * esq  = ggml_cont(C, ggml_transpose(C,
                             ggml_sum_rows(C, ggml_sqr(C, emb))));       // [512, 1]

    ggml_tensor * dist = ggml_scale(C, dots, -2.0f);
    dist = ggml_add(C, dist, zsq);      // 沿 vocab 广播
    dist = ggml_add(C, dist, esq);      // 沿位置广播

    // ggml 没有 argmin,取负再 argmax
    ggml_tensor * toks = ggml_argmax(C, ggml_scale(C, dist, -1.0f));     // I32 [16]

    ggml_cgraph * gf = ggml_new_graph_custom(C, 8192, false);
    ggml_build_forward_expand(gf, toks);
    ggml_graph_compute_with_ctx(C, gf, m.n_threads);

    memcpy(tokens, toks->data, N_TOKENS * sizeof(int32_t));
    ggml_free(C); C = nullptr;
}

// ================================================================
//  tokenizer: decode
// ================================================================

void iris_tok_decode(iris_model & m, const int32_t * tokens, float * frame_out) {
    C = begin(m.buf_tok, m.buf_tok_size);
    WC = m.wctx_tok;
    GN_EPS = m.gn_eps;

    ggml_tensor * ids = ggml_new_tensor_1d(C, GGML_TYPE_I32, N_TOKENS);
    memcpy(ids->data, tokens, N_TOKENS * sizeof(int32_t));

    // z_q = emb[tokens],再转回 [4,4,512,1]
    ggml_tensor * emb = W("embedding.weight");
    ggml_tensor * zq  = ggml_get_rows(C, emb, ids);                      // [512,16]
    zq = ggml_reshape_4d(C, ggml_cont(C, ggml_transpose(C, zq)), 4, 4, VOCAB, 1);

    ggml_tensor * h = conv(zq, "post_quant_conv", 1, 0);
    h = conv(h, "decoder.conv_in");

    h = resblock(h, "decoder.mid.block_1");
    h = attnblock(h, "decoder.mid.attn_1");
    h = resblock(h, "decoder.mid.block_2");

    char p[192];
    for (int i = 4; i >= 0; i--) {
        for (int j = 0; j < 3; j++) {                    // decoder 每层 3 个 block
            snprintf(p, sizeof(p), "decoder.up.%d.block.%d", i, j);
            h = resblock(h, p);
            if (i == 2 || i == 3) {
                snprintf(p, sizeof(p), "decoder.up.%d.attn.%d", i, j);
                h = attnblock(h, p);
            }
        }
        if (i != 0) {
            h = ggml_upscale(C, h, 2, GGML_SCALE_MODE_NEAREST);
            snprintf(p, sizeof(p), "decoder.up.%d.upsample.conv", i);
            h = conv(h, p);
        }
    }
    h = conv(ggml_silu(C, gnorm(h, "decoder.norm_out")), "decoder.conv_out");

    ggml_cgraph * gf = ggml_new_graph_custom(C, 16384, false);
    ggml_build_forward_expand(gf, h);
    ggml_graph_compute_with_ctx(C, gf, m.n_threads);

    memcpy(frame_out, h->data, FRAME_N * sizeof(float));
    ggml_free(C); C = nullptr;
}

// ================================================================
//  world_model
// ================================================================

void iris_wm_forward(iris_model & m,
                     const int32_t * tokens, int T,
                     float * logits_obs, int * n_obs_out,
                     float * logits_rew,
                     float * logits_end, int * n_act_out) {
    if (T > MAX_TOK) { fprintf(stderr, "[X] T=%d 超过 %d\n", T, MAX_TOK); exit(1); }

    C = begin(m.buf_wm, m.buf_wm_size);
    WC = m.wctx_wm;

    const int HS = EMBED / N_HEAD;
    const float LN_EPS = 1e-5f;      // 注意:world_model 用 1e-5,不是 tokenizer 的 1e-6

    // action 位置重映射到 512+a(两张表拼成 516 行之后的行号)
    std::vector<int32_t> tok(tokens, tokens + T);
    for (int p = 0; p < T; p++) if (p % TPB == TPB - 1) tok[p] += VOCAB;

    ggml_tensor * ids = ggml_new_tensor_1d(C, GGML_TYPE_I32, T);
    memcpy(ids->data, tok.data(), T * sizeof(int32_t));

    // ---- embedding + pos ----
    ggml_tensor * e_act = W("embedder.embedding_tables.0.weight");
    ggml_tensor * e_obs = W("embedder.embedding_tables.1.weight");
    ggml_tensor * pos   = W("pos_emb.weight");

    ggml_tensor * tbl = ggml_concat(C, e_obs, e_act, 1);                  // [256,516]
    ggml_tensor * x   = ggml_get_rows(C, tbl, ids);                       // [256,T]
    x = ggml_add(C, x, ggml_view_2d(C, pos, EMBED, T, pos->nb[1], 0));

    // ---- 10 个 block ----
    for (int L = 0; L < N_LAYER; L++) {
        ggml_tensor * h = layer_norm(x,
                              Wf("transformer.blocks.%d.ln1.weight", L),
                              Wf("transformer.blocks.%d.ln1.bias",   L), LN_EPS);

        ggml_tensor * q = linear(Wf("transformer.blocks.%d.attn.query.weight", L),
                                 Wf("transformer.blocks.%d.attn.query.bias",   L), h);
        ggml_tensor * k = linear(Wf("transformer.blocks.%d.attn.key.weight",   L),
                                 Wf("transformer.blocks.%d.attn.key.bias",     L), h);
        ggml_tensor * v = linear(Wf("transformer.blocks.%d.attn.value.weight", L),
                                 Wf("transformer.blocks.%d.attn.value.bias",   L), h);

        ggml_tensor * Q = ggml_permute(C, ggml_reshape_3d(C, q, HS, N_HEAD, T), 0,2,1,3);
        ggml_tensor * K = ggml_permute(C, ggml_reshape_3d(C, k, HS, N_HEAD, T), 0,2,1,3);

        ggml_tensor * KQ = ggml_scale(C, ggml_mul_mat(C, K, Q), 1.0f / sqrtf((float)HS));
        KQ = ggml_soft_max(C, ggml_diag_mask_inf(C, KQ, 0));

        ggml_tensor * V = ggml_cont(C,
            ggml_permute(C, ggml_reshape_3d(C, v, HS, N_HEAD, T), 1,2,0,3));
        ggml_tensor * KQV = ggml_mul_mat(C, V, KQ);

        ggml_tensor * y = ggml_cont_2d(C, ggml_permute(C, KQV, 0,2,1,3), EMBED, T);
        y = linear(Wf("transformer.blocks.%d.attn.proj.weight", L),
                   Wf("transformer.blocks.%d.attn.proj.bias",   L), y);
        x = ggml_add(C, x, y);

        ggml_tensor * mm = layer_norm(x,
                               Wf("transformer.blocks.%d.ln2.weight", L),
                               Wf("transformer.blocks.%d.ln2.bias",   L), LN_EPS);
        mm = linear(Wf("transformer.blocks.%d.mlp.0.weight", L),
                    Wf("transformer.blocks.%d.mlp.0.bias",   L), mm);
        mm = ggml_gelu_erf(C, mm);                     // 必须是 erf 版,不是 ggml_gelu
        mm = linear(Wf("transformer.blocks.%d.mlp.2.weight", L),
                    Wf("transformer.blocks.%d.mlp.2.bias",   L), mm);
        x = ggml_add(C, x, mm);
    }

    x = layer_norm(x, W("transformer.ln_f.weight"), W("transformer.ln_f.bias"), LN_EPS);

    // ---- 三个 head:取模切片 ----
    std::vector<int32_t> io, ia;
    for (int p = 0; p < T; p++) {
        if (p % TPB != TPB - 2) io.push_back(p);     // obs 头:跳过 p%17==15
        if (p % TPB == TPB - 1) ia.push_back(p);     // reward/end 头:只在 p%17==16
    }
    if (n_obs_out) *n_obs_out = (int)io.size();
    if (n_act_out) *n_act_out = (int)ia.size();

    ggml_tensor * id_o = ggml_new_tensor_1d(C, GGML_TYPE_I32, io.size());
    memcpy(id_o->data, io.data(), io.size() * sizeof(int32_t));
    ggml_tensor * id_a = nullptr;
    if (!ia.empty()) {
        id_a = ggml_new_tensor_1d(C, GGML_TYPE_I32, ia.size());
        memcpy(id_a->data, ia.data(), ia.size() * sizeof(int32_t));
    }

    const char * hname[3] = { "head_observations", "head_rewards", "head_ends" };
    ggml_tensor * head[3] = { nullptr, nullptr, nullptr };
    char nm[192];
    for (int i = 0; i < 3; i++) {
        ggml_tensor * idx = (i == 0) ? id_o : id_a;
        if (!idx) continue;
        ggml_tensor * s = ggml_get_rows(C, x, idx);
        snprintf(nm, sizeof(nm), "%s.head_module.0.weight", hname[i]);
        ggml_tensor * w0 = W(nm);
        snprintf(nm, sizeof(nm), "%s.head_module.0.bias", hname[i]);
        ggml_tensor * b0 = W(nm);
        snprintf(nm, sizeof(nm), "%s.head_module.2.weight", hname[i]);
        ggml_tensor * w1 = W(nm);
        snprintf(nm, sizeof(nm), "%s.head_module.2.bias", hname[i]);
        ggml_tensor * b1 = W(nm);
        s = ggml_relu(C, linear(w0, b0, s));         // head 里是 ReLU,不是 GELU
        head[i] = linear(w1, b1, s);
    }

    ggml_cgraph * gf = ggml_new_graph_custom(C, 8192, false);
    for (int i = 0; i < 3; i++) if (head[i]) ggml_build_forward_expand(gf, head[i]);
    ggml_graph_compute_with_ctx(C, gf, m.n_threads);

    if (logits_obs && head[0])
        memcpy(logits_obs, head[0]->data, io.size() * VOCAB * sizeof(float));
    if (logits_rew && head[1])
        memcpy(logits_rew, head[1]->data, ia.size() * 3 * sizeof(float));
    if (logits_end && head[2])
        memcpy(logits_end, head[2]->data, ia.size() * 2 * sizeof(float));

    ggml_free(C); C = nullptr;
}

// ================================================================
//  actor_critic
// ================================================================

void iris_ac_forward(iris_model & m, const float * frame,
                     iris_state & st, float * logits, float * value) {
    C = begin(m.buf_ac, m.buf_ac_size);
    WC = m.wctx_ac;

    ggml_tensor * inp = ggml_new_tensor_4d(C, GGML_TYPE_F32, FRAME_HW, FRAME_HW, FRAME_C, 1);
    memcpy(inp->data, frame, FRAME_N * sizeof(float));

    ggml_tensor * hx = ggml_new_tensor_2d(C, GGML_TYPE_F32, LSTM_H, 1);
    ggml_tensor * cx = ggml_new_tensor_2d(C, GGML_TYPE_F32, LSTM_H, 1);
    memcpy(hx->data, st.hx, LSTM_H * sizeof(float));
    memcpy(cx->data, st.cx, LSTM_H * sizeof(float));

    // ---- 4 x [conv3x3 -> maxpool2x2 -> relu],空间 64->32->16->8->4 ----
    ggml_tensor * cur = inp;
    for (int i = 0; i < 4; i++) {
        ggml_tensor * w = Wf("conv%d.weight", i + 1);
        ggml_tensor * b = Wf("conv%d.bias",   i + 1);
        cur = ggml_conv_2d_direct(C, w, cur, 1,1, 1,1, 1,1);
        cur = ggml_add(C, cur, ggml_reshape_4d(C, b, 1, 1, b->ne[0], 1));
        cur = ggml_pool_2d(C, cur, GGML_OP_POOL_MAX, 2,2, 2,2, 0.0f, 0.0f);
        cur = ggml_relu(C, cur);
    }

    // flatten [4,4,64,1] -> [1024,1],零成本
    ggml_tensor * flat = ggml_reshape_2d(C, cur, 1024, 1);

    // ---- LSTMCell:ggml 没有这个算子,手工拆 ----
    ggml_tensor * gates = ggml_mul_mat(C, W("lstm.weight_ih"), flat);
    gates = ggml_add(C, gates, ggml_mul_mat(C, W("lstm.weight_hh"), hx));
    gates = ggml_add(C, gates, W("lstm.bias_ih"));
    gates = ggml_add(C, gates, W("lstm.bias_hh"));

    const size_t S = LSTM_H * sizeof(float);
    // PyTorch 门序是 i, f, g, o —— 弄错不报错,只会算出错数字
    ggml_tensor * g_i = ggml_sigmoid(C, ggml_view_1d(C, gates, LSTM_H, 0*S));
    ggml_tensor * g_f = ggml_sigmoid(C, ggml_view_1d(C, gates, LSTM_H, 1*S));
    ggml_tensor * g_g = ggml_tanh   (C, ggml_view_1d(C, gates, LSTM_H, 2*S));
    ggml_tensor * g_o = ggml_sigmoid(C, ggml_view_1d(C, gates, LSTM_H, 3*S));

    ggml_tensor * cx2 = ggml_add(C, ggml_mul(C, g_f, cx), ggml_mul(C, g_i, g_g));
    ggml_tensor * hx2 = ggml_mul(C, g_o, ggml_tanh(C, cx2));

    ggml_tensor * t_logits = linear(W("actor_linear.weight"),  W("actor_linear.bias"),  hx2);
    ggml_tensor * t_value  = linear(W("critic_linear.weight"), W("critic_linear.bias"), hx2);

    ggml_cgraph * gf = ggml_new_graph(C);
    ggml_build_forward_expand(gf, t_logits);
    ggml_build_forward_expand(gf, t_value);    // 两个出口,必须分别注册
    ggml_build_forward_expand(gf, hx2);
    ggml_build_forward_expand(gf, cx2);
    ggml_graph_compute_with_ctx(C, gf, m.n_threads);

    if (logits) memcpy(logits, t_logits->data, N_ACTIONS * sizeof(float));
    if (value)  memcpy(value,  t_value->data,  sizeof(float));

    // 状态推进:拷回调用者持有的 iris_state
    memcpy(st.hx, hx2->data, LSTM_H * sizeof(float));
    memcpy(st.cx, cx2->data, LSTM_H * sizeof(float));

    ggml_free(C); C = nullptr;
}

// =================================================================
// KV Cache
// =================================================================

bool iris_kv_init(iris_kv_cache & kv){
    const int HS = EMBED / N_HEAD;

    ggml_init_params ip = { 32ull * 1024 * 1024, NULL, false};
    kv.ctx = ggml_init(ip);
    if (!kv.ctx) {fprintf(stderr,"[X] kv buf 分配失败\n"); return false;}

    for (int L = 0; L < N_LAYER; L++) {
    kv.k[L] = ggml_new_tensor_3d(kv.ctx,GGML_TYPE_F32,HS,MAX_TOK,N_HEAD);
    kv.v[L] = ggml_new_tensor_3d(kv.ctx,GGML_TYPE_F32,HS,MAX_TOK,N_HEAD);
    }
    kv.n_past = 0;
    printf("[iris] KV cache: %d 层 × 2 × [%d,%d,%d] = %.1f MB\n",
            N_LAYER,HS,MAX_TOK,N_HEAD,
            N_LAYER * 2.0 *HS * MAX_TOK * N_HEAD * 4 / 1048576.0);
    return true;
}

void iris_kv_free(iris_kv_cache & kv){
    if (kv.ctx) ggml_free(kv.ctx);
    kv = iris_kv_cache{};
}

void iris_wm_forward_cached(iris_model & m, iris_kv_cache & kv,
                            const int32_t * tokens, int n_new,
                            float * logits_obs, float * logits_rew, float * logits_end){
    const int HS = EMBED / N_HEAD;
    const float LN_EPS = 1e-5f;
    const int n_past = kv.n_past;
    const int n_tot = n_past + n_new;
    if (n_tot > MAX_TOK) {
        fprintf(stderr, "[X] n_past(%d) + n_new(%d) 超过%d\n", n_past,n_new,MAX_TOK);
        exit(1);
    }

    C = begin(m.buf_wm,m.buf_wm_size);
    WC =m.wctx_wm;

    // action位置重映射 -- 注意用 **绝对位置** n_past+i
    std::vector<int32_t> tok(tokens,tokens + n_new);
    for(int i =0; i<n_new; i++)
        if ((n_past + i ) % TPB == TPB -1 ) tok[i] += VOCAB;
    ggml_tensor * ids = ggml_new_tensor_1d(C,GGML_TYPE_I32,n_new);
    memcpy(ids->data,tok.data(),n_new * sizeof(int32_t));

    ggml_tensor * tbl = ggml_concat(C,W("embedder.embedding_tables.1.weight"),
                                      W("embedder.embedding_tables.0.weight"),1);
    ggml_tensor * x = ggml_get_rows(C,tbl,ids);

    // pos_emb从第n_past行开始取n_new行
    ggml_tensor * pos = W("pos_emb.weight");
    x = ggml_add(C,x,ggml_view_2d(C,pos,EMBED,n_new,
                                  pos->nb[1], (size_t)n_past * pos->nb[1]));
    ggml_cgraph * gf = ggml_new_graph_custom(C,8192,false);

    for(int L=0; L<N_LAYER; L++) {
        ggml_tensor *h = layer_norm(x,
                                    Wf("transformer.blocks.%d.ln1.weight",L),
                                    Wf("transformer.blocks.%d.ln1.bias",  L), LN_EPS);
        ggml_tensor *q = linear(Wf("transformer.blocks.%d.attn.query.weight",L),
                                Wf("transformer.blocks.%d.attn.query.bias",  L),h);
        ggml_tensor *k = linear(Wf("transformer.blocks.%d.attn.key.weight",  L),
                                Wf("transformer.blocks.%d.attn.key.bias",    L),h);
        ggml_tensor *v = linear(Wf("transformer.blocks.%d.attn.value.weight",L),
                                Wf("transformer.blocks.%d.attn.value.bias",  L),h);
        //---- 新K/V 写进cache的第n_past个位置 ----
        ggml_tensor *k_new = ggml_permute(C,ggml_reshape_3d(C,k,HS,N_HEAD,n_new),0,2,1,3);
        ggml_tensor *v_new = ggml_permute(C,ggml_reshape_3d(C,v,HS,N_HEAD,n_new),0,2,1,3);

        ggml_tensor * k_dst = ggml_view_3d(C,kv.k[L],HS,n_new,N_HEAD,
                                  kv.k[L]->nb[1],kv.k[L]->nb[2],
                                  (size_t)n_past * kv.k[L]->nb[1]);
        ggml_tensor * v_dst = ggml_view_3d(C,kv.v[L],HS,n_new,N_HEAD,
                                  kv.v[L]->nb[1],kv.v[L]->nb[2],
                                  (size_t)n_past * kv.v[L]->nb[1]);

        // 先注册写入，保证它在下面的读取之前执行
        ggml_build_forward_expand(gf,ggml_cpy(C,k_new,k_dst));
        ggml_build_forward_expand(gf,ggml_cpy(C,v_new,v_dst));

        // -------- 读 cache里全部n_tot个 --------
        ggml_tensor * K = ggml_view_3d(C,kv.k[L],HS,n_tot,N_HEAD,
                                 kv.k[L]->nb[1],kv.k[L]->nb[2],0);
        ggml_tensor * Vc = ggml_view_3d(C,kv.v[L],HS,n_tot,N_HEAD,
                                 kv.v[L]->nb[1],kv.v[L]->nb[2],0);
        ggml_tensor * Q = ggml_permute(C,ggml_reshape_3d(C,q,HS,N_HEAD,n_new),0,2,1,3);

        ggml_tensor * KQ = ggml_scale(C,ggml_mul_mat(C,K,Q),         //[n_tot,n_new,NH]
                                      1.0f / sqrtf((float)HS));
        KQ = ggml_soft_max(C,ggml_diag_mask_inf(C,KQ,n_past));       //<- n_past 不是0

        ggml_tensor * V = ggml_cont(C,ggml_permute(C,Vc,1,0,2,3));   //[n_tot,HS,NH]
        ggml_tensor * KQV = ggml_mul_mat(C,V,KQ);                    //[HS,n_new,NH]
        ggml_tensor * y = ggml_cont_2d(C,ggml_permute(C,KQV,0,2,1,3),EMBED,n_new);

        y = linear(Wf("transformer.blocks.%d.attn.proj.weight",L),
                   Wf("transformer.blocks.%d.attn.proj.bias",  L),y);
        x = ggml_add(C,x,y);

        ggml_tensor * mm = layer_norm(x,
                               Wf("transformer.blocks.%d.ln2.weight",L),
                               Wf("transformer.blocks.%d.ln2.bias",  L),LN_EPS);
        mm = linear(Wf("transformer.blocks.%d.mlp.0.weight",L),
                    Wf("transformer.blocks.%d.mlp.0.bias",L),mm);
        mm = ggml_gelu_erf(C,mm);
        mm = linear(Wf("transformer.blocks.%d.mlp.2.weight",L),
                    Wf("transformer.blocks.%d.mlp.2.bias",  L),mm);
        x = ggml_add(C,x,mm);
    }

    x = layer_norm(x,W("transformer.ln_f.weight"),W("transformer.ln_f.bias"),LN_EPS);

    // ---- 只取最后一个位置，三个head都算------
    ggml_tensor * last = ggml_cont(C,ggml_view_2d(C,x,EMBED,1,
                               x->nb[1],(size_t)(n_new -1) * x->nb[1]));

    const char * hname[3] ={"head_observations","head_rewards","head_ends"};
    ggml_tensor * head[3];
    char nm[192];
    for(int i=0;i<3;i++){
        snprintf(nm,sizeof(nm),"%s.head_module.0.weight",hname[i]);
        ggml_tensor * w0 = W(nm);
        snprintf(nm,sizeof(nm),"%s.head_module.0.bias",  hname[i]);
        ggml_tensor * b0 = W(nm);
        snprintf(nm,sizeof(nm),"%s.head_module.2.weight",hname[i]);
        ggml_tensor * w1 = W(nm);
        snprintf(nm,sizeof(nm),"%s.head_module.2.bias",  hname[i]);
        ggml_tensor * b1 = W(nm);
        head[i] = linear(w1,b1,ggml_relu(C,linear(w0,b0,last)));
    }

    for(int i=0;i<3;i++) ggml_build_forward_expand(gf,head[i]);
    ggml_graph_compute_with_ctx(C,gf,m.n_threads);

    if (logits_obs) memcpy(logits_obs, head[0]->data,VOCAB * sizeof(float));
    if (logits_rew) memcpy(logits_rew, head[1]->data,3 * sizeof(float));
    if (logits_end) memcpy(logits_end, head[2]->data,2 * sizeof(float));

    kv.n_past = n_tot;
    ggml_free(C);C = nullptr;
}


