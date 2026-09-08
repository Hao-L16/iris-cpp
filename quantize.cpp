// quantize.cpp
//   ./quantize <in.gguf> <out.gguf> <type> [name_filter]
//   type: f16 q8_0 q6_k q5_k q5_0 q4_k q4_0 q3_k q2_k
//   name_filter: 只量化名字含该子串的张量（E3 用，如 "blocks.3."）
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

static ggml_type parse_type(const char * s) {
    if (!strcmp(s,"f16"))  return GGML_TYPE_F16;
    if (!strcmp(s,"q8_0")) return GGML_TYPE_Q8_0;
    if (!strcmp(s,"q6_k")) return GGML_TYPE_Q6_K;
    if (!strcmp(s,"q5_k")) return GGML_TYPE_Q5_K;
    if (!strcmp(s,"q5_0")) return GGML_TYPE_Q5_0;
    if (!strcmp(s,"q4_k")) return GGML_TYPE_Q4_K;
    if (!strcmp(s,"q4_0")) return GGML_TYPE_Q4_0;
    if (!strcmp(s,"q3_k")) return GGML_TYPE_Q3_K;
    if (!strcmp(s,"q2_k")) return GGML_TYPE_Q2_K;
    return GGML_TYPE_COUNT;
}

// 机制性排除：这些张量不进 mul_mat，或不该动
static bool excluded(const char * n) {
    if (strstr(n, "pos_emb"))                       return true; // view_2d + add
    if (strstr(n, "embedding_tables"))              return true; // concat + get_rows
    if (!strcmp(n, "embedding.weight"))             return true; // VQ-VAE 码本
    if (strstr(n, "head_rewards.head_module.2"))    return true; // -> planner 打分
    if (strstr(n, "head_ends.head_module.2"))       return true; // -> gate P(done)
    if (strstr(n, "actor_linear"))                  return true; // -> 动作
    if (strstr(n, "critic_linear"))                 return true;
    return false;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <in.gguf> <out.gguf> <type> [name_filter]\n", argv[0]);
        return 1;
    }
    const char * fin    = argv[1];
    const char * fout   = argv[2];
    const ggml_type qt  = parse_type(argv[3]);
    const char * filter = (argc > 4) ? argv[4] : nullptr;

    if (qt == GGML_TYPE_COUNT) { fprintf(stderr, "bad type: %s\n", argv[3]); return 1; }

    ggml_quantize_init(qt);

    // ---- 读入 ----
    ggml_context * ctx_in = nullptr;
    gguf_init_params p = { /*no_alloc*/ false, /*ctx*/ &ctx_in };
    gguf_context * g_in = gguf_init_from_file(fin, p);
    if (!g_in) { fprintf(stderr, "open failed: %s\n", fin); return 1; }

    const int64_t nt = gguf_get_n_tensors(g_in);

    // ---- 输出 ctx：用 F32 大小作上界 ----
    size_t mem = 1u << 20;
    for (int64_t i = 0; i < nt; i++)
        mem += ggml_nbytes(ggml_get_tensor(ctx_in, gguf_get_tensor_name(g_in, i)))
             + ggml_tensor_overhead();

    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx_out = ggml_init(ip);

    gguf_context * g_out = gguf_init_empty();
    gguf_set_kv(g_out, g_in);   // 元数据原样搬过去

    const int64_t blck = ggml_blck_size(qt);
    const ggml_type_traits * tt = ggml_get_type_traits(qt);

    printf("%-44s %-6s %9s %9s %10s\n", "name", "type", "in MB", "out MB", "rel err");

    double mb_in = 0, mb_out = 0;
    int n_q = 0, n_keep = 0;
    double se_tot = 0, s0_tot = 0;

    for (int64_t i = 0; i < nt; i++) {
        const char * name = gguf_get_tensor_name(g_in, i);
        ggml_tensor * src = ggml_get_tensor(ctx_in, name);

        bool q = ggml_n_dims(src) == 2
              && src->type == GGML_TYPE_F32
              && !excluded(name)
              && src->ne[0] % blck == 0
              && (!filter || strstr(name, filter));

        ggml_type ot = q ? qt : GGML_TYPE_F32;
        ggml_tensor * dst = ggml_new_tensor(ctx_out, ot, GGML_MAX_DIMS, src->ne);
        ggml_set_name(dst, name);

        double rel = -1.0;
        if (q) {
            const int64_t n_per_row = src->ne[0];
            const int64_t nrows     = ggml_nelements(src) / n_per_row;
            ggml_quantize_chunk(qt, (const float *) src->data, dst->data,
                                0, nrows, n_per_row, nullptr);

            // 反量化回来量一下相对误差（E3 要用）
            const int64_t n = ggml_nelements(src);
            float * back = (float *) malloc(n * sizeof(float));
            tt->to_float(dst->data, back, n);
            double se = 0, s0 = 0;
            const float * a = (const float *) src->data;
            for (int64_t k = 0; k < n; k++) {
                double d = a[k] - back[k];
                se += d * d;  s0 += (double) a[k] * a[k];
            }
            free(back);
            rel = sqrt(se / s0);
            se_tot += se; s0_tot += s0;
            n_q++;
        } else {
            memcpy(dst->data, src->data, ggml_nbytes(src));
            n_keep++;
        }

        gguf_add_tensor(g_out, dst);

        mb_in  += ggml_nbytes(src) / 1048576.0;
        mb_out += ggml_nbytes(dst) / 1048576.0;

        if (rel >= 0) printf("%-44s %-6s %9.3f %9.3f %10.3e\n",
                             name, ggml_type_name(ot),
                             ggml_nbytes(src)/1048576.0, ggml_nbytes(dst)/1048576.0, rel);
    }

    if (!gguf_write_to_file(g_out, fout, false)) {
        fprintf(stderr, "write failed: %s\n", fout); return 1;
    }

    printf("\nquantized %d, kept f32 %d\n", n_q, n_keep);
    printf("OVERALL rel = %.4f (weighted over %d quantised tensors)\n", sqrt(se_tot / s0_tot),ggml_type)name(qt),n_q);
    printf("%.2f MB -> %.2f MB  (%.2fx)\n", mb_in, mb_out, mb_in / mb_out);

    gguf_free(g_out); ggml_free(ctx_out);
    gguf_free(g_in);  ggml_free(ctx_in);
    return 0;
}
