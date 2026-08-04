//test_iris.cpp — 重构回归测试
//
// 目的:证明 iris.cpp 抽出来的四个函数,和三个 test_*.cpp 算出同样的结果。
// 用的是同一批 ref/*.bin,所以任何重构错误都会立刻暴露。
//
// 编译:
//   g++ -std=c++17 test_iris.cpp iris.cpp //     -I /home/lhao16/ggml/include //     -L /home/lhao16/ggml/build/src //     -lggml -lggml-base -lggml-cpu //     -Wl,-rpath,/home/lhao16/ggml/build/src //     -lm -lpthread -o test_iris && ./test_iris

#include "iris.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

using namespace iris;

static int g_fail = 0;

static std::vector<float> read_f32(const char * path, size_t n) {
    std::vector<float> v(n);
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[X] 打不开 %s\n", path); exit(1); }
    if (fread(v.data(), sizeof(float), n, f) != n) {
        fprintf(stderr, "[X] %s 读取不足\n", path); exit(1); }
    fclose(f);
    return v;
}

static void cmp(const char * name, const float * got, const char * ref_path, size_t n) {
    std::vector<float> ref = read_f32(ref_path, n);
    double maxd = 0, amax = 0;
    for (size_t i = 0; i < n; i++) {
        maxd = fmax(maxd, fabs((double)got[i] - (double)ref[i]));
        amax = fmax(amax, fabs((double)ref[i]));
    }
    const double tol = 1e-5 + 1e-6 * amax;
    const double rel = amax > 0 ? maxd / amax : 0;
    const bool ok = maxd <= tol;
    if (!ok) g_fail++;
    printf("  %-16s n=%-7zu maxdiff=%.3e rel=%.3e  %s\n",
           name, n, maxd, rel, ok ? "PASS" : "FAIL");
}

int main() {
    iris_model m;
    if (!iris_load(m, ".")) return 1;
    m.n_threads = 8;
    printf("[iris] n_threads = %d\n", m.n_threads);

    // ---------- 1. tokenizer encode ----------
    // ref/tok_input.bin 已经是 mul(2).sub(1) 之后的
    printf("\n[1] tokenizer encode:\n");
    std::vector<float> tok_in = read_f32("ref/tok_input.bin", FRAME_N);
    int32_t toks[N_TOKENS];
    iris_tok_encode(m, tok_in.data(), toks);

    int32_t toks_ref[N_TOKENS];
    { FILE * f = fopen("ref/tok_tokens.bin", "rb");
      if (!f || fread(toks_ref, sizeof(int32_t), N_TOKENS, f) != N_TOKENS) {
          fprintf(stderr, "[X] 读不了 ref/tok_tokens.bin\n"); return 1; }
      fclose(f); }

    int bad = 0;
    printf("  ggml : ");  for (int i = 0; i < N_TOKENS; i++) {
        printf("%d ", toks[i]); if (toks[i] != toks_ref[i]) bad++; }
    printf("\n  torch: ");  for (int i = 0; i < N_TOKENS; i++) printf("%d ", toks_ref[i]);
    printf("\n  ==> %s (%d/16)\n", bad == 0 ? "PASS" : "FAIL", N_TOKENS - bad);
    if (bad) g_fail++;

    // ---------- 2. tokenizer decode ----------
    printf("\n[2] tokenizer decode:\n");
    std::vector<float> rec(FRAME_N);
    iris_tok_decode(m, toks, rec.data());
    cmp("rec", rec.data(), "ref/tok_rec.bin", FRAME_N);

    // ---------- 3. actor_critic ----------
    printf("\n[3] actor_critic:\n");
    std::vector<float> ac_in = read_f32("ref/inp.bin", FRAME_N);
    for (auto & v : ac_in) v = v * 2.0f - 1.0f;      // dump_ac.py 存的是原始 [0,1)

    iris_state st;
    std::vector<float> hx0 = read_f32("ref/hx.bin", LSTM_H);
    std::vector<float> cx0 = read_f32("ref/cx.bin", LSTM_H);
    memcpy(st.hx, hx0.data(), LSTM_H * sizeof(float));
    memcpy(st.cx, cx0.data(), LSTM_H * sizeof(float));

    float logits[N_ACTIONS], value;
    iris_ac_forward(m, ac_in.data(), st, logits, &value);

    cmp("logits", logits, "ref/logits.bin", N_ACTIONS);
    cmp("value",  &value, "ref/value.bin",  1);
    cmp("hx_out", st.hx,  "ref/hx_out.bin", LSTM_H);
    cmp("cx_out", st.cx,  "ref/cx_out.bin", LSTM_H);

    // ---------- 4. world_model ----------
    printf("\n[4] world_model:\n");
    const int T = 34;
    int32_t wm_tok[T];
    { FILE * f = fopen("ref/wm_tokens.bin", "rb");
      if (!f || fread(wm_tok, sizeof(int32_t), T, f) != (size_t)T) {
          fprintf(stderr, "[X] 读不了 ref/wm_tokens.bin\n"); return 1; }
      fclose(f); }

    std::vector<float> lo(32 * VOCAB), lr(2 * 3), le(2 * 2);
    int n_obs = 0, n_act = 0;
    iris_wm_forward(m, wm_tok, T, lo.data(), &n_obs, lr.data(), le.data(), &n_act);

    printf("  slice: obs %d, act %d  (期望 32 / 2)\n", n_obs, n_act);
    if (n_obs != 32 || n_act != 2) g_fail++;

    cmp("logits_obs",     lo.data(), "ref/wm_logits_obs.bin",     32 * VOCAB);
    cmp("logits_rewards", lr.data(), "ref/wm_logits_rewards.bin", 2 * 3);
    cmp("logits_ends",    le.data(), "ref/wm_logits_ends.bin",    2 * 2);

    // argmax 一致性 —— 比浮点容差更有意义
    {
        std::vector<float> ref = read_f32("ref/wm_logits_obs.bin", 32 * VOCAB);
        int nbad = 0;
        for (int p = 0; p < 32; p++) {
            int ag = 0, ar = 0;
            for (int j = 1; j < VOCAB; j++) {
                if (lo [p*VOCAB+j] > lo [p*VOCAB+ag]) ag = j;
                if (ref[p*VOCAB+j] > ref[p*VOCAB+ar]) ar = j;
            }
            if (ag != ar) nbad++;
        }
        printf("  argmax 一致: %d/32\n", 32 - nbad);
        if (nbad) g_fail++;
    }

    // ----------5.KV cache-------------
    printf("\n[5] KV cache:\n");

    //诊断：此刻堆还健康吗？
    void * probe = malloc(32ull*1024*1024);
    printf(" [probe] 32MB malloc: %s\n", probe ? "OK" : "FAILED");
    free(probe);

    iris_kv_cache kv;
    if(!iris_kv_init(kv)) return 1;

    // 全量前向的最后一个 obs位置是33，对应logits_obs的第31行
    const float * ref_last = lo.data()+31*VOCAB;

    //(a)一次喂34个
    std::vector<float> c1(VOCAB);
    iris_kv_reset(kv);
    iris_wm_forward_cached(m,kv,wm_tok,34,c1.data(),nullptr,nullptr);
    {
        double d = 0, a = 0;
        for(int i = 0 ;i<VOCAB; i++){
            d = fmax(d,fabs(c1[i] - ref_last[i])); a = fmax(a,fabs(ref_last[i]));
        }
        printf(" 一次喂34个：maxdiff=%.3e rel=%.3e %s\n", d,d/a,d<=1e-5+1e-6 ? "PASS" : "FAIL");
        if (d>1e-5 +1e-6*a) g_fail++;
    }

    //（b）逐个喂34次--- 这才是想象循环的实际用法
    std::vector<float> c2(VOCAB);
    iris_kv_reset(kv);
    for(int i=0;i<34;i++)
        iris_wm_forward_cached(m,kv,&wm_tok[i],1,c2.data(),nullptr,nullptr);
    {
        double d = 0, a = 0;
        for(int i = 0; i< VOCAB; i++){
            d = fmax(d,fabs(c2[i] - ref_last[i])); a = fmax(a,fabs(ref_last[i]));
        }
        printf(" 逐个喂34次：maxdiff=%.3e ref=%.3e %s\n", d, d/a,
                 d <= 1e-5 + 1e-6 *a ? "PASS" : "FAIL");
        if (d > 1e-5 + 1e-6 * a) g_fail++;
    }

    //argmax 一致性--最有意义的判据
    {
        int ag = 0, ar =0;
        for(int j = 1; j<VOCAB; j++){
            if (c2[j] > c2 [ag]) ag =j;
            if (ref_last[j] > ref_last[ar]) ar = j;
        }
        printf("  argmax: cache = %d full = %d %s\n", ag,ar,ag == ar ? "PASS":"FAIL");
        if (ag != ar) g_fail ++;
    }
    iris_kv_free(kv);


    printf("\n==> %s\n", g_fail == 0 ? "ALL PASS" : "FAIL");
    iris_free(m);
    return g_fail;
}


