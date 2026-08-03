// run_iris.cpp — 真实循环(actor 模式),trace 驱动
//
// 编译:
//   g++ -std=c++17 -O2 run_iris.cpp iris.cpp //     -I /home/lhao16/ggml/include -L /home/lhao16/ggml/build/src //     -lggml -lggml-base -lggml-cpu -Wl,-rpath,/home/lhao16/ggml/build/src //     -lm -lpthread -o run_iris && ./run_iris 200

#include "iris.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

using namespace iris;
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// uint8 HWC [0,255]  →  float CHW [-1,1]
// 复现 torchvision to_tensor(/255 + HWC→CHW) 再 mul(2).sub(1)
static void frame_to_input(const uint8_t * src, float * dst) {
    for (int c = 0; c < 3; c++)
        for (int h = 0; h < 64; h++)
            for (int w = 0; w < 64; w++) {
                float v = src[(h*64 + w)*3 + c] / 255.0f;   // HWC → [0,1]
                dst[c*64*64 + h*64 + w] = v * 2.0f - 1.0f;  // → [-1,1]
            }
}

// decode 输出 [-1,1] → postprocess (y+1)/2 → clamp(0,1) → actor 的 *2-1
// 等价于 clamp(rec,-1,1),按原链条写以便对照 Python
static void rec_to_ac_input(const float * rec, float * dst) {
    for (int i = 0; i < FRAME_N; i++) {
        float v = (rec[i] + 1.0f) * 0.5f;
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        dst[i] = v * 2.0f - 1.0f;
    }
}

int main(int argc, char ** argv) {
    const int N = (argc > 1) ? atoi(argv[1]) : 200;
    const int NT = (argc > 2) ? atoi(argv[2]) : 8;

    // ---- 读 trace ----
    FILE * f = fopen("trace/frames.bin", "rb");
    if (!f) { fprintf(stderr, "[X] 打不开 trace/frames.bin\n"); return 1; }
    fseek(f, 0, SEEK_END);
    const long total = ftell(f) / (64*64*3);
    fseek(f, 0, SEEK_SET);
    printf("[*] trace 共 %ld 帧,本次跑 %d 帧\n", total, N);
    if (N > total) { fprintf(stderr, "[X] 帧数不够\n"); return 1; }

    std::vector<uint8_t> frames((size_t)N * 64*64*3);
    if (fread(frames.data(), 1, frames.size(), f) != frames.size()) return 1;
    fclose(f);

    // ---- 载模型 ----
    iris_model m;
    if (!iris_load(m, ".")) return 1;
    m.n_threads = NT;
    printf("[*] n_threads = %d\n", m.n_threads);

    iris_state st;                   // LSTM 状态,跨步存活
    std::vector<float> x(FRAME_N), rec(FRAME_N), ac_in(FRAME_N);
    int32_t toks[N_TOKENS];
    float logits[N_ACTIONS], value;

    std::vector<int32_t> actions(N);
    double t_enc = 0, t_dec = 0, t_ac = 0;
    int action_hist[N_ACTIONS] = {0};

    auto t_all = Clock::now();
    for (int step = 0; step < N; step++) {
        frame_to_input(&frames[(size_t)step * 64*64*3], x.data());

        auto t0 = Clock::now();
        iris_tok_encode(m, x.data(), toks);          t_enc += ms_since(t0);

        t0 = Clock::now();
        iris_tok_decode(m, toks, rec.data());        t_dec += ms_since(t0);

        rec_to_ac_input(rec.data(), ac_in.data());

        t0 = Clock::now();
        iris_ac_forward(m, ac_in.data(), st, logits, &value);   t_ac += ms_since(t0);

        // argmax 而非采样 —— 量化对比需要确定性
        int a = 0;
        for (int i = 1; i < N_ACTIONS; i++) if (logits[i] > logits[a]) a = i;
        actions[step] = a;
        action_hist[a]++;

        if (step < 5 || step % 50 == 0) {
            printf("  step %4d  tok=", step);
            for (int i = 0; i < 16; i++) printf("%d ", toks[i]);
            printf(" a=%d\n", a);
        }
    }
    const double t_total = ms_since(t_all);

    // ---- 存动作序列,供量化对比 ----
    FILE * fo = fopen("trace/actions_fp32.bin", "wb");
    if (fo) { fwrite(actions.data(), sizeof(int32_t), N, fo); fclose(fo); }

    printf("\n[*] 延迟(每帧均值):\n");
    printf("    encode      %7.2f ms\n", t_enc / N);
    printf("    decode      %7.2f ms\n", t_dec / N);
    printf("    actor_critic%7.2f ms\n", t_ac  / N);
    printf("    ----------------------\n");
    printf("    合计        %7.2f ms   (%.1f fps)\n", t_total / N, 1000.0 * N / t_total);
    printf("\n[*] 动作分布: ");
    for (int i = 0; i < N_ACTIONS; i++) printf("%d:%d ", i, action_hist[i]);
    printf("\n[*] 动作序列已写入 trace/actions_fp32.bin\n");

    {
        int32_t wt[34];
        for (int i = 0; i<34;i++) wt[i] = (i % 17 == 16) ? 1:toks [i%16];
        std::vector<float> lo(32*512), lr(6),le(4);
        int no,na;
        auto t0 = Clock::now();
        for (int r=0;r<10;r++)
            iris_wm_forward(m,wt,34,lo.data(),&no,lr.data(),le.data(),&na);
        printf("\n[*] wm_forward(T=34):%.2f ms (10次均值)\n", ms_since(t0)/10);
    }

    iris_free(m);
    return 0;
}
