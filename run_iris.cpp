// run_iris.cpp — 真实推理循环,trace 驱动
//
// 用法:
//   ./run_iris [帧数] [线程数] [模式]
//     模式:  actor  = 纯反应式(默认)
//            gate   = 高门槛想象接管
//
// 例:
//   ./run_iris 50 8 actor
//   ./run_iris 50 8 gate
//
// 编译:  make

#include "iris.h"
#include "iris_plan.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

using namespace iris;
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// uint8 HWC [0,255] → float CHW [-1,1]
// 复现 torchvision to_tensor(/255 + HWC→CHW) 再 mul(2).sub(1)
static void frame_to_input(const uint8_t * src, float * dst) {
    for (int c = 0; c < 3; c++)
        for (int h = 0; h < 64; h++)
            for (int w = 0; w < 64; w++) {
                float v = src[(h*64 + w)*3 + c] / 255.0f;
                dst[c*64*64 + h*64 + w] = v * 2.0f - 1.0f;
            }
}

int main(int argc, char ** argv) {
    const int N  = (argc > 1) ? atoi(argv[1]) : 200;
    const int NT = (argc > 2) ? atoi(argv[2]) : 8;
    const std::string MODE = (argc > 3) ? argv[3] : "actor";
    const std::string TAG = (argc >4 ) ? argv[4] : "fp32";

    if (MODE != "actor" && MODE != "gate") {
        fprintf(stderr, "[X] 模式只能是 actor 或 gate\n");
        return 1;
    }

    // ---- 读 trace ----
    FILE * f = fopen("trace/frames.bin", "rb");
    if (!f) { fprintf(stderr, "[X] 打不开 trace/frames.bin\n"); return 1; }
    fseek(f, 0, SEEK_END);
    const long total = ftell(f) / (64*64*3);
    fseek(f, 0, SEEK_SET);
    printf("[*] trace 共 %ld 帧,本次跑 %d 帧,模式 %s, 标签 %s\n", total, N, MODE.c_str(),TAG.c_str());
    if (N > total) { fprintf(stderr, "[X] 帧数不够\n"); fclose(f); return 1; }

    std::vector<uint8_t> frames((size_t)N * 64*64*3);
    if (fread(frames.data(), 1, frames.size(), f) != frames.size()) { fclose(f); return 1; }
    fclose(f);

    // ---- 载模型 ----
    iris_model m;
    if (!iris_load(m, ".")) return 1;
    m.n_threads = NT;
    printf("[*] n_threads = %d\n", m.n_threads);

    // gate 模式需要 4 个独立的 KV cache(每条分支一个)
    iris_kv_cache kv[N_ACTIONS];
    if (MODE == "gate")
        for (int a = 0; a < N_ACTIONS; a++)
            if (!iris_kv_init(kv[a])) return 1;

    iris_state st;                       // 真实 LSTM 状态,跨帧存活
    std::vector<float> x(FRAME_N), rec(FRAME_N), ac_in(FRAME_N);
    int32_t toks[N_TOKENS];
    float logits[N_ACTIONS], value, scores[N_ACTIONS];

    std::vector<int32_t> actions(N);
    double t_enc = 0, t_dec = 0, t_ac = 0, t_plan = 0;
    plan::plan_stats ps;
    int action_hist[N_ACTIONS] = {0};
    int n_gate_open = 0, n_changed = 0;

    auto t_all = Clock::now();
    for (int step = 0; step < N; step++) {
        frame_to_input(&frames[(size_t)step * 64*64*3], x.data());

        auto t0 = Clock::now();
        iris_tok_encode(m, x.data(), toks);            t_enc += ms_since(t0);

        t0 = Clock::now();
        iris_tok_decode(m, toks, rec.data());          t_dec += ms_since(t0);

        plan::rec_to_ac(rec.data(), ac_in.data());

        // 真实前向:每帧恰好推进一次 LSTM
        t0 = Clock::now();
        iris_ac_forward(m, ac_in.data(), st, logits, &value);   t_ac += ms_since(t0);

        const int actor_a = plan::argmax(logits, N_ACTIONS);
        int a = actor_a;
        float spread = 0.0f;

        if (MODE == "gate") {
            t0 = Clock::now();
            plan::run(m, kv, st, toks, scores, &ps);    // st 按值传,真实状态不被污染
            t_plan += ms_since(t0);

            if (plan::gate_fires(scores, &spread)) {
                a = plan::argmax(scores, N_ACTIONS);
                n_gate_open++;
                if (a != actor_a) n_changed++;          // 真正改变了动作的次数
            }
        }

        actions[step] = a;
        action_hist[a]++;

        if (step < 3 || step % 25 == 0) {
            printf("  step %4d  actor=%d", step, actor_a);
            if (MODE == "gate")
                printf("  scores=%.3f %.3f %.3f %.3f  spread=%.3f",
                       scores[0], scores[1], scores[2], scores[3], spread);
            printf("  -> a=%d\n", a);
        }
    }
    const double t_total = ms_since(t_all);

    // ---- 存动作序列,供量化对比 ----
    const std::string out = "trace/actions_" + TAG+ "_" + MODE + ".bin";
    FILE * fo = fopen(out.c_str(), "wb");
    if (fo) { fwrite(actions.data(), sizeof(int32_t), N, fo); fclose(fo); }

    printf("\n[*] 延迟(每帧均值):\n");
    printf("    encode        %8.2f ms\n", t_enc / N);
    printf("    decode        %8.2f ms\n", t_dec / N);
    printf("    actor_critic  %8.2f ms\n", t_ac  / N);
    if (MODE == "gate") {
        printf("    plan          %8.2f ms\n", t_plan / N);
        printf("      ├ world_model %6.2f ms  (%ld 次/帧)\n", ps.t_wm  / N, ps.n_wm  / N);
        printf("      ├ decode      %6.2f ms  (%ld 次/帧)\n", ps.t_dec / N, ps.n_dec / N);
        printf("      └ actor       %6.2f ms  (%ld 次/帧)\n", ps.t_ac  / N, ps.n_ac  / N);
    }
    printf("    ------------------------------\n");
    printf("    合计          %8.2f ms   (%.2f fps)\n", t_total / N, 1000.0 * N / t_total);

    printf("\n[*] 动作分布: ");
    for (int i = 0; i < N_ACTIONS; i++) printf("%d:%d ", i, action_hist[i]);
    printf("\n");

    if (MODE == "gate")
        printf("[*] gate 开启 %d/%d (%.1f%%),其中真正改变动作 %d 次 (%.1f%%)\n",
               n_gate_open, N, 100.0 * n_gate_open / N,
               n_changed, 100.0 * n_changed / N);

    printf("[*] 动作序列已写入 %s\n", out.c_str());

    if (MODE == "gate")
        for (int a = 0; a < N_ACTIONS; a++) iris_kv_free(kv[a]);
    iris_free(m);
    return 0;
}
