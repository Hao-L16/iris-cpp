// iris_plan.h — 想象循环(planner)
//
// 结构对应 Python 侧 test_plan_clean.py 的 plan() v5:
//   K=4 条直线轨迹,只有 t=0 枚举全部动作,t>=1 由 actor 决定后续动作
//   score = Σ γ^t · alive · E[r_t] + γ^H · alive · V(s_H)
//   E[r]  = P(+1) − P(−1),来自 reward head 的 3 类 softmax
//   alive = Π (1 − P(done))
//
// ⚠️ 与 Python 的一处有意偏离:后续动作用 argmax 而不是温度采样。
//    量化对比需要确定性 —— 采样的随机性会淹没量化的影响。

#pragma once

#include "iris.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

namespace plan {

// ⚠️ 这两个值必须和 test_plan_clean.py 一致,跑之前去核对
constexpr int   H             = 5;      // 想象步数
constexpr float GAMMA         = 0.995f; // 折扣因子
constexpr float SPREAD_THRESH = 1.5f;   // gate 门槛(top − runner-up)

// 就地 softmax
inline void softmax(float * x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

inline int argmax(const float * x, int n) {
    int a = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[a]) a = i;
    return a;
}

// decode 输出 [-1,1] → postprocess (y+1)/2 → clamp(0,1) → actor 的 *2-1
// 数学上等于 clamp(rec,-1,1),按原链条写以便和 Python 对照
inline void rec_to_ac(const float * rec, float * dst) {
    for (int i = 0; i < iris::FRAME_N; i++) {
        float v = (rec[i] + 1.0f) * 0.5f;
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        dst[i] = v * 2.0f - 1.0f;
    }
}

// 计时用的累加器,便于分析瓶颈
struct plan_stats {
    double t_wm = 0, t_dec = 0, t_ac = 0;
    long   n_wm = 0, n_dec = 0, n_ac = 0;
};

// ---------------------------------------------------------------
//  对 4 个候选首动作各想象 H 步,返回 4 个分数
//
//  obs_tokens : 当前真实帧的 16 个 obs token
//  st_real    : 真实 LSTM 状态(只读,每个分支各拷一份,真实状态不被污染)
//  kv         : 4 个 KV cache,函数内部会 reset
//  scores_out : 输出 4 个分数
// ---------------------------------------------------------------
inline void run(iris_model & m,
                iris_kv_cache * kv,
                const iris_state & st_real,
                const int32_t * obs_tokens,
                float * scores_out,
                plan_stats * stats = nullptr)
{
    using namespace iris;
    using Clock = std::chrono::steady_clock;
    auto ms = [](Clock::time_point t) {
        return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
    };

    std::vector<float> lo(VOCAB), lr(3), le(2);
    std::vector<float> rec(FRAME_N), ac_in(FRAME_N);
    float ac_logits[N_ACTIONS], V;
    int32_t seq[TPB];              // 17: 16 obs + 1 action
    int32_t obs[N_TOKENS];

    for (int a = 0; a < N_ACTIONS; a++) {
        iris_kv_reset(kv[a]);
        iris_state st = st_real;            // 分支各自的 LSTM 副本

        // ---- 喂入 [16 obs] + [候选动作 a] ----
        memcpy(seq, obs_tokens, N_TOKENS * sizeof(int32_t));
        seq[N_TOKENS] = a;

        auto t0 = Clock::now();
        iris_wm_forward_cached(m, kv[a], seq, TPB, lo.data(), lr.data(), le.data());
        if (stats) { stats->t_wm += ms(t0); stats->n_wm++; }

        // 此刻最后一个位置是动作 token,它预测:
        //   lo → 下一帧的第 0 个 obs token
        //   lr → 这一步的 reward,  le → 这一步的 done

        float score = 0.0f, alive = 1.0f, g = 1.0f;

        for (int t = 0; t < H; t++) {
            // ---- 结算这一步的 reward / done ----
            float r3[3] = { lr[0], lr[1], lr[2] };
            softmax(r3, 3);
            const float E_r = r3[2] - r3[0];          // 类别是 {-1, 0, +1}

            float e2[2] = { le[0], le[1] };
            softmax(e2, 2);
            const float p_done = e2[1];

            score += g * alive * E_r;
            alive *= (1.0f - p_done);
            g     *= GAMMA;

            // ---- 自回归生成 16 个 obs token ----
            obs[0] = (int32_t) argmax(lo.data(), VOCAB);
            for (int k = 1; k < N_TOKENS; k++) {
                t0 = Clock::now();
                iris_wm_forward_cached(m, kv[a], &obs[k-1], 1, lo.data(), nullptr, nullptr);
                if (stats) { stats->t_wm += ms(t0); stats->n_wm++; }
                obs[k] = (int32_t) argmax(lo.data(), VOCAB);
            }
            // 把最后一个 obs token 也喂进去,让 cache 位置推进到动作槽
            t0 = Clock::now();
            iris_wm_forward_cached(m, kv[a], &obs[N_TOKENS-1], 1, nullptr, nullptr, nullptr);
            if (stats) { stats->t_wm += ms(t0); stats->n_wm++; }

            // ---- 解码成画面,喂给 actor ----
            t0 = Clock::now();
            iris_tok_decode(m, obs, rec.data());
            if (stats) { stats->t_dec += ms(t0); stats->n_dec++; }

            rec_to_ac(rec.data(), ac_in.data());

            t0 = Clock::now();
            iris_ac_forward(m, ac_in.data(), st, ac_logits, &V);
            if (stats) { stats->t_ac += ms(t0); stats->n_ac++; }

            // ---- 决定下一个想象动作,喂进 cache ----
            int32_t next_a = (int32_t) argmax(ac_logits, N_ACTIONS);
            t0 = Clock::now();
            iris_wm_forward_cached(m, kv[a], &next_a, 1, lo.data(), lr.data(), le.data());
            if (stats) { stats->t_wm += ms(t0); stats->n_wm++; }
        }

        // ---- 终值:γ^H · alive · V(s_H) ----
        score += g * alive * V;
        scores_out[a] = score;
    }
}

// gate 判据:planner 最高分 − 次高分 > 门槛才接管
// 注意衡量的是 planner **对自己**的确定程度,actor 的选择不参与比较
inline bool gate_fires(const float * scores, float * spread_out = nullptr) {
    int i1 = 0;
    for (int i = 1; i < iris::N_ACTIONS; i++) if (scores[i] > scores[i1]) i1 = i;
    float second = -1e30f;
    for (int i = 0; i < iris::N_ACTIONS; i++)
        if (i != i1 && scores[i] > second) second = scores[i];
    const float spread = scores[i1] - second;
    if (spread_out) *spread_out = spread;
    return spread > SPREAD_THRESH;
}

} // namespace plan
