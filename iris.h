// iris.h — IRIS ggml runner: 三个组件的统一接口
//
// 设计要点:
//   1. 权重加载一次(iris_load),之后常驻
//   2. 每个 forward 函数内部自己建图 + 执行 + 拷出结果,调用者不碰 ggml
//   3. 计算内存池在 load 时一次性 malloc,每步复用,不重复申请
//   4. LSTM 状态(hx/cx)存在 iris_state 里,由调用者持有并跨步传递
//
// 所有帧数据都是 **已预处理** 的([-1,1] 区间,即 x*2-1 之后),
// 布局是 PyTorch NCHW contiguous = ggml [W,H,C,N],3*64*64 个 float。

#pragma once

#include <cstdint>
#include <cstddef>

struct ggml_context;
struct gguf_context;

// ---------------- 常量 ----------------
namespace iris {
    constexpr int FRAME_C   = 3;
    constexpr int FRAME_HW  = 64;
    constexpr int FRAME_N   = FRAME_C * FRAME_HW * FRAME_HW;   // 12288

    constexpr int N_TOKENS  = 16;      // 每帧 16 个 obs token
    constexpr int VOCAB     = 512;     // 码本大小 / obs 词表
    constexpr int N_ACTIONS = 4;       // Breakout 动作数

    constexpr int TPB       = 17;      // tokens per block (16 obs + 1 action)
    constexpr int MAX_BLK   = 20;
    constexpr int MAX_TOK   = TPB * MAX_BLK;   // 340

    constexpr int EMBED     = 256;
    constexpr int N_HEAD    = 4;
    constexpr int N_LAYER   = 10;

    constexpr int LSTM_H    = 512;
}

// ---------------- 模型(权重 + 内存池)----------------
struct iris_model {
    // 权重上下文,三个组件各一个
    ggml_context * wctx_tok = nullptr;
    ggml_context * wctx_wm  = nullptr;
    ggml_context * wctx_ac  = nullptr;

    gguf_context * gg_tok = nullptr;
    gguf_context * gg_wm  = nullptr;
    gguf_context * gg_ac  = nullptr;

    // 计算内存池:load 时 malloc 一次,每次 forward 复用
    void * buf_tok = nullptr;  size_t buf_tok_size = 0;
    void * buf_wm  = nullptr;  size_t buf_wm_size  = 0;
    void * buf_ac  = nullptr;  size_t buf_ac_size  = 0;

    float gn_eps   = 1e-6f;    // 从 tokenizer gguf 的 metadata 读
    int   n_threads = 1;       // 调试用 1(累加顺序确定),上板后调大
};

// ---------------- actor_critic 的循环状态 ----------------
struct iris_state {
    float hx[iris::LSTM_H] = {0};
    float cx[iris::LSTM_H] = {0};
};

// ---------------- 生命周期 ----------------
// dir 是放三个 .gguf 的目录,比如 "." 或 "/home/lhao16/iris/gguf"
bool iris_load(iris_model & m, const char * dir);
void iris_free(iris_model & m);

// ---------------- 前向 ----------------

// 帧 → 16 个 token
//   frame:  FRAME_N 个 float,已预处理
//   tokens: 输出 N_TOKENS 个 int32
void iris_tok_encode(iris_model & m, const float * frame, int32_t * tokens);

// 16 个 token → 重建帧
//   frame_out: FRAME_N 个 float,[-1,1] 区间(未做 postprocess)
void iris_tok_decode(iris_model & m, const int32_t * tokens, float * frame_out);

// world_model 完整前向
//   tokens: T 个 token,obs 位置取值 [0,512),action 位置取值 [0,4)
//           (函数内部会把 action 位置重映射到 512+a)
//   T:      序列长度,必须 <= MAX_TOK
//   输出三个 head 的 logits,按位置紧密排列;n_* 返回实际位置数
//     logits_obs  : n_obs * VOCAB      位置满足 p%17 != 15
//     logits_rew  : n_act * 3          位置满足 p%17 == 16
//     logits_end  : n_act * 2          同上
//   任何一个输出指针传 nullptr 表示不需要(仍然会算,只是不拷贝)
void iris_wm_forward(iris_model & m,
                     const int32_t * tokens, int T,
                     float * logits_obs, int * n_obs,
                     float * logits_rew,
                     float * logits_end, int * n_act);

// actor_critic 前向,**会推进 LSTM 状态**
//   frame:  FRAME_N 个 float,已预处理
//   st:     入参是上一步的 hx/cx,函数返回时被更新为新的 hx/cx
//   logits: 输出 N_ACTIONS 个 float
//   value:  输出 1 个 float
void iris_ac_forward(iris_model & m, const float * frame,
                     iris_state & st, float * logits, float * value);

