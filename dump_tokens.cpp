// dump_tokens.cpp — 只跑一次，把 2000 帧编码成 token 存盘
//   ./dump_tokens
#include "iris.h"
#include <cstdio>
#include <cstdint>
#include <vector>

// —— 直接复制自 run_iris.cpp ——
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
    const char * dir = (argc > 1) ? argv[1] : ".";

    iris_model m;
    if (!iris_load(m, dir)) return 1;
    m.n_threads = 8;                     // 只是编码，快就行

    FILE * f = fopen("trace/frames.bin", "rb");
    if (!f) { fprintf(stderr, "打不开 trace/frames.bin\n"); return 1; }
    FILE * o = fopen("trace/tokens.bin", "wb");
    if (!o) { fprintf(stderr, "打不开 trace/tokens.bin\n"); return 1; }

    const size_t FRAME_BYTES = 64*64*3;
    std::vector<uint8_t> raw(FRAME_BYTES);
    std::vector<float>   in(FRAME_BYTES);
    int32_t toks[16];

    int n = 0;
    while (fread(raw.data(), 1, FRAME_BYTES, f) == FRAME_BYTES) {
        frame_to_input(raw.data(), in.data());
        iris_tok_encode(m, in.data(), toks);
        fwrite(toks, sizeof(int32_t), 16, o);
        if (++n % 100 == 0) { printf("\r%d", n); fflush(stdout); }
    }

    printf("\n%d 帧 -> trace/tokens.bin (%d 字节)\n", n, n*16*4);
    fclose(f); fclose(o); iris_free(m);
    return 0;
}
