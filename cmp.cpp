// cmp.cpp — 比较两次 teacher-forced 回放的输出
//   ./cmp <ref.bin> <test.bin>
//
// 输入格式（由 tf.cpp 产生），每个 chunk：
//   320 行 × 12 字节 = { int32 argmax, int32 top2, float gap }
//   20 × 3 float   logits_rewards
//   20 × 2 float   logits_ends
//   合计 4240 字节 / chunk
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

static const int NB   = 20;                 // 每次前向的 block 数
static const int NOBS = NB * 16;            // 320 个观测预测
static const int NACT = NB;                 // 20 个动作位置
static const size_t CHUNK_BYTES = (size_t)NOBS*12 + (size_t)NACT*3*4 + (size_t)NACT*2*4;

// ---- reset 帧（来自 tokens.bin 扫描：与首帧完全相同的帧）----
static std::vector<int> RESET_F;            //从dones.bin读入
static const int RESET_PAD = 2;             // 前后各排除几帧

static bool near_reset(int frame) {
    for (int r : RESET_F)
        if (abs(frame - r) <= RESET_PAD) return true;
    return false;
}

struct Chunk {
    int32_t am [NOBS];      // argmax
    int32_t a2 [NOBS];      // 第二名的下标
    float   gap[NOBS];      // top1 - top2
    float   lr [NACT*3];
    float   le [NACT*2];
};

static bool read_chunk(FILE * f, Chunk & c) {
    for (int i = 0; i < NOBS; i++) {
        if (fread(&c.am [i], 4, 1, f) != 1) return false;
        if (fread(&c.a2 [i], 4, 1, f) != 1) return false;
        if (fread(&c.gap[i], 4, 1, f) != 1) return false;
    }
    if (fread(c.lr, 4, NACT*3, f) != (size_t)NACT*3) return false;
    if (fread(c.le, 4, NACT*2, f) != (size_t)NACT*2) return false;
    return true;
}

// head_rewards 3 类，假定顺序 [-1, 0, +1] —— 若 iris_plan.h 用的是别的下标，改这里
static float exp_r(const float * p) {
    float m  = std::max(std::max(p[0], p[1]), p[2]);
    float e0 = expf(p[0]-m), e1 = expf(p[1]-m), e2 = expf(p[2]-m);
    return (e2 - e0) / (e0 + e1 + e2);
}
// head_ends 2 类 -> P(done)
static float p_done(const float * p) {
    float m  = std::max(p[0], p[1]);
    float e0 = expf(p[0]-m), e1 = expf(p[1]-m);
    return e1 / (e0 + e1);
}

// 排序后取分位（调用前 v 必须已排序）
static float q(const std::vector<float> & v, double p) {
    if (v.empty()) return 0.0f;
    return v[(size_t)(p * (v.size() - 1))];
}
static double mean(const std::vector<float> & v) {
    if (v.empty()) return 0.0;
    double s = 0; for (float x : v) s += x; return s / v.size();
}

// 统一打印：先排序，再取值，避免参数求值顺序问题
static void report(const char * name, std::vector<float> & v) {
    if (v.empty()) { printf("%-14s (无数据)\n", name); return; }
    double mu = mean(v);                       // 排序前算，与顺序无关
    std::sort(v.begin(), v.end());
    printf("%-14s 平均 %.5f  中位 %.5f  p95 %.5f  最大 %.5f\n",
           name, mu, q(v,0.50), q(v,0.95), v.back());
    int c05 = 0,c10 = 0,c25=0;
    for(float x:v) {if (x>0.05f) c05++; if (x > 0.10f) c10++; if (x>0.25f) c25++;}
    printf("               >0.05:%4d        >0.10:%4d       >0.25:%4d    (共 %zu)\n",
           c05,c10,c25,v.size());
}

static long fsize(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    return n;
}

static void load_dones(const char * path) {
    FILE * f = fopen(path,"rb");
    if (!f) {fprintf(stderr,"[!] 打不开 %s, 跳过reset 统计\n",path); return;}
    int32_t d;
    for( int i = 0; fread(&d,4,1,f) == 1; i++)
        if(d) RESET_F.push_back(i+1);
    fclose(f);
    printf("[*] dones.bin:%zu 个 episode边界\n", RESET_F.size());
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "用法: %s <ref.bin> <test.bin>\n", argv[0]);
        return 1;
    }

    // ---- 先验证文件大小，挡住旧格式 / 半截文件 ----
    long sa = fsize(argv[1]), sb = fsize(argv[2]);
    if (sa < 0) { fprintf(stderr, "打不开 %s\n", argv[1]); return 1; }
    if (sb < 0) { fprintf(stderr, "打不开 %s\n", argv[2]); return 1; }
    if (sa != sb) {
        fprintf(stderr, "两文件大小不同 (%ld vs %ld) —— 其中一个是旧的，重跑 tf\n", sa, sb);
        return 1;
    }
    if (sa % (long)CHUNK_BYTES != 0) {
        fprintf(stderr, "文件大小 %ld 不是 chunk 大小 %zu 的整数倍 —— 格式不符，重跑 tf\n",
                sa, CHUNK_BYTES);
        return 1;
    }
    load_dones("trace/dones.bin");
    FILE * fa = fopen(argv[1], "rb");
    FILE * fb = fopen(argv[2], "rb");

    Chunk A, B;
    int  chunk = 0;
    long n = 0, flips = 0, swap2 = 0, demote2 = 0;
    long n_ok = 0, flips_ok = 0, n_bad = 0, flips_bad = 0;

    std::vector<float> flip_gap, abs_dgap, drew, dend;
    double sum_dgap = 0;

    while (read_chunk(fa, A) && read_chunk(fb, B)) {
        for (int i = 0; i < NOBS; i++) {
            // 该位置预测的目标帧：
            //   i%16 = 0..14 -> 预测同一帧内的下一个 token
            //   i%16 = 15    -> block 的 p=16 位置，跨帧预测下一帧
            int  frame = chunk*NB + i/16 + ((i % 16) == 15 ? 1 : 0);
            bool bad   = near_reset(frame);
            bool flip  = (A.am[i] != B.am[i]);

            n++;
            if (flip) {
                flips++;
                flip_gap.push_back(A.gap[i]);
                if (B.am[i] == A.a2[i]) swap2++;      // 新冠军 = 原亚军
                if (B.a2[i] == A.am[i]) demote2++;    // 原冠军 = 新亚军
            }

            if (bad) { n_bad++; if (flip) flips_bad++; }
            else     { n_ok++;  if (flip) flips_ok++;  }

            float d = B.gap[i] - A.gap[i];
            sum_dgap += d;
            abs_dgap.push_back(fabsf(d));
        }

        for (int k = 0; k < NACT; k++) {
            drew.push_back(fabsf(exp_r (B.lr + k*3) - exp_r (A.lr + k*3)));
            dend.push_back(fabsf(p_done(B.le + k*2) - p_done(A.le + k*2)));
        }
        chunk++;
    }
    fclose(fa); fclose(fb);

    if (n == 0) { fprintf(stderr, "没读到数据\n"); return 1; }

    printf("位置数        %ld   (%d chunk)\n", n, chunk);
    printf("argmax 翻转   %ld  (%.3f%%)\n", flips, 100.0*flips/n);

    if (flips) {
        std::sort(flip_gap.begin(), flip_gap.end());
        printf("  翻转位 FP32 gap:  中位 %.4f  p90 %.4f  最大 %.4f\n",
               q(flip_gap,0.50), q(flip_gap,0.90), flip_gap.back());
        printf("  新冠军是原亚军    %ld/%ld  (%.1f%%)\n",
               swap2, flips, 100.0*swap2/flips);
        printf("  原冠军退居第二    %ld/%ld  (%.1f%%)\n",
               demote2, flips, 100.0*demote2/flips);
    }

    // ---- reset 边界排除 ----
    if(!RESET_F.empty()) {
        printf(" 干净区        %6ld/%6ld  (%.3f%%)\n",
                flips_ok,n_ok,100.0*flips_ok/n_ok);
        printf(" reset 附近    %6ld/%6ld  (%.3f%%)\n",
                flips_bad, n_bad,n_bad ? 100.0*flips_bad/n_bad : 0.0);
    }

    {
        double mu = sum_dgap / abs_dgap.size();
        std::sort(abs_dgap.begin(), abs_dgap.end());
        printf("gap 逐位变化  均值 %+.4f   |变化| 中位 %.4f  p95 %.4f\n",
               mu, q(abs_dgap,0.50), q(abs_dgap,0.95));
    }

    report("E[r] 偏差",    drew);
    report("P(done) 偏差", dend);
    return 0;
}
