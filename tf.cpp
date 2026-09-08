// tf.cpp — teacher-forced 回放
//   ./tf <model_dir> <out.bin> [n_frames]
#include "iris.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

int main(int argc, char ** argv) {
    const char * dir = (argc > 1) ? argv[1] : ".";
    const char * out = (argc > 2) ? argv[2] : "tf_out.bin";
    int nf = (argc > 3) ? atoi(argv[3]) : 2000;

    iris_model m;
    if (!iris_load(m, dir)) return 1;
    m.n_threads = 1;                 // 确定性

    // token + action
    std::vector<int32_t> tok(nf*16), act(nf);
    FILE * f = fopen("trace/tokens.bin","rb");
    if(!f) { fprintf(stderr,"打不开 trace/tokens.bin\n"); return 1;}
    size_t got = fread(tok.data(), sizeof(int32_t), nf*16, f);
    if(got != (size_t)(nf*16)) { fprintf(stderr, "tokens.bin 只有%zu\n",got); return 1;}
    fclose(f);
    f = fopen("trace/actions.bin","rb");
    got = fread(act.data(), sizeof(int32_t), nf, f);
    if(got != (size_t)nf) { fprintf(stderr, "actions_fp32只有%zu\n",got); return 1;}
    fclose(f);

    const int NB = 20, T = NB*17;    // 一次 20 个 block = 340 token
    std::vector<int32_t> seq(T);
    std::vector<float> lo(340*512), lr(20*3), le(20*2);

    const char * ref = (argc > 4) ? argv[4] : nullptr;
    std::vector<int32_t> ra1,ra2;
    if(ref) {
        FILE * rf = fopen(ref,"rb");
        if(!rf) {fprintf(stderr,"打不开%s\n",ref); return 1;}
        const int NOBS = 320;
        const int CH = NOBS * 16 + 20*3*4 + 20*2*4; //5520
        std::vector<uint8_t> buf;
        fseek(rf,0,SEEK_END); long sz = ftell(rf); fseek(rf,0,SEEK_SET);
        buf.resize(sz);
        if(fread(buf.data(),1,sz,rf) != (size_t)sz){
            fprintf(stderr,"ref短读\n"); return 1;
        }
        fclose(rf);
        long nch = sz/CH;
        for(long c=0; c<nch;c++){
            const int32_t * r = (const int32_t *)(buf.data() + c*CH);
            for(int i = 0; i<NOBS;i++){
                ra1.push_back(r[i*4+0]);
                ra2.push_back(r[i*4+1]);
            }
        }
        printf("ref loaded: %zu rows\n", ra1.size());
    }
    long row = 0;

    FILE * o = fopen(out, "wb");
    double gsum = 0; long gn = 0;
    std::vector<float> gaps;

    for (int c = 0; c + NB <= nf; c += NB) {
        for (int b = 0; b < NB; b++) {
            for (int i = 0; i < 16; i++) seq[b*17+i] = tok[(c+b)*16+i];
            seq[b*17+16] = act[c+b];
        }
        int n_obs = 0, n_act = 0;
        iris_wm_forward(m, seq.data(), T, lo.data(), &n_obs, lr.data(), le.data(), &n_act);

        for (int r = 0; r < n_obs; r++) {
            const float * p = lo.data() + r*512;
            int a1 = 0; for (int k = 1; k < 512; k++) if (p[k] > p[a1]) a1 = k;
            int a2 = -1;
            for (int k = 0; k < 512; k++) if (k != a1 && (a2<0 || p[k] > p[a2])) a2 = k;
            float gap = p[a1] - p[a2];

            int32_t rec[4];
            rec[0] = a1; rec[1] = a2;
            memcpy(&rec[2], &gap, 4);
            float gref = 0.0f;
            if(ref && row < (long)ra1.size()){
                gref = p[ra1[row]] - p[ra2[row]];     //对手固定威FP32的a1/a2，可正可负
            }
            memcpy(&rec[3],&gref,4);
            fwrite(rec, 4, 4, o);                   //每行16字节:a1(i32)a2(i32)gap(f32)gap_ref(f32)
            row++;
            gaps.push_back(gap); gsum += gap; gn++;

        }
        // 三个 head 的原始 logits 也存下来，后面算 E[r] / P(done) 偏差
        fwrite(lr.data(), 4, n_act*3, o);
        fwrite(le.data(), 4, n_act*2, o);
    }
    fclose(o); iris_free(m);

    std::sort(gaps.begin(), gaps.end());
    auto q = [&](double p){ return gaps[(size_t)(p*(gaps.size()-1))]; };
    printf("rows %ld  mean %.4f\n", gn, gsum/gn);
    printf("gap  p1 %.4f  p5 %.4f  p25 %.4f  p50 %.4f  p75 %.4f  p95 %.4f\n",
           q(.01), q(.05), q(.25), q(.50), q(.75), q(.95));
    printf("gap < 0.687 (Q8_0 扰动量级): %.2f%%\n",
           100.0 * (std::lower_bound(gaps.begin(),gaps.end(),0.687f) - gaps.begin()) / gaps.size());
    return 0;
}
