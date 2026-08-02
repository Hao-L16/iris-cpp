#pragma once
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

static int g_fail = 0;

static bool check(const char * name, const struct ggml_tensor * t) {
    const size_t n = ggml_nelements(t);
    const std::string path = "ref/" + std::string(name) + ".bin";

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { printf("  %-18s [X] 打不开 %s\n", name, path.c_str()); g_fail++; return false; }

    fseek(f, 0, SEEK_END);
    const size_t nfile = ftell(f) / sizeof(float);
    fseek(f, 0, SEEK_SET);
    if (nfile != n) {
        printf("  %-18s [X] 元素数不符: ggml=%zu, ref=%zu  <-- 形状错了\n", name, n, nfile);
        fclose(f); g_fail++; return false;
    }

    std::vector<float> ref(n);
    if (fread(ref.data(), sizeof(float), n, f) != n) { fclose(f); g_fail++; return false; }
    fclose(f);

    const float * got = (const float *) t->data;
    double maxd = 0.0, amax = 0.0;
    size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)got[i] - (double)ref[i]);
        if (d > maxd) { maxd = d; at = i; }
        double a = fabs((double)ref[i]);
        if (a > amax) amax = a;
    }

    const double tol = 1e-5 + 1e-6 * amax;                 // numpy.allclose 风格
    const double rel = (amax > 0.0) ? maxd / amax : 0.0;
    const bool   ok  = (maxd <= tol);
    if (!ok) g_fail++;

    printf("  %-18s ne=[%4lld %4lld %4lld %2lld]  maxdiff=%.3e  rel=%.3e  %s",
           name, (long long)t->ne[0], (long long)t->ne[1],
           (long long)t->ne[2], (long long)t->ne[3], maxd, rel, ok ? "PASS" : "FAIL");
    if (!ok) printf("   idx %zu (got %.6f ref %.6f)", at, got[at], ref[at]);
    printf("\n");
    return ok;
}
