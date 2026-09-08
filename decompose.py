#!/usr/bin/env python3
"""Decompose the hold-out prediction error, and test the homoscedasticity
assumption that the Gaussian model rests on.

Part 1 -- three nested predictors of neg%, each adding one approximation:
    (a) fitted c, b, fitted sigma   -> isolates the Gaussian/homoscedastic bias
    (b) c=1,  b=0,  fitted sigma    -> adds the parameter-free assumption
    (c) c=1,  b=0,  predicted sigma -> adds the sigma extrapolation  [= hold-out]
The (a)->(b) step is the cost of dropping c and b; (b)->(c) is the cost of
predicting sigma from the weight error.

Part 2 -- residual spread binned by m.  The model assumes eps ~ N(0, sigma^2)
independent of m.  If the spread grows with m, the global sigma overstates the
noise in the lower tail, which is exactly where neg events live, and neg% is
overpredicted -- the Step 0 pattern.
"""
import csv
import numpy as np
from scipy.stats import norm

NOBS, REC_BYTES = 320, 16
CH = NOBS * REC_BYTES + 20 * 3 * 4 + 20 * 2 * 4
SEEDS = [0, 1, 2, 3, 4]
FORMATS = ["q8_0", "q6_k", "q5_k", "q5_0", "q4_k", "q4_0", "q3_k", "q2_k"]
SPLITS = {"Split-1": ["q8_0", "q6_k", "q5_k"],
          "Split-2": ["q8_0", "q5_k", "q4_0", "q3_k"]}


def read_records(path):
    raw = np.fromfile(path, dtype=np.uint8)
    nblk = raw.size // CH
    recs = raw.reshape(nblk, CH)[:, :NOBS*REC_BYTES].reshape(nblk*NOBS, REC_BYTES)
    return (recs[:, 8:12].copy().view(np.float32).ravel(),
            recs[:, 12:16].copy().view(np.float32).ravel())


D = {(r["trace"], r["format"]): r for r in csv.DictReader(open("fit_per_trace.csv"))}
M = {f"s{s}": np.load(f"fp_margins_s{s}.npy").astype(np.float64) for s in SEEDS}
E = {f: float(D[("s0", f)]["rel_weight_err"]) for f in FORMATS}
g = lambda s, f, k: float(D[(f"s{s}", f)][k])
avg = lambda f, k: np.mean([g(s, f, k) for s in SEEDS])
pred = lambda m, sg, c, b: 100.0 * norm.cdf(-(c*m + b)/sg).mean()

print("PART 1 -- error decomposition (neg%, mean over five traces)")
for name, train in SPLITS.items():
    k = np.mean([avg(f, "sigma")/E[f] for f in train])
    print(f"\n{name}  train = {', '.join(train)}   k_hat = {k:.3f}")
    print(f"  {'fmt':6} {'(a) fit':>8} {'(b) c=1':>8} {'(c) +sig':>9} "
          f"{'true':>8} | {'cost c,b':>9} {'cost sig':>9} {'total':>8}")
    for f in [x for x in FORMATS if x not in train]:
        a = np.mean([pred(M[f"s{s}"], g(s,f,"sigma"), g(s,f,"c"), g(s,f,"b")) for s in SEEDS])
        b_ = np.mean([pred(M[f"s{s}"], g(s,f,"sigma"), 1.0, 0.0) for s in SEEDS])
        c_ = np.mean([pred(M[f"s{s}"], k*E[f], 1.0, 0.0) for s in SEEDS])
        t = avg(f, "neg_pct")
        print(f"  {f:6} {a:8.3f} {b_:8.3f} {c_:9.3f} {t:8.3f} | "
              f"{b_-a:+9.3f} {c_-b_:+9.3f} {100*(c_-t)/t:+7.1f}%")

print("\n\nPART 2 -- residual spread by margin decile (trace s0)")
print("The model assumes this row is flat.  A rising row means the global")
print("sigma overstates the noise in the lower tail, where neg events are.")
m0, _ = read_records("tf2_s0_f32.bin")
m0 = m0.astype(np.float64)
edges = np.percentile(m0, np.arange(0, 101, 10))
idx = np.clip(np.digitize(m0, edges[1:-1]), 0, 9)
print(f"\n{'fmt':6} {'global':>7} " + " ".join(f"{'d'+str(i+1):>7}" for i in range(10)))
for f in FORMATS:
    _, mq = read_records(f"tf2_s0_{f}.bin")
    r = mq.astype(np.float64) - (g(0,f,"c")*m0 + g(0,f,"b"))
    print(f"{f:6} {g(0,f,'sigma'):7.4f} " +
          " ".join(f"{r[idx==i].std():7.4f}" for i in range(10)))
print(f"\ndecile edges: " + " ".join(f"{e:.3f}" for e in edges))
