#!/usr/bin/env python3
"""Replace the global sigma with an empirical sigma(m) profile.

Motivation: Part 2 showed the residual spread rises monotonically with m, and
that sigma(m)/sigma_global is near-identical across all eight formats.  So the
*shape* is a property of the model and only the *scale* is a property of the
format -- the same split the dissertation already makes for the margin
distribution.  A global sigma therefore overstates the noise in the lower tail,
which is exactly where neg events live.

Part A -- shape invariance, quantified.  Per-bin normalised profiles with the
across-format spread, so "the rows coincide" becomes a number rather than an
eyeball judgement.

Part B -- in-sample: does sigma(m) remove the Step 0 bias?

Part C -- hold-out: shape averaged over TRAIN formats only, scale from k_hat.
Reported with fitted c,b and with c=1,b=0, because Part 1 showed the two differ
by 4.9 points at q2_k and that difference was masking the Gaussian bias.
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
NBIN = 20      # finer than the deciles of Part 2: neg events sit inside d1


def read_records(path):
    raw = np.fromfile(path, dtype=np.uint8)
    nblk = raw.size // CH
    recs = raw.reshape(nblk, CH)[:, :NOBS*REC_BYTES].reshape(nblk*NOBS, REC_BYTES)
    return (recs[:, 8:12].copy().view(np.float32).ravel().astype(np.float64),
            recs[:, 12:16].copy().view(np.float32).ravel().astype(np.float64))


D = {(r["trace"], r["format"]): r for r in csv.DictReader(open("fit_per_trace.csv"))}
E = {f: float(D[("s0", f)]["rel_weight_err"]) for f in FORMATS}
g = lambda s, f, k: float(D[(f"s{s}", f)][k])
avg = lambda f, k: np.mean([g(s, f, k) for s in SEEDS])

# cache margins, residual profiles and bin indices per (seed, format)
MARG, BIN, PROF = {}, {}, {}
for s in SEEDS:
    m = np.load(f"fp_margins_s{s}.npy").astype(np.float64)
    MARG[s] = m
    edges = np.percentile(m, np.linspace(0, 100, NBIN + 1))
    BIN[s] = np.clip(np.digitize(m, edges[1:-1]), 0, NBIN - 1)
for f in FORMATS:
    for s in SEEDS:
        _, mq = read_records(f"tf2_s{s}_{f}.bin")
        r = mq - (g(s, f, "c") * MARG[s] + g(s, f, "b"))
        PROF[(s, f)] = np.array([r[BIN[s] == i].std() for i in range(NBIN)])


def predict(s, prof, c, b):
    """E_m[ Phi(-(c m + b)/sigma(m)) ] with a per-position sigma."""
    return 100.0 * norm.cdf(-(c * MARG[s] + b) / prof[BIN[s]]).mean()


print(f"PART A -- normalised sigma(m)/sigma_global, first {NBIN//2} bins")
print("Row-to-row agreement is the claim; the sd row quantifies it.")
norm_prof = {f: np.mean([PROF[(s, f)] / g(s, f, "sigma") for s in SEEDS], axis=0)
             for f in FORMATS}
print(f"{'fmt':6} " + " ".join(f"{'b'+str(i+1):>6}" for i in range(NBIN//2)))
for f in FORMATS:
    print(f"{f:6} " + " ".join(f"{v:6.3f}" for v in norm_prof[f][:NBIN//2]))
stack = np.array([norm_prof[f] for f in FORMATS])
print(f"{'mean':6} " + " ".join(f"{v:6.3f}" for v in stack.mean(0)[:NBIN//2]))
print(f"{'sd':6} " + " ".join(f"{v:6.3f}" for v in stack.std(0, ddof=1)[:NBIN//2]))

print("\n\nPART B -- in-sample, global sigma vs sigma(m)  (fitted c, b)")
print(f"{'fmt':6} {'global':>9} {'sigma(m)':>9} {'true':>9} "
      f"{'dev glob':>9} {'dev s(m)':>9}")
for f in FORMATS:
    pg = np.mean([predict(s, np.full(NBIN, g(s, f, "sigma")), g(s, f, "c"), g(s, f, "b"))
                  for s in SEEDS])
    pm = np.mean([predict(s, PROF[(s, f)], g(s, f, "c"), g(s, f, "b")) for s in SEEDS])
    t = avg(f, "neg_pct")
    print(f"{f:6} {pg:9.3f} {pm:9.3f} {t:9.3f} "
          f"{100*(pg-t)/t:+8.1f}% {100*(pm-t)/t:+8.1f}%")

print("\n\nPART C -- hold-out with a transferred shape")
for name, train in SPLITS.items():
    k = np.mean([avg(f, "sigma") / E[f] for f in train])
    shape = np.mean([norm_prof[f] for f in train], axis=0)
    print(f"\n{name}  train = {', '.join(train)}   k_hat = {k:.3f}")
    print(f"  {'fmt':6} {'fit c,b':>9} {'c=1,b=0':>9} {'true':>9} "
          f"{'dev fit':>9} {'dev c=1':>9}")
    for f in [x for x in FORMATS if x not in train]:
        prof = shape * (k * E[f])
        pf = np.mean([predict(s, prof, g(s, f, "c"), g(s, f, "b")) for s in SEEDS])
        p1 = np.mean([predict(s, prof, 1.0, 0.0) for s in SEEDS])
        t = avg(f, "neg_pct")
        print(f"  {f:6} {pf:9.3f} {p1:9.3f} {t:9.3f} "
          f"{100*(pf-t)/t:+8.1f}% {100*(p1-t)/t:+8.1f}%")
