#!/usr/bin/env python3
"""Hold-out validation of the sigma / weight-error relation.

Chain (fixed before looking at the test set):
  1. k_hat = mean over TRAIN formats of sigma_i / e_i
  2. sigma_hat_j = k_hat * e_j          (c = 1, b = 0: parameter-free)
  3. neg_hat_j   = E_m[ Phi(-m / sigma_hat_j) ], m from that trace's own
                   empirical FP32 margins

Target is neg% (the original runner-up overtaking on the original pair), NOT
flip%.  The regression is fitted on m' = margin over the reference-selected
pair, so m' < 0 is exactly the neg event.  A flip can also be won by a third
candidate with m' still positive, so flip% > neg%, and the gap widens as bit
width falls -- that gap is the Section 4.5 mechanism, not a prediction error.
"""
import csv
import numpy as np
from scipy.stats import norm

SEEDS = [0, 1, 2, 3, 4]
FORMATS = ["q8_0", "q6_k", "q5_k", "q5_0", "q4_k", "q4_0", "q3_k", "q2_k"]

SPLITS = {
    "Split-1 (extrapolate down)": ["q8_0", "q6_k", "q5_k"],
    "Split-2 (interpolate)":      ["q8_0", "q5_k", "q4_0", "q3_k"],
}

rows = list(csv.DictReader(open("fit_per_trace.csv")))
D = {(r["trace"], r["format"]): r for r in rows}
M = {f"s{s}": np.load(f"fp_margins_s{s}.npy").astype(np.float64) for s in SEEDS}
E = {f: float(D[("s0", f)]["rel_weight_err"]) for f in FORMATS}


def pred_neg(m, sigma, c=1.0, b=0.0):
    """E_m[ Phi(-(c m + b)/sigma) ]  over the empirical margins."""
    return 100.0 * norm.cdf(-(c * m + b) / sigma).mean()


# --- Step 0: in-sample sanity.  Using each cell's OWN fitted c, b, sigma,
# does the Gaussian model reproduce that cell's neg%?  If not, the whole
# prediction chain is void regardless of how k_hat is estimated.
print("Step 0 -- in-sample check (own fitted c, b, sigma -> own neg%)")
print(f"{'fmt':6} {'pred neg%':>10} {'true neg%':>10} {'dev':>8}")
for f in FORMATS:
    p = np.mean([pred_neg(M[f"s{s}"], float(D[(f"s{s}", f)]["sigma"]),
                          float(D[(f"s{s}", f)]["c"]),
                          float(D[(f"s{s}", f)]["b"])) for s in SEEDS])
    t = np.mean([float(D[(f"s{s}", f)]["neg_pct"]) for s in SEEDS])
    print(f"{f:6} {p:10.3f} {t:10.3f} {100*(p-t)/t:+7.1f}%")

# --- Hold-out proper
for name, train in SPLITS.items():
    test = [f for f in FORMATS if f not in train]
    print(f"\n{name}   train = {', '.join(train)}")

    k = np.mean([np.mean([float(D[(f"s{s}", f)]["sigma"]) for s in SEEDS]) / E[f]
                 for f in train])
    print(f"  k_hat = {k:.3f}")
    print(f"  {'fmt':6} {'sig pred':>9} {'sig true':>9} {'dev':>8} "
          f"{'neg pred':>9} {'neg true':>9} {'dev':>8} {'flip true':>10}")
    for f in test:
        sh = k * E[f]
        st = np.mean([float(D[(f"s{s}", f)]["sigma"]) for s in SEEDS])
        np_ = np.mean([pred_neg(M[f"s{s}"], sh) for s in SEEDS])
        nt = np.mean([float(D[(f"s{s}", f)]["neg_pct"]) for s in SEEDS])
        ft = np.mean([float(D[(f"s{s}", f)]["flip_rate"]) for s in SEEDS])
        print(f"  {f:6} {sh:9.4f} {st:9.4f} {100*(sh-st)/st:+7.1f}% "
              f"{np_:9.3f} {nt:9.3f} {100*(np_-nt)/nt:+7.1f}% {ft:10.3f}")
