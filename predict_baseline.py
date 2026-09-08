#!/usr/bin/env python3
# ---- save as: predict_baseline.py   (run from ~/iris-cpp) ----
"""Predict the C++/PyTorch baseline argmax mismatch rate from the margin
distribution alone, before running PyTorch at all.

Reasoning: Section 3.2.2 reports intermediate activations agreeing to ~1e-4
relative, with one deliberate substitution (GELU erf vs tanh, ~3e-4 at worst).
With logits of order 10 that is an absolute logit perturbation of order 1e-3,
so the margin between the top two moves by roughly delta = 2e-3.  A position
whose full-precision margin is below that is at risk of choosing differently.

Gives an order-of-magnitude expectation to compare the measured rate against.
If PyTorch disagrees far more often than this, the cause is not floating-point
residual and something in the port differs structurally.

Outputs no files; prints only.
"""
import numpy as np

SEEDS = [0, 1, 2, 3, 4]
DELTAS = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

print("Fraction of positions with FP32 margin below delta (%)")
print("(an upper bound on the baseline mismatch rate: a margin below delta is")
print(" at risk, not certain to differ -- the perturbation is signed)")
print(f"\n{'delta':>8} " + " ".join(f"{'s'+str(s):>8}" for s in SEEDS) + f"{'mean':>9}")
for d in DELTAS:
    v = [100.0 * (np.load(f"fp_margins_s{s}.npy") < d).mean() for s in SEEDS]
    print(f"{d:8.1e} " + " ".join(f"{x:8.4f}" for x in v) + f"{np.mean(v):9.4f}")

print("\nFor scale: q8_0 flip rate is 0.967%, the smallest in the ladder.")

m = np.load("fp_margins_s0.npy")
print(f"\nPositions below 2e-3 on s0: {(m < 2e-3).sum()} of {m.size}")

# Exact ties deserve a separate look: at a tie the argmax is decided by the
# tie-breaking rule, and GGML and PyTorch are not guaranteed to agree on it.
# Such a position contributes a deterministic mismatch unrelated to rounding.
nz = int((m == 0).sum())
print(f"Exact zeros (tied top-2) on s0: {nz}")
print(f"Smallest 10 margins on s0: {np.sort(m)[:10]}")
