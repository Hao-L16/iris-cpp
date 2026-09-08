#!/usr/bin/env python3
# ---- save as: analyse_b0.py   (run from ~/iris-cpp, after run_b0.sh 2000) ----
"""Closed-loop action comparison across the format ladder.

Chapter 4 measures which single-step predictions a format changes.  This asks
what reaches behaviour.  The actor-critic is not quantised (Table 3.1), so the
gate is the only channel through which a format can change an action, and the
full-precision gate opens on 1.1% of frames and changes the action on 0.2%.
The attenuation from token flips to action changes is the quantity of interest.

Flip rates are taken from trace s0 alone (Table C.1), not the five-trace means,
because B0 runs on s0: comparing an s0 action divergence against a five-trace
flip rate would mix two different samples.

Limitation: run_iris reports the gate-open count but not which frames it opened
on, so this cannot verify that every divergent frame is a gate-open frame --
the claim that the gate is the only channel is argued from Table 3.1, not
measured here.
"""
import re
import numpy as np

FORMATS = ["q8_0", "q6_k", "q5_k", "q5_0", "q4_k", "q4_0", "q3_k", "q2_k"]

# per-trace flip rate on s0, Table C.1
FLIP_S0 = {"q8_0": 0.994, "q6_k": 2.656, "q5_k": 4.044, "q5_0": 5.769,
           "q4_k": 8.009, "q4_0": 11.547, "q3_k": 18.081, "q2_k": 41.406}


def gate_stats(path):
    """Pull '开启 N/M' and '改变动作 K 次' out of a run log."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"gate 开启 (\d+)/(\d+).*?改变动作 (\d+) 次", txt)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (None,)*3


fp = np.fromfile("b0/actions_fp32_gate.bin", dtype=np.int32)
n = fp.size
print(f"baseline: {n} frames, action histogram {np.bincount(fp, minlength=4)}")
o, t, c = gate_stats("run1.log")
if o is not None:
    print(f"baseline gate: open {o}/{t} ({100*o/t:.2f}%), "
          f"changed {c} ({100*c/t:.2f}%)")

print(f"\n{'fmt':6} {'flip% (s0)':>11} {'action diff':>12} {'diff%':>8} "
      f"{'attenuation':>12} {'gate open':>10} {'changed':>8} {'first':>7}")
print("-" * 82)

rows = []
for f in FORMATS:
    a = np.fromfile(f"b0/actions_{f}_gate.bin", dtype=np.int32)
    if a.size != n:
        print(f"{f:6}  SIZE MISMATCH {a.size} vs {n} -- skipped")
        continue
    d = a != fp
    nd = int(d.sum())
    pct = 100.0 * nd / n
    flip = FLIP_S0[f]
    # how much of the single-step error reaches behaviour
    att = f"{flip/pct:.0f}x" if nd else ">" + f"{flip*n/100:.0f}x"
    go, gt, gc = gate_stats(f"b0/{f}.log")
    first = int(np.argmax(d)) if nd else -1
    print(f"{f:6} {flip:11.3f} {nd:12d} {pct:8.3f} {att:>12} "
          f"{go if go is not None else -1:10d} "
          f"{gc if gc is not None else -1:8d} {first:7d}")
    rows.append((f, flip, nd, pct, go, gc, a))

print("\naction histograms (0..3), where they differ from baseline:")
base_h = np.bincount(fp, minlength=4)
print(f"  {'fp32':6} {base_h}")
for f, _, nd, _, _, _, a in rows:
    if nd:
        print(f"  {f:6} {np.bincount(a, minlength=4)}   "
              f"({nd} frame{'s' if nd > 1 else ''} differ)")

print("\ndivergent frames per format:")
for f, _, nd, _, _, _, a in rows:
    if nd:
        idx = np.flatnonzero(a != fp)
        show = idx[:20]
        print(f"  {f:6} {list(show)}{' ...' if nd > 20 else ''}")
        print(f"         fp32 -> quant: "
              f"{[(int(fp[i]), int(a[i])) for i in show[:8]]}")
