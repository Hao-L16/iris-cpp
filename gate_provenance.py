#!/usr/bin/env python3
# ---- save as: gate_provenance.py   (run from ~/iris-cpp) ----
"""Was actions_fp32_gate.bin produced by replaying trace_s0's frames, or by
running the environment live?

The two leave different signatures.  A replay is anchored to the trace at every
frame, so its disagreement rate with actions.bin is roughly constant over the
episode.  A live run shares only the opening frames and then follows its own
trajectory, so disagreement starts near zero and rises to chance level.

Chance level for 4 actions is 75% if the two are independent; the observed
overall rate of 9.25% is far below that, which already argues against a fully
independent trajectory -- but the shape over time distinguishes the cases much
more sharply than the average does.
"""
import numpy as np

a = np.fromfile("trace_s0/actions.bin", dtype=np.int32)
g = np.fromfile("trace_s0/actions_fp32_gate.bin", dtype=np.int32)
d = a != g

print(f"overall disagreement: {100*d.mean():.2f}%   (chance for 4 actions: 75%)")
print("\ndisagreement by block of 200 frames:")
for i in range(0, 2000, 200):
    print(f"  {i:5d}-{i+199:<5d}  {100*d[i:i+200].mean():6.2f}%")

print(f"\nfirst disagreement at frame {int(np.argmax(d))}")
print(f"longest run of agreement: "
      f"{max((len(s) for s in ''.join('01'[x] for x in d.astype(int)).split('1')), default=0)}")

print("\naction histograms (0..3):")
print(f"  actions.bin          {np.bincount(a, minlength=4)}")
print(f"  actions_fp32_gate    {np.bincount(g, minlength=4)}")
