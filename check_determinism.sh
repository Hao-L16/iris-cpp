#!/usr/bin/env bash
# ---- save as: check_determinism.sh   (run from ~/iris-cpp) ----
# Does a full-precision gate run reproduce byte-for-byte?
#
# run_iris writes to trace/actions_fp32_gate.bin unconditionally, so the
# existing baseline must be moved out of the way before rerunning: the program
# would otherwise overwrite the very file we are checking against.
#
# If the rerun reproduces the baseline exactly, the imagination sampling is
# seeded deterministically and B0 can proceed.  If not, the RNG has to be fixed
# before anything is measured across formats.
set -e
cd ~/iris-cpp

echo "=== which trace is in trace/ ==="
for s in 0 1 2 3 4; do
  if cmp -s trace/frames.bin trace_s/frames.bin; then echo "  trace/ == trace_s"; fi
done

echo "=== which model is in ./ ==="
ls -l *.gguf
md5sum *.gguf

echo "=== preserving the baseline ==="
cp -v trace/actions_fp32_gate.bin baseline_gate.bin
md5sum baseline_gate.bin

echo "=== rerun 1 ==="
./run_iris 2000 8 gate > run1.log 2>&1
tail -4 run1.log
cp trace/actions_fp32_gate.bin rerun1.bin

echo "=== rerun 2 ==="
./run_iris 2000 8 gate > run2.log 2>&1
cp trace/actions_fp32_gate.bin rerun2.bin

echo "=== verdict ==="
md5sum baseline_gate.bin rerun1.bin rerun2.bin
cmp rerun1.bin rerun2.bin && echo "  rerun1 == rerun2  : run-to-run deterministic"                           || echo "  rerun1 != rerun2  : RNG NOT FIXED -- stop"
cmp baseline_gate.bin rerun1.bin && echo "  baseline reproduced: same model, same code"                           || echo "  baseline differs   : model or code changed since Aug 31"
