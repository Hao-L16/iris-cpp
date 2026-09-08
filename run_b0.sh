#!/usr/bin/env bash
# ---- save as: run_b0.sh   (run from ~/iris-cpp) ----
#
# B0: closed-loop action comparison across the format ladder.
#
# Runs gate mode on trace_s0 under each quantised world model and records the
# chosen action sequence, so the single-step flip rates of Chapter 4 can be
# compared against what actually reaches behaviour.  The actor-critic is not
# quantised (Table 3.1), so the gate is the only channel through which a format
# can change an action.
#
# iris_load reads "./iris-worldmodel-f32.gguf", which is a SYMLINK into
# ~/iris/gguf.  Copying over it would write through the link and destroy the
# full-precision weights, so the link is removed and recreated instead, and a
# trap restores the original on any exit.
#
# The swap is verified on the INPUT, by resolving the link and checking the md5
# of the file iris_load will actually open.  The output cannot serve as that
# check: a format that changes no action is a possible result, not evidence of
# a failed swap.
set -u
cd ~/iris-cpp

FRAMES=${1:-2000}
THREADS=8
WM_LINK=iris-worldmodel-f32.gguf
ORIG=${IRIS_GGUF_DIR:-../iris/gguf}/iris-worldmodel-f32.gguf
FORMATS="q8_0 q6_k q5_k q5_0 q4_k q4_0 q3_k q2_k"

restore() {
  rm -f "$WM_LINK"
  ln -s "$ORIG" "$WM_LINK"
  echo "[restore] $WM_LINK -> $ORIG"
}
trap restore EXIT INT TERM

mkdir -p b0
: > b0/manifest.txt

echo "=== preconditions ==="
[ -L "$WM_LINK" ] || { echo "FATAL: $WM_LINK is not a symlink -- stop"; exit 1; }
echo "link now points to: $(readlink -f "$WM_LINK")"
cmp -s trace/frames.bin trace_s0/frames.bin \
  && echo "trace/ == trace_s0  ok" \
  || { echo "FATAL: trace/ is not s0 -- stop"; exit 1; }
[ -f baseline_gate.bin ] || { echo "FATAL: baseline_gate.bin missing"; exit 1; }
echo "baseline: $(md5sum baseline_gate.bin)"

for f in $FORMATS; do
  src="wm-$f.gguf"
  [ -f "$src" ] || { echo "SKIP $f: $src missing"; continue; }

  rm -f "$WM_LINK"
  ln -s "$PWD/$src" "$WM_LINK"

  # verify on the input: resolve the link, hash what iris_load will open
  resolved=$(readlink -f "$WM_LINK")
  want=$(readlink -f "$src")
  [ "$resolved" = "$want" ] || { echo "FATAL: link resolves to $resolved"; exit 1; }
  m=$(md5sum "$resolved" | cut -d' ' -f1)

  echo ""
  echo "=== $f  ($src, md5 ${m:0:8})  $(date +%H:%M:%S) ==="
  ./run_iris "$FRAMES" "$THREADS" gate "$f" > "b0/$f.log" 2>&1
  grep -E "标签|gate 开启|动作分布|已写入" "b0/$f.log"

  out="trace/actions_${f}_gate.bin"
  [ -f "$out" ] || { echo "FATAL: $out not written"; exit 1; }
  mv "$out" "b0/actions_${f}_gate.bin"
  echo "$f  $m  $(md5sum "b0/actions_${f}_gate.bin" | cut -d' ' -f1)" >> b0/manifest.txt
done

cp baseline_gate.bin b0/actions_fp32_gate.bin
echo ""
echo "=== manifest (format, model md5, action md5) ==="
cat b0/manifest.txt
