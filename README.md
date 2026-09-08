# iris-cpp

A C++/GGML reimplementation of the [IRIS](https://github.com/eloialonso/iris)
discrete-token world model, built so that low-bit quantised arithmetic actually
executes rather than being simulated in floating point.

This is the runtime half of an MSc dissertation on low-bit quantisation of
discrete-token world models. The Python half — training, GGUF export, trace
generation and the PyTorch reference — lives in
[iris-bfp](https://github.com/Hao-L16/iris-bfp).

## Why a reimplementation

Quantising a PyTorch model means rounding weights to a grid and multiplying
them back out in float32. That reproduces the rounding error but not the
arithmetic, and this study is about what integer kernels do to decisions.
GGML is a tensor library rather than an application, and llama.cpp — the
application built on it — implements a text transformer with no notion of a
convolutional tokeniser or an actor-critic. So the agent was rebuilt.

All three components are explicit GGML compute graphs: the convolutional
VQ-VAE tokeniser, the ten-layer causal Transformer, and the actor-critic.
578 tensors in total, exported from the PyTorch checkpoint with no
restructuring — each tensor keeps its name, shape and values.

## Requirements

- [GGML](https://github.com/ggerganov/ggml), built with the CPU backend.
  Measurements were taken against commit `9be31331`.
- A C++17 compiler. Development was on x86 under WSL2 with Ubuntu 22.04.
- Python 3 with NumPy and SciPy, for the analysis scripts.

## Building

Point `GGML` at your GGML checkout. It defaults to `../ggml`.

```bash
make                      # builds test_iris and run_iris
make GGML=/path/to/ggml   # if GGML lives elsewhere
```

The remaining programs are not yet in the Makefile and are built directly:

```bash
GGML=../ggml
CXXFLAGS="-std=c++17 -O2 -I $GGML/include"
LDFLAGS="-L $GGML/build/src -lggml -lggml-base -lggml-cpu \
         -Wl,-rpath,$GGML/build/src -lm -lpthread"

g++ $CXXFLAGS quantize.cpp             -o quantize    $LDFLAGS
g++ $CXXFLAGS tf.cpp iris.cpp          -o tf          $LDFLAGS
g++ $CXXFLAGS cmp.cpp                  -o cmp         $LDFLAGS
g++ $CXXFLAGS dump_tokens.cpp iris.cpp -o dump_tokens $LDFLAGS
g++ $CXXFLAGS dump_types.cpp           -o dump_types  $LDFLAGS
```

## Data this repository does not contain

Weights and recorded traces are excluded — they are large, and the IRIS
checkpoint is not mine to redistribute. You will need:

- **The three GGUF files** (`iris-worldmodel-f32.gguf`,
  `iris-tokenizer-f32.gguf`, `iris-actorcritic-f32.gguf`), produced by
  `convert_iris_to_gguf.py` in the companion repository from the published
  IRIS Breakout checkpoint.
- **Five traces** of 2000 frames each (`trace_s0/` … `trace_s4/`, each holding
  `frames.bin`, `tokens.bin`, `actions.bin`, `dones.bin`), recorded by
  `dump_trace_agent.py` in the companion repository under five environment
  seeds.

Traces are agent-driven, not random-policy. This matters: a random policy dies
quickly against the dense early wall, and on those states the head sensitivity
ordering reverses.

## Programs

### C++

| Program | What it does |
|---|---|
| `quantize` | Converts the full-precision world model to a target GGML format and prints the per-tensor and weighted relative Frobenius weight error, measured independently of any inference run. Takes lowercase format names. |
| `tf` | Replays a trace under teacher forcing and writes one 16-byte record per observation-token position — the top-two token indices, the self-selected margin, and the margin over the pair a reference run selected — in blocks of 320 records followed by that block's auxiliary head logits. |
| `run_iris` | Runs the agent end to end in either actor or gate mode, reports the per-frame latency breakdown and gate statistics, and writes the chosen action sequence. A tag argument names the model in the output filename. |
| `test_iris` | Component-level verification against activations dumped from PyTorch. |
| `cmp`, `dump_tokens`, `dump_types` | Small utilities for inspecting GGUF contents and tokeniser output. |

`tf` produces 45 files — five traces by eight formats, plus five full-precision
references. Those are the raw data for the whole analysis.

**One trap.** In a full-precision run there is no earlier reference to compare
against, so the fourth field of each record is written as zero. It is not a
margin and must not be read as one.

### Python analysis

| Script | What it does |
|---|---|
| `analyse_tf.py` | Reads all 45 record files and recomputes, in one pass, the flip rates, margin quantiles, containment fractions, maximisation statistics, reversal rates, regression coefficients and binned residual profile. |
| `holdout.py` | Hold-out validation of the σ / weight-error relation. Trains `k̂` on a subset of formats and predicts the rest, parameter-free (c = 1, b = 0). Splits and tolerances are literals at the head of the file and were fixed before any prediction was computed. |
| `replace.py` | Substitutes the empirical σ(m) profile for the single global σ. The shape turns out to be a property of the model and only the scale a property of the format. |
| `decompose.py` | Decomposes the hold-out prediction error and tests the homoscedasticity assumption the Gaussian model rests on. |
| `analyse_b0.py` | Closed-loop action comparison across the format ladder. Run after `run_b0.sh`. |
| `predict_baseline.py` | Predicts the C++/PyTorch argmax mismatch rate from the margin distribution alone, before running PyTorch at all. |
| `gate_provenance.py` | Checks whether a recorded action file came from replaying a trace or from running the environment live. The two leave different signatures. |
| `diag.py` | Record-format diagnostics. |

### Shell

| Script | What it does |
|---|---|
| `run_b0.sh` | Drives the closed-loop measurement across all eight formats. Set `IRIS_GGUF_DIR` to point at the GGUF directory. |
| `check_determinism.sh` | Confirms that two runs of the same configuration reproduce byte-identical output. |

`run_b0.sh` loads the world model from a fixed filename that is a **symbolic
link**, so switching formats replaces the link rather than copying over it — a
copy would follow the link and destroy the full-precision weights. A trap
restores the original link on any exit, and the swap is verified on the *input*
by resolving the link and recording the md5 of the file the loader will open.
The output cannot serve as that check, because a format that changes no action
is a possible result and not evidence of a failed swap.

## Reading the record format

`analyse_tf.py`'s record reader was validated against synthetic files with
known coefficients before being run on real data, because the block structure
means that reading the files as a flat array returns plausible but incorrect
numbers **without raising an error**.

The same script refuses to proceed unless each full-precision reference
reproduces that trace's published margin quantiles. Traces live in per-seed
directories and are copied into a working directory before each run, so a
filename is not evidence of which trace a file holds — a reproduced quantile
is.

## Determinism

Quantisation is round-to-nearest applied in place, with no calibration data and
no stochastic rounding: re-running the quantiser on the same input produces
byte-identical output. Inference is likewise deterministic — observation tokens
are generated by argmax rather than sampled, and thread count affects only the
order of floating-point reductions. The traces are the only stochastic element,
and they are fixed files.

## Licence and attribution

This is a reimplementation of IRIS by Micheli, Alonso and Fleuret
([ICLR 2023](https://openreview.net/forum?id=vhFu1Acb0xb)); the model
architecture and the published checkpoint are theirs. The comparison between
legacy and K-quant scaling that this work is built around was prompted by
SECDA-LLM and F-BFQ from Dr José Cano's group at the University of Glasgow.
