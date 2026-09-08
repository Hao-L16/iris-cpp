#!/usr/bin/env python3
"""Analyse the tf teacher-forced replay dumps.

File layout (from tf.cpp): each file is a sequence of fixed 5520-byte blocks.
Within one block:
    320 records x 16 bytes : a1(i32) a2(i32) gap(f32) gap_ref(f32)
     20 x 3 float32        : reward-head logits   (skipped here)
     20 x 2 float32        : ends-head logits     (skipped here)
The head logits sit BETWEEN the record runs, so the file cannot be read as a
flat (N,4) array -- doing so silently returns garbage.

Definitions used below:
    m   = gap  from the FP32 reference file   (full-precision margin)
    m'  = gap_ref from the quantised file     (same two tokens, quantised logits)
    flip = a1 differs from the FP32 file's a1

Note: in an FP32 reference file the gap_ref column is written as zero, because
there is no earlier reference to compare against.  It is unused and must not be
mistaken for a margin.

Outputs, in addition to the printed tables:
    fp_margins_s{N}.npy   per-trace FP32 self-selected margins (float32, 32000)
    fit_per_trace.csv     40 rows: per (trace, format) regression + flip stats

Usage:  python3 analyse_tf.py            # run from ~/iris-cpp
"""
import numpy as np
import os
import csv

NOBS      = 320
REC_BYTES = 16
HEAD_BYTES = 20 * 3 * 4 + 20 * 2 * 4      # 400
CH        = NOBS * REC_BYTES + HEAD_BYTES  # 5520

SEEDS   = [0, 1, 2, 3, 4]
FORMATS = ["q8_0", "q6_k", "q5_k", "q5_0", "q4_k", "q4_0", "q3_k", "q2_k"]
# The p25 criterion is taken per trace, from that trace's own FP32 margins,
# rather than fixed at one seed's value.

# Published FP32 margin quantiles (Appendix C, Table C.2), used as a provenance
# check on the reader: an exact reproduction of an already-published quantity
# catches a field-offset error, a block misread or a mislabelled seed, none of
# which an "are the numbers plausible" inspection would catch.
PUBLISHED_Q = {   # seed -> {percentile: value}
    0: {1: 0.0212, 5: 0.1178, 25: 0.7558, 50: 2.6327, 75: 5.2172, 95: 10.977},
    1: {1: 0.0199, 5: 0.1065, 25: 0.7554, 50: 2.9052, 75: 5.4109, 95: 10.797},
    2: {1: 0.0207, 5: 0.1116, 25: 0.7645, 50: 2.6871, 75: 5.1413, 95: 11.085},
    3: {1: 0.0219, 5: 0.1229, 25: 0.8068, 50: 2.7302, 75: 5.1552, 95: 10.932},
    4: {1: 0.0234, 5: 0.1186, 25: 0.7696, 50: 2.7893, 75: 5.2815, 95: 10.654},
}
QTOL = 5e-3     # published to 3-4 dp, so this is a rounding tolerance

# Weighted relative Frobenius error over the 64 quantised tensors, recorded by
#  at quantisation time.  Deterministic and independent of any
# inference run (Section 3.3.4), so there is no per-trace version.
REL_WEIGHT_ERR = {
    "q8_0": 0.0051, "q6_k": 0.0169, "q5_k": 0.0306, "q5_0": 0.0408,
    "q4_k": 0.0604, "q4_0": 0.0818, "q3_k": 0.1416, "q2_k": 0.2514,
}


def read_records(path):
    """Return (a1, a2, gap, gap_ref) as 1-D arrays, skipping the head logits."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % CH:
        raise ValueError(f"{path}: size {raw.size} is not a multiple of {CH}")
    nblk = raw.size // CH
    blocks = raw.reshape(nblk, CH)[:, :NOBS * REC_BYTES]     # drop head logits
    recs = blocks.reshape(nblk * NOBS, REC_BYTES)
    a1 = recs[:, 0:4].copy().view(np.int32).ravel()
    a2 = recs[:, 4:8].copy().view(np.int32).ravel()
    gap = recs[:, 8:12].copy().view(np.float32).ravel()
    gref = recs[:, 12:16].copy().view(np.float32).ravel()
    return a1, a2, gap, gref


def check_reference(path, seed, a1, gap, gref):
    """Provenance checks on an FP32 reference file.  Raises on failure."""
    bad = []
    if gap.size != 32000:
        bad.append(f"n = {gap.size}, expected 32000")
    if not np.isfinite(gap).all():
        bad.append(f"{(~np.isfinite(gap)).sum()} non-finite margins")
    if a1.min() < 0 or a1.max() > 511:
        bad.append(f"token index out of range [{a1.min()}, {a1.max()}]")
    # An FP32 reference writes no gap_ref; a non-zero column here means this is
    # not the file it is taken to be.
    if (gref != 0).any():
        bad.append(f"gap_ref not all zero ({(gref != 0).sum()} non-zero) "
                   f"-- is this really an FP32 reference file?")
    for q, want in PUBLISHED_Q[seed].items():
        got = float(np.percentile(gap, q))
        if abs(got - want) > QTOL:
            bad.append(f"p{q} = {got:.4f}, published {want:.4f}")
    if bad:
        raise SystemExit(f"{path}: provenance check failed\n  "
                         + "\n  ".join(bad))


def fit(m, mq):
    """Least squares m' = c*m + b; return c, b, sigma, R^2."""
    A = np.vstack([m, np.ones_like(m)]).T
    (c, b), *_ = np.linalg.lstsq(A, mq, rcond=None)
    resid = mq - (c * m + b)
    sigma = resid.std(ddof=2)
    ss_res = (resid ** 2).sum()
    ss_tot = ((mq - mq.mean()) ** 2).sum()
    return c, b, sigma, 1.0 - ss_res / ss_tot


def main():
    fit_rows = []

    ref = {}
    for s in SEEDS:
        p = f"tf2_s{s}_f32.bin"
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        a1, a2, gap, gref_fp = read_records(p)
        check_reference(p, s, a1, gap, gref_fp)
        np.save(f"fp_margins_s{s}.npy", gap.astype(np.float32))
        ref[s] = (a1, a2, gap, float(np.percentile(gap, 25)))
        print(f"seed {s}: {len(a1)} records, FP32 margin median {np.median(gap):.4f}"
              f"   [Table C.2 reproduced; wrote fp_margins_s{s}.npy]")
    print()

    hdr = (f"{'fmt':6} {'flip%':>8} {'c':>8} {'b':>9} {'sigma':>8} {'R2':>7} "
           f"{'medMgn':>8} {'<p25%':>7} {'neg%':>7}")
    print(hdr)
    print("-" * len(hdr))

    summary = {}
    for f in FORMATS:
        rows = []
        for s in SEEDS:
            p = f"tf2_s{s}_{f}.bin"
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")
            qa1, qa2, qgap, gref = read_records(p)
            ra1, ra2, m, p25 = ref[s]
            if len(qa1) != len(ra1):
                raise SystemExit(f"length mismatch: {p}")
            # A quantised file must carry a real gap_ref; an all-zero column
            # would mean an FP32 file has been picked up under a quantised name.
            if not (gref != 0).any():
                raise SystemExit(f"{p}: gap_ref is all zero -- FP32 file "
                                 f"mislabelled as {f}?")

            flip = qa1 != ra1
            flip_pct = 100.0 * flip.mean()
            med_flip_margin = float(np.median(m[flip])) if flip.any() else float("nan")
            below = 100.0 * (m[flip] < p25).mean() if flip.any() else float("nan")
            neg = 100.0 * (gref < 0).mean()
            # maximisation statistics, evaluated on flipped positions only
            top2 = 100.0 * (qa1[flip] == ra2[flip]).mean() if flip.any() else float("nan")
            demoted = 100.0 * (qa2[flip] == ra1[flip]).mean() if flip.any() else float("nan")
            dgap = float(qgap.mean() - m.mean())
            c, b, sg, r2 = fit(m.astype(np.float64), gref.astype(np.float64))
            rows.append((flip_pct, c, b, sg, r2, med_flip_margin, below, neg,
                         top2, demoted, dgap, float(qgap.mean())))
            fit_rows.append(dict(
                trace=f"s{s}", format=f, n=int(len(qa1)),
                flip_rate=flip_pct, c=c, b=b, sigma=sg, R2=r2,
                med_flip_margin=med_flip_margin, below_p25=below,
                top2=top2, demoted=demoted, dgap=dgap,
                mean_gap_q=float(qgap.mean()), neg_pct=neg,
                fp_p25=p25, rel_weight_err=REL_WEIGHT_ERR[f]))

        a = np.array(rows)
        mu, sd = a.mean(axis=0), a.std(axis=0, ddof=1)
        summary[f] = (mu, sd)
        print(f"{f:6} {mu[0]:8.3f} {mu[1]:8.4f} {mu[2]:9.4f} {mu[3]:8.4f} "
              f"{mu[4]:7.4f} {mu[5]:8.4f} {mu[6]:7.2f} {mu[7]:7.3f}")

    print("\nstandard deviations across the five seeds")
    print(hdr)
    print("-" * len(hdr))
    for f in FORMATS:
        _, sd = summary[f]
        print(f"{f:6} {sd[0]:8.3f} {sd[1]:8.4f} {sd[2]:9.4f} {sd[3]:8.4f} "
              f"{sd[4]:7.4f} {sd[5]:8.4f} {sd[6]:7.2f} {sd[7]:7.3f}")

    print("\nmaximisation statistics (mean over five seeds, sd in brackets)")
    mh = (f"{'fmt':6} {'top2%':>8} {'demoted%':>10} {'meanGap':>9} {'dGap':>9}")
    print(mh)
    print("-" * len(mh))
    for f in FORMATS:
        mu, sd = summary[f]
        print(f"{f:6} {mu[8]:8.2f} {mu[9]:10.2f} {mu[11]:9.4f} {mu[10]:9.4f}")
    print("  sd:")
    for f in FORMATS:
        _, sd = summary[f]
        print(f"{f:6} {sd[8]:8.2f} {sd[9]:10.2f} {sd[11]:9.4f} {sd[10]:9.4f}")

    ref_gap_mean = np.mean([ref[s][2].mean() for s in SEEDS])
    print(f"per-seed p25 used: " + ", ".join(f'{ref[s][3]:.4f}' for s in SEEDS))
    print(f"\nFP32 mean self-selected gap: {ref_gap_mean:.4f}")

    print("\nFP32 margin distribution (per seed, then mean and sd across seeds)")
    QS = [1, 5, 25, 50, 75, 95, 99]
    print(f"{'seed':6} " + " ".join(f"{'p'+str(q):>9}" for q in QS) + f"{'mean':>10}{'sd':>9}")
    rows = []
    for s in SEEDS:
        g = ref[s][2]
        r = [np.percentile(g, q) for q in QS] + [g.mean(), g.std(ddof=1)]
        rows.append(r)
        print(f"s{s:<5} " + " ".join(f"{v:9.4f}" for v in r[:len(QS)])
              + f"{r[-2]:10.4f}{r[-1]:9.4f}")
    a = np.array(rows)
    print(f"{'mean':6} " + " ".join(f"{v:9.4f}" for v in a.mean(axis=0)[:len(QS)])
          + f"{a.mean(axis=0)[-2]:10.4f}{a.mean(axis=0)[-1]:9.4f}")
    print(f"{'sd':6} " + " ".join(f"{v:9.4f}" for v in a.std(axis=0, ddof=1)[:len(QS)]))

    print("\nper-seed flip rate (%) -- for the appendix table")
    print(f"{'fmt':6} " + " ".join(f"{'s'+str(s):>8}" for s in SEEDS))
    for f in FORMATS:
        vals = []
        for s in SEEDS:
            qa1, _, _, _ = read_records(f"tf2_s{s}_{f}.bin")
            ra1 = ref[s][0]
            vals.append(100.0 * (qa1 != ra1).mean())
        print(f"{f:6} " + " ".join(f"{v:8.3f}" for v in vals))

    # sigma / relative-weight-error ratio: reproduces the Figure 4.3 claim
    # (9.7 to 15.3, mean 11.9) from the exported sigma column.
    print("\nsigma / relative weight error  (Figure 4.3 check)")
    print(f"{'fmt':6} {'sigma':>9} {'relErr':>9} {'ratio':>9}")
    ratios = []
    for f in FORMATS:
        sg = summary[f][0][3]
        e = REL_WEIGHT_ERR[f]
        ratios.append(sg / e)
        print(f"{f:6} {sg:9.4f} {e:9.4f} {sg / e:9.2f}")
    print(f"  min {min(ratios):.2f}   max {max(ratios):.2f}   "
          f"mean {np.mean(ratios):.2f}   (published: 9.70 / 15.30 / 11.90)")

    cols = ["trace", "format", "n", "flip_rate", "c", "b", "sigma", "R2",
            "med_flip_margin", "below_p25", "top2", "demoted", "dgap",
            "mean_gap_q", "neg_pct", "fp_p25", "rel_weight_err"]
    with open("fit_per_trace.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(fit_rows)
    print(f"\nwrote fit_per_trace.csv ({len(fit_rows)} rows)")


if __name__ == "__main__":
    main()
