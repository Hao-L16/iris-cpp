#!/usr/bin/env python3
import numpy as np

NOBS, REC_BYTES = 320, 16
HEAD_BYTES = 20*3*4 + 20*2*4
CH = NOBS*REC_BYTES + HEAD_BYTES

def read_records(path):
    raw = np.fromfile(path, dtype=np.uint8)
    nblk = raw.size // CH
    blocks = raw.reshape(nblk, CH)[:, :NOBS*REC_BYTES]
    recs = blocks.reshape(nblk*NOBS, REC_BYTES)
    return (recs[:, 0:4].copy().view(np.int32).ravel(),
            recs[:, 4:8].copy().view(np.int32).ravel(),
            recs[:, 8:12].copy().view(np.float32).ravel(),
            recs[:, 12:16].copy().view(np.float32).ravel())

a1, a2, gap, gref = read_records("tf2_s0_f32.bin")

# --- 检查 A：gap 是否复现 Table C.2 的 s0 行 ---
# 这是唯一对照已发表已知量的检查，决定字段偏移对不对。
print("A) FP32 gap vs Table C.2 (s0)")
print(f"   n = {gap.size}  (expect 32000)")
for q, v in [(1,0.0212),(5,0.1178),(25,0.7558),(50,2.6327),
             (75,5.2172),(95,10.977)]:
    got = float(np.percentile(gap, q))
    print(f"   p{q:<3} {got:10.4f}  expect {v:8.4f}  "
          f"{'ok' if abs(got-v)<5e-3 else '** MISMATCH **'}")
print(f"   mean {gap.mean():10.4f}  expect   3.5501")

# --- 检查 B：gap_ref 到底是什么 ---
# 区分 NaN / 全零 / 真实数值三种假设。
print("\nB) FP32 gap_ref content")
print(f"   NaN      : {np.isnan(gref).sum()}")
print(f"   exact 0  : {(gref == 0).sum()}")
print(f"   negative : {(gref < 0).sum()}")
finite = gref[np.isfinite(gref)]
if finite.size:
    print(f"   finite n={finite.size}  min {finite.min():.4f}  "
          f"max {finite.max():.4f}  mean {finite.mean():.4f}")
print(f"   raw bits (first 5): "
      f"{[hex(x) for x in gref[:5].view(np.uint32)]}")

# --- 检查 C：前 5 条记录原样打印 ---
print("\nC) first 5 records  (a1, a2, gap, gap_ref)")
for i in range(5):
    print(f"   {a1[i]:5d} {a2[i]:5d} {gap[i]:12.6f} {gref[i]:12.6f}")

# --- 检查 D：量化文件的 gref 是否正常 ---
# 这才是回归真正用到的那一列。它正常，则分析管线不受影响。
qa1, qa2, qgap, qgref = read_records("tf2_s0_q4_k.bin")
print("\nD) q4_k gap_ref (the column the regression actually uses)")
print(f"   NaN {np.isnan(qgref).sum()}   exact 0 {(qgref==0).sum()}   "
      f"negative {(qgref<0).sum()}")
print(f"   mean {qgref.mean():.4f}   median {np.median(qgref):.4f}")
print(f"   flip rate vs f32: {100.0*(qa1!=a1).mean():.3f}%  (expect 8.009)")
print(f"   token range: a1 in [{a1.min()}, {a1.max()}] "
      f"(expect within [0, 511])")
