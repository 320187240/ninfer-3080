#!/usr/bin/env python3
"""Compare TP prefill layer traces against the single-GPU baseline.

Usage: python tp_trace_compare.py <tp_trace_dir> <sg_trace_dir> [tolerance]

Reads the raw BF16 dumps written by the NINFER_TP_TRACE hook: per rank
("sg" or "tp0"/"tp1") one "<rank>_emb_D*_T*.bf16" plus "<rank>_L<nn>_D*_T*.bf16" after
every layer, and optional "<rank>_logits_*" / "<rank>_scalars.txt". For each stage it
reports max-abs-diff and rel-L2 of the TP rank vs the single-GPU baseline and prints the
first stage whose relative error exceeds the tolerance. Tolerance guidance: a single
bf16 allreduce exchange rounds at ~4e-3 and 64 layers accumulate two exchanges each, so
TP-vs-SG parity sits in the low 1e-2 band by construction.
"""
import glob
import os
import re
import sys

import numpy as np


def load(rank: str, stage: str, trace_dir: str):
    pattern = os.path.join(trace_dir, f"{rank}_{stage}_D*_T*.bf16")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    path = matches[-1]  # last write wins (multi-chunk prefill keeps the final chunk)
    dims = re.search(r"_D(\d+)_T(\d+)\.bf16$", path)
    d, t = int(dims.group(1)), int(dims.group(2))
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size < d * t:
        return None
    return bf16(raw[: d * t])


def bf16(raw: np.ndarray) -> np.ndarray:
    bits = raw.astype(np.uint32) << 16
    return bits.view(np.float32)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    trace_dir, base_dir = sys.argv[1], sys.argv[2]
    tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-2

    stages = []
    for path in glob.glob(os.path.join(base_dir, "sg_L??_D*_T*.bf16")) + glob.glob(
        os.path.join(base_dir, "sg_emb_D*_T*.bf16")
    ):
        name = re.search(r"sg_(.*?)_D\d+_T\d+\.bf16$", path)
        if name:
            stages.append(name.group(1))
    stages = sorted(set(stages), key=lambda s: (s != "emb", int(s[1:]) if s != "emb" else -1))

    first_bad = None
    print(f"{'stage':>6} {'shape':>12} {'max_abs':>12} {'rel_l2':>10}  verdict")
    for stage in stages:
        base = load("sg", stage, base_dir)
        if base is None:
            continue
        for rank in ("tp0", "tp1"):
            tp = load(rank, stage, trace_dir)
            if tp is None:
                continue
            n = min(base.size, tp.size)
            diff = tp[:n].astype(np.float64) - base[:n].astype(np.float64)
            max_abs = float(np.max(np.abs(diff)))
            denom = float(np.linalg.norm(base[:n].astype(np.float64)))
            rel_l2 = float(np.linalg.norm(diff)) / (denom if denom > 0 else 1.0)
            verdict = "OK" if rel_l2 <= tolerance else "DIVERGED"
            if rel_l2 > tolerance and first_bad is None:
                first_bad = (stage, rank)
            print(
                f"{stage:>6}[{rank}] {n:>10} {max_abs:12.5f} {rel_l2:10.3e}  {verdict}"
            )
            break  # ranks are mirrored; tp0 is representative

    for rank in ("sg", "tp0", "tp1"):
        scalars = os.path.join(trace_dir, f"{rank}_scalars.txt")
        if os.path.exists(scalars):
            with open(scalars) as handle:
                lines = handle.read().strip().splitlines()
            print(f"{rank} scalars: " + ", ".join(lines[-2:]))

    if first_bad:
        print(f"\nfirst divergent stage: {first_bad[0]} ({first_bad[1]})")
        return 1
    print("\nall traced stages within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
