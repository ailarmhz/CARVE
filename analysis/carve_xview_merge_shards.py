#!/usr/bin/env python3
r"""
Merge sharded outputs into the per-(dataset,method,seed) files the aggregator
expects, and the diag per-backend CSV. Scan order is irrelevant for the metrics
(AUROC / paired bootstrap operate per-scan), so shards are simply concatenated.

  end-task:  results_carve_xview/<ds>_<method>_seed<s>_sh*of<N>.pt
             -> results_carve_xview/<ds>_<method>_seed<s>.pt
  diag:      paper_submission/xview_diag_<backend>_sh*of<N>.csv
             -> paper_submission/xview_diag_<backend>.csv

    python carve_xview_merge_shards.py --endtask_dir results_carve_xview
    python carve_xview_merge_shards.py --diag_dir paper_submission
"""
from __future__ import annotations
import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

_SHARD = re.compile(r"^(.*)_sh\d+of\d+$")
CAT_KEYS = ["probs", "y_true", "sigma_pred", "k_hat", "gate_applied", "wallclock_s", "peak_mem_mb"]


def merge_endtask(d: Path):
    groups = defaultdict(list)
    for f in sorted(d.glob("*_sh*of*.pt")):
        m = _SHARD.match(f.stem)
        if m:
            groups[m.group(1)].append(f)
    for base, fs in sorted(groups.items()):
        parts = [torch.load(f, map_location="cpu", weights_only=False) for f in fs]
        merged = dict(parts[0])
        merged["ids"] = [i for p in parts for i in p["ids"]]
        for k in CAT_KEYS:
            if all(k in p for p in parts):          # robust to non-TTA outputs (e.g. prompt-ensemble)
                merged[k] = torch.cat([p[k] for p in parts], dim=0)
            else:
                merged.pop(k, None)                 # drop partial fields (e.g. k_hat on mixed shards)
        merged.pop("shard_idx", None); merged.pop("num_shards", None)
        out = d / f"{base}.pt"
        torch.save(merged, out)
        print(f"merged {len(fs)} shards -> {out}  (n={merged['probs'].shape[0]})")


def merge_diag(d: Path):
    groups = defaultdict(list)
    for f in sorted(d.glob("xview_diag_*_sh*of*.csv")):
        m = _SHARD.match(f.stem)
        if m:
            groups[m.group(1)].append(f)
    for base, fs in sorted(groups.items()):
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        out = d / f"{base}.csv"
        df.to_csv(out, index=False)
        print(f"merged {len(fs)} shards -> {out}  ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endtask_dir", type=Path, default=None)
    ap.add_argument("--diag_dir", type=Path, default=None)
    args = ap.parse_args()
    if args.endtask_dir:
        merge_endtask(args.endtask_dir)
    if args.diag_dir:
        merge_diag(args.diag_dir)
    if not args.endtask_dir and not args.diag_dir:
        ap.error("pass --endtask_dir and/or --diag_dir")


if __name__ == "__main__":
    main()
