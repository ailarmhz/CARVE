#!/usr/bin/env python3
"""Aggregate the best-possible-results matrix across {CT-CLIP variants, fVLM} x
{internal, external} x {RadChest, CC-II, LUNA} x {zeroshot, pl_selftrain}.

For each cell we report the RANK metric (macro AUROC -- threshold-free, where the
base model matters) plus the operating-point story (macro F1 at three thresholds):
  - F1@0.5             : default decision threshold.
  - F1@prevalence      : per-label threshold set so the predicted positive rate
                         matches a reference prevalence pi_j. This only uses the
                         RANK of the scores, so it recovers F1 from a strong but
                         squashed rank (the fVLM cosine-margin case). Leakage-clean
                         when pi comes from SOURCE; for CC-II/LUNA (no matched
                         source labels) we use the cell's own prevalence and FLAG
                         it as an oracle-prevalence upper bound.
  - F1@oracle_thr      : per-label best-F1 threshold on the eval set (upper bound).

Seed-ensembles by averaging probs across all seeds present for the (dataset,method).
"""
from __future__ import annotations
import argparse, glob, os, re
from collections import defaultdict
import numpy as np
import torch


def auroc(y, s):
    p = (y == 1); npos = int(p.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    o = np.argsort(s); ss = s[o]; r = np.empty(len(s)); i = 0; rk = 1.0
    while i < len(ss):
        j = i + 1
        while j < len(ss) and ss[j] == ss[i]:
            j += 1
        r[o[i:j]] = (rk + rk + (j - i) - 1) / 2.0; rk += j - i; i = j
    return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def f1_bin(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); d = 2 * tp + fp + fn
    return 2.0 * tp / d if d > 0 else np.nan


def macro_auroc(pr, y):
    return float(np.nanmean([auroc(y[:, j], pr[:, j]) for j in range(y.shape[1])]))


def f1_at_half(pr, y):
    return float(np.nanmean([f1_bin((pr[:, j] > 0.5).astype(int), y[:, j]) for j in range(y.shape[1])]))


def f1_prevalence(pr, y, prevalence):
    """Threshold each label at the (1 - pi_j) quantile of its scores (rank-only)."""
    f1s = []
    for j in range(y.shape[1]):
        pi = float(prevalence[j]); s = pr[:, j]
        if pi <= 0:
            pred = np.zeros_like(s, dtype=int)
        elif pi >= 1:
            pred = np.ones_like(s, dtype=int)
        else:
            thr = np.quantile(s, 1.0 - pi); pred = (s > thr).astype(int)
        f1s.append(f1_bin(pred, y[:, j]))
    return float(np.nanmean(f1s))


def f1_oracle(pr, y):
    f1s = []
    for j in range(y.shape[1]):
        s = pr[:, j]; cand = np.unique(s)
        best = np.nan
        for t in cand:
            v = f1_bin((s > t).astype(int), y[:, j])
            if not np.isnan(v) and (np.isnan(best) or v > best):
                best = v
        f1s.append(best)
    return float(np.nanmean(f1s))


def seed_ensemble(paths):
    """Average probs across seed files; return (pr[N,L], y[N,L], labels)."""
    prs = []; y0 = None; labels = None
    for p in sorted(paths):
        d = torch.load(p, map_location="cpu", weights_only=False)
        pr = np.asarray(d["probs"], dtype=float); y = np.asarray(d["y_true"], dtype=int)
        if y0 is None:
            y0 = y; labels = d["label_cols"]
        elif y.shape != y0.shape:
            continue
        prs.append(pr)
    if not prs:
        return None
    return np.mean(prs, axis=0), y0, labels


def merged_seed_paths(d, dataset, method):
    """Prefer merged (non-shard) per-seed files; fall back to shard merge if needed."""
    pats = sorted(glob.glob(os.path.join(d, f"{dataset}_{method}_seed*.pt")))
    return [p for p in pats if "_sh" not in os.path.basename(p)]


# (model, internal/external, dataset, variant, results_dir)
CELLS = [
    ("CT-CLIP", "internal", "ctrate",   "zeroshot",  "results_endtask_internal_zeroshot"),
    ("CT-CLIP", "internal", "ctrate",   "vocabfine", "results_endtask_internal_vocabfine"),
    ("CT-CLIP", "internal", "ctrate",   "classfine", "results_endtask_internal_classfine"),
    ("CT-CLIP", "external", "radchest", "zeroshot",  "results_endtask_external_zeroshot"),
    ("CT-CLIP", "external", "radchest", "vocabfine", "results_endtask_external_vocabfine"),
    ("CT-CLIP", "external", "radchest", "classfine", "results_endtask_external_classfine"),
    ("CT-CLIP", "external", "ccii",     "zeroshot",  "results_carve_xview_z240"),
    ("CT-CLIP", "external", "luna",     "zeroshot",  "results_carve_xview_z240"),
    ("fVLM",    "internal", "ctrate",   "-",         "results_fvlm_internal_zeroshot"),
    ("fVLM",    "external", "radchest", "-",         "results_fvlm_external_zeroshot"),
    # new-matrix dirs (filled in by the launched jobs); harmless if absent
    ("fVLM",    "external", "luna",     "-",         "results_fvlm_matrix_luna"),
    ("fVLM",    "external", "ccii",     "-",         "results_fvlm_matrix_ccii"),
    ("fVLM",    "internal", "ctrate",   "-",         "results_fvlm_matrix_ctrate"),
    ("fVLM",    "external", "radchest", "-",         "results_fvlm_matrix_radchest"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="zeroshot,pl_selftrain")
    ap.add_argument("--out_csv", default="results/bestresults_matrix.csv")
    args = ap.parse_args()
    methods = args.methods.split(",")

    # source prevalence (CT-RATE internal) per label, for prevalence-matched F1 on
    # the matched multi-label external set (RadChest). Built from the internal y.
    src = seed_ensemble(merged_seed_paths("results_endtask_internal_zeroshot", "ctrate", "zeroshot"))
    src_prev = {}
    if src is not None:
        _, sy, slabels = src
        for j, lab in enumerate(slabels):
            src_prev[lab] = float(sy[:, j].mean())

    rows = []
    for model, split, dataset, variant, d in CELLS:
        if not os.path.isdir(d):
            continue
        for method in methods:
            paths = merged_seed_paths(d, dataset, method)
            if not paths:
                continue
            res = seed_ensemble(paths)
            if res is None:
                continue
            pr, y, labels = res
            A = macro_auroc(pr, y)
            f_half = f1_at_half(pr, y)
            # prevalence vector: source for matched labels else self (flagged)
            if all(l in src_prev for l in labels):
                prev = np.array([src_prev[l] for l in labels]); prev_src = "source"
            else:
                prev = y.mean(axis=0); prev_src = "self(oracle)"
            f_prev = f1_prevalence(pr, y, prev)
            f_orac = f1_oracle(pr, y)
            rows.append(dict(model=model, split=split, dataset=dataset, variant=variant,
                             method=method, n=pr.shape[0], L=pr.shape[1], nseeds=len(paths),
                             AUROC=round(A, 4), F1_half=round(f_half, 4),
                             F1_prev=round(f_prev, 4), prev_src=prev_src,
                             F1_oracle=round(f_orac, 4)))

    # print
    hdr = f"{'model':7} {'split':8} {'dataset':8} {'variant':9} {'method':12} {'n':>4} {'L':>2} {'sd':>2} {'AUROC':>7} {'F1@.5':>6} {'F1prev':>6} {'F1orac':>6} prev"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['model']:7} {r['split']:8} {r['dataset']:8} {r['variant']:9} {r['method']:12} "
              f"{r['n']:>4} {r['L']:>2} {r['nseeds']:>2} {r['AUROC']:>7.4f} {r['F1_half']:>6.3f} "
              f"{r['F1_prev']:>6.3f} {r['F1_oracle']:>6.3f} {r['prev_src']}")
    if rows:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.out_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
