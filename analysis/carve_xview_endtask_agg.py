#!/usr/bin/env python3
r"""
carve_xview_endtask_agg.py -- aggregate end-task preds into metrics + paired
bootstrap CIs on every delta, with the sigma_pred>tau stratification (CPU).

Reads preds written by carve_xview_adapt.py, named  <dataset>_<method>_seed<seed>.pt
(keys: ids, probs[N,L], y_true[N,L], sigma_pred[N], gate_applied[N], wallclock_s[N],
peak_mem_mb[N]).  Methods are seed-ensembled (probs averaged over seeds per scan),
then:

  * per-method macro AUROC / F1 / precision / recall (threshold 0.5),
  * PAIRED scan bootstrap CIs on deltas vs zero-shot, and gate vs always-on,
    overall and restricted to the sigma_pred>tau (Delta_sep>0 likely) subpopulation,
  * mean wall-clock (s) and peak GPU memory (MB) per method.

Hypothesis check: the gated variant beats both always-on and zero-shot, with the
gain concentrated in the sigma_pred>tau subpopulation.

    python carve_xview_endtask_agg.py --preds_dir results_carve_xview --dataset radchest \
        --gate_tau 0.091 --n_boot 2000
"""
from __future__ import annotations
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

METHODS = ["zeroshot", "tent", "ml_tta", "bem", "carve_xview", "carve_xview_gate"]


def micro_auroc(p, y):
    p = p.ravel(); y = y.ravel()
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort"); ps = p[order]
    ranks = np.empty(len(p)); i, rk = 0, 1.0
    while i < len(ps):
        j = i + 1
        while j < len(ps) and ps[j] == ps[i]:
            j += 1
        ranks[order[i:j]] = (rk + rk + (j - i) - 1) / 2.0
        rk += j - i; i = j
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_auroc(p, y):
    return float(np.nanmean([micro_auroc(p[:, j], y[:, j]) for j in range(y.shape[1])]))


def macro_prf(p, y, thr=0.5):
    pred = (p >= thr).astype(int)
    f1s, prs, rcs = [], [], []
    for j in range(y.shape[1]):
        tp = int(((pred[:, j] == 1) & (y[:, j] == 1)).sum())
        fp = int(((pred[:, j] == 1) & (y[:, j] == 0)).sum())
        fn = int(((pred[:, j] == 0) & (y[:, j] == 1)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        prs.append(pr); rcs.append(rc); f1s.append(f1)
    return float(np.mean(f1s)), float(np.mean(prs)), float(np.mean(rcs))


def macro_accuracy(p, y, thr=0.5):
    """Per-label accuracy at threshold, macro-averaged over labels (multi-label)."""
    pred = (p >= thr).astype(int)
    return float(np.mean([(pred[:, j] == y[:, j]).mean() for j in range(y.shape[1])]))


def metrics(p, y):
    f1, pr, rc = macro_prf(p, y)
    return {"AUROC": macro_auroc(p, y), "Accuracy": macro_accuracy(p, y),
            "F1": f1, "Precision": pr, "Recall": rc}


def paired_boot(pa, pb, y, mask, n_boot, rng, metric=macro_auroc):
    """CI on metric(pa)-metric(pb) over scans in mask (paired: same resample)."""
    idx_pool = np.where(mask)[0]
    if len(idx_pool) < 5:
        return float("nan"), float("nan"), float("nan")
    point = metric(pa[idx_pool], y[idx_pool]) - metric(pb[idx_pool], y[idx_pool])
    deltas = []
    for _ in range(n_boot):
        s = rng.choice(idx_pool, size=len(idx_pool), replace=True)
        deltas.append(metric(pa[s], y[s]) - metric(pb[s], y[s]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", type=Path, required=True)
    ap.add_argument("--dataset", default="radchest")
    ap.add_argument("--gate_tau", type=float, default=0.091)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "paper_submission")
    ap.add_argument("--tag", default="", help="output filename tag (default = dataset). "
                    "e.g. internal_zeroshot -> endtask_metrics_internal_zeroshot.csv")
    args = ap.parse_args()
    TAG = args.tag if args.tag else args.dataset
    rng = np.random.default_rng(args.seed)

    # collect seeds per method
    files = defaultdict(list)
    pat = re.compile(rf"{re.escape(args.dataset)}_([a-z_]+)_seed(\d+)\.pt$")
    for f in sorted(args.preds_dir.glob(f"{args.dataset}_*_seed*.pt")):
        m = pat.search(f.name)
        if m:
            files[m.group(1)].append(f)
    if not files:
        raise SystemExit(f"no preds matching {args.dataset}_<method>_seed<seed>.pt in {args.preds_dir}")

    # seed-ensemble probs per method; align on ids of zeroshot
    ens, meta = {}, {}
    y_ref = ids_ref = None
    for method, fs in files.items():
        ps, sig, wall, mem, gate = [], None, [], [], None
        ids0 = None
        for f in fs:
            d = torch.load(f, map_location="cpu", weights_only=False)
            ps.append(d["probs"].numpy())
            ids0 = [str(x) for x in d["ids"]]
            sig = d["sigma_pred"].numpy()
            gate = d["gate_applied"].numpy()
            wall.append(float(d["wallclock_s"].mean())); mem.append(float(d["peak_mem_mb"].mean()))
        ens[method] = np.mean(ps, axis=0)
        meta[method] = {"sigma": sig, "gate": gate, "n_seed": len(fs),
                        "wall_s": float(np.mean(wall)), "peak_mb": float(np.mean(mem))}
        if method == "zeroshot" or y_ref is None:
            d0 = torch.load(fs[0], map_location="cpu", weights_only=False)
            y_ref = d0["y_true"].numpy().astype(int); ids_ref = ids0

    y = y_ref
    sigma_ref = meta.get("carve_xview_gate", meta[list(meta)[0]])["sigma"]
    strat_hi = sigma_ref > args.gate_tau
    full = np.ones(len(y), dtype=bool)

    # ── per-method metrics table ──────────────────────────────────────────────
    rows = []
    for method in METHODS:
        if method not in ens:
            continue
        mm = metrics(ens[method], y)
        rows.append({"method": method, "n_seed": meta[method]["n_seed"],
                     **{k: round(v, 4) for k, v in mm.items()},
                     "wall_s": round(meta[method]["wall_s"], 3),
                     "peak_mb": round(meta[method]["peak_mb"], 0)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(args.out_dir / f"endtask_metrics_{TAG}.csv", index=False)
    print("\n== Per-method metrics (seed-ensembled) =="); print(tbl.to_string(index=False))

    # ── paired bootstrap deltas (AUROC + Accuracy) ────────────────────────────
    def report(pa_name, pb_name, mask, tag):
        if pa_name not in ens or pb_name not in ens:
            return None
        pt, lo, hi = paired_boot(ens[pa_name], ens[pb_name], y, mask, args.n_boot, rng)
        pa_, la_, ha_ = paired_boot(ens[pa_name], ens[pb_name], y, mask, args.n_boot, rng,
                                    metric=macro_accuracy)
        sig = "*" if (np.isfinite(lo) and (lo > 0 or hi < 0)) else " "
        sga = "*" if (np.isfinite(la_) and (la_ > 0 or ha_ < 0)) else " "
        print(f"  {pa_name:>16} - {pb_name:<12} [{tag:<10}] "
              f"dAUROC={pt:+.4f} [{lo:+.4f}, {hi:+.4f}] {sig}  "
              f"dAcc={pa_:+.4f} [{la_:+.4f}, {ha_:+.4f}] {sga}")
        return {"contrast": f"{pa_name}-{pb_name}", "stratum": tag,
                "delta_auroc": round(pt, 5), "auroc_ci_lo": round(lo, 5), "auroc_ci_hi": round(hi, 5),
                "auroc_significant": sig.strip() == "*",
                "delta_acc": round(pa_, 5), "acc_ci_lo": round(la_, 5), "acc_ci_hi": round(ha_, 5),
                "acc_significant": sga.strip() == "*"}

    print("\n== Paired scan-bootstrap dAUROC + dAccuracy (95% CI) ==")
    drows = []
    for tag, mask in [("full", full), ("sigma>tau", strat_hi), ("sigma<=tau", ~strat_hi)]:
        for a, b in [("carve_xview_gate", "zeroshot"),
                     ("carve_xview_gate", "carve_xview"),
                     ("carve_xview", "zeroshot"),
                     ("carve_xview", "bem"),
                     ("bem", "zeroshot"),
                     ("tent", "zeroshot"),
                     ("ml_tta", "zeroshot")]:
            r = report(a, b, mask, tag)
            if r:
                drows.append(r)
        print("")
    pd.DataFrame(drows).to_csv(args.out_dir / f"endtask_deltas_{TAG}.csv", index=False)
    frac = float(strat_hi.mean())
    print(f"sigma_pred>tau covers {frac:.1%} of scans (tau={args.gate_tau}). "
          f"Hypothesis: gate>zeroshot AND gate>=always-on, gain concentrated in sigma>tau.")


if __name__ == "__main__":
    main()
