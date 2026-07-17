#!/usr/bin/env python3
r"""
carve_xview_adapt.py -- end-task two-stage TTA driver (CT-CLIP), 6 methods + gate.

Implements the CARVE two-stage protocol per scan, episodic (reset to source each
scan), reusing the proven building blocks from the production pipeline (same as
margin_bem_sim.py):

  Stage 1 (no grad) on the retained low-entropy views:
    - score V weak 3D views, keep low-entropy K (CARVE selection),
    - p_bar = mean retained probs;  k_hat = clip(round(sum p_bar), 1, 8),
    - consensus mask + cross-view stability w  (compute_reliability_and_cardinality),
    - sigma_pred = std over labels of p_bar  (label-free gate proxy).
  Stage 2 (grad) on the retained views:
    - method-specific loss; update ONLY selected visual-norm params; then
    - recompute the prediction on the unperturbed volume.

Methods (--method):
  zeroshot    : no update (source prediction).
  tent        : mean binary-entropy minimisation over retained views.
  ml_tta      : single-view BEM top-k^ (V=1) -- the ML-TTA / CARVE-w/o-MV analogue.
  bem         : multi-view BEM top-k^.
  carve_xview : multi-view reliability-weighted BEM (mode=abstain), always-on.
  carve_xview_gate : carve_xview with the sigma_pred gate -- skip the update (return
                     zero-shot) when sigma_pred(x) <= --gate_tau.

Logs per scan: final probs, sigma_pred, gate_applied, wall-clock (s) and peak GPU
memory (MB). Saves one preds .pt per (method, seed). Aggregate with
carve_xview_endtask_agg.py (paired bootstrap CIs, sigma_pred strata, mem/time).

DATASETS (all wired in load_dataset()):
  radchest : RadChestNPZDataset (16-label). root /datasets/ctrate/Radchest.
  luna     : LUNANIfTIDataset (binary nodule). root /datasets/ctrate/LUNA/luna16/
             all_subsets, labels image_id,label, prompt name "nodule".
  ccii     : LUNANIfTIDataset slice-directory loader (3-class). 0/1/2 one-hot
             expanded to class_0/1/2 (matches the existing CC-CCII zero-shot runs).
             root /home/ailarmz/projects/aip-lsigal/ailarmz/CT-CLIP/CC-CCII/images/
             demo data, labels metadata/train_list_CCII.csv (img_list,label).

Run via run_carve_xview_endtask.sh (GPU).
"""
from __future__ import annotations
import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
_TM_REPO = str(_SCRIPT_DIR.parent / "third_party" / "transformer_maskgit")
sys.path.insert(0, str(_SCRIPT_DIR / "pipeline"))
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, _TM_REPO)
for _mod in list(sys.modules):
    if _mod == "transformer_maskgit" or _mod.startswith("transformer_maskgit."):
        del sys.modules[_mod]

from radchest_zero_shot_npz_with_preprocess_tent_label_chunk_grouping import (  # noqa: E402
    build_ctclip, configure_model_for_tent, collect_norm_params,
    RadChestNPZDataset, repeat_batchencoding, select_low_entropy_view_ids,
    estimate_k_hat_from_probs, weak_3d_augment, bem_topk_loss,
    binary_entropy_from_probs,
)
from carve_xview import (  # noqa: E402
    compute_reliability_and_cardinality, carve_xview_loss,
    build_pseudo_targets, pseudolabel_loss,
)


def _load_pipeline_module(subdir, modname):
    """Load a sibling pipeline variant under a unique module name (avoids clashing
    with the scripts/ module already imported). LUNA + CC-CCII reuse the LUNA
    NIfTI/MetaImage/slice-directory loader from these variants."""
    import importlib.util
    p = _SCRIPT_DIR / subdir / "radchest_zero_shot_npz_with_preprocess_tent_label_chunk_grouping.py"
    spec = importlib.util.spec_from_file_location(modname, str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_luna_like(mod, data_root, df, id_col, label_cols, args):
    return mod.LUNANIfTIDataset(
        data_root=data_root, df=df, id_col=id_col, label_cols=label_cols,
        target_shape=(args.target_d, args.target_h, args.target_w),
        apply_rescale_if_available=True, resample_if_available=True,
        hu_min=-1000.0, hu_max=1000.0, scale_divisor=1000.0,
        target_z_spacing=1.5, target_x_spacing=0.75, target_y_spacing=0.75,
    )


class CTRateInternalDataset:
    """CT-RATE validation (internal) — same official depth-240 preprocessing as the
    fixed zero-shot scorer (per-scan slope/intercept, resample to 1.5/0.75/0.75,
    /1000, crop/pad to (1,Z,480,480)). Returns (vname, x[1,Z,480,480], y[L])."""

    def __init__(self, labels_csv, meta_csv, target_d, subset_ids=None):
        import ctclip_zeroshot_official as _Z
        self._Z = _Z; _Z.Z_DEPTH = int(target_d)
        self.df = pd.read_csv(labels_csv)
        if subset_ids is not None:
            self.df = self.df[self.df.iloc[:, 0].astype(str).isin(set(subset_ids))].reset_index(drop=True)
        self.id_col = self.df.columns[0]
        self.label_cols = list(self.df.columns[1:])
        self.meta = pd.read_csv(meta_csv).set_index("VolumeName")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]; v = str(row[self.id_col])
        path = self._Z.ctrate_path(v)
        mr = self.meta.loc[v]
        slope = float(mr["RescaleSlope"]); inter = float(mr["RescaleIntercept"])
        xy = float(str(mr["XYSpacing"]).strip("[]").split(",")[0]); z = float(mr["ZSpacing"])
        x = self._Z.official_preprocess(path, slope, inter, xy, z)   # [1,Z,480,480]
        y = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return v, x, y


def _read_subset_ids(path):
    if not path:
        return None
    if path.endswith(".csv"):
        return [str(x) for x in pd.read_csv(path).iloc[:, 0].tolist()]
    return [ln.strip() for ln in open(path) if ln.strip()]


def load_dataset(args):
    """Return (dataset, label_cols, prompt_names).
    label_cols index the GT columns; prompt_names build the pos/neg text prompts."""
    if args.dataset == "ctrate":
        ds = CTRateInternalDataset(args.labels_csv, args.ctrate_meta_csv, args.target_d,
                                   subset_ids=_read_subset_ids(args.subset_ids))
        return ds, ds.label_cols, ds.label_cols
    if args.dataset == "radchest":
        df = pd.read_csv(args.test_csv)
        id_col = df.columns[0]; label_cols = list(df.columns[1:])
        ds = RadChestNPZDataset(
            radchest_root=args.radchest_root, df=df, id_col=id_col, label_cols=label_cols,
            target_shape=(args.target_d, args.target_h, args.target_w),
            apply_rescale_if_available=True, resample_if_available=True,
            hu_min=-1000.0, hu_max=1000.0, scale_divisor=1000.0,
            target_z_spacing=1.5, target_x_spacing=0.75, target_y_spacing=0.75,
            metadata_csv=args.metadata_csv,
        )
        return ds, label_cols, label_cols
    elif args.dataset == "luna":
        mod = _load_pipeline_module("pipeline_luna", "luna_pipeline_mod")
        df = pd.read_csv(args.test_csv)            # columns: image_id, label (0/1)
        id_col = df.columns[0]; label_cols = list(df.columns[1:])   # ["label"]
        prompt_names = [args.luna_class_name]      # binary: prompt uses class name ("nodule")
        return _build_luna_like(mod, args.radchest_root, df, id_col, label_cols, args), label_cols, prompt_names
    elif args.dataset == "ccii":
        mod = _load_pipeline_module("pipeline_ccii", "ccii_pipeline_mod")
        df = pd.read_csv(args.test_csv)            # columns: img_list, label (0/1/2)
        df = mod.maybe_expand_multiclass_labels(df)   # -> img_list, class_0, class_1, class_2
        id_col = df.columns[0]; label_cols = list(df.columns[1:])   # ["class_0","class_1","class_2"]
        prompt_names = label_cols                   # matches the existing CC-CCII zero-shot runs
        return _build_luna_like(mod, args.radchest_root, df, id_col, label_cols, args), label_cols, prompt_names
    raise ValueError(args.dataset)

POS_T, NEG_T = "There is {label}.", "There is no {label}."
NUM_VIEWS = 8
KEEP_FRAC = 0.25
AUG_HU_STD, AUG_CROP = 0.02, 0.03
K_MIN, K_MAX = 1, 8
NEG_WEIGHT = 0.8
LR, STEPS = 1e-5, 2

MULTIVIEW = {"bem", "carve_xview", "carve_xview_gate", "tent"}
GATED = {"carve_xview_gate"}


def build_label_tokens(tokenizer, label_cols):
    pos, neg = [], []
    for ln in label_cols:
        pos.append(tokenizer([POS_T.format(label=ln)], padding=True, truncation=True, return_tensors="pt"))
        neg.append(tokenizer([NEG_T.format(label=ln)], padding=True, truncation=True, return_tensors="pt"))
    return pos, neg


# ---- fast image-once encoding -----------------------------------------------
# The CTCLIP forward (return_loss=False, use_all_token_embeds=False) reduces to a
# normalized dot product:  s = (l2norm(to_text_latent(text_cls)) .
#                                l2norm(to_visual_latent(mean_pool(visual(img))))) * temp
# Calling model(text,img) once per label re-encodes the IMAGE L times (the OOM at
# z=240). Since the text tower is FROZEN during TTA (only visual-norm params train),
# the L pos/neg text latents are constant: encode them once, encode the image once
# per view, and contract. This is numerically identical to the per-label model()
# path and ~2L x fewer image-encoder forwards.  Toggle with --fast_encode (default 1).
FAST = True
TXT = {"pos": None, "neg": None, "temp": None}    # cached [L,dim] text latents + temp


def _l2n(t):
    return F.normalize(t, dim=-1)


@torch.no_grad()
def _encode_text_latents(model, toks, device):
    lats = []
    for tok in toks:
        t = repeat_batchencoding(tok, 1).to(device)
        enc = model.text_transformer(t.input_ids, attention_mask=t.attention_mask)[0][:, 0, :]
        lats.append(_l2n(model.to_text_latent(enc)))
    return torch.cat(lats, dim=0)                 # [L, dim]


def _encode_image_latent(model, x):
    """l2-normalized image latent [1,dim]; carries grad iff x's graph is live."""
    enc = model.visual_transformer(x, return_encoded_tokens=True)
    enc = torch.mean(enc, dim=1)
    enc = enc.reshape(enc.shape[0], -1)
    return _l2n(model.to_visual_latent(enc))       # [1, dim]


def disable_dropout(model):
    """Set every Dropout module to eval -> deterministic scoring while the model
    stays in train() mode (gradient checkpointing + BN batch-stats still work).
    Must be re-applied after each model.train() reset (train() re-enables dropout)."""
    for m in model.modules():
        if "Drop" in m.__class__.__name__:        # nn.Dropout*, PatchDropout, DropPath, ...
            m.eval()


def cache_text_latents(model, pos_toks, neg_toks, device):
    TXT["pos"] = _encode_text_latents(model, pos_toks, device)   # [L,dim] constant
    TXT["neg"] = _encode_text_latents(model, neg_toks, device)
    TXT["temp"] = model.temperature.exp().detach()


@torch.no_grad()
def probs_ng(model, x, pos_toks, neg_toks, device):
    L = len(pos_toks); p = torch.zeros(L, device=device)
    if FAST:
        img = _encode_image_latent(model, x)                    # [1,dim], no grad
        s_pos = (TXT["pos"] * img).sum(-1) * TXT["temp"]        # [L]
        s_neg = (TXT["neg"] * img).sum(-1) * TXT["temp"]        # [L]
        return F.softmax(torch.stack([s_neg, s_pos], dim=-1), dim=-1)[:, 1]
    for j in range(L):
        s_pos = model(repeat_batchencoding(pos_toks[j], 1).to(device), x, device=device)
        s_neg = model(repeat_batchencoding(neg_toks[j], 1).to(device), x, device=device)
        p[j] = F.softmax(torch.stack([s_neg, s_pos], dim=-1), dim=-1)[0, 1]
    return p


def margins_grad(model, x, pos_toks, neg_toks, device):
    """Per-label logit margin d = s_pos - s_neg, WITH grad. returns [L]."""
    if FAST:
        img = _encode_image_latent(model, x)                    # [1,dim], grad
        return ((TXT["pos"] - TXT["neg"]) * img).sum(-1) * TXT["temp"]   # [L]
    L = len(pos_toks); ds = []
    for j in range(L):
        s_pos = model(repeat_batchencoding(pos_toks[j], 1).to(device), x, device=device)
        s_neg = model(repeat_batchencoding(neg_toks[j], 1).to(device), x, device=device)
        ds.append((s_pos - s_neg).reshape(()))
    return torch.stack(ds)                       # [L]


def probs_grad(model, x, pos_toks, neg_toks, device):
    """Per-label prob WITH grad (for BEM/Tent which take probs). returns [1,L].
    Image-grad flows only through the positive prompt (negative detached), matching
    the per-label path where s_neg is computed under no_grad."""
    if FAST:
        img = _encode_image_latent(model, x)                    # [1,dim], grad
        s_pos = (TXT["pos"] * img).sum(-1) * TXT["temp"]        # [L] grad
        s_neg = (TXT["neg"] * img.detach()).sum(-1) * TXT["temp"]   # [L] no grad
        return F.softmax(torch.stack([s_neg, s_pos], dim=-1), dim=-1)[:, 1].unsqueeze(0)
    L = len(pos_toks); ps = []
    for j in range(L):
        with torch.no_grad():
            s_neg = model(repeat_batchencoding(neg_toks[j], 1).to(device), x, device=device)
        s_pos = model(repeat_batchencoding(pos_toks[j], 1).to(device), x, device=device)
        ps.append(F.softmax(torch.stack([s_neg, s_pos], dim=-1), dim=-1)[0:1, 1])
    return torch.stack(ps, dim=-1)               # [1, L]


def select_adapt_params(model, target):
    """Track-E adaptation-target ablation. Returns the param list the optimizer trains.
    norm_all = visual-transformer LayerNorm scale+bias (current default);
    ln_scale / ln_bias = only the LN weight / bias; final_block = the last temporal
    transformer block (most expressive feature target)."""
    import re
    params, names, *_ = collect_norm_params(model, image_only=True, visual_scope="transformer_only")
    nm = dict(zip(names, params))
    if target == "norm_all":
        return list(nm.values())
    if target == "ln_scale":
        return [p for n, p in nm.items() if n.endswith(".weight")]
    if target == "ln_bias":
        return [p for n, p in nm.items() if n.endswith(".bias")]
    if target == "final_block":
        ps = []
        for n, p in model.named_parameters():
            if re.search(r"enc_temporal_transformer\.layers\.3\.", n):
                p.requires_grad_(True); ps.append(p)
        return ps
    raise ValueError(f"unknown adapt_target {target!r}")


def adapt_one(model, x, pos_toks, neg_toks, optimizer, device, method, gate_tau,
              card_mode="sum_round", oracle_k=None, fixed_k=None):
    """Run one scan's adaptation; returns (final_probs[L], sigma_pred, gate_applied, khat).

    card_mode selects the cardinality used by the BEM / CARVE objective (Track A):
      sum_round (default) = clip(round(sum p_bar)); oracle = true #positives (uses
      labels -> upper bound only, NOT deployable); gap = largest-gap estimator;
      fixed = constant fixed_k. zeroshot/tent ignore cardinality entirely.

    Mode discipline: SCORING (stage-1 views + the final saved prediction) runs in
    eval() so it is fully deterministic and exactly reproduces the per-label model()
    path (CTViT/BERT are stochastic in train() even with dropout disabled). The
    gradient UPDATE runs in train()+disable_dropout (where backward builds a graph);
    the encoder stochasticity there is only a gradient signal, not an output."""
    # ---- Stage 1: no-grad view scoring, selection, consensus, sigma_pred ----
    model.eval()
    with torch.no_grad():
        V = 1 if method == "ml_tta" else NUM_VIEWS
        vp = []
        for v in range(V):
            xv = x if v == 0 else weak_3d_augment(x, hu_jitter_std=AUG_HU_STD,
                                                  crop_shift_ratio=AUG_CROP, seed=SEED_BASE * 1000 + v)
            vp.append(probs_ng(model, xv, pos_toks, neg_toks, device))
            if v > 0:
                del xv
            if x.is_cuda:
                torch.cuda.empty_cache()
        vp = torch.stack(vp, dim=0)                          # [V, L]
        # cross-view disagreement (std across views, averaged over labels) for augsweep
        view_dis = float(vp.std(dim=0).mean().item()) if V > 1 else 0.0
        if V > 1:
            keep_ids, _ = select_low_entropy_view_ids(vp.unsqueeze(1), keep_frac=KEEP_FRAC)
            keep = keep_ids[:, 0]
        else:
            keep = torch.zeros(1, dtype=torch.long, device=device)
        sel = vp[keep, :]                                    # [K, L]
        p_bar = sel.mean(0)
        sigma_pred = float(p_bar.std().item())
        k_hat = estimate_k_hat_from_probs(p_bar.unsqueeze(0), min_k=K_MIN, max_k=K_MAX)  # [1]
        w, khat_i, pbar_mask = compute_reliability_and_cardinality(sel, k_min=K_MIN, k_max=K_MAX)

        # ---- Track A: cardinality override (oracle / largest-gap / fixed) ----
        ko = None
        if card_mode == "oracle" and oracle_k is not None:
            ko = int(oracle_k)
        elif card_mode == "gap":
            ps = torch.sort(p_bar, descending=True).values            # [L]
            gaps = (ps[:-1] - ps[1:])                                 # gap after rank k (1-indexed)
            kc = list(range(K_MIN, min(K_MAX, p_bar.numel() - 1) + 1))
            ko = max(kc, key=lambda k: float(gaps[k - 1]))
        elif card_mode == "fixed" and fixed_k is not None:
            ko = int(fixed_k)
        if ko is not None:
            ko = int(max(K_MIN, min(K_MAX, p_bar.numel(), ko)))
            k_hat = torch.tensor([ko], device=device, dtype=torch.long)
            topk_idx = sel.topk(ko, dim=1).indices                    # [K, ko]
            member = torch.zeros_like(sel, dtype=torch.bool); member.scatter_(1, topk_idx, True)
            w = member.float().mean(dim=0)                            # cross-view stability @ ko
            pbar_mask = torch.zeros(p_bar.numel(), dtype=torch.bool, device=device)
            pbar_mask[p_bar.topk(ko).indices] = True
            khat_i = ko

        # ---- pseudo-label self-training target (fixed from the consensus) ----
        pl_yhat, pl_mask, _ = build_pseudo_targets(
            p_bar, khat=int(k_hat.item()), mode=PL_MODE, margin=PL_MARGIN,
            k_min=K_MIN, k_max=K_MAX)

    gate_applied = method in GATED and sigma_pred <= gate_tau
    do_update = (method != "zeroshot") and not gate_applied

    if do_update:
        model.train(); disable_dropout(model)            # grad-enabled; deterministic ops off
        for _ in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            K = sel.shape[0]
            scale = 1.0 / max(K, 1)
            for ki in range(K):
                vidx = int(keep[ki].item())
                xv = x if vidx == 0 else weak_3d_augment(x, hu_jitter_std=AUG_HU_STD,
                                                         crop_shift_ratio=AUG_CROP, seed=SEED_BASE * 1000 + vidx)
                if method in ("carve_xview", "carve_xview_gate"):
                    d = margins_grad(model, xv, pos_toks, neg_toks, device)        # [L]
                    loss = carve_xview_loss(d.unsqueeze(0), w, pbar_mask,
                                            lambda_neg=NEG_WEIGHT, mode="abstain")
                elif method == "tent":
                    pv = probs_grad(model, xv, pos_toks, neg_toks, device)          # [1,L]
                    loss = binary_entropy_from_probs(pv).mean()
                elif method == "pl_selftrain":
                    pv = probs_grad(model, xv, pos_toks, neg_toks, device)          # [1,L]
                    loss = pseudolabel_loss(pv, pl_yhat, pl_mask, neg_weight=NEG_WEIGHT)
                else:  # bem, ml_tta
                    pv = probs_grad(model, xv, pos_toks, neg_toks, device)
                    loss, _ = bem_topk_loss(pv, k_hat=k_hat, neg_weight=NEG_WEIGHT)
                (loss * scale).backward()
                del loss
                if vidx != 0:
                    del xv
                if x.is_cuda:
                    torch.cuda.empty_cache()
            optimizer.step()

    model.eval()                                         # deterministic final prediction
    with torch.no_grad():
        final = probs_ng(model, x, pos_toks, neg_toks, device)
    khat = float(k_hat.item()) if hasattr(k_hat, "item") else float(k_hat)
    return final.detach().cpu().numpy(), sigma_pred, bool(gate_applied), khat, view_dis


ALL_METHODS = ["zeroshot", "tent", "ml_tta", "bem", "carve_xview", "carve_xview_gate",
               "pl_selftrain"]

# pseudo-label self-training config (set from CLI in main())
PL_MODE = "conf_hard"
PL_MARGIN = 0.1

# Per-view augmentation base seed. weak_3d_augment reseeds on the view index; without
# folding in the global --seed, every --seed produced byte-identical predictions
# (seed-ensembling was a no-op). SEED_BASE makes the augmented views depend on --seed
# (view seed = SEED_BASE*1000 + v) so seeds actually vary; view 0 stays the clean image.
SEED_BASE = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--methods", default=",".join(ALL_METHODS),
                    help="comma list; all 6 run per scan so the (expensive) volume load is shared.")
    ap.add_argument("--gate_tau", type=float, default=0.091, help="sigma_pred gate (CARVE-Gated sweep).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="radchest", choices=["radchest", "luna", "ccii", "ctrate"],
                    help="radchest (16-label), luna (binary), or ccii (3-class one-hot).")
    ap.add_argument("--radchest_root", default="/datasets/ctrate/Radchest",
                    help="data root; for luna pass all_subsets root, for ccii the CC-CCII images root.")
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--metadata_csv", default="", help="RAD-ChestCT only; unused for luna/ccii.")
    ap.add_argument("--luna_class_name", default="nodule",
                    help="For --dataset luna: class name used in the text prompts.")
    ap.add_argument("--num_shards", type=int, default=1, help="strided scan sharding (jobs).")
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--max_scans", type=int, default=-1, help="cap scans (smoke tests).")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--num_views", type=int, default=8,
                    help="weak 3D views (K retained = round(num_views*0.25)); lower to fit memory at z=240.")
    ap.add_argument("--keep_frac", type=float, default=0.25,
                    help="CARVE low-entropy view-selection fraction; 1.0 = no selection (ablation).")
    ap.add_argument("--lambda_neg", type=float, default=0.8, help="negative-term weight in BEM/CARVE loss.")
    ap.add_argument("--aug_hu_std", type=float, default=0.02, help="weak-view HU Gaussian jitter std (augsweep).")
    ap.add_argument("--aug_crop", type=float, default=0.03, help="weak-view spatial shift ratio (augsweep).")
    ap.add_argument("--target_d", type=int, default=40)
    ap.add_argument("--target_h", type=int, default=480)
    ap.add_argument("--target_w", type=int, default=480)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--card_mode", default="sum_round",
                    choices=["sum_round", "oracle", "gap", "fixed"],
                    help="cardinality for BEM/CARVE: sum_round (default), oracle (uses labels; "
                         "upper-bound only), gap (largest-gap), fixed (--fixed_k).")
    ap.add_argument("--fixed_k", type=int, default=3, help="k for --card_mode fixed.")
    ap.add_argument("--adapt_target", default="norm_all",
                    choices=["norm_all", "ln_scale", "ln_bias", "final_block"],
                    help="Track-E: which params the optimizer adapts.")
    ap.add_argument("--steps", type=int, default=2, help="TTA gradient steps per scan.")
    ap.add_argument("--pl_mode", default="conf_hard",
                    choices=["conf_hard", "topk_hard", "topk_conf"],
                    help="pseudo-label self-training target (pl_selftrain method).")
    ap.add_argument("--pl_margin", type=float, default=0.1,
                    help="confidence margin |p-0.5| for pl_selftrain conf/topk_conf modes.")
    ap.add_argument("--score_temp_scale", type=float, default=1.0,
                    help="AUDIT: divide the applied logit-scale (model temp) by this factor "
                         "(temperature scaling). >1 softens toward 0.5; fit on SOURCE only. "
                         "Affects BOTH the scored prediction and the adaptation gradient.")
    ap.add_argument("--fast_encode", type=int, default=1,
                    help="1 = image-once encoding (cache frozen text latents); 0 = per-label model() path.")
    ap.add_argument("--labels_csv", default="", help="ctrate internal: predicted-labels CSV.")
    ap.add_argument("--ctrate_meta_csv", default="", help="ctrate internal: validation metadata CSV.")
    ap.add_argument("--subset_ids", default="", help="optional .txt/.csv of scan ids to restrict to (fixed subset).")
    ap.add_argument("--radchest_spacing", type=float, default=0.8,
                    help="RAD-ChestCT: override per-scan spacing isotropically (mm). NPZ is at "
                         "final_spacing=0.8 iso but headers give original DICOM spacing -> over-pad. "
                         "0 = use metadata (buggy) spacing.")
    args = ap.parse_args()

    if args.dataset == "radchest" and args.radchest_spacing > 0:
        import radchest_zero_shot_npz_with_preprocess_tent_label_chunk_grouping as _radmod
        _orig_extract = _radmod._extract_spacing_and_rescale
        _sp = args.radchest_spacing
        def _extract_fixed(row):
            slope, inter, xy, z = _orig_extract(row)
            return slope, inter, _sp, _sp            # corrected: NPZ is 0.8 iso
        _radmod._extract_spacing_and_rescale = _extract_fixed
        print(f"[radchest] spacing overridden to {_sp} mm iso (corrected pipeline)")

    global NUM_VIEWS, FAST, STEPS, PL_MODE, PL_MARGIN, SEED_BASE, KEEP_FRAC, NEG_WEIGHT, AUG_HU_STD, AUG_CROP
    NUM_VIEWS = args.num_views
    KEEP_FRAC = args.keep_frac
    NEG_WEIGHT = args.lambda_neg
    AUG_HU_STD = args.aug_hu_std
    AUG_CROP = args.aug_crop
    FAST = bool(args.fast_encode)
    STEPS = args.steps
    PL_MODE = args.pl_mode
    PL_MARGIN = args.pl_margin
    SEED_BASE = args.seed
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    assert all(m in ALL_METHODS for m in methods), f"bad --methods {methods}"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[seed={args.seed} shard {args.shard_idx}/{args.num_shards}] loading CT-CLIP ...")
    model, tokenizer = build_ctclip(args.weights, device=device, checkpoint_during_training=True)
    configure_model_for_tent(model)
    disable_dropout(model)                       # deterministic scoring (model stays train())

    ds, label_cols, prompt_names = load_dataset(args)
    pos_toks, neg_toks = build_label_tokens(tokenizer, prompt_names)
    if FAST:
        model.eval()                                  # deterministic text latents
        cache_text_latents(model, pos_toks, neg_toks, device)
        if args.score_temp_scale != 1.0:
            TXT["temp"] = TXT["temp"] / float(args.score_temp_scale)
            print(f"[AUDIT] score_temp_scale={args.score_temp_scale} -> effective temp={float(TXT['temp']):.4f}")
        print(f"[fast_encode] cached {TXT['pos'].shape[0]} text latents (dim={TXT['pos'].shape[1]}), "
              f"temp={float(TXT['temp']):.4f}")
    n_all = len(ds) if args.max_scans <= 0 else min(args.max_scans, len(ds))
    # strided shard of scan indices (balances heterogeneous load across jobs)
    shard = [i for i in range(n_all) if i % args.num_shards == args.shard_idx]
    print(f"dataset={args.dataset} L={len(label_cols)} methods={methods} "
          f"scans={len(shard)}/{n_all} (shard {args.shard_idx}/{args.num_shards})")

    base_state = deepcopy(model.state_dict())
    # per-method accumulators
    acc = {m: {"ids": [], "probs": [], "ys": [], "sig": [], "gate": [], "wall": [], "mem": [], "khat": []}
           for m in methods}

    for si, i in enumerate(shard):
        sid, x_raw, y_raw = ds[i]                # load + preprocess volume ONCE
        x = x_raw.unsqueeze(0).to(device)
        y_np = y_raw.numpy().astype(int)
        for method in methods:
            model.load_state_dict(base_state, strict=True); model.train()  # episodic reset
            disable_dropout(model)                                          # keep scoring deterministic
            params = select_adapt_params(model, args.adapt_target)
            if not params:
                params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=args.lr)
            if device == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            p_final, sigma_pred, gate_applied, khat, view_dis = adapt_one(
                model, x, pos_toks, neg_toks, optimizer, device, method, args.gate_tau,
                card_mode=args.card_mode, oracle_k=int(y_np.sum()), fixed_k=args.fixed_k)
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0.0
            a = acc[method]
            a["ids"].append(str(sid)); a["probs"].append(p_final); a["ys"].append(y_np)
            a["sig"].append(sigma_pred); a["gate"].append(gate_applied)
            a["wall"].append(dt); a["mem"].append(peak_mb); a["khat"].append(khat)
            a.setdefault("vdis", []).append(view_dis)
            del optimizer
            if device == "cuda":
                torch.cuda.empty_cache()
        del x
        if device == "cuda":
            torch.cuda.empty_cache()
        if si % 10 == 0:
            print(f"  scan {si+1}/{len(shard)} (idx {i})", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_sh{args.shard_idx}of{args.num_shards}" if args.num_shards > 1 else ""
    for method in methods:
        a = acc[method]
        out = {
            "ids": a["ids"],
            "probs": torch.tensor(np.stack(a["probs"]), dtype=torch.float32),
            "y_true": torch.tensor(np.stack(a["ys"]), dtype=torch.float32),
            "label_cols": label_cols,
            "sigma_pred": torch.tensor(a["sig"], dtype=torch.float32),
            "k_hat": torch.tensor(a["khat"], dtype=torch.float32),
            "gate_applied": torch.tensor(a["gate"], dtype=torch.bool),
            "wallclock_s": torch.tensor(a["wall"], dtype=torch.float32),
            "peak_mem_mb": torch.tensor(a["mem"], dtype=torch.float32),
            "view_disagreement": torch.tensor(a.get("vdis", []), dtype=torch.float32),
            "method": method, "seed": args.seed, "gate_tau": args.gate_tau,
            "lambda_neg": NEG_WEIGHT, "aug_hu_std": AUG_HU_STD, "aug_crop": AUG_CROP,
            "num_views": NUM_VIEWS, "keep_frac": KEEP_FRAC,
            "num_shards": args.num_shards, "shard_idx": args.shard_idx,
        }
        fp = args.out_dir / f"{args.dataset}_{method}_seed{args.seed}{suffix}.pt"
        torch.save(out, fp)
        print(f"  wrote {fp}  (n={len(a['probs'])}, mean t={np.mean(a['wall']):.2f}s, "
              f"mean peak={np.mean(a['mem']):.0f}MB, gated={int(np.sum(a['gate']))})")


if __name__ == "__main__":
    main()
