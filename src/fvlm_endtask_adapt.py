#!/usr/bin/env python3
r"""fVLM end-task TTA driver (second architecture). Mirrors carve_xview_adapt.py's protocol
(episodic per-scan, 8 views, keep 25% low-entropy, Adam lr 1e-5, 2 steps, gate tau=0.091,
lambda_neg 0.8) and saves the SAME .pt format, so carve_xview_endtask_agg.py / metrics_panel.py
/ the oracle scripts work unchanged. Uses the IDENTICAL objective functions as CT-CLIP.

fVLM specifics: lavis 3D ViT + 4 organ cross-attention heads; native input (1,1,112,256,352)
in [0,1] with HU window [-1150,350]; gradients flow in eval mode (no checkpointing), 86
Dropout modules disabled for determinism; adapt the 50 visual-encoder LayerNorm params.
"""
from __future__ import annotations
import argparse, sys, time, glob, os
from copy import deepcopy
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import nibabel as nib

R = Path(__file__).resolve().parent
for p in [str(R), str(R/"pipeline"), str(R.parent/"external"/"transformer_maskgit")]:
    sys.path.insert(0, p)
for m in list(sys.modules):
    if m.startswith("transformer_maskgit"):
        del sys.modules[m]
from fvlm_wrapper import build_fvlm, _FVLM_IMG_SIZE
import ctclip_zeroshot_official as Z
from radchest_zero_shot_npz_with_preprocess_tent_label_chunk_grouping import (  # identical objectives
    binary_entropy_from_probs, estimate_k_hat_from_probs, bem_topk_loss,
    select_low_entropy_view_ids, weak_3d_augment, find_volume_path, load_npz_volume,
    _extract_spacing_and_rescale)
from carve_xview import (compute_reliability_and_cardinality, carve_xview_loss,
                         build_pseudo_targets, pseudolabel_loss)

FVLM_W = os.path.join(os.environ.get("FVLM_ROOT", "."), "model.pth")
POS_T, NEG_T = "there is absolutely {label} present.", "there is absolutely not {label} present."
NUM_VIEWS, KEEP_FRAC, K_MIN, K_MAX = 8, 0.25, 1, 8
NEG_WEIGHT, LR, STEPS, AUG_HU_STD, AUG_CROP = 0.8, 1e-5, 2, 0.02, 0.03
ALL_METHODS = ["zeroshot", "tent", "ml_tta", "bem", "carve_xview", "carve_xview_gate",
               "pl_selftrain"]
MULTIVIEW = {"bem", "carve_xview", "carve_xview_gate", "tent"}
PL_MODE = "conf_hard"
PL_MARGIN = 0.1
SEED_BASE = 0   # per-view aug base seed; view seed = SEED_BASE*1000+v so --seed varies preds
GATED = {"carve_xview_gate"}
TXT = {"pos": None, "neg": None}


def hu_window(hu):
    x = np.clip(hu, -1150.0, 350.0); return (x + 1150.0) / 1500.0     # official fVLM [0,1]


def fvlm_input_from_hu(hu_dhw):
    t = torch.from_numpy(hu_window(hu_dhw).astype(np.float32))[None, None]
    return F.interpolate(t, size=_FVLM_IMG_SIZE, mode="trilinear", align_corners=False)[0]  # (1,112,256,352)


def select_fvlm_params(model, target):
    """Track-fVLM adaptation targets (§7). Returns trainable param list; enables grad."""
    for p in model.parameters():
        p.requires_grad_(False)
    core = model.core
    ps = []
    if target == "norm_all":
        for nm, mm in core.visual_encoder.named_modules():
            if isinstance(mm, nn.LayerNorm):
                for pn, pp in mm.named_parameters(recurse=False):
                    if pn in ("weight", "bias"): ps.append(pp)
    elif target == "organ_query":
        ps = [core.query_tokens]
    elif target == "organ_proj":
        for vp in core.vision_projs:
            ps += list(vp.parameters())
    elif target == "final_attn":
        ps = list(core.attention.parameters())
    elif target in ("temperature", "per_label_bias"):
        ps = []                                  # handled as extra params in main (operating-point targets)
    else:
        raise ValueError(target)
    for p in ps: p.requires_grad_(True)
    return ps


def disable_dropout(model):
    for m in model.modules():
        if "Drop" in m.__class__.__name__:
            m.eval()


@torch.no_grad()
def cache_text(model, tok, prompt_names, device):
    def feats(tpl):
        fs = []
        for n in prompt_names:
            t = tok([tpl.format(label=n)], return_tensors="pt", padding=True, truncation=True).to(device)
            fs.append(model._encode_text(t))
        return torch.cat(fs, 0)
    TXT["pos"] = feats(POS_T); TXT["neg"] = feats(NEG_T)


def img_feat(model, x, grad):
    if grad:
        return model._encode_image_organs(x).mean(0)        # [1,E] grad
    with torch.no_grad():
        return model._encode_image_organs(x).mean(0)


def probs_ng(model, x):
    f = img_feat(model, x, False)
    sp = (TXT["pos"] * f).sum(-1); sn = (TXT["neg"] * f).sum(-1)
    return F.softmax(torch.stack([sn, sp], 0), 0)[1]          # [L]


def margins_grad(model, x):
    f = img_feat(model, x, True)
    return ((TXT["pos"] - TXT["neg"]) * f).sum(-1)            # [L]


def probs_grad(model, x):
    f = img_feat(model, x, True)
    sp = (TXT["pos"] * f).sum(-1); sn = (TXT["neg"] * f.detach()).sum(-1)
    return F.softmax(torch.stack([sn, sp], 0), 0)[1].unsqueeze(0)  # [1,L]


def adapt_one(model, x, optimizer, device, method, gate_tau, card_mode="sum_round", oracle_k=None):
    model.eval()
    with torch.no_grad():
        V = 1 if method == "ml_tta" else NUM_VIEWS
        vp = []
        for v in range(V):
            xv = x if v == 0 else weak_3d_augment(x, hu_jitter_std=AUG_HU_STD, crop_shift_ratio=AUG_CROP, seed=SEED_BASE*1000+v)
            vp.append(probs_ng(model, xv))
        vp = torch.stack(vp, 0)                                # [V,L]
        if V > 1:
            keep = select_low_entropy_view_ids(vp.unsqueeze(1), keep_frac=KEEP_FRAC)[0][:, 0]
        else:
            keep = torch.zeros(1, dtype=torch.long, device=device)
        sel = vp[keep, :]; p_bar = sel.mean(0)
        sigma_pred = float(p_bar.std().item())
        k_hat = estimate_k_hat_from_probs(p_bar.unsqueeze(0), min_k=K_MIN, max_k=K_MAX)
        w, khat_i, pbar_mask = compute_reliability_and_cardinality(sel, k_min=K_MIN, k_max=K_MAX)
        ko = int(oracle_k) if (card_mode == "oracle" and oracle_k is not None) else None
        if ko is not None:
            ko = max(K_MIN, min(K_MAX, p_bar.numel(), ko)); k_hat = torch.tensor([ko], device=device)
            ti = sel.topk(ko, 1).indices; mem = torch.zeros_like(sel, dtype=torch.bool); mem.scatter_(1, ti, True)
            w = mem.float().mean(0); pbar_mask = torch.zeros(p_bar.numel(), dtype=torch.bool, device=device)
            pbar_mask[p_bar.topk(ko).indices] = True
        pl_yhat, pl_mask, _ = build_pseudo_targets(
            p_bar, khat=int(k_hat.item()), mode=PL_MODE, margin=PL_MARGIN, k_min=K_MIN, k_max=K_MAX)
    gate_applied = method in GATED and sigma_pred <= gate_tau
    do_update = (method != "zeroshot") and not gate_applied
    if do_update:
        for _ in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            K = sel.shape[0]; scale = 1.0 / max(K, 1)
            for ki in range(K):
                vidx = int(keep[ki].item())
                xv = x if vidx == 0 else weak_3d_augment(x, hu_jitter_std=AUG_HU_STD, crop_shift_ratio=AUG_CROP, seed=SEED_BASE*1000+vidx)
                if method in ("carve_xview", "carve_xview_gate"):
                    d = margins_grad(model, xv); loss = carve_xview_loss(d.unsqueeze(0), w, pbar_mask, lambda_neg=NEG_WEIGHT, mode="abstain")
                elif method == "tent":
                    loss = binary_entropy_from_probs(probs_grad(model, xv)).mean()
                elif method == "pl_selftrain":
                    loss = pseudolabel_loss(probs_grad(model, xv), pl_yhat, pl_mask, neg_weight=NEG_WEIGHT)
                else:
                    loss, _ = bem_topk_loss(probs_grad(model, xv), k_hat=k_hat, neg_weight=NEG_WEIGHT)
                (loss * scale).backward(); del loss
                torch.cuda.empty_cache()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        final = probs_ng(model, x)
    return final.detach().cpu().numpy(), sigma_pred, bool(gate_applied), float(k_hat.item())


def load_ctrate(args):
    df = pd.read_csv(args.labels_csv); meta = pd.read_csv(args.meta_csv).set_index("VolumeName")
    sub = set(l.strip() for l in open(args.subset_ids)) if args.subset_ids else None
    if sub: df = df[df.iloc[:, 0].astype(str).isin(sub)].reset_index(drop=True)
    labels = list(df.columns[1:])
    def get(i):
        v = str(df.iloc[i, 0]); mr = meta.loc[v]
        slope = float(mr["RescaleSlope"]); inter = float(mr["RescaleIntercept"])
        img = nib.load(Z.ctrate_path(v)).get_fdata().astype(np.float32) * slope + inter
        hu = np.transpose(img, (2, 1, 0))                     # (Z,H,W)
        y = df.iloc[i][labels].values.astype(np.float32)
        return v, fvlm_input_from_hu(hu), torch.tensor(y)
    return df, labels, labels, get


def load_radchest(args):
    df = pd.read_csv(args.labels_csv); meta = pd.read_csv(args.meta_csv)
    labels = list(df.columns[1:])
    def get(i):
        v = str(df.iloc[i, 0]); path = find_volume_path(args.radchest_root, v)
        rm = meta[meta["VolumeAcc_DEID"] == (v + ".npz")]
        slope, inter, xy, z = _extract_spacing_and_rescale(rm)
        ct = load_npz_volume(path).astype(np.float32)         # already HU, (Z,H,W)
        if slope is not None: ct = ct  # npz ct already HU
        y = df.iloc[i][labels].values.astype(np.float32)
        return v, fvlm_input_from_hu(ct), torch.tensor(y)
    return df, labels, labels, get


def _load_pipeline_module(subdir, modname):
    """Load the LUNA/CC-CCII sibling pipeline loader under a unique module name."""
    import importlib.util
    p = R / subdir / "radchest_zero_shot_npz_with_preprocess_tent_label_chunk_grouping.py"
    spec = importlib.util.spec_from_file_location(modname, str(p))
    mod = importlib.util.module_from_spec(spec); sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_luna_like(args, subdir, modname, expand_multiclass, prompt_names_from):
    """Shared LUNA/CC-CCII loader for fVLM. Reuses the dataset's file IO + spatial
    resampling but sets the fVLM intensity window directly (hu [-1150,350], divisor
    1.0, target_shape = fVLM size) so __getitem__ emits fVLM-windowed HU that
    fvlm_input_from_hu maps to [0,1] with an identity resize. prompt_names_from:
    'class' -> use --luna_class_name (binary); 'cols' -> use the label columns."""
    mod = _load_pipeline_module(subdir, modname)
    df = pd.read_csv(args.labels_csv)
    if expand_multiclass:
        df = mod.maybe_expand_multiclass_labels(df)
    id_col = df.columns[0]; label_cols = list(df.columns[1:])
    prompt_names = [args.luna_class_name] if prompt_names_from == "class" else label_cols
    D, H, W = _FVLM_IMG_SIZE
    ds = mod.LUNANIfTIDataset(
        data_root=args.radchest_root, df=df, id_col=id_col, label_cols=label_cols,
        target_shape=(D, H, W), apply_rescale_if_available=True, resample_if_available=True,
        hu_min=-1150.0, hu_max=350.0, scale_divisor=1.0,
        target_z_spacing=1.5, target_x_spacing=0.75, target_y_spacing=0.75)
    def get(i):
        sid, x, y = ds[i]                                     # x = [1,D,H,W] fVLM-windowed HU
        hu = np.asarray(x).squeeze()                          # (D,H,W) HU
        return str(sid), fvlm_input_from_hu(hu), torch.tensor(np.asarray(y, dtype=np.float32))
    return df, label_cols, prompt_names, get


def load_luna(args):
    return _load_luna_like(args, "pipeline_luna", "luna_fvlm_mod",
                           expand_multiclass=False, prompt_names_from="class")


def load_ccii(args):
    return _load_luna_like(args, "pipeline_ccii", "ccii_fvlm_mod",
                           expand_multiclass=True, prompt_names_from="cols")


def main():
    global STEPS, PL_MODE, PL_MARGIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ctrate", "radchest", "luna", "ccii"])
    ap.add_argument("--labels_csv", required=True); ap.add_argument("--meta_csv", default="")
    ap.add_argument("--subset_ids", default=""); ap.add_argument("--radchest_root", default="/datasets/ctrate/Radchest")
    ap.add_argument("--luna_class_name", default="nodule", help="binary prompt class name for --dataset luna.")
    ap.add_argument("--methods", default=",".join(ALL_METHODS)); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate_tau", type=float, default=0.091); ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--adapt_target", default="norm_all",
                    choices=["norm_all", "organ_query", "organ_proj", "temperature", "per_label_bias", "final_attn"])
    ap.add_argument("--card_mode", default="sum_round", choices=["sum_round", "oracle"])
    ap.add_argument("--pl_mode", default="conf_hard", choices=["conf_hard", "topk_hard", "topk_conf"])
    ap.add_argument("--pl_margin", type=float, default=0.1)
    ap.add_argument("--num_shards", type=int, default=1); ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--max_scans", type=int, default=-1); ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    PL_MODE = args.pl_mode; PL_MARGIN = args.pl_margin
    global SEED_BASE; SEED_BASE = args.seed
    torch.manual_seed(args.seed); np.random.seed(args.seed); device = "cuda"
    model, tok = build_fvlm(FVLM_W, device=device); model.eval(); disable_dropout(model)
    loader = {"ctrate": load_ctrate, "radchest": load_radchest,
              "luna": load_luna, "ccii": load_ccii}[args.dataset]
    df, labels, prompt_names, get = loader(args)
    cache_text(model, tok, prompt_names, device)
    print(f"[fVLM] dataset={args.dataset} L={len(labels)} prompts={prompt_names} "
          f"methods={methods} cached {TXT['pos'].shape}")
    n_all = len(df) if args.max_scans <= 0 else min(args.max_scans, len(df))
    shard = [i for i in range(n_all) if i % args.num_shards == args.shard_idx]
    STEPS = args.steps
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    norm_params = select_fvlm_params(model, args.adapt_target)
    print(f"[fVLM] adapt_target={args.adapt_target} lr={args.lr} steps={STEPS} "
          f"trainable_params={len(norm_params)} elems={sum(p.numel() for p in norm_params)}")
    acc = {m: {"ids": [], "probs": [], "ys": [], "sig": [], "gate": [], "wall": [], "mem": [], "khat": []} for m in methods}
    for si, i in enumerate(shard):
        sid, x_raw, y_raw = get(i); x = x_raw.unsqueeze(0).to(device); y_np = y_raw.numpy().astype(int)
        for method in methods:
            model.load_state_dict(base, strict=True); model.eval(); disable_dropout(model)
            for pp in norm_params: pp.requires_grad_(True)
            opt = torch.optim.Adam(norm_params, lr=args.lr)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
            p_final, sig, gate, khat = adapt_one(model, x, opt, device, method, args.gate_tau,
                                                 card_mode=args.card_mode, oracle_k=int(y_np.sum()))
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            a = acc[method]; a["ids"].append(str(sid)); a["probs"].append(p_final); a["ys"].append(y_np)
            a["sig"].append(sig); a["gate"].append(gate); a["wall"].append(dt)
            a["mem"].append(torch.cuda.max_memory_allocated() / 1e6); a["khat"].append(khat)
            del opt; torch.cuda.empty_cache()
        del x; torch.cuda.empty_cache()
        if si % 10 == 0: print(f"  scan {si+1}/{len(shard)}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suf = f"_sh{args.shard_idx}of{args.num_shards}" if args.num_shards > 1 else ""
    for m in methods:
        a = acc[m]
        torch.save({"ids": a["ids"], "probs": torch.tensor(np.stack(a["probs"]), dtype=torch.float32),
                    "y_true": torch.tensor(np.stack(a["ys"]), dtype=torch.float32), "label_cols": labels,
                    "sigma_pred": torch.tensor(a["sig"]), "k_hat": torch.tensor(a["khat"]),
                    "gate_applied": torch.tensor(a["gate"], dtype=torch.bool),
                    "wallclock_s": torch.tensor(a["wall"]), "peak_mem_mb": torch.tensor(a["mem"]),
                    "method": m, "seed": args.seed},
                   args.out_dir / f"{args.dataset}_{m}_seed{args.seed}{suf}.pt")
        print(f"  wrote {m}: n={len(a['probs'])} meanT={np.mean(a['wall']):.2f}s peak={np.mean(a['mem']):.0f}MB gated={int(np.sum(a['gate']))}")


if __name__ == "__main__":
    main()
