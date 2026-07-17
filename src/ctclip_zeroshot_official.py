#!/usr/bin/env python3
r"""
ctclip_zeroshot_official.py -- faithful CT-CLIP zero-shot reproduction.

Replicates the OFFICIAL recipe (ibrahimhamamci/CT-CLIP, scripts/data_inference_nii.py
+ zero_shot.py) EXACTLY, to fix the broken 0.577/0.512 reproduction. Standalone:
does NOT touch any CARVE/TTA code.

Official recipe (verbatim from the repo):
  - per-scan slope/intercept from metadata:  HU = slope*raw + intercept
  - HU clip [-1000, 1000]
  - resample to target spacing (z,x,y) = (1.5, 0.75, 0.75)  (trilinear)
  - normalize:  / 1000   -> [-1, 1]
  - center crop/pad to (H,W,D) = (480, 480, 240), PAD VALUE = -1
  - final tensor (1, 240, 480, 480)            <-- 240 z-slices (we had 40!)
  - prompt pair:  "{label} is present." / "{label} is not present."
    present prob = softmax([s_present, s_not_present])[0]

Usage (sharded; 240-depth forward is ~6x our old 40-depth):
  python ctclip_zeroshot_official.py --dataset ctrate --num_shards 20 --shard_idx 0 \
      --out results_zsfix/ctrate_official.pt
"""
from __future__ import annotations
import argparse, glob, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib

_SCRIPT_DIR = Path(__file__).resolve().parent
_TM = str(_SCRIPT_DIR / "transformer_maskgit")
sys.path.insert(0, str(_SCRIPT_DIR / "scripts")); sys.path.insert(0, str(_SCRIPT_DIR)); sys.path.insert(0, _TM)
for _m in list(sys.modules):
    if _m == "transformer_maskgit" or _m.startswith("transformer_maskgit."):
        del sys.modules[_m]
from radchest_zero_shot_npz_with_preprocess_tent_label_chunk_grouping import (  # noqa
    build_ctclip, repeat_batchencoding, load_npz_volume, _extract_spacing_and_rescale, find_volume_path)


def radchest_preprocess(npz_path, z_spacing, xy_spacing):
    """Official recipe applied to a RAD-ChestCT NPZ ('ct', already HU-clipped int16).
    Resample to target spacing, /1000, official crop/pad to (480,480,Z_DEPTH), pad=-1."""
    vol = load_npz_volume(npz_path).astype(np.float32)            # (z, y, x) HU in [-1000,1000]
    vol = np.clip(vol, -1000, 1000)
    t = torch.tensor(vol).unsqueeze(0).unsqueeze(0)               # (1,1,z,y,x)
    vol = resize_array(t, (z_spacing, xy_spacing, xy_spacing), TARGET_SPACING)[0][0]  # (z',y',x')
    vol = np.transpose(vol, (1, 2, 0))                            # (y', x', z')
    vol = (vol / 1000).astype(np.float32)
    return _crop_pad_permute(torch.tensor(vol))

TARGET_SPACING = (1.5, 0.75, 0.75)   # (z, x, y)
CTRATE_ROOT = "/datasets/ctrate/Validation/Data/dataset/valid"
# ablation knobs (defaults = official). Z_DEPTH/PROMPT set in main() from args.
Z_DEPTH = 240
PROMPT_STYLE = "official"  # official: "{l} is present." | old: "there is absolutely {l} present."


def resize_array(array, current_spacing, target_spacing):
    """OFFICIAL resize_array (verbatim)."""
    orig = array.shape[2:]
    sf = [current_spacing[i] / target_spacing[i] for i in range(len(orig))]
    new = [int(orig[i] * sf[i]) for i in range(len(orig))]
    return F.interpolate(array, size=new, mode="trilinear", align_corners=False).cpu().numpy()


def official_preprocess(nii_path, slope, intercept, xy_spacing, z_spacing):
    """OFFICIAL nii_img_to_tensor (verbatim from data_inference_nii.py)."""
    img = nib.load(str(nii_path)).get_fdata()
    img = slope * img + intercept
    img = np.clip(img, -1000, 1000)
    img = img.transpose(2, 0, 1)                                   # -> (z, x, y)
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    img = resize_array(t, (z_spacing, xy_spacing, xy_spacing), TARGET_SPACING)
    img = img[0][0]
    img = np.transpose(img, (1, 2, 0))                             # -> (x, y, z)
    img = (img / 1000).astype(np.float32)
    return _crop_pad_permute(torch.tensor(img))


def _crop_pad_permute(t):
    """OFFICIAL center crop/pad to (480,480,Z_DEPTH) pad=-1, then permute -> (1,Zd,480,480)."""
    dh, dw, dd = (480, 480, Z_DEPTH)
    h, w, d = t.shape
    hs = max((h - dh) // 2, 0); he = min(hs + dh, h)
    ws = max((w - dw) // 2, 0); we = min(ws + dw, w)
    ds_ = max((d - dd) // 2, 0); de = min(ds_ + dd, d)
    t = t[hs:he, ws:we, ds_:de]
    ph = (dh - t.size(0)) // 2; pw = (dw - t.size(1)) // 2; pd = (dd - t.size(2)) // 2
    t = F.pad(t, (pd, dd - t.size(2) - pd, pw, dw - t.size(1) - pw, ph, dh - t.size(0) - ph), value=-1)
    return t.permute(2, 0, 1).unsqueeze(0)                         # (1, Z_DEPTH, 480, 480)


def ctrate_path(vname):
    stem = vname.replace(".nii.gz", "")
    p = stem.split("_")                                            # valid, N, a, M
    cand = os.path.join(CTRATE_ROOT, f"{p[0]}_{p[1]}", f"{p[0]}_{p[1]}_{p[2]}", vname)
    if os.path.exists(cand):
        return cand
    g = glob.glob(os.path.join(CTRATE_ROOT, "**", vname), recursive=True)
    return g[0] if g else None


def macro_auroc(probs, y):
    from sklearn.metrics import roc_auc_score
    a = [roc_auc_score(y[:, j], probs[:, j]) for j in range(y.shape[1]) if y[:, j].min() != y[:, j].max()]
    return float(np.mean(a)) if a else float("nan")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/datasets/ctrate/CT-CLIP-weights/models/CT-CLIP-Related/CT-CLIP_v2.pt")
    ap.add_argument("--dataset", default="ctrate", choices=["ctrate", "radchest"])
    ap.add_argument("--labels_csv", default="/home/ailarmz/projects/aip-lsigal/ailarmz/CT-CLIP/metadata/valid_predicted_labels.csv")
    ap.add_argument("--meta_csv", default="/home/ailarmz/projects/aip-lsigal/ailarmz/CT-CLIP/metadata/validation_metadata.csv")
    ap.add_argument("--radchest_root", default="/datasets/ctrate/Radchest")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--max_scans", type=int, default=-1)
    ap.add_argument("--z_depth", type=int, default=240, help="ablation: 240 official, 40 = old bug.")
    ap.add_argument("--prompt_style", default="official", choices=["official", "old"])
    ap.add_argument("--radchest_spacing", type=float, default=-1.0,
                    help="if >0, override RAD spacing isotropically (mm). Fix: NPZ is at final_spacing=0.8 iso, "
                         "but _extract_spacing_and_rescale returns the ORIGINAL DICOM spacing -> over-padding.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    global Z_DEPTH, PROMPT_STYLE
    Z_DEPTH = args.z_depth; PROMPT_STYLE = args.prompt_style
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = build_ctclip(args.weights, device=device)
    # suspect #2: assert load completeness
    ckpt = torch.load(args.weights, map_location="cpu"); state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"[ckpt] keys={len(state)} missing={len(miss)} unexpected={len(unexp)} (unexpected ok if just position_ids)")
    model.eval()

    df = pd.read_csv(args.labels_csv)
    id_col = df.columns[0]; label_cols = list(df.columns[1:])
    if args.dataset == "ctrate":
        meta = pd.read_csv(args.meta_csv).set_index("VolumeName")
    else:
        meta_df = pd.read_csv(args.meta_csv)                       # RAD-ChestCT metadata (VolumeAcc_DEID)
    n_all = len(df) if args.max_scans <= 0 else min(args.max_scans, len(df))
    shard = [i for i in range(n_all) if i % args.num_shards == args.shard_idx]
    print(f"dataset={args.dataset} labels L={len(label_cols)} scans={len(shard)}/{n_all} (shard {args.shard_idx}/{args.num_shards})")

    ids, probs, ys = [], [], []
    for si, i in enumerate(shard):
        row = df.iloc[i]; vname = str(row[id_col])
        try:
            if args.dataset == "ctrate":
                path = ctrate_path(vname)
                if path is None or vname not in meta.index:
                    continue
                mr = meta.loc[vname]
                slope = float(mr["RescaleSlope"]); intercept = float(mr["RescaleIntercept"])
                xy = float(str(mr["XYSpacing"]).strip("[]").split(",")[0]); z = float(mr["ZSpacing"])
                img = official_preprocess(path, slope, intercept, xy, z).to(device)
            else:  # radchest NPZ
                path = find_volume_path(args.radchest_root, vname)
                rm = meta_df[meta_df["VolumeAcc_DEID"] == (vname + ".npz")]
                _, _, xy, z = _extract_spacing_and_rescale(rm)
                if args.radchest_spacing > 0:
                    xy = z = args.radchest_spacing       # corrected: NPZ is at final_spacing iso
                img = radchest_preprocess(path, z, xy).to(device)
        except Exception as e:
            print(f"  skip {vname}: {e}"); continue
        img = img.unsqueeze(0)                                     # (1,1,Z,480,480) match official collate
        if si == 0:
            print(f"  input shape {tuple(img.shape)}  range[{img.min():.2f},{img.max():.2f}]")
        p = np.zeros(len(label_cols), np.float32)
        for j, lab in enumerate(label_cols):
            if PROMPT_STYLE == "old":
                text = [f"there is absolutely {lab} present.", f"there is absolutely not {lab} present."]
            else:
                text = [f"{lab} is present.", f"{lab} is not present."]
            tok = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            out = model(tok, img, device=device)                  # [2] similarities
            p[j] = torch.softmax(out, dim=0)[0].item()             # present = index 0
        ids.append(vname); probs.append(p); ys.append(row[label_cols].values.astype(int))
        if si % 10 == 0:
            print(f"  {si+1}/{len(shard)} {vname}", flush=True)

    probs = np.stack(probs); ys = np.stack(ys)
    suffix = f"_sh{args.shard_idx}of{args.num_shards}" if args.num_shards > 1 else ""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fp = args.out.with_name(args.out.stem + suffix + args.out.suffix)
    torch.save({"ids": ids, "probs": torch.tensor(probs), "y_true": torch.tensor(ys.astype(np.float32)),
                "label_cols": label_cols}, fp)
    auc = macro_auroc(probs, ys)
    print(f"wrote {fp}  n={len(ids)}  >>> macro AUROC (this shard) = {auc:.4f} <<<")


if __name__ == "__main__":
    main()
