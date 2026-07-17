#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
radchest_zero_shot_npz_with_preprocess_tent_label_chunk_grouping.py

CT-CLIP zero-shot classification with:
1) plain zero-shot inference
2) anchor-guided multi-label TTA
3) ML-TTA-inspired multi-view BEM-style TTA (memory-safe)

Notes:
- This is a CT adaptation of the multi-label TTA idea, not an exact repo port.
- The ML-TTA branch is implemented in a memory-safe way:
    * build weak views
    * score all views without grad
    * keep lowest-entropy views per sample
    * recompute each kept view once per batch (not per sample)
"""

import os
import glob
import json
import re
import argparse
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
from copy import deepcopy
from contextlib import nullcontext

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel

from transformer_maskgit import CTViT
from ct_clip import CTCLIP


def _check_deepspeed():
    try:
        import deepspeed
        return deepspeed, True
    except Exception:
        return None, False


logger = logging.getLogger("radchest_zero_shot_npz_tent")

DEFAULT_WEIGHTS = "/datasets/ctrate/CT-CLIP-weights/models/CT-CLIP-Related/CT-CLIP_v2.pt"
DEFAULT_RADCHEST_ROOT = "/datasets/ctrate/Radchest"

DEFAULT_TRAIN_CSV = "/home/ailarmz/projects/aip-lsigal/ailarmz/CT-CLIP/CT_CLIP/Radchest-labels/train_list.csv"
DEFAULT_VAL_CSV   = "/home/ailarmz/projects/aip-lsigal/ailarmz/CT-CLIP/CT_CLIP/Radchest-labels/val_list.csv"
DEFAULT_TEST_CSV  = "/home/ailarmz/projects/aip-lsigal/ailarmz/CT-CLIP/CT_CLIP/Radchest-labels/test_list.csv"

DEFAULT_RESULTS_DIR = "./results_radchest_zero_shot_tent/"


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def find_volume_path(radchest_root: str, sample_id: str) -> str:
    direct = os.path.join(radchest_root, f"{sample_id}.npz")
    if os.path.exists(direct):
        return direct
    matches = glob.glob(os.path.join(radchest_root, "**", f"{sample_id}.npz"), recursive=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find NPZ for id={sample_id} under {radchest_root}")


def load_npz_volume(path: str) -> np.ndarray:
    z = np.load(path)
    keys = list(z.keys())
    if len(keys) == 1:
        arr = z[keys[0]]
    else:
        for k in ["arr_0", "volume", "ct", "data", "img", "image"]:
            if k in z:
                arr = z[k]
                break
        else:
            arr = z[keys[0]]
    return arr.astype(np.float32)


def _ensure_dhw(vol: np.ndarray) -> np.ndarray:
    if vol.ndim == 4:
        if vol.shape[-1] <= 4:
            vol = vol[..., 0]
        elif vol.shape[0] <= 4:
            vol = vol[0, ...]
        else:
            vol = vol[..., 0]

    if vol.ndim != 3:
        raise ValueError(f"Expected 3D (or 4D w/ channel) NPZ volume, got shape={vol.shape}")

    dims = vol.shape
    d_idx = int(np.argmin(dims))
    if d_idx != 0:
        vol = np.moveaxis(vol, d_idx, 0)
    return vol


_float_pat = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_float_any(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    m = _float_pat.findall(str(x))
    return float(m[0]) if m else None


def _extract_spacing_and_rescale(row: pd.DataFrame):
    if row is None or len(row) == 0:
        return None, None, None, None

    slope = row["orig_slope"].iloc[0] if "orig_slope" in row.columns else None
    intercept = row["orig_inter"].iloc[0] if "orig_inter" in row.columns else None
    xy = row["orig_yxspacing"].iloc[0] if "orig_yxspacing" in row.columns else None
    z = row["SliceThickness"].iloc[0] if "SliceThickness" in row.columns else None

    return (
        _parse_float_any(slope),
        _parse_float_any(intercept),
        _parse_float_any(xy),
        _parse_float_any(z),
    )


def _resample_to_spacing(
    tensor_5d: torch.Tensor,
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
) -> torch.Tensor:
    if tensor_5d.ndim != 5:
        raise ValueError(f"Expected 5D tensor [N,C,D,H,W], got {tensor_5d.shape}")

    _, _, D, H, W = tensor_5d.shape
    old_shape = (D, H, W)

    scaling = [
        float(current_spacing[0]) / float(target_spacing[0]),
        float(current_spacing[1]) / float(target_spacing[1]),
        float(current_spacing[2]) / float(target_spacing[2]),
    ]
    new_shape = [max(1, int(round(old_shape[i] * scaling[i]))) for i in range(3)]
    return F.interpolate(tensor_5d, size=new_shape, mode="trilinear", align_corners=False)


def preprocess_ct(
    vol: np.ndarray,
    target_shape: Tuple[int, int, int] = (240, 480, 480),
    apply_rescale_if_available: bool = True,
    slope: Optional[float] = None,
    intercept: Optional[float] = None,
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
    scale_divisor: float = 1000.0,
    resample_if_available: bool = True,
    z_spacing: Optional[float] = None,
    xy_spacing: Optional[float] = None,
    target_z_spacing: float = 1.5,
    target_x_spacing: float = 0.75,
    target_y_spacing: float = 0.75,
) -> torch.Tensor:
    vol = _ensure_dhw(vol).astype(np.float32)

    if apply_rescale_if_available and (slope is not None) and (intercept is not None):
        vol = slope * vol + intercept

    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol / float(scale_divisor)).astype(np.float32)

    t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
    if resample_if_available and (z_spacing is not None) and (xy_spacing is not None):
        current = (float(z_spacing), float(xy_spacing), float(xy_spacing))
        target = (float(target_z_spacing), float(target_x_spacing), float(target_y_spacing))
        t = _resample_to_spacing(t, current, target)

    t = F.interpolate(t, size=target_shape, mode="trilinear", align_corners=False)
    return t.squeeze(0)


# -------------------------
# Dataset
# -------------------------
class RadChestNPZDataset(Dataset):
    def __init__(
        self,
        radchest_root: str,
        df: pd.DataFrame,
        id_col: str,
        label_cols: List[str],
        target_shape: Tuple[int, int, int],
        apply_rescale_if_available: bool,
        resample_if_available: bool,
        hu_min: float,
        hu_max: float,
        scale_divisor: float,
        target_z_spacing: float,
        target_x_spacing: float,
        target_y_spacing: float,
        metadata_csv: str,
    ):
        self.radchest_root = radchest_root
        self.df = df.reset_index(drop=True)
        self.id_col = id_col
        self.label_cols = label_cols
        self.target_shape = target_shape

        self.apply_rescale_if_available = apply_rescale_if_available
        self.resample_if_available = resample_if_available
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.scale_divisor = scale_divisor
        self.target_z_spacing = target_z_spacing
        self.target_x_spacing = target_x_spacing
        self.target_y_spacing = target_y_spacing

        self.df_metadata = pd.read_csv(metadata_csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sid = str(row[self.id_col])

        row_metadata = self.df_metadata[self.df_metadata["VolumeAcc_DEID"] == (sid + ".npz")]
        slope, intercept, xy_spacing, z_spacing = _extract_spacing_and_rescale(row_metadata)

        vol_path = find_volume_path(self.radchest_root, sid)
        vol = load_npz_volume(vol_path)

        x = preprocess_ct(
            vol,
            target_shape=self.target_shape,
            apply_rescale_if_available=self.apply_rescale_if_available,
            slope=slope,
            intercept=intercept,
            hu_min=self.hu_min,
            hu_max=self.hu_max,
            scale_divisor=self.scale_divisor,
            resample_if_available=self.resample_if_available,
            z_spacing=z_spacing,
            xy_spacing=xy_spacing,
            target_z_spacing=self.target_z_spacing,
            target_x_spacing=self.target_x_spacing,
            target_y_spacing=self.target_y_spacing,
        )

        y = torch.from_numpy(row[self.label_cols].values.astype(np.float32))
        return sid, x, y


def collate_fn(batch):
    sids, xs, ys = zip(*batch)
    return list(sids), torch.stack(xs, dim=0), torch.stack(ys, dim=0)


# -------------------------
# Model builder
# -------------------------
def build_ctclip(weights_path: str, device: str, checkpoint_during_training: bool = False):
    tokenizer = BertTokenizer.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized",
        do_lower_case=True,
    )
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    text_encoder.resize_token_embeddings(len(tokenizer))

    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
    )

    clip = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=294912,
        dim_text=768,
        dim_latent=512,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False,
        checkpoint_during_training=checkpoint_during_training,
    )

    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    for k in ["text_transformer.embeddings.position_ids", "text_encoder.embeddings.position_ids"]:
        state.pop(k, None)

    missing, unexpected = clip.load_state_dict(state, strict=False)
    logger.info(f"[ckpt load] missing={len(missing)} unexpected={len(unexpected)}")

    clip.to(device)
    clip.eval()
    return clip, tokenizer


# -------------------------
# Metrics
# -------------------------
def _auc_roc_binary_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    n = y_true.shape[0]
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(y_score)
    scores_sorted = y_score[order]

    ranks = np.empty(n, dtype=np.float64)
    i, rank = 0, 1.0
    while i < n:
        j = i + 1
        while j < n and scores_sorted[j] == scores_sorted[i]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        ranks[order[i:j]] = avg_rank
        rank += (j - i)
        i = j

    rank_sum_pos = ranks[y_true == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def multilabel_metrics_from_probs(probs: torch.Tensor, y_true: torch.Tensor,
                                  label_cols: List[str], thr: float = 0.5):
    probs_np = probs.detach().cpu().float().numpy()
    y_np = y_true.detach().cpu().float().numpy().astype(np.int64)
    pred_np = (probs_np >= thr).astype(np.int64)

    N, L = y_np.shape
    per_label, aucs, f1s, accs, precs, recs = {}, [], [], [], [], []

    for j, name in enumerate(label_cols):
        yj, pj, sj = y_np[:, j], pred_np[:, j], probs_np[:, j]

        tp = int(((pj == 1) & (yj == 1)).sum())
        fp = int(((pj == 1) & (yj == 0)).sum())
        fn = int(((pj == 0) & (yj == 1)).sum())
        tn = int(((pj == 0) & (yj == 0)).sum())

        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = (2 * prec * rec) / (prec + rec + 1e-12)
        acc  = (tp + tn) / (tp + tn + fp + fn + 1e-12)
        auc  = _auc_roc_binary_numpy(yj, sj)

        per_label[name] = {
            "auroc": float(auc) if not np.isnan(auc) else None,
            "f1": float(f1),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "support_pos": int(yj.sum()),
            "support_neg": int((1 - yj).sum()),
        }

        if not np.isnan(auc):
            aucs.append(auc)
        f1s.append(f1)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)

    macro_auroc = float(np.mean(aucs)) if aucs else None
    summary = {
        "threshold": thr,
        "num_samples": int(N),
        "num_labels": int(L),
        "labels_with_valid_auroc": int(len(aucs)),
        "micro": {},
        "macro": {
            "auroc": macro_auroc,
            "f1": float(np.mean(f1s)),
            "accuracy": float(np.mean(accs)),
            "precision": float(np.mean(precs)),
            "recall": float(np.mean(recs)),
        },
        "subset_acc": float((pred_np == y_np).all(axis=1).mean()),
    }

    y_flat = y_np.reshape(-1)
    p_flat = pred_np.reshape(-1)
    s_flat = probs_np.reshape(-1)

    tp = int(((p_flat == 1) & (y_flat == 1)).sum())
    fp = int(((p_flat == 1) & (y_flat == 0)).sum())
    fn = int(((p_flat == 0) & (y_flat == 1)).sum())
    tn = int(((p_flat == 0) & (y_flat == 0)).sum())

    micro_prec = tp / (tp + fp + 1e-12)
    micro_rec = tp / (tp + fn + 1e-12)
    summary["micro"] = {
        "auroc": float(_auc_roc_binary_numpy(y_flat.astype(np.int64), s_flat)) if (y_flat.sum() > 0 and y_flat.sum() < len(y_flat)) else None,
        "f1": float((2 * micro_prec * micro_rec) / (micro_prec + micro_rec + 1e-12)),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn + 1e-12)),
        "precision": float(micro_prec),
        "recall": float(micro_rec),
    }
    return summary, per_label


# -------------------------
# Helpers
# -------------------------
def repeat_batchencoding(tok, B: int):
    tok_rep = {}
    for k, v in tok.items():
        tok_rep[k] = v.repeat(B, 1) if torch.is_tensor(v) else v
    return type(tok)(tok_rep)


NORM_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
)


def copy_model_and_optimizer(model, optimizer):
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model_for_tent(model: nn.Module):
    model.train()
    model.requires_grad_(False)

    norm_count = 0
    bn_count = 0
    for m in model.modules():
        if isinstance(m, NORM_TYPES):
            norm_count += 1
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                bn_count += 1
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    logger.info(f"[Tent] configured model: norm_layers={norm_count}, batchnorm_layers={bn_count}")
    return model


def is_text_module_name(module_name: str) -> bool:
    return module_name.startswith("text_transformer")


def is_visual_module_name(module_name: str) -> bool:
    return module_name.startswith("visual_transformer") or "visual_transformer." in module_name


def visual_scope_match(module_name: str, scope: str) -> bool:
    if scope == "all":
        return is_visual_module_name(module_name)
    if scope == "patch_only":
        return (
            module_name.startswith("visual_transformer.to_patch_emb")
            or module_name.startswith("visual_transformer.to_patch_emb_first_frame")
        )
    if scope == "transformer_only":
        return (
            module_name.startswith("visual_transformer.enc_spatial_transformer")
            or module_name.startswith("visual_transformer.enc_temporal_transformer")
        )
    if scope == "patch_and_norm_out":
        return (
            module_name.startswith("visual_transformer.to_patch_emb")
            or module_name.startswith("visual_transformer.to_patch_emb_first_frame")
            or module_name.endswith("enc_spatial_transformer.norm_out")
            or module_name.endswith("enc_temporal_transformer.norm_out")
            or module_name == "visual_transformer.enc_spatial_transformer.norm_out"
            or module_name == "visual_transformer.enc_temporal_transformer.norm_out"
        )
    raise ValueError(f"Unknown visual scope: {scope}")


def collect_norm_params(model: nn.Module, image_only: bool = False, visual_scope: str = "all"):
    params, names = [], []
    text_names, visual_names, other_names = [], [], []

    for module_name, m in model.named_modules():
        if not isinstance(m, NORM_TYPES):
            continue

        keep_module = True
        if image_only:
            keep_module = visual_scope_match(module_name, visual_scope)

        for param_name, p in m.named_parameters(recurse=False):
            if param_name not in {"weight", "bias"} or p is None:
                continue
            if keep_module:
                p.requires_grad_(True)
                params.append(p)
                full_name = f"{module_name}.{param_name}"
                names.append(full_name)

                if is_text_module_name(module_name):
                    text_names.append(full_name)
                elif is_visual_module_name(module_name):
                    visual_names.append(full_name)
                else:
                    other_names.append(full_name)

    return params, names, text_names, visual_names, other_names


def check_model_for_tent(model: nn.Module):
    assert model.training, "Tent needs train mode"
    param_grads = [p.requires_grad for p in model.parameters()]
    assert any(param_grads), "Tent needs some params to update"
    assert not all(param_grads), "Tent should not update all params"
    assert any(isinstance(m, NORM_TYPES) for m in model.modules()), "Tent needs normalization layers"


def pairwise_entropy_from_logits(pair_logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(pair_logits, dim=-1)
    logp = F.log_softmax(pair_logits, dim=-1)
    return -(p * logp).sum(dim=-1)


def parse_label_groups(group_str: Optional[str], num_labels: int) -> List[List[int]]:
    if group_str is None or str(group_str).strip() == "":
        return [[i] for i in range(num_labels)]

    groups = []
    for chunk in str(group_str).split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        idxs = [int(x.strip()) for x in chunk.split(",") if x.strip() != ""]
        if len(idxs) == 0:
            continue
        groups.append(sorted(list(set(idxs))))

    flat = sorted([i for g in groups for i in g])
    if flat != list(range(num_labels)):
        raise ValueError(
            f"Invalid --tent_label_groups. Must cover each label index exactly once. "
            f"Got flat={flat}, expected={list(range(num_labels))}"
        )
    return groups


def estimate_group_cardinality_priors(df: pd.DataFrame, label_cols: List[str], groups: List[List[int]]):
    y = df[label_cols].values.astype(np.float32)
    mus, sigmas = [], []
    for g in groups:
        cg = y[:, g].sum(axis=1)
        mus.append(float(np.mean(cg)))
        sigmas.append(max(float(np.std(cg)), 0.5))
    return mus, sigmas


def binary_entropy_from_probs(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def weak_3d_augment(
    x: torch.Tensor,
    hu_jitter_std: float = 0.03,
    crop_shift_ratio: float = 0.04,
    seed: Optional[int] = None,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
        if x.is_cuda:
            torch.cuda.manual_seed(seed)

    out = x
    if hu_jitter_std > 0:
        out = out + torch.randn_like(out) * hu_jitter_std

    _, _, D, H, W = out.shape
    max_sd = max(1, int(round(D * crop_shift_ratio)))
    max_sh = max(1, int(round(H * crop_shift_ratio)))
    max_sw = max(1, int(round(W * crop_shift_ratio)))

    sd = int(torch.randint(-max_sd, max_sd + 1, (1,), device=x.device).item())
    sh = int(torch.randint(-max_sh, max_sh + 1, (1,), device=x.device).item())
    sw = int(torch.randint(-max_sw, max_sw + 1, (1,), device=x.device).item())

    out = torch.roll(out, shifts=(sd, sh, sw), dims=(2, 3, 4))
    return out


def symmetric_kl_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)

    kl_pq = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    kl_qp = q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))
    return 0.5 * (kl_pq + kl_qp)


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    if mask is None:
        return x.mean()
    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum().clamp_min(eps)
    return (x * mask_f).sum() / denom


def safe_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(x.clamp(min=eps, max=1.0 - eps))


def build_anchor_sets(anchor_probs: torch.Tensor, pos_thr: float = 0.55, neg_thr: float = 0.35, topk_pos: int = 2):
    B, L = anchor_probs.shape
    device = anchor_probs.device

    k_hat = torch.full((B,), min(topk_pos, L), dtype=torch.long, device=device)
    _, top_idx = torch.topk(anchor_probs, k=min(topk_pos, L), dim=1, largest=True, sorted=True)

    pos_mask = torch.zeros_like(anchor_probs, dtype=torch.bool)
    for b in range(B):
        pos_mask[b, top_idx[b]] = True
    pos_mask = pos_mask & (anchor_probs >= pos_thr)

    neg_mask = anchor_probs <= neg_thr
    neg_mask = neg_mask & (~pos_mask)
    return pos_mask, neg_mask, k_hat


def structured_multilabel_tent_loss(
    probs_pos: torch.Tensor,
    groups: List[List[int]],
    group_mus: List[float],
    group_sigmas: List[float],
    reliability_mask: Optional[torch.Tensor] = None,
    use_anchor_topk: bool = True,
    anchor_probs: Optional[torch.Tensor] = None,
    anchor_pos_thr: float = 0.55,
    anchor_neg_thr: float = 0.35,
    topk_pos: int = 2,
    topk_alpha_neg: float = 2.0,
    lambda_topk: float = 1.0,
    lambda_label_entropy: float = 0.02,
    lambda_group_entropy: float = 0.05,
    lambda_cardinality: float = 0.05,
    lambda_anchor_kl: float = 0.0,
):
    device = probs_pos.device
    dtype = probs_pos.dtype

    if reliability_mask is None:
        reliability_mask = torch.ones_like(probs_pos, dtype=torch.bool)

    topk_loss = torch.tensor(0.0, device=device, dtype=dtype)
    pos_count = 0
    neg_count = 0
    mean_k_hat = 0.0

    if use_anchor_topk and anchor_probs is not None:
        pos_mask, neg_mask, k_hat = build_anchor_sets(
            anchor_probs=anchor_probs,
            pos_thr=anchor_pos_thr,
            neg_thr=anchor_neg_thr,
            topk_pos=topk_pos,
        )
        pos_mask = pos_mask & reliability_mask
        neg_mask = neg_mask & reliability_mask

        pos_count = int(pos_mask.sum().item())
        neg_count = int(neg_mask.sum().item())
        mean_k_hat = float(k_hat.float().mean().item())

        pos_term = torch.tensor(0.0, device=device, dtype=dtype)
        neg_term = torch.tensor(0.0, device=device, dtype=dtype)

        if pos_count > 0:
            pos_term = -masked_mean(safe_log(probs_pos), pos_mask)
        if neg_count > 0:
            neg_term = -masked_mean(safe_log(1.0 - probs_pos), neg_mask)

        topk_loss = pos_term + topk_alpha_neg * neg_term

    label_ent = masked_mean(binary_entropy_from_probs(probs_pos), reliability_mask)

    group_ent_terms = []
    card_terms = []
    for gi, g in enumerate(groups):
        pg = probs_pos[:, g]
        rg = reliability_mask[:, g]
        valid_group = rg.any(dim=1)

        if valid_group.any():
            rg_f = rg.to(dtype=dtype)
            group_prob = (pg * rg_f).sum(dim=1) / rg_f.sum(dim=1).clamp_min(1.0)
            group_ent_terms.append(binary_entropy_from_probs(group_prob[valid_group]).mean())
        else:
            group_ent_terms.append(torch.tensor(0.0, device=device, dtype=dtype))

        c_g = (pg * rg.to(dtype=dtype)).sum(dim=1)
        mu = torch.tensor(group_mus[gi], dtype=dtype, device=device)
        sigma = torch.tensor(group_sigmas[gi], dtype=dtype, device=device)
        card_terms.append(((c_g - mu) ** 2 / (sigma ** 2 + 1e-6)).mean())

    group_ent = torch.stack(group_ent_terms).mean() if group_ent_terms else torch.tensor(0.0, device=device, dtype=dtype)
    card_loss = torch.stack(card_terms).mean() if card_terms else torch.tensor(0.0, device=device, dtype=dtype)

    anchor_kl = torch.tensor(0.0, device=device, dtype=dtype)
    if anchor_probs is not None and lambda_anchor_kl > 0:
        anchor_kl = symmetric_kl_probs(probs_pos, anchor_probs).mean()

    total = (
        lambda_topk * topk_loss
        + lambda_label_entropy * label_ent
        + lambda_group_entropy * group_ent
        + lambda_cardinality * card_loss
        + lambda_anchor_kl * anchor_kl
    )

    stats = {
        "topk_loss": float(topk_loss.detach().item()),
        "label_entropy": float(label_ent.detach().item()),
        "group_entropy": float(group_ent.detach().item()),
        "cardinality_loss": float(card_loss.detach().item()),
        "anchor_kl": float(anchor_kl.detach().item()),
        "anchor_pos_count": int(pos_count),
        "anchor_neg_count": int(neg_count),
        "mean_k_hat": float(mean_k_hat),
        "mean_selected_view_entropy": 0.0,
    }
    return total, stats


# -------------------------
# ML-TTA-inspired helpers
# -------------------------
def select_low_entropy_view_ids(view_probs: torch.Tensor, keep_frac: float = 0.5):
    view_entropy = binary_entropy_from_probs(view_probs).mean(dim=-1)  # [V, B]
    V, B = view_entropy.shape
    K = max(1, int(round(V * keep_frac)))
    keep_ids = torch.argsort(view_entropy, dim=0, descending=False)[:K]  # [K, B]
    return keep_ids, view_entropy


def estimate_k_hat_from_probs(
    probs: torch.Tensor, min_k: int = 1, max_k: int = 3, scale: float = 1.0
) -> torch.Tensor:
    k_hat = torch.round(probs.sum(dim=1) * scale).long()
    k_hat = torch.clamp(k_hat, min=min_k, max=max_k)
    return k_hat


def build_topk_masks_from_khat(probs: torch.Tensor, k_hat: torch.Tensor):
    B, L = probs.shape
    max_k = int(k_hat.max().item())
    _, top_idx = torch.topk(probs, k=max_k, dim=1, largest=True, sorted=True)

    pos_mask = torch.zeros_like(probs, dtype=torch.bool)
    for b in range(B):
        kb = int(k_hat[b].item())
        pos_mask[b, top_idx[b, :kb]] = True

    neg_mask = ~pos_mask
    return pos_mask, neg_mask


def bem_topk_loss(
    probs: torch.Tensor,
    k_hat: torch.Tensor,
    neg_weight: float = 0.5,
    eps: float = 1e-6,
    sample_weights: Optional[torch.Tensor] = None,
):
    probs = probs.clamp(min=eps, max=1.0 - eps)
    pos_mask, neg_mask = build_topk_masks_from_khat(probs, k_hat)

    pos_count = pos_mask.sum(dim=1).clamp_min(1)
    neg_count = neg_mask.sum(dim=1).clamp_min(1)

    pos_loss = -(torch.log(probs) * pos_mask.float()).sum(dim=1) / pos_count.float()
    neg_loss = -(torch.log(1.0 - probs) * neg_mask.float()).sum(dim=1) / neg_count.float()

    per_sample = pos_loss + neg_weight * neg_loss
    if sample_weights is not None:
        per_sample = per_sample * sample_weights
        total = per_sample.sum() / (sample_weights.sum() + eps)
    else:
        total = per_sample.mean()
    stats = {
        "mean_pos_count": float(pos_count.float().mean().item()),
        "mean_neg_count": float(neg_count.float().mean().item()),
    }
    return total, stats


def bem_topk_hinge_loss(
    probs: torch.Tensor,
    k_hat: torch.Tensor,
    neg_weight: float = 0.5,
    margin: float = 0.1,
    eps: float = 1e-6,
    sample_weights: Optional[torch.Tensor] = None,
):
    """Hinge-margin BEM: push top-k probs above (1-margin), rest below margin."""
    probs = probs.clamp(min=eps, max=1.0 - eps)
    pos_mask, neg_mask = build_topk_masks_from_khat(probs, k_hat)

    pos_count = pos_mask.sum(dim=1).clamp_min(1)
    neg_count = neg_mask.sum(dim=1).clamp_min(1)

    pos_loss = (F.relu((1.0 - margin) - probs) * pos_mask.float()).sum(dim=1) / pos_count.float()
    neg_loss = (F.relu(probs - margin) * neg_mask.float()).sum(dim=1) / neg_count.float()

    per_sample = pos_loss + neg_weight * neg_loss
    if sample_weights is not None:
        per_sample = per_sample * sample_weights
        total = per_sample.sum() / (sample_weights.sum() + eps)
    else:
        total = per_sample.mean()
    stats = {
        "mean_pos_count": float(pos_count.float().mean().item()),
        "mean_neg_count": float(neg_count.float().mean().item()),
    }
    return total, stats


def make_coronal_view(x: torch.Tensor) -> torch.Tensor:
    """x: [B,1,D,H,W] → coronal view [B,1,D,H,W] with H as the slice axis."""
    D, H, W = x.shape[2], x.shape[3], x.shape[4]
    return F.interpolate(
        x.permute(0, 1, 3, 2, 4).contiguous(),
        size=(D, H, W), mode="trilinear", align_corners=False,
    )


def make_sagittal_view(x: torch.Tensor) -> torch.Tensor:
    """x: [B,1,D,H,W] → sagittal view [B,1,D,H,W] with W as the slice axis."""
    D, H, W = x.shape[2], x.shape[3], x.shape[4]
    return F.interpolate(
        x.permute(0, 1, 4, 2, 3).contiguous(),
        size=(D, H, W), mode="trilinear", align_corners=False,
    )


def infer_view_probs(
    model,
    x: torch.Tensor,
    pos_toks: list,
    neg_toks: list,
    device: torch.device,
) -> torch.Tensor:
    """Compute [B, L] zero-shot probs for a single view, label by label."""
    B, L = x.shape[0], len(pos_toks)
    probs_out = torch.empty((B, L), device=device, dtype=torch.float32)
    for j in range(L):
        tok_pos = repeat_batchencoding(pos_toks[j], B).to(device)
        tok_neg = repeat_batchencoding(neg_toks[j], B).to(device)
        sim_pos = model(tok_pos, x, device=device)
        sim_neg = model(tok_neg, x, device=device)
        pair_logits = torch.stack([sim_neg, sim_pos], dim=-1)
        probs_out[:, j] = F.softmax(pair_logits, dim=-1)[:, 1]
        del tok_pos, tok_neg, sim_pos, sim_neg, pair_logits
    return probs_out


def triplanar_consistency_bem_loss(
    probs: torch.Tensor,          # [B, L]  — with grad
    mean_probs_ng: torch.Tensor,  # [B, L]  — triplanar mean, no grad
    var_probs_ng: torch.Tensor,   # [B, L]  — triplanar variance, no grad
    k_hat: torch.Tensor,          # [B,]
    neg_weight: float = 0.5,
    var_percentile: float = 50.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """
    BEM loss restricted to high-consistency labels (low triplanar variance).

    For each sample:
      1. Threshold variance at var_percentile → high_cons mask [B, L]
      2. Within high_cons, rank by mean_prob and take top k_hat → pseudo-positives
      3. Apply CE loss: pos → 1, remaining high_cons → 0
    """
    B, L = probs.shape
    probs = probs.clamp(min=eps, max=1.0 - eps)

    # per-sample variance threshold
    thr = torch.quantile(var_probs_ng, var_percentile / 100.0, dim=1, keepdim=True)  # [B,1]
    high_cons = var_probs_ng < thr                                                    # [B, L]

    # within high-consistency labels, pick top k_hat as pseudo-positives
    masked_mean = mean_probs_ng.clone()
    masked_mean[~high_cons] = -1.0
    pos_mask = torch.zeros(B, L, dtype=torch.bool, device=probs.device)
    for b in range(B):
        k = int(k_hat[b].item())
        top_idx = torch.argsort(masked_mean[b], descending=True)[:k]
        pos_mask[b, top_idx] = True
    pos_mask = pos_mask & high_cons
    neg_mask = (~pos_mask) & high_cons

    pos_count = pos_mask.sum(dim=1).clamp(min=1).float()
    neg_count = neg_mask.sum(dim=1).clamp(min=1).float()
    pos_loss  = -(torch.log(probs) * pos_mask.float()).sum(dim=1) / pos_count
    neg_loss  = -(torch.log(1.0 - probs) * neg_mask.float()).sum(dim=1) / neg_count

    total = (pos_loss + neg_weight * neg_loss).mean()
    stats = {
        "mean_high_cons":  float(high_cons.float().sum(dim=1).mean().item()),
        "mean_pos_count":  float(pos_count.mean().item()),
        "mean_neg_count":  float(neg_count.mean().item()),
    }
    return total, stats


def build_setpl_masks(
    selected_ng: torch.Tensor,  # [K, B, L] selected-view probs (no grad)
    k_hat: torch.Tensor,        # [B,] per-sample cardinality estimate
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build intersection and outside-union masks from K selected views.

    For each view k and sample b, rank labels by probability and take the top
    k_hat[b] as that view's pseudo-positive set.  Then:
      inter_mask   [B, L] — labels selected by ALL K views  (high-confidence positive)
      outside_mask [B, L] — labels selected by NO  view     (high-confidence negative)
    The uncertain zone (union minus intersection) receives no gradient.
    """
    K, B, L = selected_ng.shape
    view_masks = torch.zeros(K, B, L, dtype=torch.bool, device=selected_ng.device)
    for k in range(K):
        for b in range(B):
            kb = int(k_hat[b].item())
            top_idx = torch.argsort(selected_ng[k, b], descending=True)[:kb]
            view_masks[k, b, top_idx] = True
    inter_mask   = view_masks.all(dim=0)   # [B, L]
    outside_mask = ~view_masks.any(dim=0)  # [B, L]
    return inter_mask, outside_mask


def setpl_view_loss(
    probs: torch.Tensor,          # [B, L] with gradients (single view)
    inter_mask: torch.Tensor,     # [B, L] bool — all views agreed positive
    outside_mask: torch.Tensor,   # [B, L] bool — all views agreed negative
    neg_weight: float = 0.8,
    margin: float = 0.1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """
    Three-zone loss:
      • intersection  → cross-entropy toward 1  (only if p < 1 − margin)
      • outside-union → cross-entropy toward 0  (only if p > margin)
      • uncertain zone → zero gradient
    """
    probs = probs.clamp(min=eps, max=1.0 - eps)

    # Positive arm — active where confidence is still low
    pos_active = inter_mask & (probs.detach() < (1.0 - margin))
    pos_count  = pos_active.sum(dim=1).clamp(min=1).float()
    pos_loss   = -(torch.log(probs) * pos_active.float()).sum(dim=1) / pos_count

    # Negative arm — active where score is still high
    neg_active = outside_mask & (probs.detach() > margin)
    neg_count  = neg_active.sum(dim=1).clamp(min=1).float()
    neg_loss   = -(torch.log(1.0 - probs) * neg_active.float()).sum(dim=1) / neg_count

    total = (pos_loss + neg_weight * neg_loss).mean()
    stats = {
        "setpl_pos_loss":       float(pos_loss.mean().item()),
        "setpl_neg_loss":       float(neg_loss.mean().item()),
        "setpl_inter_active":   float(pos_active.float().sum(dim=1).mean().item()),
        "setpl_outside_active": float(neg_active.float().sum(dim=1).mean().item()),
    }
    return total, stats


# -------------------------
# Tent wrapper
# -------------------------
class TentZeroShot(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        label_cols: List[str],
        pos_template: str,
        neg_template: str,
        optimizer: Optional[torch.optim.Optimizer],
        steps: int = 1,
        episodic: bool = False,
        device: str = "cuda",
        uncertainty_filter: bool = False,
        uncertainty_low: float = 0.50,
        uncertainty_high: float = 0.85,
        gradient_accumulation_steps: int = 1,
        engine=None,
        groups: Optional[List[List[int]]] = None,
        group_mus: Optional[List[float]] = None,
        group_sigmas: Optional[List[float]] = None,
        use_anchor_topk: bool = True,
        anchor_pos_thr: float = 0.55,
        anchor_neg_thr: float = 0.35,
        topk_pos: int = 2,
        topk_alpha_neg: float = 2.0,
        lambda_topk: float = 1.0,
        lambda_label_entropy: float = 0.02,
        lambda_group_entropy: float = 0.05,
        lambda_cardinality: float = 0.05,
        lambda_anchor_kl: float = 0.0,
        use_mltta_bem: bool = False,
        num_views: int = 2,
        view_keep_frac: float = 0.5,
        bem_min_k: int = 1,
        bem_max_k: int = 3,
        bem_neg_weight: float = 0.5,
        aug_hu_jitter_std: float = 0.02,
        aug_crop_shift_ratio: float = 0.03,
        # set-theoretic pseudo-labeling loss
        use_set_pl: bool = False,
        setpl_neg_weight: float = 0.8,
        setpl_margin: float = 0.1,
        setpl_lambda: float = 1.0,
        # BEM loss type
        bem_loss_type: str = "ce",      # "ce" or "hinge"
        bem_hinge_margin: float = 0.1,
        # triplanar consistency BEM
        use_triplanar_consistency: bool = False,
        triplanar_var_percentile: float = 50.0,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self._engine = engine
        self.tokenizer = tokenizer
        self.label_cols = label_cols
        self.pos_template = pos_template
        self.neg_template = neg_template
        self.steps = steps
        self.episodic = episodic
        self.device = device

        self.uncertainty_filter = uncertainty_filter
        self.uncertainty_low = uncertainty_low
        self.uncertainty_high = uncertainty_high
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)

        self.groups = groups if groups is not None else [[i] for i in range(len(label_cols))]
        self.group_mus = group_mus if group_mus is not None else [1.0 for _ in self.groups]
        self.group_sigmas = group_sigmas if group_sigmas is not None else [1.0 for _ in self.groups]

        self.use_anchor_topk = use_anchor_topk
        self.anchor_pos_thr = anchor_pos_thr
        self.anchor_neg_thr = anchor_neg_thr
        self.topk_pos = topk_pos
        self.topk_alpha_neg = topk_alpha_neg
        self.lambda_topk = lambda_topk
        self.lambda_label_entropy = lambda_label_entropy
        self.lambda_group_entropy = lambda_group_entropy
        self.lambda_cardinality = lambda_cardinality
        self.lambda_anchor_kl = lambda_anchor_kl

        self.use_mltta_bem = use_mltta_bem
        self.num_views = max(1, num_views)
        self.view_keep_frac = view_keep_frac
        self.bem_min_k = bem_min_k
        self.bem_max_k = bem_max_k
        self.bem_neg_weight = bem_neg_weight
        self.aug_hu_jitter_std = aug_hu_jitter_std
        self.aug_crop_shift_ratio = aug_crop_shift_ratio

        self.use_set_pl = use_set_pl
        self.setpl_neg_weight = setpl_neg_weight
        self.setpl_margin = setpl_margin
        self.setpl_lambda = setpl_lambda
        self.bem_loss_type = bem_loss_type
        self.bem_hinge_margin = bem_hinge_margin
        self.use_triplanar_consistency = use_triplanar_consistency
        self.triplanar_var_percentile  = triplanar_var_percentile

        assert steps > 0
        assert optimizer is not None or engine is not None, "Need optimizer or DeepSpeed engine"

        _model_for_reset = engine.module if engine is not None else model
        _opt_for_reset = engine.optimizer if engine is not None else optimizer
        self._model_for_reset = _model_for_reset
        self._opt_for_reset = _opt_for_reset
        self.model_state, self.optimizer_state = copy_model_and_optimizer(_model_for_reset, _opt_for_reset)

        self.pos_tok_single = []
        self.neg_tok_single = []
        for ln in self.label_cols:
            self.pos_tok_single.append(
                self.tokenizer([self.pos_template.format(label=ln)], padding=True, truncation=True, return_tensors="pt")
            )
            self.neg_tok_single.append(
                self.tokenizer([self.neg_template.format(label=ln)], padding=True, truncation=True, return_tensors="pt")
            )

    def reset(self):
        load_model_and_optimizer(self._model_for_reset, self._opt_for_reset, self.model_state, self.optimizer_state)

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        L = len(self.label_cols)
        pair_logits = torch.empty((B, L, 2), device=x.device, dtype=torch.float32)

        for j in range(L):
            tok_pos = repeat_batchencoding(self.pos_tok_single[j], B).to(self.device)
            tok_neg = repeat_batchencoding(self.neg_tok_single[j], B).to(self.device)

            sim_pos = self.model(tok_pos, x, device=self.device)
            sim_neg = self.model(tok_neg, x, device=self.device)

            pair_logits[:, j, 0] = sim_neg
            pair_logits[:, j, 1] = sim_pos

            del tok_pos, tok_neg, sim_pos, sim_neg

        return pair_logits

    def _forward_probs(self, x: torch.Tensor) -> torch.Tensor:
        pair_logits = self._forward_logits(x)
        probs_pair = F.softmax(pair_logits, dim=-1)
        return probs_pair[..., 1]

    @torch.enable_grad()
    def forward_and_adapt(self, x: torch.Tensor, accumulate_only: bool = False, loss_scale: float = 1.0):
        amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if x.is_cuda else nullcontext()

        with amp_ctx:
            pair_logits = self._forward_logits(x)
            probs_pair = F.softmax(pair_logits, dim=-1)
            probs_before = probs_pair[..., 1]
            anchor_probs = probs_before.detach()
            ent = pairwise_entropy_from_logits(pair_logits)

            if self.uncertainty_filter:
                conf = probs_pair.max(dim=-1).values
                reliability_mask = (conf >= self.uncertainty_low) & (conf <= self.uncertainty_high)
                active_pairs = int(reliability_mask.sum().item())
            else:
                reliability_mask = torch.ones_like(probs_before, dtype=torch.bool)
                active_pairs = int(reliability_mask.numel())

            if self.use_mltta_bem:
                B, L = x.shape[0], len(self.label_cols)

                if self.use_triplanar_consistency:
                    # ── Triplanar consistency BEM ──────────────────────────────────────────
                    # Reuse the already-computed initial forward as the axial view, then
                    # detach pair_logits/probs_pair/ent to free the 15-20 GB backward graph
                    # before running coronal and sagittal passes. This keeps L40S (44 GB)
                    # from OOMing: model params + 2 no-grad fp16 passes + 1 grad pass ≈ 35 GB.
                    p_ax         = probs_before.detach()
                    probs_before = p_ax          # drop grad_fn so initial graph can be freed
                    ent          = ent.detach()  # likewise
                    del pair_logits, probs_pair
                    if x.is_cuda:
                        torch.cuda.empty_cache()

                    with torch.no_grad():
                        x_cor = make_coronal_view(x)
                        p_cor = self._forward_probs(x_cor); del x_cor
                        if x.is_cuda:
                            torch.cuda.empty_cache()
                        x_sag = make_sagittal_view(x)
                        p_sag = self._forward_probs(x_sag); del x_sag
                        if x.is_cuda:
                            torch.cuda.empty_cache()
                        planes_ng = torch.stack([p_ax, p_cor, p_sag], dim=0)  # [3,B,L]
                        del p_ax, p_cor, p_sag
                        mean_ng   = planes_ng.mean(dim=0)   # [B, L]
                        var_ng    = planes_ng.var(dim=0)    # [B, L]
                        del planes_ng
                        k_hat     = estimate_k_hat_from_probs(
                            mean_ng, min_k=self.bem_min_k,
                            max_k=min(self.bem_max_k, 5), scale=0.6,
                        )
                        if x.is_cuda:
                            torch.cuda.empty_cache()

                    _opt = self._engine.optimizer if self._engine is not None else self.optimizer
                    if not accumulate_only:
                        _opt.zero_grad(set_to_none=True)

                    pv = self._forward_probs(x)
                    loss_bem, _tc_stats = triplanar_consistency_bem_loss(
                        pv, mean_ng, var_ng, k_hat,
                        neg_weight=self.bem_neg_weight,
                        var_percentile=self.triplanar_var_percentile,
                    )
                    total_loss = float(loss_bem.detach().item())

                    if not accumulate_only:
                        (loss_bem * loss_scale).backward()
                        if self._engine is not None:
                            self._engine.step()
                        else:
                            _opt.step()
                    del pv, loss_bem
                    if x.is_cuda:
                        torch.cuda.empty_cache()

                    struct_stats = {
                        "topk_loss":      total_loss,
                        "label_entropy":  float(binary_entropy_from_probs(mean_ng).mean().item()),
                        "group_entropy":  0.0,
                        "cardinality_loss": 0.0,
                        "anchor_kl":      0.0,
                        "anchor_pos_count": int(round(B * float(k_hat.float().mean().item()))),
                        "anchor_neg_count": int(round(B * (L - float(k_hat.float().mean().item())))),
                        "mean_k_hat":     float(k_hat.float().mean().item()),
                        "mean_selected_view_entropy": 0.0,
                        "setpl_pos_loss": 0.0, "setpl_neg_loss": 0.0,
                        "setpl_inter_active": 0.0, "setpl_outside_active": 0.0,
                        "setpl_inter_size": 0.0, "setpl_union_size": 0.0,
                    }
                    probs_before   = mean_ng
                    active_pairs   = int(B * L)
                    struct_loss    = None
                    loss_value_bem = torch.tensor(total_loss, device=x.device)

                else:
                    # ── Standard augmented-view BEM ───────────────────────────────────────
                    V = self.num_views

                    # No-grad: score views one-at-a-time
                    with torch.no_grad():
                        view_probs_ng = []
                        for i in range(V):
                            if i == 0:
                                xv = x
                            else:
                                xv = weak_3d_augment(
                                    x,
                                    hu_jitter_std=self.aug_hu_jitter_std,
                                    crop_shift_ratio=self.aug_crop_shift_ratio,
                                    seed=i,
                                )
                            pv = self._forward_probs(xv)
                            view_probs_ng.append(pv)
                            if i > 0:
                                del xv
                            if x.is_cuda:
                                torch.cuda.empty_cache()
                        view_probs_ng = torch.stack(view_probs_ng, dim=0)  # [V, B, L]
                        keep_ids, view_entropy = select_low_entropy_view_ids(
                            view_probs_ng, keep_frac=self.view_keep_frac
                        )
                        K = keep_ids.shape[0]
                        batch_ids = torch.arange(B, device=x.device)
                        selected_ng = torch.stack(
                            [view_probs_ng[keep_ids[k], batch_ids, :] for k in range(K)], dim=0
                        )
                        probs_sel_mean_ng = selected_ng.mean(dim=0)  # [B, L]
                        k_hat = estimate_k_hat_from_probs(
                            probs_sel_mean_ng,
                            min_k=self.bem_min_k,
                            max_k=self.bem_max_k,
                        )
                        if self.use_set_pl:
                            setpl_inter_mask, setpl_outside_mask = build_setpl_masks(
                                selected_ng, k_hat
                            )
                        else:
                            setpl_inter_mask = setpl_outside_mask = None
                    del view_probs_ng, selected_ng
                    if x.is_cuda:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                    # Grad: one selected view at a time, accumulate gradients, then step
                    _opt = self._engine.optimizer if self._engine is not None else self.optimizer
                    if not accumulate_only:
                        _opt.zero_grad(set_to_none=True)

                    scale = 1.0 / max(K, 1)
                    total_loss = 0.0
                    _setpl_stats_accum: dict = {}
                    for k in range(K):
                        view_idx = keep_ids[k, 0].item()
                        if view_idx == 0:
                            xv = x
                        else:
                            xv = weak_3d_augment(
                                x,
                                hu_jitter_std=self.aug_hu_jitter_std,
                                crop_shift_ratio=self.aug_crop_shift_ratio,
                                seed=view_idx,
                            )
                        pv = self._forward_probs(xv)  # [B, L]
                        if self.bem_loss_type == "hinge":
                            loss_bem, bem_stats_k = bem_topk_hinge_loss(
                                pv, k_hat=k_hat, neg_weight=self.bem_neg_weight,
                                margin=self.bem_hinge_margin,
                            )
                        else:
                            loss_bem, bem_stats_k = bem_topk_loss(
                                pv, k_hat=k_hat, neg_weight=self.bem_neg_weight
                            )
                        if self.use_set_pl and setpl_inter_mask is not None:
                            loss_setpl, setpl_stats_k = setpl_view_loss(
                                pv,
                                setpl_inter_mask,
                                setpl_outside_mask,
                                neg_weight=self.setpl_neg_weight,
                                margin=self.setpl_margin,
                            )
                            loss_k = loss_bem + self.setpl_lambda * loss_setpl
                            for sk, sv in setpl_stats_k.items():
                                _setpl_stats_accum[sk] = _setpl_stats_accum.get(sk, 0.0) + sv / max(K, 1)
                        else:
                            loss_k = loss_bem
                        total_loss = total_loss + loss_k.detach().item()
                        if not accumulate_only:
                            (loss_k * scale * loss_scale).backward()
                        del pv, loss_k, loss_bem
                        if self.use_set_pl and setpl_inter_mask is not None:
                            del loss_setpl
                        if view_idx != 0:
                            del xv
                        if x.is_cuda:
                            torch.cuda.empty_cache()

                    if not accumulate_only:
                        if self._engine is not None:
                            self._engine.step()
                        else:
                            _opt.step()

                    struct_stats = {
                        "topk_loss": total_loss / max(K, 1),
                        "label_entropy": float(binary_entropy_from_probs(probs_sel_mean_ng).mean().item()),
                        "group_entropy": 0.0,
                        "cardinality_loss": 0.0,
                        "anchor_kl": 0.0,
                        "anchor_pos_count": int(round(B * (k_hat.float().mean().item()))),
                        "anchor_neg_count": int(round(B * (L - k_hat.float().mean().item()))),
                        "mean_k_hat": float(k_hat.float().mean().item()),
                        "mean_selected_view_entropy": float(torch.gather(view_entropy, 0, keep_ids).mean().item()),
                        "setpl_pos_loss":       _setpl_stats_accum.get("setpl_pos_loss", 0.0),
                        "setpl_neg_loss":       _setpl_stats_accum.get("setpl_neg_loss", 0.0),
                        "setpl_inter_active":   _setpl_stats_accum.get("setpl_inter_active", 0.0),
                        "setpl_outside_active": _setpl_stats_accum.get("setpl_outside_active", 0.0),
                        "setpl_inter_size": float(setpl_inter_mask.float().sum(dim=1).mean().item())
                                            if setpl_inter_mask is not None else 0.0,
                        "setpl_union_size": float((~setpl_outside_mask).float().sum(dim=1).mean().item())
                                            if setpl_outside_mask is not None else 0.0,
                    }
                    probs_before   = probs_sel_mean_ng
                    active_pairs   = int(B * L)
                    struct_loss    = None
                    loss_value_bem = torch.tensor(total_loss / max(K, 1), device=x.device)

            else:
                struct_loss, struct_stats = structured_multilabel_tent_loss(
                    probs_pos=probs_before,
                    groups=self.groups,
                    group_mus=self.group_mus,
                    group_sigmas=self.group_sigmas,
                    reliability_mask=reliability_mask,
                    use_anchor_topk=self.use_anchor_topk,
                    anchor_probs=anchor_probs,
                    anchor_pos_thr=self.anchor_pos_thr,
                    anchor_neg_thr=self.anchor_neg_thr,
                    topk_pos=self.topk_pos,
                    topk_alpha_neg=self.topk_alpha_neg,
                    lambda_topk=self.lambda_topk,
                    lambda_label_entropy=self.lambda_label_entropy,
                    lambda_group_entropy=self.lambda_group_entropy,
                    lambda_cardinality=self.lambda_cardinality,
                    lambda_anchor_kl=self.lambda_anchor_kl,
                )

            loss = None if (self.uncertainty_filter and active_pairs == 0) else struct_loss

        _opt = self._engine.optimizer if self._engine is not None else self.optimizer
        if not accumulate_only:
            _opt.zero_grad(set_to_none=True)

        did_update = False
        if loss is not None:
            if self._engine is not None:
                self._engine.backward(loss * loss_scale)
            else:
                (loss * loss_scale).backward()

            if not accumulate_only:
                if self._engine is not None:
                    self._engine.step()
                else:
                    _opt.step()
            did_update = True
            loss_value = loss.detach()
        else:
            loss_value = torch.tensor(0.0, device=x.device)
            if self.use_mltta_bem:
                loss_value = loss_value_bem
                did_update = True  # BEM path already stepped above

        stats = {
            "did_update": did_update,
            "num_active_pairs": active_pairs,
            "num_total_pairs": int(reliability_mask.numel()),
            "topk_loss": float(struct_stats.get("topk_loss", 0.0)),
            "label_entropy": float(struct_stats.get("label_entropy", 0.0)),
            "group_entropy": float(struct_stats.get("group_entropy", 0.0)),
            "cardinality_loss": float(struct_stats.get("cardinality_loss", 0.0)),
            "anchor_kl": float(struct_stats.get("anchor_kl", 0.0)),
            "anchor_pos_count": int(struct_stats.get("anchor_pos_count", 0)),
            "anchor_neg_count": int(struct_stats.get("anchor_neg_count", 0)),
            "mean_k_hat": float(struct_stats.get("mean_k_hat", 0.0)),
            "mean_selected_view_entropy": float(struct_stats.get("mean_selected_view_entropy", 0.0)),
            "view_loss": 0.0,
            "view_disagreement": 0.0,
            "view_gate_pass": 1,
        }
        # Return post-adaptation probs so evaluation uses the adapted model (ML-TTA / Tent convention)
        if did_update and not accumulate_only:
            with torch.no_grad():
                probs_out = self._forward_probs(x)
            return probs_out.detach(), loss_value, ent.detach(), stats
        return probs_before.detach(), loss_value, ent.detach(), stats

    def forward(self, x: torch.Tensor):
        if self.episodic:
            self.reset()

        B = x.shape[0]
        accum = self.gradient_accumulation_steps

        if accum > 1 and B > 1:
            micro_batch_size = max(1, B // accum)
            num_chunks = (B + micro_batch_size - 1) // micro_batch_size
            probs_list, loss_val, ent_out = [], None, None
            stats_out = {
                "did_update": 0,
                "num_active_pairs": 0,
                "num_total_pairs": 0,
                "topk_loss": 0.0,
                "label_entropy": 0.0,
                "group_entropy": 0.0,
                "cardinality_loss": 0.0,
                "anchor_kl": 0.0,
                "anchor_pos_count": 0,
                "anchor_neg_count": 0,
                "mean_k_hat": 0.0,
                "mean_selected_view_entropy": 0.0,
                "view_loss": 0.0,
                "view_disagreement": 0.0,
                "view_gate_pass": 0,
            }

            _opt = self._engine.optimizer if self._engine is not None else self.optimizer

            for _ in range(self.steps):
                _opt.zero_grad(set_to_none=True)
                step_probs, step_loss, step_ent = [], None, None
                step_stats_accum = {k: 0.0 for k in stats_out.keys()}
                step_stats_accum["did_update"] = 0
                step_stats_accum["view_gate_pass"] = 0

                num_used_chunks = 0
                for start in range(0, B, micro_batch_size):
                    x_chunk = x[start:start + micro_batch_size]
                    p, l, e, s = self.forward_and_adapt(x_chunk, accumulate_only=True, loss_scale=1.0 / num_chunks)
                    step_probs.append(p)
                    step_loss = l if step_loss is None else step_loss
                    step_ent = e if step_ent is None else torch.cat([step_ent, e], dim=0)

                    for k in step_stats_accum.keys():
                        step_stats_accum[k] += float(s[k]) if isinstance(step_stats_accum[k], float) else int(s[k])

                    num_used_chunks += 1
                    if x_chunk.is_cuda:
                        torch.cuda.empty_cache()

                if self._engine is not None:
                    self._engine.step()
                else:
                    _opt.step()

                for k in ["topk_loss", "label_entropy", "group_entropy", "cardinality_loss",
                          "anchor_kl", "mean_k_hat", "mean_selected_view_entropy",
                          "view_loss", "view_disagreement"]:
                    step_stats_accum[k] /= max(1, num_used_chunks)

                probs_list.append(torch.cat(step_probs, dim=0))
                loss_val = step_loss
                ent_out = step_ent
                stats_out = step_stats_accum

            # Return post-adaptation probs: one no-grad forward on full batch
            with torch.no_grad():
                probs_final = self._forward_probs(x)
            return probs_final.detach(), loss_val, ent_out, stats_out

        probs, loss, ent, stats = None, None, None, None
        for _ in range(self.steps):
            probs, loss, ent, stats = self.forward_and_adapt(x)
            if x.is_cuda:
                torch.cuda.empty_cache()
        return probs, loss, ent, stats


# -------------------------
# Config
# -------------------------
@dataclass
class RunCfg:
    weights: str
    radchest_root: str
    train_csv: str
    val_csv: str
    test_csv: str
    split: str
    results_dir: str
    batch_size: int
    num_workers: int
    max_samples: Optional[int]
    seed: int
    thr: float

    target_d: int
    target_h: int
    target_w: int

    apply_rescale_if_available: bool
    resample_if_available: bool
    hu_min: float
    hu_max: float
    scale_divisor: float
    target_z_spacing: float
    target_x_spacing: float
    target_y_spacing: float

    pos_template: str
    neg_template: str

    use_triplanar_mean: bool

    use_tent: bool
    tent_lr: float
    tent_steps: int

    tent_image_only: bool
    tent_uncertainty_filter: bool
    tent_uncertainty_low: float
    tent_uncertainty_high: float
    tent_episodic: bool
    tent_visual_scope: str
    tent_gradient_accumulation_steps: int
    use_deepspeed: bool

    tent_label_groups: Optional[str]
    tent_use_anchor_topk: bool
    tent_anchor_pos_thr: float
    tent_anchor_neg_thr: float
    tent_topk_pos: int
    tent_topk_alpha_neg: float
    tent_lambda_topk: float
    tent_lambda_label_entropy: float
    tent_lambda_group_entropy: float
    tent_lambda_cardinality: float
    tent_lambda_anchor_kl: float

    tent_use_mltta_bem: bool
    tent_num_views: int
    tent_view_keep_frac: float
    tent_bem_min_k: int
    tent_bem_max_k: int
    tent_bem_neg_weight: float
    tent_aug_hu_jitter_std: float
    tent_aug_crop_shift_ratio: float

    tent_use_set_pl: bool
    tent_setpl_neg_weight: float
    tent_setpl_margin: float
    tent_setpl_lambda: float

    tent_bem_loss_type: str
    tent_bem_hinge_margin: float

    tent_use_triplanar_consistency: bool
    tent_triplanar_var_percentile: float

    metadata_csv: str


# -------------------------
# Core
# -------------------------
def run_zero_shot(cfg: RunCfg):
    ensure_dir(cfg.results_dir)
    set_seed(cfg.seed)

    df_tr = pd.read_csv(cfg.train_csv)
    df_va = pd.read_csv(cfg.val_csv)
    df_te = pd.read_csv(cfg.test_csv)

    if cfg.split == "train":
        df = df_tr
    elif cfg.split == "val":
        df = df_va
    elif cfg.split == "test":
        df = df_te
    elif cfg.split == "all":
        df = pd.concat([df_tr, df_va, df_te], axis=0, ignore_index=True)
    else:
        raise ValueError("--split must be one of train/val/test/all")

    id_col = df.columns[0]
    label_cols = list(df.columns[1:])

    label_groups = parse_label_groups(cfg.tent_label_groups, len(label_cols))
    group_mus, group_sigmas = estimate_group_cardinality_priors(df_tr, label_cols, label_groups)

    if cfg.max_samples is not None and cfg.max_samples < len(df):
        df = df.sample(n=cfg.max_samples, random_state=cfg.seed).reset_index(drop=True)

    logger.info(f"Split: {cfg.split} | #samples={len(df)} | #labels={len(label_cols)}")

    ds = RadChestNPZDataset(
        radchest_root=cfg.radchest_root,
        df=df,
        id_col=id_col,
        label_cols=label_cols,
        target_shape=(cfg.target_d, cfg.target_h, cfg.target_w),
        apply_rescale_if_available=cfg.apply_rescale_if_available,
        resample_if_available=cfg.resample_if_available,
        hu_min=cfg.hu_min,
        hu_max=cfg.hu_max,
        scale_divisor=cfg.scale_divisor,
        target_z_spacing=cfg.target_z_spacing,
        target_x_spacing=cfg.target_x_spacing,
        target_y_spacing=cfg.target_y_spacing,
        metadata_csv=cfg.metadata_csv,
    )

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = build_ctclip(cfg.weights, device=device, checkpoint_during_training=cfg.use_tent)

    if cfg.use_tent:
        model = configure_model_for_tent(model)
        params, param_names, text_names, visual_names, other_names = collect_norm_params(
            model,
            image_only=cfg.tent_image_only,
            visual_scope=cfg.tent_visual_scope,
        )
        check_model_for_tent(model)

        if len(params) == 0:
            raise RuntimeError("No normalization affine parameters selected for Tent.")

        logger.info(f"[Tent] #adapt_params={len(params)}")
        logger.info(f"[Tent] selected_text_params={len(text_names)}")
        logger.info(f"[Tent] selected_visual_params={len(visual_names)}")
        logger.info(f"[Tent] selected_other_params={len(other_names)}")

        optimizer = torch.optim.Adam(params, lr=cfg.tent_lr)
        engine = None

        if cfg.use_deepspeed:
            deepspeed, has_ds = _check_deepspeed()
            if not has_ds:
                raise RuntimeError("--use_deepspeed requires deepspeed.")
            ds_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds_config_zero_offload.json")
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=params,
                config=ds_config_path,
                dist_init_required=False,
            )
            engine = model_engine
            model = model_engine

        tented_model = TentZeroShot(
            model=model,
            tokenizer=tokenizer,
            label_cols=label_cols,
            pos_template=cfg.pos_template,
            neg_template=cfg.neg_template,
            optimizer=optimizer,
            steps=cfg.tent_steps,
            episodic=cfg.tent_episodic,
            device=device,
            uncertainty_filter=cfg.tent_uncertainty_filter,
            uncertainty_low=cfg.tent_uncertainty_low,
            uncertainty_high=cfg.tent_uncertainty_high,
            gradient_accumulation_steps=cfg.tent_gradient_accumulation_steps,
            engine=engine,
            groups=label_groups,
            group_mus=group_mus,
            group_sigmas=group_sigmas,
            use_anchor_topk=cfg.tent_use_anchor_topk,
            anchor_pos_thr=cfg.tent_anchor_pos_thr,
            anchor_neg_thr=cfg.tent_anchor_neg_thr,
            topk_pos=cfg.tent_topk_pos,
            topk_alpha_neg=cfg.tent_topk_alpha_neg,
            lambda_topk=cfg.tent_lambda_topk,
            lambda_label_entropy=cfg.tent_lambda_label_entropy,
            lambda_group_entropy=cfg.tent_lambda_group_entropy,
            lambda_cardinality=cfg.tent_lambda_cardinality,
            lambda_anchor_kl=cfg.tent_lambda_anchor_kl,
            use_mltta_bem=cfg.tent_use_mltta_bem,
            num_views=cfg.tent_num_views,
            view_keep_frac=cfg.tent_view_keep_frac,
            bem_min_k=cfg.tent_bem_min_k,
            bem_max_k=cfg.tent_bem_max_k,
            bem_neg_weight=cfg.tent_bem_neg_weight,
            aug_hu_jitter_std=cfg.tent_aug_hu_jitter_std,
            aug_crop_shift_ratio=cfg.tent_aug_crop_shift_ratio,
            use_set_pl=cfg.tent_use_set_pl,
            setpl_neg_weight=cfg.tent_setpl_neg_weight,
            setpl_margin=cfg.tent_setpl_margin,
            setpl_lambda=cfg.tent_setpl_lambda,
            bem_loss_type=cfg.tent_bem_loss_type,
            bem_hinge_margin=cfg.tent_bem_hinge_margin,
            use_triplanar_consistency=cfg.tent_use_triplanar_consistency,
            triplanar_var_percentile=cfg.tent_triplanar_var_percentile,
        )
    else:
        model.eval()
        tented_model = None
        pos_tok_single, neg_tok_single = [], []
        for ln in label_cols:
            pos_tok_single.append(tokenizer([cfg.pos_template.format(label=ln)], padding=True, truncation=True, return_tensors="pt"))
            neg_tok_single.append(tokenizer([cfg.neg_template.format(label=ln)], padding=True, truncation=True, return_tensors="pt"))

    all_ids, all_probs, all_y = [], [], []
    all_batch_losses = []
    all_active_pairs, all_total_pairs, all_updates = 0, 0, 0
    all_topk_loss, all_label_entropy, all_group_entropy = [], [], []
    all_cardinality_loss, all_anchor_kl = [], []
    all_anchor_pos_count, all_anchor_neg_count = [], []
    all_mean_k_hat, all_mean_selected_view_entropy = [], []

    for step, (sids, x, y) in enumerate(dl):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if cfg.use_tent:
            probs_bl, loss, ent, stats = tented_model(x)
            all_batch_losses.append(float(loss.item()))
            all_active_pairs += stats["num_active_pairs"]
            all_total_pairs += stats["num_total_pairs"]
            all_updates += int(stats["did_update"])
            all_topk_loss.append(float(stats["topk_loss"]))
            all_label_entropy.append(float(stats["label_entropy"]))
            all_group_entropy.append(float(stats["group_entropy"]))
            all_cardinality_loss.append(float(stats["cardinality_loss"]))
            all_anchor_kl.append(float(stats["anchor_kl"]))
            all_anchor_pos_count.append(int(stats["anchor_pos_count"]))
            all_anchor_neg_count.append(int(stats["anchor_neg_count"]))
            all_mean_k_hat.append(float(stats["mean_k_hat"]))
            all_mean_selected_view_entropy.append(float(stats["mean_selected_view_entropy"]))
        else:
            with torch.no_grad():
                B = x.shape[0]
                probs_bl = infer_view_probs(model, x, pos_tok_single, neg_tok_single, device)
                if cfg.use_triplanar_mean:
                    x_cor = make_coronal_view(x)
                    p_cor = infer_view_probs(model, x_cor, pos_tok_single, neg_tok_single, device)
                    del x_cor
                    if x.is_cuda:
                        torch.cuda.empty_cache()
                    x_sag = make_sagittal_view(x)
                    p_sag = infer_view_probs(model, x_sag, pos_tok_single, neg_tok_single, device)
                    del x_sag
                    if x.is_cuda:
                        torch.cuda.empty_cache()
                    probs_bl = (probs_bl + p_cor + p_sag) / 3.0
                    del p_cor, p_sag

        all_ids.extend(sids)
        all_probs.append(probs_bl.detach().cpu())
        all_y.append(y.detach().cpu())

        if "cuda" in str(device):
            torch.cuda.empty_cache()

        if (step + 1) % 10 == 0:
            if cfg.use_tent:
                logger.info(
                    f"Processed {len(all_ids)} samples | "
                    f"last_total_loss={all_batch_losses[-1]:.6f} | "
                    f"topk={all_topk_loss[-1]:.6f} | "
                    f"label_ent={all_label_entropy[-1]:.6f} | "
                    f"khat={all_mean_k_hat[-1]:.3f} | "
                    f"sel_view_ent={all_mean_selected_view_entropy[-1]:.6f} | "
                    f"active_pairs={all_active_pairs}/{all_total_pairs} | "
                    f"updates={all_updates}"
                )
            else:
                logger.info(f"Processed {len(all_ids)} samples...")

    probs = torch.cat(all_probs, dim=0)
    y_true = torch.cat(all_y, dim=0)

    summary, per_label = multilabel_metrics_from_probs(probs, y_true, label_cols, thr=cfg.thr)
    if cfg.use_tent:
        summary["tent"] = {
            "enabled": True,
            "lr": cfg.tent_lr,
            "steps": cfg.tent_steps,
            "image_only": cfg.tent_image_only,
            "uncertainty_filter": cfg.tent_uncertainty_filter,
            "uncertainty_low": cfg.tent_uncertainty_low,
            "uncertainty_high": cfg.tent_uncertainty_high,
            "episodic": cfg.tent_episodic,
            "visual_scope": cfg.tent_visual_scope,
            "mean_batch_total_loss": float(np.mean(all_batch_losses)) if all_batch_losses else None,
            "active_pairs": int(all_active_pairs),
            "total_pairs": int(all_total_pairs),
            "num_batches_with_updates": int(all_updates),
            "label_groups": label_groups,
            "group_cardinality_mu": group_mus,
            "group_cardinality_sigma": group_sigmas,
            "use_anchor_topk": cfg.tent_use_anchor_topk,
            "anchor_pos_thr": cfg.tent_anchor_pos_thr,
            "anchor_neg_thr": cfg.tent_anchor_neg_thr,
            "topk_pos": cfg.tent_topk_pos,
            "topk_alpha_neg": cfg.tent_topk_alpha_neg,
            "lambda_topk": cfg.tent_lambda_topk,
            "lambda_label_entropy": cfg.tent_lambda_label_entropy,
            "lambda_group_entropy": cfg.tent_lambda_group_entropy,
            "lambda_cardinality": cfg.tent_lambda_cardinality,
            "lambda_anchor_kl": cfg.tent_lambda_anchor_kl,
            "use_mltta_bem": cfg.tent_use_mltta_bem,
            "num_views": cfg.tent_num_views,
            "view_keep_frac": cfg.tent_view_keep_frac,
            "bem_min_k": cfg.tent_bem_min_k,
            "bem_max_k": cfg.tent_bem_max_k,
            "bem_neg_weight": cfg.tent_bem_neg_weight,
            "mean_topk_loss": float(np.mean(all_topk_loss)) if all_topk_loss else None,
            "mean_label_entropy": float(np.mean(all_label_entropy)) if all_label_entropy else None,
            "mean_group_entropy": float(np.mean(all_group_entropy)) if all_group_entropy else None,
            "mean_cardinality_loss": float(np.mean(all_cardinality_loss)) if all_cardinality_loss else None,
            "mean_anchor_kl": float(np.mean(all_anchor_kl)) if all_anchor_kl else None,
            "mean_anchor_pos_count": float(np.mean(all_anchor_pos_count)) if all_anchor_pos_count else None,
            "mean_anchor_neg_count": float(np.mean(all_anchor_neg_count)) if all_anchor_neg_count else None,
            "mean_k_hat": float(np.mean(all_mean_k_hat)) if all_mean_k_hat else None,
            "mean_selected_view_entropy": float(np.mean(all_mean_selected_view_entropy)) if all_mean_selected_view_entropy else None,
        }
    else:
        summary["tent"] = {"enabled": False}

    logger.info(f"SUMMARY: {summary}")

    with open(os.path.join(cfg.results_dir, "summary_zero_shot.json"), "w") as f:
        json.dump(
            {"cfg": cfg.__dict__, "label_cols": label_cols, "summary": summary, "per_label": per_label},
            f,
            indent=2,
        )

    torch.save(
        {"ids": all_ids, "probs": probs, "y_true": y_true, "label_cols": label_cols},
        os.path.join(cfg.results_dir, "preds_zero_shot.pt"),
    )
    logger.info(f"Saved results to: {cfg.results_dir}")


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    p.add_argument("--radchest_root", type=str, default=DEFAULT_RADCHEST_ROOT)
    p.add_argument("--train_csv", type=str, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--val_csv", type=str, default=DEFAULT_VAL_CSV)
    p.add_argument("--test_csv", type=str, default=DEFAULT_TEST_CSV)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thr", type=float, default=0.5)

    p.add_argument("--target_d", type=int, default=240)
    p.add_argument("--target_h", type=int, default=480)
    p.add_argument("--target_w", type=int, default=480)

    p.add_argument("--apply_rescale_if_available", action="store_true", default=True)
    p.add_argument("--resample_if_available", action="store_true", default=True)
    p.add_argument("--hu_min", type=float, default=-1000.0)
    p.add_argument("--hu_max", type=float, default=1000.0)
    p.add_argument("--scale_divisor", type=float, default=1000.0)

    p.add_argument("--target_z_spacing", type=float, default=1.5)
    p.add_argument("--target_x_spacing", type=float, default=0.75)
    p.add_argument("--target_y_spacing", type=float, default=0.75)

    p.add_argument("--pos_template", type=str, default="there is absolutely {label} present.")
    p.add_argument("--neg_template", type=str, default="there is absolutely not {label} present.")

    p.add_argument("--use_triplanar_mean", action="store_true",
                   help="Zero-shot baseline: average predictions from axial/coronal/sagittal views.")

    p.add_argument("--use_tent", action="store_true")
    p.add_argument("--tent_lr", type=float, default=1e-5)
    p.add_argument("--tent_steps", type=int, default=1)

    p.add_argument("--tent_image_only", action="store_true")
    p.add_argument("--tent_uncertainty_filter", action="store_true")
    p.add_argument("--tent_uncertainty_low", type=float, default=0.50)
    p.add_argument("--tent_uncertainty_high", type=float, default=0.85)
    p.add_argument("--tent_episodic", action="store_true")
    p.add_argument("--tent_visual_scope", type=str, default="all",
                   choices=["all", "patch_only", "transformer_only", "patch_and_norm_out"])
    p.add_argument("--tent_gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--use_deepspeed", action="store_true")

    p.add_argument("--tent_label_groups", type=str, default="")
    p.add_argument("--tent_use_anchor_topk", action="store_true")
    p.add_argument("--tent_anchor_pos_thr", type=float, default=0.55)
    p.add_argument("--tent_anchor_neg_thr", type=float, default=0.35)
    p.add_argument("--tent_topk_pos", type=int, default=2)
    p.add_argument("--tent_topk_alpha_neg", type=float, default=2.0)
    p.add_argument("--tent_lambda_topk", type=float, default=1.0)
    p.add_argument("--tent_lambda_label_entropy", type=float, default=0.02)
    p.add_argument("--tent_lambda_group_entropy", type=float, default=0.05)
    p.add_argument("--tent_lambda_cardinality", type=float, default=0.05)
    p.add_argument("--tent_lambda_anchor_kl", type=float, default=0.0)

    p.add_argument("--tent_use_mltta_bem", action="store_true")
    p.add_argument("--tent_num_views", type=int, default=2)
    p.add_argument("--tent_view_keep_frac", type=float, default=0.5)
    p.add_argument("--tent_bem_min_k", type=int, default=1)
    p.add_argument("--tent_bem_max_k", type=int, default=3)
    p.add_argument("--tent_bem_neg_weight", type=float, default=0.5)
    p.add_argument("--tent_aug_hu_jitter_std", type=float, default=0.02)
    p.add_argument("--tent_aug_crop_shift_ratio", type=float, default=0.03)

    # set-theoretic pseudo-labeling loss (set-PL)
    p.add_argument("--tent_use_set_pl", action="store_true",
                   help="Enable set-theoretic pseudo-label loss alongside BEM.")
    p.add_argument("--tent_setpl_neg_weight", type=float, default=0.8,
                   help="Weight for outside-union (negative) arm of set-PL loss.")
    p.add_argument("--tent_setpl_margin", type=float, default=0.1,
                   help="Margin delta: skip gradient if prediction already in safe zone.")
    p.add_argument("--tent_setpl_lambda", type=float, default=1.0,
                   help="Scale factor for set-PL loss relative to BEM loss.")

    # hinge-margin BEM
    p.add_argument("--tent_bem_loss_type", type=str, default="ce", choices=["ce", "hinge"],
                   help="BEM loss: cross-entropy (ce) or hinge margin (hinge).")
    p.add_argument("--tent_bem_hinge_margin", type=float, default=0.1,
                   help="Hinge margin δ: push positives above 1-δ, negatives below δ.")

    # triplanar consistency BEM
    p.add_argument("--tent_use_triplanar_consistency", action="store_true",
                   help="Replace augmented-view BEM with triplanar consistency pseudo-labeling.")
    p.add_argument("--tent_triplanar_var_percentile", type=float, default=50.0,
                   help="Labels with variance below this percentile are high-consistency (used for BEM loss).")

    p.add_argument(
        "--metadata_csv",
        type=str,
        default="/home/ailarmz/projects/aip-lsigal/ailarmz/CT-CLIP/CT_CLIP/Radchest-labels/CT_Scan_Metadata_Complete_35747.csv",
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    if args.tent_uncertainty_low > args.tent_uncertainty_high:
        raise ValueError("--tent_uncertainty_low must be <= --tent_uncertainty_high")
    if args.tent_anchor_neg_thr >= args.tent_anchor_pos_thr:
        raise ValueError("--tent_anchor_neg_thr must be < --tent_anchor_pos_thr")
    if args.tent_bem_min_k > args.tent_bem_max_k:
        raise ValueError("--tent_bem_min_k must be <= --tent_bem_max_k")
    if not (0.0 < args.tent_view_keep_frac <= 1.0):
        raise ValueError("--tent_view_keep_frac must be in (0, 1]")

    cfg = RunCfg(
        weights=args.weights,
        radchest_root=args.radchest_root,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        split=args.split,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        seed=args.seed,
        thr=args.thr,

        target_d=args.target_d,
        target_h=args.target_h,
        target_w=args.target_w,

        apply_rescale_if_available=args.apply_rescale_if_available,
        resample_if_available=args.resample_if_available,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        scale_divisor=args.scale_divisor,
        target_z_spacing=args.target_z_spacing,
        target_x_spacing=args.target_x_spacing,
        target_y_spacing=args.target_y_spacing,

        pos_template=args.pos_template,
        neg_template=args.neg_template,

        use_triplanar_mean=args.use_triplanar_mean,

        use_tent=args.use_tent,
        tent_lr=args.tent_lr,
        tent_steps=args.tent_steps,

        tent_image_only=args.tent_image_only,
        tent_uncertainty_filter=args.tent_uncertainty_filter,
        tent_uncertainty_low=args.tent_uncertainty_low,
        tent_uncertainty_high=args.tent_uncertainty_high,
        tent_episodic=args.tent_episodic,
        tent_visual_scope=args.tent_visual_scope,
        tent_gradient_accumulation_steps=args.tent_gradient_accumulation_steps,
        use_deepspeed=args.use_deepspeed,

        tent_label_groups=args.tent_label_groups,
        tent_use_anchor_topk=args.tent_use_anchor_topk,
        tent_anchor_pos_thr=args.tent_anchor_pos_thr,
        tent_anchor_neg_thr=args.tent_anchor_neg_thr,
        tent_topk_pos=args.tent_topk_pos,
        tent_topk_alpha_neg=args.tent_topk_alpha_neg,
        tent_lambda_topk=args.tent_lambda_topk,
        tent_lambda_label_entropy=args.tent_lambda_label_entropy,
        tent_lambda_group_entropy=args.tent_lambda_group_entropy,
        tent_lambda_cardinality=args.tent_lambda_cardinality,
        tent_lambda_anchor_kl=args.tent_lambda_anchor_kl,

        tent_use_mltta_bem=args.tent_use_mltta_bem,
        tent_num_views=args.tent_num_views,
        tent_view_keep_frac=args.tent_view_keep_frac,
        tent_bem_min_k=args.tent_bem_min_k,
        tent_bem_max_k=args.tent_bem_max_k,
        tent_bem_neg_weight=args.tent_bem_neg_weight,
        tent_aug_hu_jitter_std=args.tent_aug_hu_jitter_std,
        tent_aug_crop_shift_ratio=args.tent_aug_crop_shift_ratio,

        tent_use_set_pl=args.tent_use_set_pl,
        tent_setpl_neg_weight=args.tent_setpl_neg_weight,
        tent_setpl_margin=args.tent_setpl_margin,
        tent_setpl_lambda=args.tent_setpl_lambda,

        tent_bem_loss_type=args.tent_bem_loss_type,
        tent_bem_hinge_margin=args.tent_bem_hinge_margin,

        tent_use_triplanar_consistency=args.tent_use_triplanar_consistency,
        tent_triplanar_var_percentile=args.tent_triplanar_var_percentile,

        metadata_csv=args.metadata_csv,
    )
    run_zero_shot(cfg)


if __name__ == "__main__":
    main()