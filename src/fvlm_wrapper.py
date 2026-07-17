#!/usr/bin/env python3
"""
fVLM inference wrapper matching our CT-CLIP interface.

The exposed public API mirrors `build_ctclip(weights_path, device) → (model, tokenizer)`
and `model(text_tokens, image, device=device) → [B] similarity`, so downstream
analysis scripts (`infer_view_probs`, `triplanar_variance_filter.py`, etc.)
can be swapped to use fVLM by replacing only the builder call.

Key transforms handled inside the wrapper:

  Image format
    CT-CLIP:  [B, 1, 40, 480, 480], float in [-1, 1] (HU / 1000, clipped)
    fVLM:     [B, 1, 112, 256, 352], float in [0, 1]  (shift+scale)

  Text encoder
    fVLM was trained with `XBertEncoder.forward_text(BatchEncoding)` — same
    tokenizer (BiomedVLP-CXR-BERT-specialized) and BatchEncoding interface
    as CT-CLIP, so `repeat_batchencoding(...).to(device)` works unchanged.

  Similarity scoring
    fVLM is an ORGAN-specific model: it has 4 learned query tokens
    (lung / heart / esophagus / aorta) and 4 per-organ vision projections.
    For whole-image zero-shot we attend over ALL patch tokens (no mask),
    compute a per-organ image_feat, cosine-sim with text_feat, then
    aggregate over organs. Default is `"mean"`; pass aggregate="max" for
    "rely on the most-confident organ for each label".
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── fVLM imports (path setup must come before lavis imports) ──────────────────
_FVLM_ROOT = Path(os.environ.get("FVLM_ROOT", "/project/6101771/ailarmz/fvlm"))
if str(_FVLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_FVLM_ROOT))
_LAVIS = _FVLM_ROOT / "lavis"
if str(_LAVIS) not in sys.path:
    sys.path.insert(0, str(_LAVIS))

# Purge any stale lavis imports
for _m in list(sys.modules):
    if _m.startswith("lavis"):
        del sys.modules[_m]

from lavis.models.blip_models.vit import ViT                       # noqa: E402
from transformers import BertConfig, BertModel, BertTokenizer       # noqa: E402


# ── fVLM fixed hyperparameters (from pretrain_ct.yaml + load_fvlm_model.py) ──
_FVLM_IMG_SIZE   = (112, 256, 352)
_FVLM_PATCH_SIZE = (16, 16, 32)
_FVLM_EMBED_DIM  = 256
_FVLM_VISION_W   = 768
_FVLM_N_ORGANS   = 4
_FVLM_MAX_TXT    = 384
_FVLM_TOKENIZER  = "microsoft/BiomedVLP-CXR-BERT-specialized"


class _FVLMCore(nn.Module):
    """Minimal module that mirrors the subset of `BlipPretrain` submodule names
    we need at inference — so the fVLM checkpoint's state_dict (which uses those
    exact names) loads cleanly. Bypassing BlipPretrain/XBertEncoder avoids a
    chain of lavis registry + MLM-head issues that only matter at training time.

    Submodule names here match fVLM's saved state_dict exactly:
        visual_encoder.*         — 3D ViT
        text_encoder.*           — plain BertModel (no MLM head, no cross-attn)
        text_proj.*              — Linear(768, 256)
        vision_projs.[0..3].*    — 4 x Linear(768, 256)
        query_tokens             — Parameter(4, 768)
        attention.*              — MultiheadAttention(768, 4 heads)
        temp                     — scalar Parameter
        image_queue / text_queue / queue_ptr — momentum-queue buffers (unused at inference)
    """

    def __init__(self):
        super().__init__()
        self.visual_encoder = ViT(
            in_channels=1,
            img_size=_FVLM_IMG_SIZE,
            patch_size=_FVLM_PATCH_SIZE,
            num_classes=0,
            dropout_rate=0.1,
            qkv_bias=True,
        )
        bert_cfg = BertConfig.from_pretrained(_FVLM_TOKENIZER)
        self.text_encoder = BertModel(bert_cfg, add_pooling_layer=False)

        self.text_proj = nn.Linear(bert_cfg.hidden_size, _FVLM_EMBED_DIM)
        self.vision_projs = nn.ModuleList([
            nn.Linear(_FVLM_VISION_W, _FVLM_EMBED_DIM) for _ in range(_FVLM_N_ORGANS)
        ])
        self.query_tokens = nn.Parameter(torch.zeros(_FVLM_N_ORGANS, _FVLM_VISION_W))
        self.attention = nn.MultiheadAttention(
            embed_dim=_FVLM_VISION_W, num_heads=4, dropout=0.1, batch_first=True,
        )
        self.temp = nn.Parameter(torch.tensor(0.07))

        # No momentum-queue buffers here: in the saved checkpoint image_queue/
        # text_queue have shape [256, 0] (training-time fields we don't need).
        # They get stripped in build_fvlm() before load_state_dict.


def _build_fvlm_core(device: str) -> _FVLMCore:
    return _FVLMCore().to(device)


class FVLMForZeroShot(nn.Module):
    """Drop-in replacement for CT-CLIP's `CTCLIP` at call sites that do
    `sim = model(text_tokens, image, device=device)`.

    Returns a [B] tensor of (temperature-scaled) similarities — used as logits
    by the downstream pair-softmax code to produce per-label Bernoulli probs.
    """

    def __init__(self, core: _FVLMCore, tokenizer, aggregate: str = "mean"):
        super().__init__()
        assert aggregate in ("mean", "max")
        self.core = core
        self.aggregate = aggregate
        self.tokenizer = tokenizer

    # ── image preprocessing ──────────────────────────────────────────────
    def _prepare_image(self, x: torch.Tensor) -> torch.Tensor:
        """CT-CLIP format [B,1,40,480,480] in [-1,1]  →  fVLM [B,1,112,256,352] in [0,1]."""
        x = (x + 1.0) * 0.5
        x = x.clamp_(0.0, 1.0)
        if tuple(x.shape[-3:]) != _FVLM_IMG_SIZE:
            x = F.interpolate(x, size=_FVLM_IMG_SIZE, mode="trilinear", align_corners=False)
        return x

    # ── text encoding ────────────────────────────────────────────────────
    def _encode_text(self, text_tokens) -> torch.Tensor:
        out = self.core.text_encoder(
            input_ids=text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        cls = out.last_hidden_state[:, 0, :]
        return F.normalize(self.core.text_proj(cls), dim=-1)    # [B, embed_dim]

    # ── image encoding (ensemble over 4 organ queries, no mask) ──────────
    def _encode_image_organs(self, image: torch.Tensor) -> torch.Tensor:
        """Returns [n_organs, B, embed_dim] of normalized image features."""
        image_embeds, _ = self.core.visual_encoder(image)   # [B, N, 768]
        B = image_embeds.shape[0]
        n_organs = len(self.core.vision_projs)

        feats = []
        for organ_id in range(n_organs):
            query = self.core.query_tokens[organ_id].view(1, 1, -1).expand(B, 1, -1)
            updated, _ = self.core.attention(query, image_embeds, image_embeds)  # [B,1,768]
            proj = self.core.vision_projs[organ_id](updated.squeeze(1))          # [B, embed_dim]
            feats.append(F.normalize(proj, dim=-1))
        return torch.stack(feats, dim=0)    # [n_organs, B, embed_dim]

    def forward(self, text_tokens, image: torch.Tensor, device=None) -> torch.Tensor:
        # Match CT-CLIP's `model(tok, x, device=device)` signature.
        if device is not None:
            image = image.to(device)
        img = self._prepare_image(image)

        text_feat = self._encode_text(text_tokens)                      # [B, E]
        img_feats = self._encode_image_organs(img)                      # [O, B, E]

        sims_per_organ = (img_feats * text_feat.unsqueeze(0)).sum(-1)   # [O, B]
        sim = sims_per_organ.max(0).values if self.aggregate == "max" \
              else sims_per_organ.mean(0)
        return sim / self.core.temp                                     # [B]


def build_fvlm(
    weights_path: str,
    device: str = "cuda",
    aggregate: str = "mean",
):
    """Mirrors `build_ctclip(weights_path, device)` → (model, tokenizer).

    Args
      weights_path: path to fVLM `model.pth` (dict with 'model' key or raw state_dict).
      device: "cuda" or "cpu".
      aggregate: how to combine per-organ similarities.
        "mean" — average across 4 organ heads (default, symmetric).
        "max"  — take the strongest organ signal per (sample, label).

    The returned `model` is callable as `model(text_tokens, image, device=device)`
    and returns a `[B]` similarity tensor, so `infer_view_probs` works unchanged.
    """
    core = _build_fvlm_core(device)

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    # Drop training-only momentum-queue tensors (empty at save time)
    for _k in ("image_queue", "text_queue", "queue_ptr"):
        state.pop(_k, None)
    missing, unexpected = core.load_state_dict(state, strict=False)
    print(f"[fVLM load] missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"  first missing: {missing[:5]}")
    if unexpected:
        print(f"  first unexpected: {unexpected[:5]}")

    # fVLM was trained with `add_type_embeddings=False`, but HuggingFace
    # `BertModel` always instantiates `token_type_embeddings` and uses row 0 for
    # every token (token_type_ids defaults to 0). Zero this so it acts as the
    # identity — equivalent to not having the layer at all.
    with torch.no_grad():
        core.text_encoder.embeddings.token_type_embeddings.weight.zero_()

    core.eval()
    tokenizer = BertTokenizer.from_pretrained(_FVLM_TOKENIZER, do_lower_case=True)
    wrapped = FVLMForZeroShot(core, tokenizer, aggregate=aggregate).to(device)
    wrapped.eval()
    return wrapped, tokenizer


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(_FVLM_ROOT / "model.pth"))
    ap.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--aggregate", default="mean", choices=["mean", "max"])
    args = ap.parse_args()

    print(f"Loading fVLM from {args.weights}  (device={args.device})")
    model, tokenizer = build_fvlm(args.weights, device=args.device,
                                  aggregate=args.aggregate)
    print("Model + tokenizer loaded.")

    # Fake batch in CT-CLIP format
    B = 1
    x = torch.zeros(B, 1, 40, 480, 480, device=args.device)   # HU=0 → CT-CLIP normalized 0
    labels = ["Cardiomegaly", "Lung nodule", "Pleural effusion"]
    pos_tpl = "there is absolutely {label} present."
    neg_tpl = "there is absolutely not {label} present."

    with torch.no_grad():
        for lbl in labels:
            tok_p = tokenizer([pos_tpl.format(label=lbl)], padding=True,
                              truncation=True, return_tensors="pt").to(args.device)
            tok_n = tokenizer([neg_tpl.format(label=lbl)], padding=True,
                              truncation=True, return_tensors="pt").to(args.device)
            s_p = model(tok_p, x, device=args.device)
            s_n = model(tok_n, x, device=args.device)
            pair = torch.stack([s_n, s_p], dim=-1)
            prob = F.softmax(pair, dim=-1)[:, 1]
            print(f"  {lbl:25s}  sim_pos={s_p.item():+.4f}  "
                  f"sim_neg={s_n.item():+.4f}  p(pos)={prob.item():.4f}")
