"""
carve_xview.py -- Cross-view reliability-weighted BEM objective (CARVE, Prompt B)

WHY (from the gradient-disentanglement diagnostic):
  - Loss SHAPE (entropy / BEM / hinge) governs gradient MAGNITUDE and its
    profile vs distance from the boundary |p - 0.5|.
  - The PSEUDO-LABEL (shared by BEM and BEM-Hinge) governs gradient DIRECTION,
    which tracks the per-scan TP/TN separation Delta_sep. Changing the loss
    shape did NOT move direction (BEM-Hinge == BEM on the direction axis).
  => To move direction we must change the PSEUDO-LABEL, not the loss.

WHAT IT DOES:
  Re-weights each label's BEM term by how consistently the V weak 3D views agree
  on its top-k-hat membership (cross-view stability w_j in [0,1]).

VERIFIED BEHAVIOUR -- READ BEFORE INTERPRETING (see self-test at bottom):
  Reliability weighting is an AMPLIFIER of the consensus along the Delta_sep
  axis. It RAISES effective gradient-direction accuracy where Delta_sep > 0 and
  LOWERS it where Delta_sep < 0 (under inversion all views agree on the wrong
  ranking, so trusting agreement amplifies the error). No unsupervised loss can
  repair inversion. Therefore carve_xview must be GATED by the label-free proxy
  sigma_pred: apply it only when sigma_pred(x) > tau (Delta_sep likely > 0), and
  return the zero-shot prediction otherwise. This is the gradient-level reason
  the CARVE-Gated mechanism is necessary, not optional.

  Per-slot update SIGN is fixed by the consensus assignment, so the *unweighted*
  per-slot direction accuracy equals BEM by construction. carve_xview changes
  the RELIABILITY-WEIGHTED (effective) direction accuracy -- log BOTH.

WEIGHTING MODES:
  "trust"   m_j = w_j (positives) / (1-w_j) (negatives): re-weight toward the
            consensus. Largest gain at Delta_sep>0, largest harm at Delta_sep<0.
  "abstain" m_j = max(w_j, 1-w_j): only down-weight labels that flip across
            views. Gentler both ways; more robust to a wrong gate decision.
            Recommended default.
"""


def compute_reliability_and_cardinality(probs_views, k_min=1, k_max=8):
    """No-grad stage on the retained low-entropy views.

    probs_views: (V, L) detached per-label presence probs.
    returns: w (L,) cross-view stability, khat int, pbar_mask (L,) bool.
    """
    import torch
    with torch.no_grad():
        V, L = probs_views.shape
        p_bar = probs_views.mean(dim=0)
        khat = int(torch.clamp(torch.round(p_bar.sum()),
                               min=k_min, max=min(k_max, L)).item())
        topk_idx = probs_views.topk(khat, dim=1).indices
        member = torch.zeros_like(probs_views, dtype=torch.bool)
        member.scatter_(1, topk_idx, True)
        w = member.float().mean(dim=0)
        pbar_mask = torch.zeros(L, dtype=torch.bool, device=probs_views.device)
        pbar_mask[p_bar.topk(khat).indices] = True
    return w, khat, pbar_mask


def carve_xview_loss(margins_views, w, pbar_mask, lambda_neg=0.8,
                     mode="abstain", eps=1e-6):
    """Grad stage. Reliability-weighted BEM over the retained views.

    margins_views: (V, L) logit margins d = s_pos - s_neg, WITH grad.
    w, pbar_mask : from compute_reliability_and_cardinality (detached).
    mode         : "abstain" (default) or "trust".
    """
    import torch
    import torch.nn.functional as F
    w = w.detach()
    pos_mask = pbar_mask.detach().float()
    neg_mask = (~pbar_mask).detach().float()

    if mode == "trust":
        pos_w = w * pos_mask
        neg_w = (1.0 - w) * neg_mask
    elif mode == "abstain":
        r = torch.maximum(w, 1.0 - w)          # decisiveness of cross-view vote
        pos_w = r * pos_mask
        neg_w = r * neg_mask
    else:
        raise ValueError(f"unknown mode {mode!r}")

    logp = F.logsigmoid(margins_views)
    log1mp = F.logsigmoid(-margins_views)
    pos_term = -(logp * pos_w).sum(dim=1) / pos_w.sum().clamp_min(eps)
    neg_term = -(log1mp * neg_w).sum(dim=1) / neg_w.sum().clamp_min(eps)
    return (pos_term + lambda_neg * neg_term).mean()


def bem_loss(margins_views, pbar_mask, lambda_neg=0.8, eps=1e-6):
    """Baseline BEM top-khat (uniform weights)."""
    import torch
    import torch.nn.functional as F
    pos_mask = pbar_mask.detach().float()
    neg_mask = (~pbar_mask).detach().float()
    logp = F.logsigmoid(margins_views)
    log1mp = F.logsigmoid(-margins_views)
    pos_term = -(logp * pos_mask).sum(dim=1) / pos_mask.sum().clamp_min(eps)
    neg_term = -(log1mp * neg_mask).sum(dim=1) / neg_mask.sum().clamp_min(eps)
    return (pos_term + lambda_neg * neg_term).mean()


# --------------------------------------------------------------------------- #
# Pseudo-label SELF-TRAINING objective (non-entropy).                          #
#                                                                              #
# Motivation (this file's header): the loss SHAPE (entropy/BEM/CARVE) is       #
# direction-equivalent; what governs the gradient DIRECTION is the PSEUDO-     #
# LABEL. Entropy has NO fixed target -- its per-label sign is just            #
# sign(0.5 - p), so it only sharpens the current prediction and cannot move    #
# cross-scan ranking (AUROC). Self-training fixes a HARD pseudo-target y_hat    #
# from the no-grad consensus and minimises a class-balanced cross-entropy      #
# toward it. The gradient is (y_hat - p): it pushes the believed-present       #
# labels UP and the believed-absent labels DOWN. When y_hat correlates with    #
# truth (per-scan TP/TN separation Delta_sep > 0, i.e. base AUROC > 0.5) this   #
# raises positives and lowers negatives across scans -> AUROC can improve.      #
# Where base is at/below chance (Delta_sep <= 0) it amplifies the wrong        #
# ranking -- consistent with "no unsupervised loss repairs inversion" above.   #
# So the predicted win is concentrated on the stronger base under shift.        #
# --------------------------------------------------------------------------- #


def build_pseudo_targets(p_bar, khat=None, mode="conf_hard", margin=0.1,
                         k_min=1, k_max=8):
    """No-grad. Build a hard pseudo-target y_hat (L,) bool and a confidence mask
    (L,) bool from the consensus presence probs p_bar (L,).

    mode:
      "conf_hard" : y_hat = (p_bar > 0.5); train only labels with
                    |p_bar - 0.5| >= margin (confident either way).
      "topk_hard" : y_hat = top-khat labels positive (cardinality-structured);
                    train all L labels. khat from sum-round if not given.
      "topk_conf" : top-khat positives AND confidence mask (intersection of the
                    two above) -- structured + confident.
    Returns (y_hat[L] bool, mask[L] bool, khat int).
    """
    import torch
    with torch.no_grad():
        L = p_bar.numel()
        if khat is None:
            khat = int(torch.clamp(torch.round(p_bar.sum()),
                                   min=k_min, max=min(k_max, L)).item())
        khat = int(max(k_min, min(min(k_max, L), khat)))
        conf = (p_bar - 0.5).abs() >= margin
        topk = torch.zeros(L, dtype=torch.bool, device=p_bar.device)
        topk[p_bar.topk(khat).indices] = True
        if mode == "conf_hard":
            y_hat = p_bar > 0.5
            mask = conf
        elif mode == "topk_hard":
            y_hat = topk
            mask = torch.ones(L, dtype=torch.bool, device=p_bar.device)
        elif mode == "topk_conf":
            y_hat = topk
            mask = conf
        else:
            raise ValueError(f"unknown pseudo-label mode {mode!r}")
        # never train on an all-empty mask (would give a zero-grad no-op)
        if not bool(mask.any()):
            mask = torch.ones(L, dtype=torch.bool, device=p_bar.device)
    return y_hat, mask, khat


def pseudolabel_loss(probs_views, y_hat, mask, neg_weight=0.8, eps=1e-6):
    """Grad stage. Class-balanced, confidence-masked cross-entropy to the fixed
    hard pseudo-target y_hat over the retained views.

    probs_views: (V, L) presence probs WITH grad.
    y_hat, mask: (L,) detached bool from build_pseudo_targets.
    """
    import torch
    probs = probs_views.clamp(min=eps, max=1.0 - eps)
    m = mask.detach().float()
    pos = (y_hat.detach().float() * m)
    neg = ((1.0 - y_hat.detach().float()) * m)
    pos_term = -(torch.log(probs) * pos).sum(dim=1) / pos.sum().clamp_min(eps)
    neg_term = -(torch.log(1.0 - probs) * neg).sum(dim=1) / neg.sum().clamp_min(eps)
    return (pos_term + neg_weight * neg_term).mean()


# --------------------------------------------------------------------------- #
# Self-test (pure numpy): demonstrates the amplifier behaviour and the regime
# split that mandates gating. Run:  python3 carve_xview.py
# --------------------------------------------------------------------------- #
def _np_topk_mask(p, k):
    import numpy as np
    m = np.zeros_like(p, dtype=bool)
    m[np.argpartition(-p, k - 1)[:k]] = True
    return m


def _simulate(L, k_pos, dsep, view_noise, n_views, systematic, rng):
    import numpy as np
    y = np.zeros(L, dtype=int); y[:k_pos] = 1
    base = np.where(y == 1, dsep, -dsep)
    noise = rng.normal(0, 0.05 if systematic else view_noise, size=(n_views, L))
    return 1.0 / (1.0 + np.exp(-(base[None, :] + noise))), y


def _eff_diracc(probs, y, khat, mode):
    import numpy as np
    p_bar = probs.mean(0); pm = _np_topk_mask(p_bar, khat)
    correct = np.where(pm, y == 1, y == 0)
    member = np.stack([_np_topk_mask(probs[v], khat) for v in range(len(probs))])
    w = member.mean(0)
    if mode == "bem":      m = np.ones_like(p_bar)
    elif mode == "trust":  m = np.where(pm, w, 1.0 - w)
    elif mode == "abstain": m = np.maximum(w, 1.0 - w)
    return float((m * correct).sum() / max(m.sum(), 1e-9))


def _run_selftest():
    import numpy as np
    rng = np.random.default_rng(0)
    L, k_pos, n_views, n_scans, view_noise = 16, 5, 8, 400, 0.9
    print(f"{'regime':<13}{'dsep':>6}{'BEM':>8}{'trust':>8}{'abstain':>9}"
          f"{'t-BEM':>8}{'a-BEM':>8}")
    print("-" * 60)
    for systematic, name in [(False, "rand-noise"), (True, "systematic")]:
        for dsep in (-0.30, -0.10, 0.00, 0.10, 0.30):
            b = t = a = 0.0
            for _ in range(n_scans):
                probs, y = _simulate(L, k_pos, dsep, view_noise, n_views,
                                     systematic, rng)
                khat = int(np.clip(round(probs.mean(0).sum()), 1, 8))
                b += _eff_diracc(probs, y, khat, "bem")
                t += _eff_diracc(probs, y, khat, "trust")
                a += _eff_diracc(probs, y, khat, "abstain")
            b, t, a = b / n_scans, t / n_scans, a / n_scans
            print(f"{name:<13}{dsep:>6.2f}{b:>8.3f}{t:>8.3f}{a:>9.3f}"
                  f"{t - b:>+8.3f}{a - b:>+8.3f}")
        print("-" * 60)
    print("Amplifier: gains at dsep>0, harm at dsep<0 => MUST gate by sigma_pred.")


if __name__ == "__main__":
    _run_selftest()
