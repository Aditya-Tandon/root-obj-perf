"""Gradient-based and permutation feature importance for ParticleTransformer.

Note: mask shape is (N, P), grads shape is (N, P, F). Use np.broadcast_to
for boolean indexing (see CLAUDE.md Problem 2).
"""

import gc
import numpy as np
import torch


def _flush():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def gradient_feature_importance(model, x, mask):
    """Gradient-based feature importance.

    Computes mean |grad(sigmoid(logits))| per feature, averaged over valid
    (masked) constituents.

    Parameters
    ----------
    model : ParticleTransformer
    x : Tensor (N, P, F)
    mask : Tensor (N, P) boolean

    Returns
    -------
    importance : ndarray (F,) normalised to sum to 1, or None if no valid entries.
    """
    x = x.detach().clone().requires_grad_(True)
    logits = model(x, particle_mask=mask)["classification"].view(-1)
    torch.sigmoid(logits).mean().backward()
    g = x.grad.detach().abs().cpu().numpy()
    mn = mask.detach().cpu().numpy()
    # Broadcast mask (N, P) -> (N, P, F) for correct indexing
    mn_broad = np.broadcast_to(mn[..., None], g.shape)
    gv = g[mn_broad]
    if gv.size == 0:
        return None
    # Reshape to (n_valid, F) then average over valid constituents
    n_features = g.shape[-1]
    gv = gv.reshape(-1, n_features)
    imp = gv.mean(axis=0)
    return imp / (imp.sum() + 1e-9)


def permutation_importance(model, x, mask, labels):
    """Permutation importance — permutes each feature in-place to save memory.

    For each feature, permutes it across all jets, re-runs inference, and measures
    the drop in accuracy relative to baseline.

    Parameters
    ----------
    model : ParticleTransformer
    x : Tensor (N, P, F)
    mask : Tensor (N, P) boolean
    labels : Tensor (N,) or ndarray

    Returns
    -------
    importance : ndarray (F,) normalised to sum to 1.
    """
    model.eval()
    device = x.device
    with torch.no_grad():
        base_logits = torch.sigmoid(
            model(x, particle_mask=mask)["classification"].view(-1)
        )
        base_preds = (base_logits > 0.5).cpu().numpy()

    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy().astype(int).flatten()
    else:
        labels_np = np.asarray(labels).astype(int).flatten()

    base_acc = (base_preds == labels_np).mean()
    n_features = x.shape[-1]
    drops = np.zeros(n_features)

    for f in range(n_features):
        orig = x[:, :, f].clone()
        perm_idx = torch.randperm(x.shape[0], device=device)
        x[:, :, f] = x[perm_idx, :, f]

        with torch.no_grad():
            perm_logits = torch.sigmoid(
                model(x, particle_mask=mask)["classification"].view(-1)
            )
            perm_preds = (perm_logits > 0.5).cpu().numpy()

        perm_acc = (perm_preds == labels_np).mean()
        drops[f] = base_acc - perm_acc
        x[:, :, f] = orig
        del orig, perm_logits, perm_preds
    _flush()

    drops = np.maximum(drops, 0)
    return drops / (drops.sum() + 1e-9)
