"""Attention extraction, activation analysis, and separability metrics for ParticleTransformer."""

import gc
import numpy as np
import torch
import torch.nn as nn


class AttentionHook:
    """Hook to capture attention weights from ParticleAttentionBlock."""

    def __init__(self):
        self.attention_weights = []
        self.handles = []

    def hook_fn(self, module, input, output):
        pass

    def clear(self):
        self.attention_weights = []
        for h in self.handles:
            h.remove()
        self.handles = []


def compute_pairwise_features(x, mask=None):
    """Compute raw pairwise features before the MLP.

    Returns a dict of pairwise feature tensors and the particle mask.
    """
    from model.parT import to_rapidity

    B, N, C = x.shape
    if mask is None:
        mask = x[..., 0] != 0

    r = to_rapidity(x).unsqueeze(2)
    phi = x[..., 3].unsqueeze(2)
    pt = x[..., 1].unsqueeze(2)

    delta = torch.sqrt(
        (r - r.transpose(1, 2)) ** 2 + (phi - phi.transpose(1, 2)) ** 2 + 1e-8
    )
    k_t = torch.min(pt, pt.transpose(1, 2)) * delta
    z = torch.min(pt, pt.transpose(1, 2)) / (pt + pt.transpose(1, 2) + 1e-8)
    m_2 = (
        2
        * pt
        * pt.transpose(1, 2)
        * (torch.cosh(r - r.transpose(1, 2)) - torch.cos(phi - phi.transpose(1, 2)))
        + 1e-8
    )

    dxy = x[..., 4].unsqueeze(2)
    z0 = x[..., 5].unsqueeze(2)
    d_dxy = dxy - dxy.transpose(1, 2)
    d_z0 = z0 - z0.transpose(1, 2)

    q = x[..., 6].unsqueeze(2)
    pt_jet = pt.sum(dim=1, keepdim=True)
    q_weighted = q * pt / (pt_jet + 1e-8)
    q_ij = q_weighted * q_weighted.transpose(1, 2)

    delta_log = torch.log(torch.clamp(delta, min=1e-8))
    k_t_log = torch.log(torch.clamp(k_t, min=1e-8))
    z_log = torch.log(torch.clamp(z, min=1e-8))
    m_2_log = torch.log(torch.clamp(m_2, min=1e-8))

    return {
        "delta_R": delta,
        "log_delta_R": delta_log,
        "k_t": k_t,
        "log_k_t": k_t_log,
        "z": z,
        "log_z": z_log,
        "m_2": m_2,
        "log_m_2": m_2_log,
        "d_dxy": d_dxy,
        "d_z0": d_z0,
        "q_ij": q_ij,
    }, mask


def forward_with_attention(model, x, particle_mask=None):
    """Forward pass returning attention weights from all layers.

    Manually extracts QKV and computes attention scores with pairwise bias.

    Returns
    -------
    attention_maps : dict with "particle_attn" and "class_attn" lists of tensors
    u_ij : Tensor — pairwise embedding (CPU)
    """
    model.eval()
    attention_maps = {"particle_attn": [], "class_attn": []}

    B = x.shape[0]
    x_raw = x.clone()

    if model.use_batch_norm:
        x_proc = x.transpose(1, 2)
        x_proc = model.input_bn(x_proc)
        x_proc = x_proc.transpose(1, 2)
    else:
        x_proc = model.input_ln(x)

    u_ij = model.pair_embed(x_raw, particle_mask)
    x_embed = model.embed(x_proc)

    # Particle attention blocks
    x_curr = x_embed
    for block in model.part_atten_blocks:
        B_curr, N, C = x_curr.shape
        residual = x_curr
        x_norm = block.norm1(x_curr)

        qkv = block.qkv(x_norm).reshape(
            B_curr, N, 3, block.num_heads, C // block.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * block.scale
        if particle_mask is not None:
            attn_mask = particle_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~attn_mask, float("-inf"))
        if u_ij is not None:
            pair_bias = block.pair_proj(u_ij).permute(0, 3, 1, 2)
            attn = attn + pair_bias

        attn_weights = attn.softmax(dim=-1)
        attention_maps["particle_attn"].append(attn_weights.detach().cpu())
        x_curr = block(x_curr, u_ij, particle_mask)

    # Add CLS token
    cls_tokens = model.cls_token.expand(B, -1, -1)
    x_with_cls = torch.cat((cls_tokens, x_curr), dim=1)

    # Class attention blocks
    for block in model.cls_atten_blocks:
        B_curr, N_plus_1, C = x_with_cls.shape
        x_norm = block.norm1(x_with_cls)

        x_cls_norm = x_norm[:, 0:1, :]
        q_cls = (
            block.q(x_cls_norm)
            .reshape(B_curr, 1, block.num_heads, C // block.num_heads)
            .transpose(1, 2)
        )

        kv = (
            block.kv(x_norm)
            .reshape(B_curr, N_plus_1, 2, block.num_heads, C // block.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = (q_cls @ k.transpose(-2, -1)) * block.scale
        if particle_mask is not None:
            cls_mask = torch.ones(
                (B_curr, 1), dtype=particle_mask.dtype, device=particle_mask.device
            )
            full_mask = torch.cat([cls_mask, particle_mask], dim=1)
            attn_mask = full_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~attn_mask, float("-inf"))

        attn_weights = attn.softmax(dim=-1)
        attention_maps["class_attn"].append(attn_weights.detach().cpu())
        x_with_cls = block(x_with_cls, particle_mask)

    return attention_maps, u_ij.detach().cpu()


def forward_with_activations(model, x, particle_mask=None):
    """Forward pass capturing intermediate layer activations.

    Returns
    -------
    activations : dict with keys: input, input_processed, embedding, pairwise_embed,
        particle_attn_layers, pre_cls, cls_attn_layers, final_cls_token, output_logits.
    """
    model.eval()
    activations = {
        "input": x.detach().cpu(),
        "input_processed": None,
        "embedding": None,
        "pairwise_embed": None,
        "particle_attn_layers": [],
        "pre_cls": None,
        "cls_attn_layers": [],
        "final_cls_token": None,
        "output_logits": None,
    }

    B = x.shape[0]
    x_raw = x.clone()

    if model.use_batch_norm:
        x_proc = x.transpose(1, 2)
        x_proc = model.input_bn(x_proc)
        x_proc = x_proc.transpose(1, 2)
    else:
        x_proc = model.input_ln(x)

    activations["input_processed"] = x_proc.detach().cpu()

    u_ij = model.pair_embed(x_raw, particle_mask)
    activations["pairwise_embed"] = u_ij.detach().cpu()

    x_embed = model.embed(x_proc)
    activations["embedding"] = x_embed.detach().cpu()

    x_curr = x_embed
    for i, block in enumerate(model.part_atten_blocks):
        x_curr = block(x_curr, u_ij, particle_mask)
        activations["particle_attn_layers"].append(x_curr.detach().cpu())

    cls_tokens = model.cls_token.expand(B, -1, -1)
    x_with_cls = torch.cat((cls_tokens, x_curr), dim=1)
    activations["pre_cls"] = x_with_cls.detach().cpu()

    for i, block in enumerate(model.cls_atten_blocks):
        x_with_cls = block(x_with_cls, particle_mask)
        activations["cls_attn_layers"].append(x_with_cls.detach().cpu())

    cls_token_final = x_with_cls[:, 0]
    activations["final_cls_token"] = cls_token_final.detach().cpu()

    logits = model.head(cls_token_final)
    activations["output_logits"] = logits.detach().cpu()

    return activations


def compute_separability(activations_arr, labels):
    """Compute Fisher's discriminant ratio as a measure of class separability.

    Returns
    -------
    mean_fisher, max_fisher, fisher_per_neuron : float, float, ndarray
    """
    sig_act = activations_arr[labels == 1]
    bkg_act = activations_arr[labels == 0]

    sig_mean = sig_act.mean(axis=0)
    bkg_mean = bkg_act.mean(axis=0)

    sig_var = sig_act.var(axis=0) + 1e-8
    bkg_var = bkg_act.var(axis=0) + 1e-8

    fisher = (sig_mean - bkg_mean) ** 2 / (sig_var + bkg_var)
    return fisher.mean(), fisher.max(), fisher


def _flush():
    """Free GPU memory."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def forward_with_attention_batched(
    model, loader, device, n_samples=500, batch_size=512, need_attn=True, need_acts=True
):
    """Memory-safe batched attention extraction.

    Only layer-0 particle attention is extracted (head-averaged).

    Parameters
    ----------
    model : ParticleTransformer
    loader : DataLoader
    device : torch.device
    n_samples : int
        Number of jets to process.
    batch_size : int
        Sub-batch size for GPU processing.
    need_attn : bool
        Whether to extract attention maps.
    need_acts : bool
        Whether to extract activation norms.

    Returns
    -------
    layer0_attn_mean : ndarray (N, P, P) or None
    act_norms : ndarray (N, P) or None
    x_cpu : Tensor (N, P, F) — raw input features (CPU)
    mask_np : ndarray (N, P)
    labels_np : ndarray (N,)
    """
    model.eval()
    xs, ms, ys = [], [], []
    count = 0
    for xb, yb, mb, *_ in loader:
        xs.append(xb)
        ms.append(mb)
        ys.append(yb.view(-1))
        count += xb.shape[0]
        if count >= n_samples:
            break
    x_all = torch.cat(xs)[:n_samples]
    m_all = torch.cat(ms)[:n_samples]
    labels_np = torch.cat(ys)[:n_samples].numpy()
    del xs, ms, ys

    N = x_all.shape[0]
    attn_parts = [] if need_attn else None
    act_parts = [] if need_acts else None

    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        x = x_all[s:e].to(device)
        mask = m_all[s:e].to(device)

        with torch.no_grad():
            xr = x
            if model.use_batch_norm:
                xp = model.input_bn(x.transpose(1, 2)).transpose(1, 2)
            else:
                xp = model.input_ln(x)
            u_ij = model.pair_embed(xr, mask)
            xp = model.embed(xp)

            for li, block in enumerate(model.part_atten_blocks):
                if li == 0 and need_attn:
                    B2, N2, C2 = xp.shape
                    qkv = (
                        block.qkv(xp)
                        .reshape(B2, N2, 3, block.num_heads, C2 // block.num_heads)
                        .permute(2, 0, 3, 1, 4)
                    )
                    q, k = qkv[0], qkv[1]
                    att = (q @ k.transpose(-2, -1)) * block.scale
                    if mask is not None:
                        att = att.masked_fill(
                            ~mask.unsqueeze(1).unsqueeze(2), float("-inf")
                        )
                    att = att + block.pair_proj(u_ij).permute(0, 3, 1, 2)
                    aw = att.softmax(dim=-1)
                    attn_parts.append(aw.mean(dim=1).cpu().numpy())
                    del qkv, q, k, att, aw
                xp = block(xp, u_ij, mask)

            if need_acts:
                act_parts.append(torch.norm(xp, dim=-1).cpu().numpy())

        del x, mask, xr, u_ij, xp
        _flush()

    attn_mean = np.concatenate(attn_parts) if need_attn else None
    act_norms = np.concatenate(act_parts) if need_acts else None
    x_cpu = x_all
    mask_np = m_all.numpy()
    del x_all, m_all
    _flush()
    return attn_mean, act_norms, x_cpu, mask_np, labels_np
