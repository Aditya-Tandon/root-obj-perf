"""Attention map and class-attention visualisation for ParticleTransformer."""

import numpy as np
import matplotlib.pyplot as plt


def plot_attention_maps(
    attention_maps, sample_idx, sample_mask, is_signal, layer_idx=0
):
    """Plot per-head attention maps for a single jet.

    Parameters
    ----------
    attention_maps : dict
        As returned by ``evaluation.attention.forward_with_attention``.
    sample_idx : int
        Index of the jet in the batch.
    sample_mask : Tensor or ndarray (B, N)
        Particle mask.
    is_signal : bool
    layer_idx : int
        Which particle-attention layer to visualise.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_heads = attention_maps["particle_attn"][layer_idx].shape[1]
    n_valid = int(sample_mask[sample_idx].sum())

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    attn = attention_maps["particle_attn"][layer_idx][sample_idx, :, :n_valid, :n_valid]

    for h in range(min(n_heads, 8)):
        ax = axes[h]
        attn_head = attn[h].numpy() if hasattr(attn[h], "numpy") else attn[h]
        im = ax.imshow(
            attn_head, cmap="viridis", aspect="auto", vmin=0, vmax=attn_head.max()
        )
        ax.set_xlabel("Key Particle")
        ax.set_ylabel("Query Particle")
        ax.set_title(f"Head {h+1}")
        plt.colorbar(im, ax=ax, fraction=0.046)

    for h in range(min(n_heads, 8), len(axes)):
        axes[h].axis("off")

    label = "Signal (b-jet)" if is_signal else "Background"
    fig.suptitle(
        f"Particle Attention Maps - Layer {layer_idx+1} - {label}", fontsize=14
    )
    plt.tight_layout()
    return fig


def plot_class_attention(
    attention_maps, sample_mask, n_signal, n_background, layer_idx=-1, figsize=None
):
    """Plot class-token attention weights for signal and background jets.

    Parameters
    ----------
    attention_maps : dict
    sample_mask : Tensor or ndarray (B, N)
    n_signal, n_background : int
        Number of signal/background jets in the batch (signal first).
    layer_idx : int
        Which class-attention layer to use (default: last).
    """
    n_samples = n_signal
    if figsize is None:
        figsize = (4 * n_samples, 8)
    fig, axes = plt.subplots(2, n_samples, figsize=figsize, squeeze=False)

    for i in range(n_samples):
        # Signal
        cls_attn = (
            attention_maps["class_attn"][layer_idx][i, :, 0, 1:]
            .mean(dim=0)
            .numpy()
        )
        n_valid = int(sample_mask[i].sum())
        ax = axes[0, i]
        ax.bar(range(n_valid), cls_attn[:n_valid])
        ax.set_xlabel("Constituent Index")
        ax.set_ylabel("Attention Weight")
        ax.set_title(f"Signal Jet {i+1}")

        # Background
        cls_attn = (
            attention_maps["class_attn"][layer_idx][n_signal + i, :, 0, 1:]
            .mean(dim=0)
            .numpy()
        )
        n_valid = int(sample_mask[n_signal + i].sum())
        ax = axes[1, i]
        ax.bar(range(n_valid), cls_attn[:n_valid], color="red")
        ax.set_xlabel("Constituent Index")
        ax.set_ylabel("Attention Weight")
        ax.set_title(f"Background Jet {i+1}")

    fig.suptitle(
        "Class Token Attention Weights (Which particles the classifier focuses on)",
        fontsize=14,
    )
    plt.tight_layout()
    return fig
