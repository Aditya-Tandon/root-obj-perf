"""Luminosity scaling utilities for HH->4b vs QCD analysis.

Provides functions to convert raw cross-section weights into expected event
counts at a target integrated luminosity.

Physics:
    per-event weight = sigma_pb * L_pb / N_gen
    where L_pb = luminosity_fb * 1000  (1 fb^-1 = 1000 pb^-1)
"""

import json
import numpy as np

DEFAULT_LUMINOSITY_FB = 1000.0
DEFAULT_HH4B_XSEC_PB = 0.0113  # sigma(pp->HH) * BR(HH->bbbb) at 14 TeV NLO


def load_physics_config(config_path="hh-bbbb-obj-config.json"):
    """Load luminosity, signal cross-section, and n_gen from config JSON.

    Returns
    -------
    dict with keys: luminosity_fb, signal_xsec_pb, n_gen_signal
    """
    with open(config_path) as f:
        cfg = json.load(f)
    phys = cfg.get("physics", {})
    return {
        "luminosity_fb": phys.get("luminosity_fb", DEFAULT_LUMINOSITY_FB),
        "signal_xsec_pb": phys.get("signal_xsec_pb", DEFAULT_HH4B_XSEC_PB),
        "n_gen_signal": phys.get("n_gen_signal"),
    }


def lumi_pb(luminosity_fb):
    """Convert luminosity from fb^-1 to pb^-1."""
    return luminosity_fb * 1000.0


def signal_weight(
    n_events,
    luminosity_fb=DEFAULT_LUMINOSITY_FB,
    signal_xsec_pb=DEFAULT_HH4B_XSEC_PB,
    n_gen_signal=None,
):
    """Per-event signal weight array for expected yield at given luminosity.

    Parameters
    ----------
    n_events : int
        Number of signal events (after selection).
    luminosity_fb : float
        Target integrated luminosity in fb^-1.
    signal_xsec_pb : float
        Signal cross-section in pb.
    n_gen_signal : int or None
        Total generated signal events (before selection).
        If None, uses n_events.

    Returns
    -------
    np.ndarray of shape (n_events,), float64
    """
    if n_gen_signal is None:
        n_gen_signal = n_events
    w = signal_xsec_pb * lumi_pb(luminosity_fb) / n_gen_signal
    return np.full(n_events, w, dtype=np.float64)


def qcd_weight_per_event(sigma_bin_pb, n_gen_bin, luminosity_fb=DEFAULT_LUMINOSITY_FB):
    """Single per-event QCD weight for a given pT bin.

    Parameters
    ----------
    sigma_bin_pb : float
        Bin cross-section in pb.
    n_gen_bin : int
        Total generated events in this bin.
    luminosity_fb : float
        Target integrated luminosity in fb^-1.

    Returns
    -------
    float
    """
    return sigma_bin_pb * lumi_pb(luminosity_fb) / n_gen_bin


def scale_qcd_weights_raw(
    raw_weights, sigma_to_ngen, luminosity_fb=DEFAULT_LUMINOSITY_FB
):
    """Scale Convention-C QCD weights (raw sigma_bin per event) to expected yields.

    In Convention C (notebooks loading directly from ROOT), each event carries
    weight = sigma_bin (the bin cross-section). This function converts to
    w = sigma_bin * L_pb / N_gen_bin.

    Parameters
    ----------
    raw_weights : np.ndarray
        Per-event weights where each event has weight = sigma_bin.
    sigma_to_ngen : dict
        Maps sigma_bin (float) -> n_gen (int) for each QCD pT bin.
    luminosity_fb : float

    Returns
    -------
    np.ndarray — luminosity-scaled per-event weights
    """
    L = lumi_pb(luminosity_fb)
    raw_weights = np.asarray(raw_weights).ravel()
    # Default: keep original value (signal jets have weight=1.0 and won't match any sigma bin)
    scaled = raw_weights.astype(np.float64)
    matched = np.zeros(len(raw_weights), dtype=bool)
    for sigma_bin, n_gen in sigma_to_ngen.items():
        mask = np.isclose(raw_weights, sigma_bin, rtol=1e-6)
        scaled[mask] = sigma_bin * L / n_gen
        matched |= mask
    n_unmatched_nonunit = int(np.sum(~matched & ~np.isclose(raw_weights, 1.0)))
    if n_unmatched_nonunit > 0:
        import warnings

        warnings.warn(
            f"scale_qcd_weights_raw: {n_unmatched_nonunit} weights did not match any "
            "sigma_bin and are not 1.0 (signal). Check sigma_to_ngen dict.",
            RuntimeWarning,
            stacklevel=2,
        )
    return scaled


def scale_qcd_weights_per_event(stored_weights, luminosity_fb=DEFAULT_LUMINOSITY_FB):
    """Scale Convention-B QCD weights (sigma_bin / N_loaded per event) to expected yields.

    In Convention B (make_event_dataset.py), stored weights are already
    sigma_bin / N_loaded. Multiply by L_pb to get expected event counts.

    Parameters
    ----------
    stored_weights : np.ndarray
        Per-event weights where each event has weight = sigma_bin / N_loaded.
    luminosity_fb : float

    Returns
    -------
    np.ndarray — luminosity-scaled per-event weights
    """
    return stored_weights * lumi_pb(luminosity_fb)


def build_eval_weights(
    qcd_weights_raw,
    sigma_to_ngen,
    n_signal,
    mode="qcd_only",
    luminosity_fb=DEFAULT_LUMINOSITY_FB,
    signal_xsec_pb=DEFAULT_HH4B_XSEC_PB,
    n_gen_signal=None,
):
    """Build a combined sample_weight array for ROC/AUC evaluation.

    The returned array has length ``n_signal + len(qcd_weights_raw)`` and is
    ordered [signal, QCD].  Three weighting modes are supported:

    * ``"unweighted"`` — all weights 1.0.
    * ``"qcd_only"``   — signal = 1.0, QCD = sigma_bin * L / N_gen
      (correct within-QCD composition; HH b-jets keep equal emphasis).
    * ``"full_physics"``— signal = sigma_HH * L / N_gen_signal,
      QCD = sigma_bin * L / N_gen  (true expected yields).

    Parameters
    ----------
    qcd_weights_raw : np.ndarray
        Convention-C weights (raw sigma_bin) for QCD jets.
    sigma_to_ngen : dict
        Maps sigma_bin (float) -> n_gen (int) for each QCD pT bin.
    n_signal : int
        Number of signal jets.
    mode : str
        One of ``"unweighted"``, ``"qcd_only"``, ``"full_physics"``.
    luminosity_fb : float
    signal_xsec_pb : float
    n_gen_signal : int or None
        Total generated signal events.  Required for ``"full_physics"`` mode.

    Returns
    -------
    np.ndarray of shape (n_signal + len(qcd_weights_raw),), float64
    """
    if mode == "unweighted":
        return np.ones(n_signal + len(qcd_weights_raw), dtype=np.float64)

    qcd_scaled = scale_qcd_weights_raw(qcd_weights_raw, sigma_to_ngen, luminosity_fb)

    if mode == "qcd_only":
        sig_w = np.ones(n_signal, dtype=np.float64)
    elif mode == "full_physics":
        sig_w = signal_weight(n_signal, luminosity_fb, signal_xsec_pb, n_gen_signal)
    else:
        raise ValueError(
            f"Unknown eval weight mode {mode!r}. "
            "Choose from 'unweighted', 'qcd_only', 'full_physics'."
        )

    return np.concatenate([sig_w, qcd_scaled])
