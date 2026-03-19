"""Jet pT resolution fitting: Gaussian fits to response distributions in kinematic bins."""

import numpy as np
from scipy.optimize import curve_fit


def gaussian(x, mu, sigma, A):
    """Simple Gaussian for fitting response distributions."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_response_in_bin(response_values, bins=np.linspace(0, 2, 51)):
    """Fit a Gaussian to the response distribution in a single bin.

    Returns
    -------
    popt : tuple (mu, sigma, A)
    pcov : ndarray (3, 3) covariance matrix
    """
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    counts, _ = np.histogram(response_values, bins=bins)
    initial_guess = [1.0, 0.1, np.max(counts)]
    try:
        popt, pcov = curve_fit(
            gaussian, bin_centers, counts, absolute_sigma=True, p0=initial_guess
        )
        return popt, pcov
    except RuntimeError:
        return (np.nan, np.nan, np.nan), np.zeros((3, 3))


def get_resolution_vs_var(gen_var, pt_response, var_bins):
    """Compute Gaussian mu (scale) and sigma (resolution) of the pT response
    in bins of a given variable (gen_pt or gen_eta).

    Returns
    -------
    bin_centers, mus, sigmas, mu_errs, sigma_errs : tuple of ndarrays
    """
    bin_centers = 0.5 * (var_bins[1:] + var_bins[:-1])
    mus, sigmas = [], []
    mu_errs, sigma_errs = [], []
    for i in range(len(var_bins) - 1):
        mask = (gen_var > var_bins[i]) & (gen_var <= var_bins[i + 1])
        vals = pt_response[mask]
        if len(vals) > 20:
            (mu, sigma, A), cov = fit_response_in_bin(vals)
            mus.append(mu if not np.isnan(mu) else np.nan)
            sigmas.append(abs(sigma) if not np.isnan(sigma) else np.nan)
            mu_errs.append(np.sqrt(cov[0, 0]) if cov[0, 0] > 0 else 0.0)
            sigma_errs.append(np.sqrt(cov[1, 1]) if cov[1, 1] > 0 else 0.0)
        else:
            mus.append(np.nan)
            sigmas.append(np.nan)
            mu_errs.append(0.0)
            sigma_errs.append(0.0)
    return (
        bin_centers,
        np.array(mus),
        np.array(sigmas),
        np.array(mu_errs),
        np.array(sigma_errs),
    )
