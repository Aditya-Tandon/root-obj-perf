"""Matching, resampling, and reweighting utilities for jet datasets."""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
from typing import Tuple, List

from data_pipeline.datasets import Subset, StratifiedJetDataset


def match_dataset_sizes_stratified(
    dataset_large: Subset,
    dataset_small: Subset,
    original_dataset_large: StratifiedJetDataset,
    random_state: int = 42,
    verbose: bool = True,
) -> Subset:
    """
    MODE 1: Subsample the larger dataset to match the size of the smaller one
    while preserving ORIGINAL class distributions.

    Args:
        dataset_large: The larger Subset to be downsampled
        dataset_small: The smaller Subset (target size)
        original_dataset_large: The original StratifiedJetDataset for the larger set
        random_state: Random seed for reproducibility
        verbose: Whether to print statistics

    Returns:
        Downsampled Subset with preserved class distribution,
        Labels of the selected samples
    """
    target_size = len(dataset_small)
    large_indices = np.array(dataset_large.indices)

    # Get labels for the large dataset samples
    large_labels = original_dataset_large.labels[large_indices]

    # Calculate how many samples of each class we need
    subsample_ratio = target_size / len(large_indices)

    # Use stratified sampling to select subset
    _, selected_indices, _, _ = train_test_split(
        large_indices,
        large_labels,
        test_size=subsample_ratio,
        stratify=large_labels,
        random_state=random_state,
    )

    subsampled_ds = Subset(original_dataset_large, selected_indices)

    selected_labels = original_dataset_large.labels[selected_indices]
    if verbose:
        print(
            f"\nDownsampled dataset from {len(dataset_large)} to {len(subsampled_ds)} samples"
        )
        for c in np.unique(selected_labels):
            count = np.sum(selected_labels == c)
            print(f"  Class {c}: {count} ({100*count/len(subsampled_ds):.1f}%)")

    return subsampled_ds, selected_labels


def match_sizes_and_class_ratios(
    dataset1: Subset,
    dataset2: Subset,
    original_dataset1: StratifiedJetDataset,
    original_dataset2: StratifiedJetDataset,
    target_class1_ratio: float = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[Subset, Subset]:
    """
    MODE 2: Match both total events AND class ratios between two datasets
    while preserving feature distributions within each class.

    Strategy:
    - Use the smaller dataset's size as target
    - For each class, compute the average class ratio between the two datasets
    - Resample each dataset to have the target size with matched class ratios
    - Preserve feature distributions by random sampling within each class

    Args:
        dataset1: First Subset (e.g., PF)
        dataset2: Second Subset (e.g., PUPPI)
        original_dataset1: Original StratifiedJetDataset for dataset1
        original_dataset2: Original StratifiedJetDataset for dataset2
        target_class1_ratio: Desired ratio of class 1 (signal) samples (0-1).,
        random_state: Random seed for reproducibility
        verbose: Whether to print statistics

    Returns:
        Tuple of (resampled_dataset1, resampled_dataset2, final_labels1, final_labels2)
    """
    np.random.seed(random_state)

    indices1 = np.array(dataset1.indices)
    indices2 = np.array(dataset2.indices)
    labels1 = original_dataset1.labels[indices1]
    labels2 = original_dataset2.labels[indices2]

    # Compute class ratios for both datasets
    n1_class0 = np.sum(labels1 == 0)
    n1_class1 = np.sum(labels1 == 1)
    n2_class0 = np.sum(labels2 == 0)
    n2_class1 = np.sum(labels2 == 1)

    ratio1_class1 = n1_class1 / len(labels1)  # Signal fraction in dataset1
    ratio2_class1 = n2_class1 / len(labels2)  # Signal fraction in dataset2

    # Target class ratio: ratio_class1 (or could be user-specified)
    if target_class1_ratio is not None:
        target_ratio_class1 = target_class1_ratio
    else:
        target_ratio_class1 = ratio1_class1
    target_ratio_class0 = 1 - target_ratio_class1

    # Target size: use the smaller dataset
    target_size = min(len(dataset1), len(dataset2))

    # Target counts per class
    target_n_class1 = int(target_size * target_ratio_class1)
    target_n_class0 = target_size - target_n_class1

    if verbose:
        print(f"\nMatching sizes and class ratios:")
        print(
            f"  Dataset1 original: {len(dataset1)} samples, Class1 ratio: {ratio1_class1:.1%}"
        )
        print(
            f"  Dataset2 original: {len(dataset2)} samples, Class1 ratio: {ratio2_class1:.1%}"
        )
        print(
            f"  Target size: {target_size}, Target Class1 ratio: {target_ratio_class1:.1%}"
        )
        print(f"  Target Class 0: {target_n_class0}, Target Class 1: {target_n_class1}")

    def resample_dataset(indices, labels, orig_dataset, name):
        """Resample a dataset to target size and class ratios."""
        class0_indices = indices[labels == 0]
        class1_indices = indices[labels == 1]

        # Sample from each class (with replacement if needed)
        if len(class0_indices) >= target_n_class0:
            selected_class0 = np.random.choice(
                class0_indices, target_n_class0, replace=False
            )
        else:
            # Need to oversample
            selected_class0 = np.random.choice(
                class0_indices, target_n_class0, replace=True
            )
            if verbose:
                print(
                    f"  Warning: {name} Class 0 oversampled ({len(class0_indices)} -> {target_n_class0})"
                )

        if len(class1_indices) >= target_n_class1:
            selected_class1 = np.random.choice(
                class1_indices, target_n_class1, replace=False
            )
        else:
            # Need to oversample
            selected_class1 = np.random.choice(
                class1_indices, target_n_class1, replace=True
            )
            if verbose:
                print(
                    f"  Warning: {name} Class 1 oversampled ({len(class1_indices)} -> {target_n_class1})"
                )

        selected_indices = np.concatenate([selected_class0, selected_class1])
        np.random.shuffle(selected_indices)

        return Subset(orig_dataset, selected_indices.tolist())

    resampled1 = resample_dataset(indices1, labels1, original_dataset1, "Dataset1")
    resampled2 = resample_dataset(indices2, labels2, original_dataset2, "Dataset2")

    final_labels1 = original_dataset1.labels[np.array(resampled1.indices)]
    final_labels2 = original_dataset2.labels[np.array(resampled2.indices)]
    if verbose:
        # Verify final distributions
        print(
            f"\n  Final Dataset1: {len(resampled1)} samples, "
            f"Class0={np.sum(final_labels1==0)}, Class1={np.sum(final_labels1==1)}"
        )
        print(
            f"  Final Dataset2: {len(resampled2)} samples, "
            f"Class0={np.sum(final_labels2==0)}, Class1={np.sum(final_labels2==1)}"
        )

    return resampled1, resampled2, final_labels1, final_labels2


def _compute_jet_features(dataset: StratifiedJetDataset, indices: np.ndarray) -> dict:
    """
    Return jet-level pT and eta for a set of indices.

    Uses the pre-computed ``jet_pt`` / ``jet_eta`` arrays stored on the
    dataset (always in memory, even in lazy-HDF5 mode).  This avoids
    reading the full constituent tensor.

    Returns:
        dict with 'pt' and 'eta' arrays of shape (n_samples,)
    """
    return {
        "pt": dataset.jet_pt[indices].numpy(),
        "eta": dataset.jet_eta[indices].numpy(),
    }


def _resample_to_match_distribution(
    source_indices: np.ndarray,
    source_features: np.ndarray,
    target_features: np.ndarray,
    target_n: int,
    n_bins: int = 50,
    random_state: int = 42,
) -> np.ndarray:
    """
    Resample source_indices so that the distribution of source_features
    matches target_features, returning exactly target_n indices.

    Uses acceptance-rejection sampling based on histogram ratio weights.

    Args:
        source_indices: Array of indices into the original dataset
        source_features: 1D array of feature values for source samples
        target_features: 1D array of feature values defining the target distribution
        target_n: Number of samples to select
        n_bins: Number of histogram bins
        random_state: Random seed

    Returns:
        Selected indices array of length target_n
    """
    rng = np.random.RandomState(random_state)

    # Compute histogram range from union of both distributions
    feat_min = min(source_features.min(), target_features.min())
    feat_max = max(source_features.max(), target_features.max())
    bin_edges = np.linspace(feat_min, feat_max, n_bins + 1)

    # Compute normalised histograms
    hist_source, _ = np.histogram(source_features, bins=bin_edges, density=True)
    hist_target, _ = np.histogram(target_features, bins=bin_edges, density=True)

    # Compute per-sample weight = target_density / source_density
    bin_assignments = np.digitize(source_features, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    weights = np.ones(len(source_indices), dtype=np.float64)
    for i in range(len(source_indices)):
        b = bin_assignments[i]
        if hist_source[b] > 0:
            weights[i] = hist_target[b] / hist_source[b]
        else:
            weights[i] = 0.0

    # Normalise weights to probabilities
    total_weight = weights.sum()
    if total_weight > 0:
        probs = weights / total_weight
    else:
        probs = np.ones(len(source_indices)) / len(source_indices)

    # Sample with replacement, weighted by distribution ratio
    selected_idx = rng.choice(len(source_indices), size=target_n, replace=True, p=probs)

    return source_indices[selected_idx]


def match_sizes_ratios_and_distributions(
    dataset1: Subset,
    dataset2: Subset,
    original_dataset1: StratifiedJetDataset,
    original_dataset2: StratifiedJetDataset,
    target_class1_ratio: float = None,
    match_feature: str = "pt",
    n_bins: int = 200,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[Subset, Subset, np.ndarray, np.ndarray]:
    """
    MODE 3: Match total events, class ratios, AND feature distributions
    between two datasets.

    Strategy:
        1. Determine target size (min of both) and target class ratio.
        2. For each class, compute the feature distribution that both datasets
           share (the intersection / minimum envelope of the two distributions).
        3. Use acceptance-rejection resampling on each dataset per class so that
           both end up with the same feature distribution.

    Args:
        dataset1: First Subset (e.g., PUPPI)
        dataset2: Second Subset (e.g., PF)
        original_dataset1: Original StratifiedJetDataset for dataset1
        original_dataset2: Original StratifiedJetDataset for dataset2
        target_class1_ratio: Desired ratio of class 1 samples. If None, use the class1 ratio of the first dataset.
        match_feature: Which jet feature to match on ('pt' or 'eta')
        n_bins: Number of histogram bins for distribution matching
        random_state: Random seed for reproducibility
        verbose: Whether to print statistics

    Returns:
        Tuple of (resampled_dataset1, resampled_dataset2, labels1, labels2)
    """
    np.random.seed(random_state)

    indices1 = np.array(dataset1.indices)
    indices2 = np.array(dataset2.indices)
    labels1 = original_dataset1.labels[indices1]
    labels2 = original_dataset2.labels[indices2]

    # Compute jet-level features for both datasets
    jet_feat1 = _compute_jet_features(original_dataset1, indices1)
    jet_feat2 = _compute_jet_features(original_dataset2, indices2)

    feat1 = jet_feat1[match_feature]
    feat2 = jet_feat2[match_feature]

    # Determine target size and class ratio
    ratio1_class1 = np.sum(labels1 == 1) / len(labels1)
    ratio2_class1 = np.sum(labels2 == 1) / len(labels2)

    if target_class1_ratio is not None:
        target_ratio_class1 = target_class1_ratio
    else:
        target_ratio_class1 = ratio1_class1

    target_size = min(len(dataset1), len(dataset2))
    target_n_class1 = int(target_size * target_ratio_class1)
    target_n_class0 = target_size - target_n_class1

    if verbose:
        print(f"\nMatching sizes, class ratios, AND {match_feature} distributions:")
        print(f"  Dataset1: {len(dataset1)} samples, Class1 ratio: {ratio1_class1:.1%}")
        print(f"  Dataset2: {len(dataset2)} samples, Class1 ratio: {ratio2_class1:.1%}")
        print(
            f"  Target size: {target_size}, Target Class1 ratio: {target_ratio_class1:.1%}"
        )
        print(f"  Target Class 0: {target_n_class0}, Target Class 1: {target_n_class1}")
        print(f"  Matching on: jet-level {match_feature}")

    selected_indices1 = []
    selected_indices2 = []

    for cls, target_n in [(0, target_n_class0), (1, target_n_class1)]:
        # Get per-class indices and features
        cls_mask1 = labels1 == cls
        cls_mask2 = labels2 == cls

        cls_indices1 = indices1[cls_mask1]
        cls_indices2 = indices2[cls_mask2]
        cls_feat1 = feat1[cls_mask1]
        cls_feat2 = feat2[cls_mask2]

        if len(cls_indices1) == 0 or len(cls_indices2) == 0:
            if verbose:
                print(
                    f"  Warning: Class {cls} empty in one dataset, using uniform sampling"
                )
            if len(cls_indices1) > 0:
                sel1 = np.random.choice(cls_indices1, target_n, replace=True)
            else:
                sel1 = np.array([], dtype=int)
            if len(cls_indices2) > 0:
                sel2 = np.random.choice(cls_indices2, target_n, replace=True)
            else:
                sel2 = np.array([], dtype=int)
            selected_indices1.append(sel1)
            selected_indices2.append(sel2)
            continue

        # Build a common target distribution: the minimum envelope
        # (normalised) of both distributions, which is the largest
        # distribution both can reproduce without oversampling too heavily.
        feat_min = min(cls_feat1.min(), cls_feat2.min())
        feat_max = max(cls_feat1.max(), cls_feat2.max())
        bin_edges = np.linspace(feat_min, feat_max, n_bins + 1)

        hist1, _ = np.histogram(cls_feat1, bins=bin_edges, density=True)
        hist2, _ = np.histogram(cls_feat2, bins=bin_edges, density=True)

        # Target = element-wise minimum (the shared region)
        target_hist = np.minimum(hist1, hist2)
        # Smooth to remove step artefacts at distribution crossing points
        # sigma in units of bins; ~2% of n_bins gives a gentle smoothing
        # sigma_bins = max(1, n_bins // 50)
        # target_hist = gaussian_filter1d(target_hist, sigma=sigma_bins)
        # target_hist = np.maximum(target_hist, 0)  # ensure non-negative
        # Re-normalise
        bin_widths = np.diff(bin_edges)
        total = np.sum(target_hist * bin_widths)
        if total > 0:
            target_hist = target_hist / total

        # Resample both datasets to match the common target distribution
        sel1 = _resample_with_target_hist(
            cls_indices1,
            cls_feat1,
            target_hist,
            bin_edges,
            target_n,
            random_state=random_state + cls,
        )
        sel2 = _resample_with_target_hist(
            cls_indices2,
            cls_feat2,
            target_hist,
            bin_edges,
            target_n,
            random_state=random_state + cls + 100,
        )

        selected_indices1.append(sel1)
        selected_indices2.append(sel2)

        if verbose:
            print(
                f"  Class {cls}: resampled {len(cls_indices1)}->{target_n} (ds1), "
                f"{len(cls_indices2)}->{target_n} (ds2)"
            )

    # Combine and shuffle
    all_sel1 = np.concatenate(selected_indices1)
    all_sel2 = np.concatenate(selected_indices2)
    np.random.shuffle(all_sel1)
    np.random.shuffle(all_sel2)

    resampled1 = Subset(original_dataset1, all_sel1.tolist())
    resampled2 = Subset(original_dataset2, all_sel2.tolist())

    final_labels1 = original_dataset1.labels[all_sel1]
    final_labels2 = original_dataset2.labels[all_sel2]

    if verbose:
        print(
            f"\n  Final Dataset1: {len(resampled1)} samples, "
            f"Class0={np.sum(final_labels1==0)}, Class1={np.sum(final_labels1==1)}"
        )
        print(
            f"  Final Dataset2: {len(resampled2)} samples, "
            f"Class0={np.sum(final_labels2==0)}, Class1={np.sum(final_labels2==1)}"
        )

    return resampled1, resampled2, final_labels1, final_labels2


def _resample_with_target_hist(
    source_indices: np.ndarray,
    source_features: np.ndarray,
    target_hist: np.ndarray,
    bin_edges: np.ndarray,
    target_n: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Resample source indices so the feature distribution matches target_hist.

    Args:
        source_indices: Indices into the original dataset
        source_features: Feature values for each source sample
        target_hist: Target density histogram (already normalised)
        bin_edges: Bin edges for the histogram
        target_n: Number of output samples
        random_state: Random seed

    Returns:
        Array of selected indices of length target_n
    """
    rng = np.random.RandomState(random_state)
    n_bins = len(target_hist)

    # Compute source histogram (density)
    hist_source, _ = np.histogram(source_features, bins=bin_edges, density=True)

    # Assign each sample to a bin
    bin_assignments = np.digitize(source_features, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    # Compute per-sample weight = target / source
    weights = np.zeros(len(source_indices), dtype=np.float64)
    for b in range(n_bins):
        mask = bin_assignments == b
        if hist_source[b] > 0:
            weights[mask] = target_hist[b] / hist_source[b]
        else:
            weights[mask] = 0.0

    # Normalise to probabilities
    total = weights.sum()
    if total > 0:
        probs = weights / total
    else:
        probs = np.ones(len(source_indices)) / len(source_indices)

    selected_idx = rng.choice(len(source_indices), size=target_n, replace=True, p=probs)
    return source_indices[selected_idx]


def _resample_with_target_hist_2d(
    source_indices: np.ndarray,
    source_feat_a: np.ndarray,
    source_feat_b: np.ndarray,
    target_hist_2d: np.ndarray,
    bin_edges_a: np.ndarray,
    bin_edges_b: np.ndarray,
    target_n: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Resample source indices so the 2D (feat_a, feat_b) distribution matches
    target_hist_2d.

    Args:
        source_indices: Indices into the original dataset
        source_feat_a: 1D array of first feature values (e.g. pT)
        source_feat_b: 1D array of second feature values (e.g. eta)
        target_hist_2d: Target 2D density histogram (n_bins_a, n_bins_b), normalised
        bin_edges_a: Bin edges for feature a
        bin_edges_b: Bin edges for feature b
        target_n: Number of output samples
        random_state: Random seed

    Returns:
        Array of selected indices of length target_n
    """
    rng = np.random.RandomState(random_state)
    n_bins_a = len(bin_edges_a) - 1
    n_bins_b = len(bin_edges_b) - 1

    # Compute source 2D histogram (density)
    hist_source, _, _ = np.histogram2d(
        source_feat_a, source_feat_b, bins=[bin_edges_a, bin_edges_b], density=True
    )

    # Assign each sample to a 2D bin
    bin_a = np.digitize(source_feat_a, bin_edges_a) - 1
    bin_a = np.clip(bin_a, 0, n_bins_a - 1)
    bin_b = np.digitize(source_feat_b, bin_edges_b) - 1
    bin_b = np.clip(bin_b, 0, n_bins_b - 1)

    # Compute per-sample weight = target / source
    weights = np.zeros(len(source_indices), dtype=np.float64)
    for i in range(len(source_indices)):
        ba, bb = bin_a[i], bin_b[i]
        if hist_source[ba, bb] > 0:
            weights[i] = target_hist_2d[ba, bb] / hist_source[ba, bb]
        else:
            weights[i] = 0.0

    # Normalise to probabilities
    total = weights.sum()
    if total > 0:
        probs = weights / total
    else:
        probs = np.ones(len(source_indices)) / len(source_indices)

    selected_idx = rng.choice(len(source_indices), size=target_n, replace=True, p=probs)
    return source_indices[selected_idx]


def match_sizes_ratios_and_distributions_2d(
    dataset1: Subset,
    dataset2: Subset,
    original_dataset1: StratifiedJetDataset,
    original_dataset2: StratifiedJetDataset,
    target_class1_ratio: float = None,
    n_bins_pt: int = 50,
    n_bins_eta: int = 50,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[Subset, Subset, np.ndarray, np.ndarray]:
    """
    MODE 4: Match total events, class ratios, AND both pT and eta
    distributions simultaneously between two datasets.

    Strategy:
        1. Determine target size (min of both) and target class ratio
           (uses dataset1's class1 ratio).
        2. For each class, build a 2D (pT, eta) histogram for both datasets,
           take the element-wise minimum envelope, smooth, and use as the
           common target.
        3. Use acceptance-rejection resampling in 2D on each dataset per class.

    Args:
        dataset1: First Subset (e.g., PUPPI)
        dataset2: Second Subset (e.g., PF)
        original_dataset1: Original StratifiedJetDataset for dataset1
        original_dataset2: Original StratifiedJetDataset for dataset2
        target_class1_ratio: Desired ratio of class 1 samples.
            If None, uses dataset1's class1 ratio.
        n_bins_pt: Number of histogram bins for pT
        n_bins_eta: Number of histogram bins for eta
        random_state: Random seed for reproducibility
        verbose: Whether to print statistics

    Returns:
        Tuple of (resampled_dataset1, resampled_dataset2, labels1, labels2)
    """

    np.random.seed(random_state)

    indices1 = np.array(dataset1.indices)
    indices2 = np.array(dataset2.indices)
    labels1 = original_dataset1.labels[indices1]
    labels2 = original_dataset2.labels[indices2]

    # Compute jet-level features for both datasets
    jet_feat1 = _compute_jet_features(original_dataset1, indices1)
    jet_feat2 = _compute_jet_features(original_dataset2, indices2)

    pt1, eta1 = jet_feat1["pt"], jet_feat1["eta"]
    pt2, eta2 = jet_feat2["pt"], jet_feat2["eta"]

    # Determine target size and class ratio
    ratio1_class1 = np.sum(labels1 == 1) / len(labels1)
    ratio2_class1 = np.sum(labels2 == 1) / len(labels2)

    if target_class1_ratio is not None:
        target_ratio_class1 = target_class1_ratio
    else:
        target_ratio_class1 = ratio1_class1

    target_size = min(len(dataset1), len(dataset2))
    target_n_class1 = int(target_size * target_ratio_class1)
    target_n_class0 = target_size - target_n_class1

    if verbose:
        print(f"\nMatching sizes, class ratios, AND pT+eta distributions (2D):")
        print(f"  Dataset1: {len(dataset1)} samples, Class1 ratio: {ratio1_class1:.1%}")
        print(f"  Dataset2: {len(dataset2)} samples, Class1 ratio: {ratio2_class1:.1%}")
        print(
            f"  Target size: {target_size}, Target Class1 ratio: {target_ratio_class1:.1%}"
        )
        print(f"  Target Class 0: {target_n_class0}, Target Class 1: {target_n_class1}")
        print(
            f"  Matching on: jet-level pT ({n_bins_pt} bins) x eta ({n_bins_eta} bins)"
        )

    selected_indices1 = []
    selected_indices2 = []

    for cls, target_n in [(0, target_n_class0), (1, target_n_class1)]:
        cls_mask1 = labels1 == cls
        cls_mask2 = labels2 == cls

        cls_indices1 = indices1[cls_mask1]
        cls_indices2 = indices2[cls_mask2]
        cls_pt1, cls_eta1 = pt1[cls_mask1], eta1[cls_mask1]
        cls_pt2, cls_eta2 = pt2[cls_mask2], eta2[cls_mask2]

        if len(cls_indices1) == 0 or len(cls_indices2) == 0:
            if verbose:
                print(
                    f"  Warning: Class {cls} empty in one dataset, using uniform sampling"
                )
            if len(cls_indices1) > 0:
                sel1 = np.random.choice(cls_indices1, target_n, replace=True)
            else:
                sel1 = np.array([], dtype=int)
            if len(cls_indices2) > 0:
                sel2 = np.random.choice(cls_indices2, target_n, replace=True)
            else:
                sel2 = np.array([], dtype=int)
            selected_indices1.append(sel1)
            selected_indices2.append(sel2)
            continue

        # Build common bin edges
        pt_min = min(cls_pt1.min(), cls_pt2.min())
        pt_max = max(cls_pt1.max(), cls_pt2.max())
        eta_min = min(cls_eta1.min(), cls_eta2.min())
        eta_max = max(cls_eta1.max(), cls_eta2.max())
        bin_edges_pt = np.linspace(pt_min, pt_max, n_bins_pt + 1)
        bin_edges_eta = np.linspace(eta_min, eta_max, n_bins_eta + 1)

        # 2D histograms (density normalised)
        hist1_2d, _, _ = np.histogram2d(
            cls_pt1, cls_eta1, bins=[bin_edges_pt, bin_edges_eta], density=True
        )
        hist2_2d, _, _ = np.histogram2d(
            cls_pt2, cls_eta2, bins=[bin_edges_pt, bin_edges_eta], density=True
        )

        # Minimum envelope in 2D
        target_hist_2d = np.minimum(hist1_2d, hist2_2d)
        # Smooth with 2D Gaussian to avoid step artefacts
        # sigma_pt = max(1, n_bins_pt // 50)
        # sigma_eta = max(1, n_bins_eta // 50)
        # target_hist_2d = gaussian_filter(target_hist_2d, sigma=[sigma_pt, sigma_eta])
        # target_hist_2d = np.maximum(target_hist_2d, 0)
        # Re-normalise
        bin_widths_pt = np.diff(bin_edges_pt)
        bin_widths_eta = np.diff(bin_edges_eta)
        bin_areas = np.outer(bin_widths_pt, bin_widths_eta)
        total = np.sum(target_hist_2d * bin_areas)
        if total > 0:
            target_hist_2d = target_hist_2d / total

        # Resample both datasets to match the common 2D target distribution
        sel1 = _resample_with_target_hist_2d(
            cls_indices1,
            cls_pt1,
            cls_eta1,
            target_hist_2d,
            bin_edges_pt,
            bin_edges_eta,
            target_n,
            random_state=random_state + cls,
        )
        sel2 = _resample_with_target_hist_2d(
            cls_indices2,
            cls_pt2,
            cls_eta2,
            target_hist_2d,
            bin_edges_pt,
            bin_edges_eta,
            target_n,
            random_state=random_state + cls + 100,
        )

        selected_indices1.append(sel1)
        selected_indices2.append(sel2)

        if verbose:
            print(
                f"  Class {cls}: resampled {len(cls_indices1)}->{target_n} (ds1), "
                f"{len(cls_indices2)}->{target_n} (ds2)"
            )

    # Combine and shuffle
    all_sel1 = np.concatenate(selected_indices1)
    all_sel2 = np.concatenate(selected_indices2)
    np.random.shuffle(all_sel1)
    np.random.shuffle(all_sel2)

    resampled1 = Subset(original_dataset1, all_sel1.tolist())
    resampled2 = Subset(original_dataset2, all_sel2.tolist())

    final_labels1 = original_dataset1.labels[all_sel1]
    final_labels2 = original_dataset2.labels[all_sel2]

    if verbose:
        print(
            f"\n  Final Dataset1: {len(resampled1)} samples, "
            f"Class0={np.sum(final_labels1==0)}, Class1={np.sum(final_labels1==1)}"
        )
        print(
            f"  Final Dataset2: {len(resampled2)} samples, "
            f"Class0={np.sum(final_labels2==0)}, Class1={np.sum(final_labels2==1)}"
        )

    return resampled1, resampled2, final_labels1, final_labels2


def _reweight_to_common_target(
    datasets_and_subsets: List[Tuple[StratifiedJetDataset, Subset]],
    ref_pt: np.ndarray,
    ref_eta: np.ndarray,
    n_bins_pt: int = 100,
    n_bins_eta: int = 50,
    max_pt: float = 500.0,
    clip_max: float = 50.0,
    smooth_sigma: float = 1.0,
    verbose: bool = True,
) -> None:
    """
    Reweight EACH CLASS within every subset so its (pT, eta) distribution
    matches a single common reference distribution.
    """
    from scipy.ndimage import gaussian_filter as _gaussian_filter

    # Fixed bin edges for all subsets
    pt_bins = np.linspace(0, max_pt, n_bins_pt + 1)
    eta_bins = np.linspace(-2.0, 2.0, n_bins_eta + 1)

    # Reference 2-D density histogram (This is your Target)
    H_ref, _, _ = np.histogram2d(
        ref_pt, ref_eta, bins=[pt_bins, eta_bins], density=True
    )
    H_ref = np.maximum(H_ref, 1e-10)

    for dataset, subset in datasets_and_subsets:
        indices = np.array(subset.indices)
        labels = dataset.labels[indices]

        for cls in [0, 1]:
            # 1. Identify indices for this class
            cls_mask = labels == cls
            cls_indices = indices[cls_mask]

            if len(cls_indices) == 0:
                continue

            # 2. Compute features ONLY for this class
            jet_feats = _compute_jet_features(dataset, cls_indices)
            pt = jet_feats["pt"]
            eta = jet_feats["eta"]

            # 3. Compute Source Histogram for THIS CLASS
            H_src, _, _ = np.histogram2d(
                pt, eta, bins=[pt_bins, eta_bins], density=True
            )
            H_src = np.maximum(H_src, 1e-10)

            # 4. Calculate Ratio: Reference / Class_Source
            ratio_map = H_ref / H_src

            # Apply Gaussian smoothing
            if smooth_sigma > 0:
                ratio_map = _gaussian_filter(
                    ratio_map, sigma=smooth_sigma, mode="nearest"
                )

            # 5. Map weights to individual events of this class
            pt_idx = np.clip(np.searchsorted(pt_bins, pt) - 1, 0, n_bins_pt - 1)
            eta_idx = np.clip(np.searchsorted(eta_bins, eta) - 1, 0, n_bins_eta - 1)
            weights = ratio_map[pt_idx, eta_idx].astype(np.float32)
            weights = np.clip(weights, 0.1, clip_max)

            # 6. Write in-place to the specific class indices
            # Note: We must be careful with indexing. cls_indices points to the dataset rows.
            dataset.weights[cls_indices] = (
                torch.from_numpy(weights).float().unsqueeze(1)
            )

            if verbose:
                cls_name = "Signal" if cls == 1 else "Background"
                print(
                    f"    {cls_name}: mean={weights.mean():.3f}, "
                    f"std={weights.std():.3f}, max={weights.max():.3f}"
                )


def _reweight_bkg_to_signal(
    dataset: "StratifiedJetDataset",
    subset: Subset,
    n_bins_pt: int = 100,
    n_bins_eta: int = 50,
    max_pt: float = 500.0,
    clip_max: float = 50.0,
    smooth_sigma: float = 1.0,
    verbose: bool = True,
) -> None:
    """
    Reweight the **background** jets in a single (dataset, subset) so that
    their (pT, eta) distribution matches the **signal** jets in the same
    subset.  Signal weights are left at 1.0.

    weight_i(bkg) = P(pT,eta | signal) / P(pT,eta | background)

    Weights are clipped to [0.1, clip_max] and written in-place.

    Args:
        dataset: The StratifiedJetDataset that owns the data.
        subset: torch Subset whose .indices point into *dataset*.
        n_bins_pt: Number of pT bins for the 2-D histogram.
        n_bins_eta: Number of eta bins for the 2-D histogram.
        max_pt: Upper pT edge for binning.
        clip_max: Maximum weight after clipping.
        smooth_sigma: Gaussian smoothing sigma for ratio map (0 = no smoothing).
        verbose: Print per-subset weight statistics.
    """

    indices = np.array(subset.indices)
    labels = dataset.labels[indices]  # 0 or 1

    sig_mask = labels == 1
    bkg_mask = labels == 0

    sig_idx = indices[sig_mask]
    bkg_idx = indices[bkg_mask]

    if len(sig_idx) == 0 or len(bkg_idx) == 0:
        if verbose:
            print("    Skipping: no signal or background jets in subset.")
        return

    # Compute jet features for signal and background
    sig_feats = _compute_jet_features(dataset, sig_idx)
    bkg_feats = _compute_jet_features(dataset, bkg_idx)

    # Fixed bin edges
    pt_bins = np.linspace(0, max_pt, n_bins_pt + 1)
    eta_bins = np.linspace(-2.5, 2.5, n_bins_eta + 1)

    # Signal (reference) 2-D density
    H_sig, _, _ = np.histogram2d(
        sig_feats["pt"], sig_feats["eta"], bins=[pt_bins, eta_bins], density=True
    )
    H_sig = np.maximum(H_sig, 1e-10)

    # Background (source) 2-D density
    H_bkg, _, _ = np.histogram2d(
        bkg_feats["pt"], bkg_feats["eta"], bins=[pt_bins, eta_bins], density=True
    )
    H_bkg = np.maximum(H_bkg, 1e-10)

    ratio_map = H_sig / H_bkg

    if smooth_sigma > 0:
        ratio_map = gaussian_filter(ratio_map, sigma=smooth_sigma, mode="nearest")

    # Map weights to individual background events
    pt_idx = np.clip(np.searchsorted(pt_bins, bkg_feats["pt"]) - 1, 0, n_bins_pt - 1)
    eta_idx = np.clip(
        np.searchsorted(eta_bins, bkg_feats["eta"]) - 1, 0, n_bins_eta - 1
    )
    bkg_weights = ratio_map[pt_idx, eta_idx].astype(np.float32)
    bkg_weights = np.clip(bkg_weights, 0.1, clip_max)

    # Write background weights in-place; signal weights stay at 1.0
    dataset.weights[sig_idx] = 1.0
    dataset.weights[bkg_idx] = torch.from_numpy(bkg_weights).float().unsqueeze(1)

    if verbose:
        sig_w = dataset.weights[sig_idx].squeeze().numpy()
        print(
            f"    Signal: mean={sig_w.mean():.3f}, "
            f"std={sig_w.std():.3f}, max={sig_w.max():.3f}  (kept at 1.0)"
        )
        print(
            f"    Background: mean={bkg_weights.mean():.3f}, "
            f"std={bkg_weights.std():.3f}, max={bkg_weights.max():.3f}"
        )
