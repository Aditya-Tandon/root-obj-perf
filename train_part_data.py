import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List, Dict
import vector
from scipy.ndimage import gaussian_filter1d


# --- Dataset Class ---
class L1JetDataset(Dataset):
    def __init__(self, filepath):
        print(f"Loading data from {filepath}...")
        data = np.load(filepath)
        # X: (N, n_constituents, Features)
        self.X = torch.from_numpy(data["x"]).float()
        self.y = torch.from_numpy(data["y"]).float().unsqueeze(1)

        # Load particle mask if available, otherwise infer from non-zero energy
        if "mask" in data.files:
            self.mask = torch.from_numpy(data["mask"]).bool()
            print("Loaded particle mask from dataset.")
        else:
            # Fallback: infer mask from non-zero particles (E != 0)
            self.mask = self.X[..., 0] != 0
            print("No particle_mask in dataset, inferring from non-zero energy")

        if "weights" in data.files:
            print("Loaded weights from dataset.")
            self.weights = torch.from_numpy(data["weights"]).float().unsqueeze(1)
        else:
            print("No weights in dataset, using uniform weights of 1.0")
            self.weights = torch.ones_like(self.y)

        print(f"Data loaded: {self.X.shape} samples")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx], self.weights[idx]


# --- Stratified Split Helper ---
def stratified_split(dataset, val_split, num_classes, random_state=42, verbose=True):
    """
    Perform a stratified train/validation split on a dataset.

    Args:
        dataset: Dataset object with a .y attribute containing labels
        val_split: Fraction of data to use for validation (0-1)
        num_classes: Number of classes (1 for binary classification)
        random_state: Random seed for reproducibility
        verbose: Whether to print class distribution statistics

    Returns:
        train_ds: Training subset
        val_ds: Validation subset
        train_indices: Indices of training samples
        val_indices: Indices of validation samples
        stratify_labels: Labels used for stratification
    """
    # Get labels for stratification
    # For binary classification (num_classes=1), use the binary labels directly
    # For multi-class, the labels should already be class indices
    stratify_labels = dataset.y.squeeze().numpy().astype(int)

    # Create indices and perform stratified split
    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        stratify=stratify_labels,
        random_state=random_state,
    )

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    if verbose:
        # Print class distribution for verification
        train_labels = stratify_labels[train_indices]
        val_labels = stratify_labels[val_indices]
        print(f"Stratified split complete:")
        print(f"  Train set: {len(train_ds)} samples")
        print(f"  Val set: {len(val_ds)} samples")
        for c in range(max(num_classes, 2)):
            train_count = np.sum(train_labels == c)
            val_count = np.sum(val_labels == c)
            print(
                f"  Class {c}: Train={train_count} ({100*train_count/len(train_ds):.1f}%), "
                f"Val={val_count} ({100*val_count/len(val_ds):.1f}%)"
            )

    return train_ds, val_ds, train_indices, val_indices, stratify_labels


class StratifiedJetDataset(Dataset):
    """
    A Dataset class for jet constituent data that supports stratified splitting
    while preserving underlying feature distributions.

    Attributes:
        X: Tensor of shape (N, n_constituents, n_features) - constituent features
        y: Tensor of shape (N, 1) - binary labels
        mask: Tensor of shape (N, n_constituents) - particle mask
        weights: Tensor of shape (N, 1) - sample weights
    """

    def __init__(self, filepath: str, mode: str = None):
        """
        Load dataset from .npz file.

        Args:
            filepath: Path to the .npz file containing x, y, mask, weights
            mode: Data loading mode of the dataloader. If mode="match_distributions", weights are set to 1.0
        """
        print(f"Loading data from {filepath}...")
        data = np.load(filepath)

        self.X = torch.from_numpy(data["x"]).float()
        self.y = torch.from_numpy(data["y"]).float().unsqueeze(1)

        if "mask" in data.files:
            self.mask = torch.from_numpy(data["mask"]).bool()
        else:
            self.mask = self.X[..., 1] != 0
            print("No mask in dataset, inferring from non-zero second feature")

        if "weights" in data.files:
            self.weights = torch.from_numpy(data["weights"]).float().unsqueeze(1)
        else:
            self.weights = torch.ones_like(self.y)
            print("No weights in dataset, using uniform weights of 1.0")

        if mode in (
            "match_distributions",
            "match_distributions_2d",
            "match_distributions_2d_reweight",
            "reweight_to_pf_signal",
        ):
            self.weights = torch.ones_like(self.y)
            print("Setting weights to 1.0 (will be recomputed after resampling).")

        print(
            f"Data loaded: {self.X.shape[0]} samples, {self.X.shape[1]} constituents, {self.X.shape[2]} features"
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.mask[idx], self.weights[idx]

    @property
    def labels(self) -> np.ndarray:
        """Get labels as numpy array for stratification."""
        return self.y.squeeze().numpy().astype(int)

    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of classes in the dataset."""
        labels = self.labels
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))


def stratified_train_val_split(
    dataset: StratifiedJetDataset,
    val_split: float = 0.3,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[Subset, Subset, np.ndarray, np.ndarray]:
    """
    Perform stratified train/validation split preserving class distributions.

    This ensures that both the train and validation sets have the same
    proportion of each class as the original dataset, which also preserves
    the feature distributions associated with each class.

    Args:
        dataset: StratifiedJetDataset instance
        val_split: Fraction of data for validation (0-1)
        random_state: Random seed for reproducibility
        verbose: Whether to print distribution statistics

    Returns:
        train_ds: Training Subset
        val_ds: Validation Subset
        train_indices: Array of training sample indices
        val_indices: Array of validation sample indices
        train_labels: Array of training labels (for verification)
        val_labels: Array of validation labels (for verification)
    """
    labels = dataset.labels
    indices = np.arange(len(dataset))

    train_indices, val_indices = train_test_split(
        indices, test_size=val_split, stratify=labels, random_state=random_state
    )

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    if verbose:

        print(f"Stratified split complete:")
        print(f"  Train set: {len(train_ds)} samples")
        print(f"  Val set: {len(val_ds)} samples")

        for c in np.unique(labels):
            train_count = np.sum(train_labels == c)
            val_count = np.sum(val_labels == c)
            print(
                f"  Class {c}: Train={train_count} ({100*train_count/len(train_ds):.1f}%), "
                f"Val={val_count} ({100*val_count/len(val_ds):.1f}%)"
            )

    return train_ds, val_ds, train_indices, val_indices, train_labels, val_labels


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
    Compute jet-level pT and eta from constituent 4-vectors for a set of indices.

    Assumes feature ordering: [mass, pt, eta, phi, ...] in the constituent array.

    Returns:
        dict with 'pt' and 'eta' arrays of shape (n_samples,)
    """
    X = dataset.X[indices]  # (N, n_const, n_feat)
    mask = dataset.mask[indices]  # (N, n_const)

    const_mass = X[:, :, 0].numpy()
    const_pt = X[:, :, 1].numpy()
    const_eta = X[:, :, 2].numpy()
    const_phi = X[:, :, 3].numpy()

    # Zero out masked constituents
    mask_np = mask.numpy()
    const_mass = np.where(mask_np, const_mass, 0)
    const_pt = np.where(mask_np, const_pt, 0)
    const_eta = np.where(mask_np, const_eta, 0)
    const_phi = np.where(mask_np, const_phi, 0)

    const_vectors = vector.array(
        {
            "pt": const_pt,
            "eta": const_eta,
            "phi": const_phi,
            "mass": const_mass,
        }
    )
    jet_vectors = const_vectors.sum(axis=1)

    return {"pt": np.array(jet_vectors.pt), "eta": np.array(jet_vectors.eta)}


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
    from scipy.ndimage import gaussian_filter

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
        sigma_pt = max(1, n_bins_pt // 50)
        sigma_eta = max(1, n_bins_eta // 50)
        target_hist_2d = gaussian_filter(target_hist_2d, sigma=[sigma_pt, sigma_eta])
        target_hist_2d = np.maximum(target_hist_2d, 0)
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
    Reweight **every** subset so its (pT, eta) distribution matches a
    single common reference distribution.

    For each subset:
        weight_i = P(pT,eta | ref) / P(pT,eta | subset)

    Weights are clipped to [0.1, clip_max] and written in-place into
    each dataset's .weights tensor.

    Args:
        datasets_and_subsets: List of (StratifiedJetDataset, Subset) pairs.
        ref_pt: 1-D array of pT values defining the reference distribution.
        ref_eta: 1-D array of eta values defining the reference distribution.
        n_bins_pt: Number of pT bins for the 2-D histogram.
        n_bins_eta: Number of eta bins for the 2-D histogram.
        max_pt: Upper pT edge for binning.
        clip_max: Maximum weight after clipping.
        smooth_sigma: Gaussian smoothing sigma for ratio map (0 = no smoothing).
        verbose: Print per-subset weight statistics.
    """
    from scipy.ndimage import gaussian_filter

    # Fixed bin edges for all subsets
    pt_bins = np.linspace(0, max_pt, n_bins_pt + 1)
    eta_bins = np.linspace(-2.0, 2.0, n_bins_eta + 1)

    # Reference 2-D density histogram
    H_ref, _, _ = np.histogram2d(
        ref_pt, ref_eta, bins=[pt_bins, eta_bins], density=True
    )
    H_ref = np.maximum(H_ref, 1e-10)

    for dataset, subset in datasets_and_subsets:
        indices = np.array(subset.indices)
        jet_feats = _compute_jet_features(dataset, indices)
        pt = jet_feats["pt"]
        eta = jet_feats["eta"]

        # Source 2-D density
        H_src, _, _ = np.histogram2d(pt, eta, bins=[pt_bins, eta_bins], density=True)
        H_src = np.maximum(H_src, 1e-10)

        ratio_map = H_ref / H_src

        # Apply Gaussian smoothing to reduce spikiness in low-statistics regions
        if smooth_sigma > 0:
            ratio_map = gaussian_filter(ratio_map, sigma=smooth_sigma, mode="nearest")

        # Map weights to individual events
        pt_idx = np.clip(np.searchsorted(pt_bins, pt) - 1, 0, n_bins_pt - 1)
        eta_idx = np.clip(np.searchsorted(eta_bins, eta) - 1, 0, n_bins_eta - 1)
        weights = ratio_map[pt_idx, eta_idx].astype(np.float32)
        weights = np.clip(weights, 0.1, clip_max)

        # Write in-place
        dataset.weights[indices] = torch.from_numpy(weights).float().unsqueeze(1)

        if verbose:
            labels = dataset.labels[indices]
            for cls, name in [(1, "Signal"), (0, "Background")]:
                cls_w = weights[labels == cls]
                if len(cls_w) > 0:
                    print(
                        f"    {name}: mean={cls_w.mean():.3f}, "
                        f"std={cls_w.std():.3f}, max={cls_w.max():.3f}"
                    )


class CombinedJetDataLoader:
    """
    A DataLoader wrapper that handles PF and PUPPI datasets together,
    with five matching modes:

    Mode 1 (match_mode='size_only'):
        Match total events, preserve original class ratios in each dataset

    Mode 2 (match_mode='size_and_ratio'):
        Match total events AND class ratios while preserving feature distributions

    Mode 3 (match_mode='match_distributions'):
        Match total events, class ratios, AND per-class feature distributions
        for a single feature (e.g. pT). Weights set to 1.0.

    Mode 4 (match_mode='match_distributions_2d'):
        Match total events, class ratios, AND both pT and eta distributions
        simultaneously (2D matching). Weights set to 1.0.

    Mode 5 (match_mode='match_distributions_2d_reweight'):
        Like Mode 4 (2D distribution matching), then reweight all 4
        subsets to the combined signal distribution.

    Mode 6 (match_mode='reweight_to_pf_signal'):
        Like Mode 4 (2D distribution matching), then reweight all 4
        subsets (PF sig, PF bkg, PUPPI sig, PUPPI bkg) to the
        PF signal distribution specifically.
    """

    def __init__(
        self,
        pf_data_path: str,
        puppi_data_path: str,
        val_split: float = 0.3,
        batch_size: int = 512,
        match_mode: str = None,  # None, 'size_only', 'size_and_ratio', 'match_distributions', 'match_distributions_2d', 'match_distributions_2d_reweight', or 'reweight_to_pf_signal'
        num_workers: int = 0,  # Default to 0 for notebook compatibility
        random_state: int = 42,
        verbose: bool = True,
        target_class1_ratio: Optional[float] = None,
        match_feature: str = "pt",  # Feature to match distributions on (Mode 3)
        n_bins: int = 200,  # Number of bins for distribution matching (Mode 3)
        n_bins_pt: int = 50,  # pT bins for 2D matching (Mode 4/5)
        n_bins_eta: int = 50,  # eta bins for 2D matching (Mode 4/5)
        reweight_max_pt: float = 500.0,  # Max pT for weight computation (Mode 5)
        reweight_clip_max: float = 50.0,  # Max weight clip value (Mode 5)
    ):
        """
        Initialize the combined data loader.

        Args:
            pf_data_path: Path to PF dataset .npz file
            puppi_data_path: Path to PUPPI dataset .npz file
            val_split: Validation split fraction
            batch_size: Batch size for dataloaders
            match_mode:
                - None: No matching, use original sizes and ratios
                - 'size_only': Match sizes, keep original class ratios (Mode 1)
                - 'size_and_ratio': Match sizes AND class ratios (Mode 2)
                - 'match_distributions': Match sizes, class ratios, AND feature distributions (Mode 3)
                - 'match_distributions_2d': Match sizes, class ratios, AND pT+eta distributions (Mode 4)
                - 'match_distributions_2d_reweight': Mode 4 + reweight all to combined signal (Mode 5)
                - 'reweight_to_pf_signal': Mode 4 + reweight all to PF signal distribution (Mode 6)
            num_workers: Number of dataloader workers (use 0 in notebooks)
            random_state: Random seed for reproducibility
            verbose: Whether to print loading and matching statistics
            target_class1_ratio: Desired ratio of class 1 (signal) samples (0-1).
                If None, uses dataset1's class1 ratio.
            match_feature: Which jet feature to match ('pt' or 'eta') for Mode 3
            n_bins: Number of histogram bins for Mode 3 distribution matching
            n_bins_pt: Number of pT bins for Mode 4/5/6 (2D matching)
            n_bins_eta: Number of eta bins for Mode 4/5/6 (2D matching)
            reweight_max_pt: Maximum pT for weight binning (Mode 5/6)
            reweight_clip_max: Maximum clipping value for weights (Mode 5/6)
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self.match_mode = match_mode
        self.verbose = verbose
        self.target_class1_ratio = target_class1_ratio
        self.match_feature = match_feature
        self.n_bins = n_bins
        self.n_bins_pt = n_bins_pt
        self.n_bins_eta = n_bins_eta
        self.reweight_max_pt = reweight_max_pt
        self.reweight_clip_max = reweight_clip_max

        # Load datasets (pass mode so Mode 3/4/5 set weights to 1.0)
        print("\n" + "=" * 60)
        print("Loading PF Dataset")
        print("=" * 60)
        self.pf_dataset = StratifiedJetDataset(pf_data_path, mode=match_mode)

        print("\n" + "=" * 60)
        print("Loading PUPPI Dataset")
        print("=" * 60)
        self.puppi_dataset = StratifiedJetDataset(puppi_data_path, mode=match_mode)

        # Perform stratified splits
        print("\n" + "=" * 60)
        print("Performing Stratified Split - PF Dataset")
        print("=" * 60)
        (
            self.train_pf,
            self.val_pf,
            self.train_pf_indices,
            self.val_pf_indices,
            self.train_labels_pf,
            self.val_labels_pf,
        ) = stratified_train_val_split(
            self.pf_dataset, val_split, random_state, verbose=verbose
        )

        print("\n" + "=" * 60)
        print("Performing Stratified Split - PUPPI Dataset")
        print("=" * 60)
        (
            self.train_puppi,
            self.val_puppi,
            self.train_puppi_indices,
            self.val_puppi_indices,
            self.train_labels_puppi,
            self.val_labels_puppi,
        ) = stratified_train_val_split(
            self.puppi_dataset, val_split, random_state, verbose=verbose
        )

        # Apply matching based on mode
        if match_mode == "size_only":
            print("\n" + "=" * 60)
            print("MODE 1: Matching Sizes (Preserving Original Class Ratios)")
            print("=" * 60)
            self._match_sizes_preserve_ratios()

        elif match_mode == "size_and_ratio":
            print("\n" + "=" * 60)
            print("MODE 2: Matching Sizes AND Class Ratios")
            print("=" * 60)
            self._match_sizes_and_ratios()

        elif match_mode == "match_distributions":
            print("\n" + "=" * 60)
            print(
                f"MODE 3: Matching Sizes, Class Ratios, AND {match_feature} Distributions"
            )
            print("=" * 60)
            self._match_distributions()

        elif match_mode == "match_distributions_2d":
            print("\n" + "=" * 60)
            print("MODE 4: Matching Sizes, Class Ratios, AND pT+eta Distributions (2D)")
            print("=" * 60)
            self._match_distributions_2d()

        elif match_mode == "match_distributions_2d_reweight":
            print("\n" + "=" * 60)
            print(
                "MODE 5: Matching Distributions (2D) + Reweighting to Combined Signal"
            )
            print("=" * 60)
            self._match_distributions_2d_reweight()

        elif match_mode == "reweight_to_pf_signal":
            print("\n" + "=" * 60)
            print("MODE 6: Matching Distributions (2D) + Reweighting to PF Signal")
            print("=" * 60)
            self._reweight_to_pf_signal()

    def _match_sizes_preserve_ratios(self):
        """Mode 1: Match sizes while preserving original class ratios."""
        # Training sets
        if len(self.train_pf) > len(self.train_puppi):
            print("\nDownsampling PF training set to match PUPPI size...")
            self.train_pf, self.train_labels_pf = match_dataset_sizes_stratified(
                self.train_pf, self.train_puppi, self.pf_dataset, self.random_state
            )
        elif len(self.train_puppi) > len(self.train_pf):
            print("\nDownsampling PUPPI training set to match PF size...")
            self.train_puppi, self.train_labels_puppi = match_dataset_sizes_stratified(
                self.train_puppi, self.train_pf, self.puppi_dataset, self.random_state
            )

        # Validation sets
        if len(self.val_pf) > len(self.val_puppi):
            print("\nDownsampling PF validation set to match PUPPI size...")
            self.val_pf, self.val_labels_pf = match_dataset_sizes_stratified(
                self.val_pf, self.val_puppi, self.pf_dataset, self.random_state
            )
        elif len(self.val_puppi) > len(self.val_pf):
            print("\nDownsampling PUPPI validation set to match PF size...")
            self.val_puppi, self.val_labels_puppi = match_dataset_sizes_stratified(
                self.val_puppi, self.val_pf, self.puppi_dataset, self.random_state
            )

    def _match_sizes_and_ratios(self):
        """Mode 2: Match both sizes AND class ratios."""
        print("\nMatching training sets...")
        (
            self.train_puppi,
            self.train_pf,
            self.train_labels_puppi,
            self.train_labels_pf,
        ) = match_sizes_and_class_ratios(
            self.train_puppi,
            self.train_pf,
            self.puppi_dataset,
            self.pf_dataset,
            self.target_class1_ratio,  # Use dataset1's class ratio as target
            self.random_state,
            verbose=self.verbose,
        )

        print("\nMatching validation sets...")
        self.val_puppi, self.val_pf, self.val_labels_puppi, self.val_labels_pf = (
            match_sizes_and_class_ratios(
                self.val_puppi,
                self.val_pf,
                self.puppi_dataset,
                self.pf_dataset,
                self.target_class1_ratio,  # Use dataset1's class ratio as target
                self.random_state,
                verbose=self.verbose,
            )
        )

    def _match_distributions(self):
        """Mode 3: Match sizes, class ratios, AND per-class feature distributions."""
        print("\nMatching training sets...")
        (
            self.train_puppi,
            self.train_pf,
            self.train_labels_puppi,
            self.train_labels_pf,
        ) = match_sizes_ratios_and_distributions(
            self.train_puppi,
            self.train_pf,
            self.puppi_dataset,
            self.pf_dataset,
            target_class1_ratio=self.target_class1_ratio,
            match_feature=self.match_feature,
            n_bins=self.n_bins,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        print("\nMatching validation sets...")
        (
            self.val_puppi,
            self.val_pf,
            self.val_labels_puppi,
            self.val_labels_pf,
        ) = match_sizes_ratios_and_distributions(
            self.val_puppi,
            self.val_pf,
            self.puppi_dataset,
            self.pf_dataset,
            target_class1_ratio=self.target_class1_ratio,
            match_feature=self.match_feature,
            n_bins=self.n_bins,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def _match_distributions_2d(self):
        """Mode 4: Match sizes, class ratios, AND per-class pT+eta distributions (2D)."""
        print("\nMatching training sets...")
        (
            self.train_puppi,
            self.train_pf,
            self.train_labels_puppi,
            self.train_labels_pf,
        ) = match_sizes_ratios_and_distributions_2d(
            self.train_puppi,
            self.train_pf,
            self.puppi_dataset,
            self.pf_dataset,
            target_class1_ratio=self.target_class1_ratio,
            n_bins_pt=self.n_bins_pt,
            n_bins_eta=self.n_bins_eta,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        print("\nMatching validation sets...")
        (
            self.val_puppi,
            self.val_pf,
            self.val_labels_puppi,
            self.val_labels_pf,
        ) = match_sizes_ratios_and_distributions_2d(
            self.val_puppi,
            self.val_pf,
            self.puppi_dataset,
            self.pf_dataset,
            target_class1_ratio=self.target_class1_ratio,
            n_bins_pt=self.n_bins_pt,
            n_bins_eta=self.n_bins_eta,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def _match_distributions_2d_reweight(self):
        """Mode 5: Mode 4 (2D distribution matching) + reweight ALL 4
        subsets to a single common (pT, eta) reference distribution.

        The reference is built from the combined signal of both datasets
        (after the Mode 4 resampling step), so all four distributions
        (PF sig, PF bkg, PUPPI sig, PUPPI bkg) end up matching one another.
        """
        # Step 1: Do 2D distribution matching (same as Mode 4)
        self._match_distributions_2d()

        # Step 2: Build a single reference (pT, eta) from combined signal
        # of both datasets (after resampling)
        print("\nBuilding common reference distribution from combined signal...")
        pf_train_idx = np.array(self.train_pf.indices)
        puppi_train_idx = np.array(self.train_puppi.indices)

        pf_labels = self.pf_dataset.labels[pf_train_idx]
        puppi_labels = self.puppi_dataset.labels[puppi_train_idx]

        pf_sig_idx = pf_train_idx[pf_labels == 1]
        puppi_sig_idx = puppi_train_idx[puppi_labels == 1]

        pf_sig_feats = _compute_jet_features(self.pf_dataset, pf_sig_idx)
        puppi_sig_feats = _compute_jet_features(self.puppi_dataset, puppi_sig_idx)

        ref_pt = np.concatenate([pf_sig_feats["pt"], puppi_sig_feats["pt"]])
        ref_eta = np.concatenate([pf_sig_feats["eta"], puppi_sig_feats["eta"]])
        print(f"  Reference signal pool: {len(ref_pt)} jets (PF sig + PUPPI sig)")

        # Step 3: Reweight ALL subsets (train + val, both datasets) to the
        # common reference
        print("\nReweighting training subsets to common reference...")
        print("  PF training:")
        _reweight_to_common_target(
            [(self.pf_dataset, self.train_pf)],
            ref_pt,
            ref_eta,
            n_bins_pt=self.n_bins_pt,
            n_bins_eta=self.n_bins_eta,
            max_pt=self.reweight_max_pt,
            clip_max=self.reweight_clip_max,
            smooth_sigma=0.0,
            verbose=self.verbose,
        )
        print("  PUPPI training:")
        _reweight_to_common_target(
            [(self.puppi_dataset, self.train_puppi)],
            ref_pt,
            ref_eta,
            n_bins_pt=self.n_bins_pt,
            n_bins_eta=self.n_bins_eta,
            max_pt=self.reweight_max_pt,
            clip_max=self.reweight_clip_max,
            smooth_sigma=0.0,
            verbose=self.verbose,
        )

        # Validation sets — use the SAME reference
        # Build val reference from combined val signal
        pf_val_idx = np.array(self.val_pf.indices)
        puppi_val_idx = np.array(self.val_puppi.indices)
        pf_val_labels = self.pf_dataset.labels[pf_val_idx]
        puppi_val_labels = self.puppi_dataset.labels[puppi_val_idx]
        pf_val_sig_feats = _compute_jet_features(
            self.pf_dataset, pf_val_idx[pf_val_labels == 1]
        )
        puppi_val_sig_feats = _compute_jet_features(
            self.puppi_dataset, puppi_val_idx[puppi_val_labels == 1]
        )
        ref_pt_val = np.concatenate([pf_val_sig_feats["pt"], puppi_val_sig_feats["pt"]])
        ref_eta_val = np.concatenate(
            [pf_val_sig_feats["eta"], puppi_val_sig_feats["eta"]]
        )

        print("\nReweighting validation subsets to common reference...")
        print("  PF validation:")
        _reweight_to_common_target(
            [(self.pf_dataset, self.val_pf)],
            ref_pt_val,
            ref_eta_val,
            n_bins_pt=self.n_bins_pt,
            n_bins_eta=self.n_bins_eta,
            max_pt=self.reweight_max_pt,
            clip_max=self.reweight_clip_max,
            smooth_sigma=0.0,
            verbose=self.verbose,
        )
        print("  PUPPI validation:")
        _reweight_to_common_target(
            [(self.puppi_dataset, self.val_puppi)],
            ref_pt_val,
            ref_eta_val,
            n_bins_pt=self.n_bins_pt,
            n_bins_eta=self.n_bins_eta,
            max_pt=self.reweight_max_pt,
            clip_max=self.reweight_clip_max,
            smooth_sigma=0.0,
            verbose=self.verbose,
        )

    def _reweight_to_pf_signal(self):
        """Mode 6: Size/ratio matching (Mode 2) + reweight ALL 4 subsets
        to the PF signal (pT, eta) distribution in a single reweighting step.

        Unlike Mode 5 (which uses combined PF+PUPPI signal as reference),
        this uses PF signal only. Unlike the old Mode 6, this skips the
        Mode 4 acceptance-rejection step to avoid artifacts from double
        distribution manipulation.
        """
        # Step 1: Size and ratio matching only (no distribution resampling)
        self._match_sizes_and_ratios()

        # Step 2: Build reference from PF signal only
        print("\nBuilding reference distribution from PF signal...")
        pf_train_idx = np.array(self.train_pf.indices)
        pf_labels = self.pf_dataset.labels[pf_train_idx]
        pf_sig_idx = pf_train_idx[pf_labels == 1]
        pf_sig_feats = _compute_jet_features(self.pf_dataset, pf_sig_idx)
        ref_pt = pf_sig_feats["pt"]
        ref_eta = pf_sig_feats["eta"]
        print(f"  Reference pool: {len(ref_pt)} PF signal jets")

        # Step 3: Reweight ALL training subsets to PF signal reference (single step)
        print("\nReweighting training subsets to PF signal reference...")
        print("  PF training:")
        _reweight_to_common_target(
            [(self.pf_dataset, self.train_pf)],
            ref_pt,
            ref_eta,
            n_bins_pt=self.n_bins_pt,
            n_bins_eta=self.n_bins_eta,
            max_pt=self.reweight_max_pt,
            clip_max=self.reweight_clip_max,
            smooth_sigma=1.5,
            verbose=self.verbose,
        )
        print("  PUPPI training:")
        _reweight_to_common_target(
            [(self.puppi_dataset, self.train_puppi)],
            ref_pt,
            ref_eta,
            n_bins_pt=self.n_bins_pt,
            n_bins_eta=self.n_bins_eta,
            max_pt=self.reweight_max_pt,
            clip_max=self.reweight_clip_max,
            smooth_sigma=1.5,
            verbose=self.verbose,
        )

        # Validation sets — reference from PF val signal
        pf_val_idx = np.array(self.val_pf.indices)
        pf_val_labels = self.pf_dataset.labels[pf_val_idx]
        pf_val_sig_feats = _compute_jet_features(
            self.pf_dataset, pf_val_idx[pf_val_labels == 1]
        )
        ref_pt_val = pf_val_sig_feats["pt"]
        ref_eta_val = pf_val_sig_feats["eta"]

        print("\nReweighting validation subsets to PF signal reference...")
        print("  PF validation:")
        _reweight_to_common_target(
            [(self.pf_dataset, self.val_pf)],
            ref_pt_val,
            ref_eta_val,
            n_bins_pt=self.n_bins_pt,
            n_bins_eta=self.n_bins_eta,
            max_pt=self.reweight_max_pt,
            clip_max=self.reweight_clip_max,
            smooth_sigma=1.5,
            verbose=self.verbose,
        )
        print("  PUPPI validation:")
        _reweight_to_common_target(
            [(self.puppi_dataset, self.val_puppi)],
            ref_pt_val,
            ref_eta_val,
            n_bins_pt=self.n_bins_pt,
            n_bins_eta=self.n_bins_eta,
            max_pt=self.reweight_max_pt,
            clip_max=self.reweight_clip_max,
            smooth_sigma=1.5,
            verbose=self.verbose,
        )

    def get_train_loaders(self, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        Get training DataLoaders for both datasets.

        Returns:
            Tuple of (pf_train_loader, puppi_train_loader)
        """
        pf_loader = DataLoader(
            self.train_pf,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
        puppi_loader = DataLoader(
            self.train_puppi,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
        return pf_loader, puppi_loader

    def get_val_loaders(self, shuffle: bool = False) -> Tuple[DataLoader, DataLoader]:
        """
        Get validation DataLoaders for both datasets.

        Returns:
            Tuple of (pf_val_loader, puppi_val_loader)
        """
        pf_loader = DataLoader(
            self.val_pf,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
        puppi_loader = DataLoader(
            self.val_puppi,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
        return pf_loader, puppi_loader

    def get_pf_loaders(self, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        Get training and validation indices and DataLoaders for PF dataset.

        Returns:
            Tuple of (pf_train_loader, pf_val_loader)
        """
        pf_train_loader = DataLoader(
            self.train_pf,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
        pf_val_loader = DataLoader(
            self.val_pf,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
        return (
            pf_train_loader,
            self.train_pf_indices,
            pf_val_loader,
            self.val_pf_indices,
            self.train_labels_pf,
            self.val_labels_pf,
        )

    def get_puppi_loaders(self, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        Get training and validation DataLoaders and indices for PUPPI dataset.

        Returns:
            Tuple of (puppi_train_loader, puppi_val_loader)
        """
        puppi_train_loader = DataLoader(
            self.train_puppi,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
        puppi_val_loader = DataLoader(
            self.val_puppi,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
        return (
            puppi_train_loader,
            self.train_puppi_indices,
            puppi_val_loader,
            self.val_puppi_indices,
            self.train_labels_puppi,
            self.val_labels_puppi,
        )

    def summary(self):
        """Print a summary of dataset sizes and class distributions."""
        print("\n" + "=" * 60)
        print(f"DATASET SUMMARY (match_mode='{self.match_mode}')")
        print("=" * 60)

        for name, train_ds, val_ds, orig_ds in [
            ("PF", self.train_pf, self.val_pf, self.pf_dataset),
            ("PUPPI", self.train_puppi, self.val_puppi, self.puppi_dataset),
        ]:
            print(f"\n{name} Dataset:")
            print(f"  Original size: {len(orig_ds)}")
            print(f"  Training samples: {len(train_ds)}")
            print(f"  Validation samples: {len(val_ds)}")

            # Get class distribution for training set
            train_labels = orig_ds.labels[np.array(train_ds.indices)]
            for c in np.unique(train_labels):
                count = np.sum(train_labels == c)
                print(f"  Train Class {c}: {count} ({100*count/len(train_ds):.1f}%)")
