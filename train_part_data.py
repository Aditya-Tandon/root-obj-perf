import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List, Dict


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

    def __init__(self, filepath: str):
        """
        Load dataset from .npz file.

        Args:
            filepath: Path to the .npz file containing x, y, mask, weights
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
        Downsampled Subset with preserved class distribution
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

    if verbose:
        selected_labels = original_dataset_large.labels[selected_indices]
        print(
            f"\nDownsampled dataset from {len(dataset_large)} to {len(subsampled_ds)} samples"
        )
        for c in np.unique(selected_labels):
            count = np.sum(selected_labels == c)
            print(f"  Class {c}: {count} ({100*count/len(subsampled_ds):.1f}%)")

    return subsampled_ds


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
        Tuple of (resampled_dataset1, resampled_dataset2)
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

    if verbose:
        # Verify final distributions
        final_labels1 = original_dataset1.labels[np.array(resampled1.indices)]
        final_labels2 = original_dataset2.labels[np.array(resampled2.indices)]
        print(
            f"\n  Final Dataset1: {len(resampled1)} samples, "
            f"Class0={np.sum(final_labels1==0)}, Class1={np.sum(final_labels1==1)}"
        )
        print(
            f"  Final Dataset2: {len(resampled2)} samples, "
            f"Class0={np.sum(final_labels2==0)}, Class1={np.sum(final_labels2==1)}"
        )

    return resampled1, resampled2


class CombinedJetDataLoader:
    """
    A DataLoader wrapper that handles PF and PUPPI datasets together,
    with two matching modes:

    Mode 1 (match_mode='size_only'):
        Match total events, preserve original class ratios in each dataset

    Mode 2 (match_mode='size_and_ratio'):
        Match total events AND class ratios while preserving feature distributions
    """

    def __init__(
        self,
        pf_data_path: str,
        puppi_data_path: str,
        val_split: float = 0.3,
        batch_size: int = 512,
        match_mode: str = None,  # None, 'size_only', or 'size_and_ratio'
        num_workers: int = 0,  # Default to 0 for notebook compatibility
        random_state: int = 42,
        verbose: bool = True,
        target_class1_ratio: Optional[float] = None,
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
            num_workers: Number of dataloader workers (use 0 in notebooks)
            random_state: Random seed for reproducibility
            verbose: Whether to print loading and matching statistics
            target_class1_ratio: Desired ratio of class 1 (signal) samples (0-1) for Mode 2. If None, use the PUPPI dataset's (dataset1) ratio.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self.match_mode = match_mode
        self.verbose = verbose
        self.target_class1_ratio = target_class1_ratio

        # Load datasets
        print("\n" + "=" * 60)
        print("Loading PF Dataset")
        print("=" * 60)
        self.pf_dataset = StratifiedJetDataset(pf_data_path)

        print("\n" + "=" * 60)
        print("Loading PUPPI Dataset")
        print("=" * 60)
        self.puppi_dataset = StratifiedJetDataset(puppi_data_path)

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

    def _match_sizes_preserve_ratios(self):
        """Mode 1: Match sizes while preserving original class ratios."""
        # Training sets
        if len(self.train_pf) > len(self.train_puppi):
            print("\nDownsampling PF training set to match PUPPI size...")
            self.train_pf = match_dataset_sizes_stratified(
                self.train_pf, self.train_puppi, self.pf_dataset, self.random_state
            )
        elif len(self.train_puppi) > len(self.train_pf):
            print("\nDownsampling PUPPI training set to match PF size...")
            self.train_puppi = match_dataset_sizes_stratified(
                self.train_puppi, self.train_pf, self.puppi_dataset, self.random_state
            )

        # Validation sets
        if len(self.val_pf) > len(self.val_puppi):
            print("\nDownsampling PF validation set to match PUPPI size...")
            self.val_pf = match_dataset_sizes_stratified(
                self.val_pf, self.val_puppi, self.pf_dataset, self.random_state
            )
        elif len(self.val_puppi) > len(self.val_pf):
            print("\nDownsampling PUPPI validation set to match PF size...")
            self.val_puppi = match_dataset_sizes_stratified(
                self.val_puppi, self.val_pf, self.puppi_dataset, self.random_state
            )

    def _match_sizes_and_ratios(self):
        """Mode 2: Match both sizes AND class ratios."""
        print("\nMatching training sets...")
        self.train_puppi, self.train_pf = match_sizes_and_class_ratios(
            self.train_puppi,
            self.train_pf,
            self.puppi_dataset,
            self.pf_dataset,
            self.target_class1_ratio,  # Use dataset1's class ratio as target
            self.random_state,
            verbose=self.verbose,
        )

        print("\nMatching validation sets...")
        self.val_puppi, self.val_pf = match_sizes_and_class_ratios(
            self.val_puppi,
            self.val_pf,
            self.puppi_dataset,
            self.pf_dataset,
            self.target_class1_ratio,  # Use dataset1's class ratio as target
            self.random_state,
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
