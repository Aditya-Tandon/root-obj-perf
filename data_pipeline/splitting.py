"""Stratified train/validation splitting utilities."""

import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

from data_pipeline.datasets import Subset, StratifiedJetDataset


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
