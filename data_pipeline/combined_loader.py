"""CombinedJetDataLoader — unified PF + PUPPI data pipeline."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional
import vector

from data_pipeline.datasets import (
    L1JetDataset,
    StratifiedJetDataset,
    WeakShuffleSampler,
    Subset,
    precollated_collate,
    _load_arrays,
)
from data_pipeline.splitting import stratified_train_val_split
from data_pipeline.matching import (
    match_dataset_sizes_stratified,
    match_sizes_and_class_ratios,
    match_sizes_ratios_and_distributions,
    match_sizes_ratios_and_distributions_2d,
    _compute_jet_features,
    _resample_with_target_hist_2d,
    _reweight_to_common_target,
    _reweight_bkg_to_signal,
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
        Mode 4 (2D distribution matching), then reweight each dataset's
        background to match its own matched signal distribution.
        Signal weights are kept at 1.0.

    Mode 7 (match_mode='match_pf_to_puppi_and_reweight')
        Mode 4 (2D distribution matching), then reweight each dataset's
        background to match its own matched signal distribution.
        Signal weights are kept at 1.0.

    Mode 8 (match_mode='reweight_to_puppi_bkg')
        Match sizes and class ratios (Mode 2), then reweight every
        per-class subset (PF sig, PF bkg, PUPPI sig) so its (pT, eta)
        count distribution matches the PUPPI background.  PUPPI
        background weights are kept at 1.0.

    Mode 9 (match_mode='reweight_to_puppi_bkg_relative')
        Mode 8 (reweight in original pT × eta), then replace the first
        4 constituent features (m, pT, eta, phi) with relative features
        (delta_m, rel_pT, delta_eta, delta_phi) computed w.r.t. the
        jet-level 4-vector.
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
        smooth_sigma: float = 1.0,  # Gaussian smoothing sigma for ratio maps (Mode 5/6
        pt_min: float = 0.0,  # Minimum pT for filtering
        pt_max: float = np.inf,  # Maximum pT for filtering
        eta_min: float = -np.inf,  # Minimum eta for filtering
        eta_max: float = np.inf,  # Maximum eta for filtering
        pt_regression: bool = False,  # Whether to add a regression target for pT (Mode 9)
        use_dataset: str = "both",  # 'pf', 'puppi', or 'both'
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
                - 'reweight_to_pf_signal': Mode 4 + reweight bkg→signal per dataset (Mode 6)
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
            smooth_sigma: Gaussian smoothing sigma for ratio maps (Mode 5/6)
            pt_min: Minimum pT for filtering jets before matching
            pt_max: Maximum pT for filtering jets before matching
            eta_min: Minimum eta for filtering jets before matching
            eta_max: Maximum eta for filtering jets before matching
            use_dataset: Which dataset(s) to load.  'pf', 'puppi', or 'both'
                (default 'both').  When match_mode is None and only one dataset
                is needed, pass 'pf' or 'puppi' to skip loading the other and
                save significant memory.
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
        self.smooth_sigma = smooth_sigma
        self.pt_min = pt_min
        self.pt_max = pt_max
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.use_dataset = use_dataset

        # When match_mode requires both datasets, override use_dataset
        needs_both = match_mode is not None
        load_pf = use_dataset in ("pf", "both") or needs_both
        load_puppi = use_dataset in ("puppi", "both") or needs_both

        # Load datasets (pass mode so Mode 3/4/5 set weights to 1.0)
        if load_pf:
            print("\n" + "=" * 60)
            print("Loading PF Dataset")
            print("=" * 60)
            self.pf_dataset = StratifiedJetDataset(
                pf_data_path, mode=match_mode, pt_regression=pt_regression
            )
        else:
            self.pf_dataset = None
            print("\nSkipping PF dataset (not needed for this configuration).")

        if load_puppi:
            print("\n" + "=" * 60)
            print("Loading PUPPI Dataset")
            print("=" * 60)
            self.puppi_dataset = StratifiedJetDataset(
                puppi_data_path, mode=match_mode, pt_regression=pt_regression
            )
        else:
            self.puppi_dataset = None
            print("\nSkipping PUPPI dataset (not needed for this configuration).")

        print(
            f"\nFiltering datasets to range: pT[{pt_min}, {pt_max}], |eta|[{eta_min}, {eta_max}]"
        )
        if self.pf_dataset is not None:
            self.pf_dataset.apply_kinematic_filter(pt_min, pt_max, eta_min, eta_max)
        if self.puppi_dataset is not None:
            self.puppi_dataset.apply_kinematic_filter(pt_min, pt_max, eta_min, eta_max)

        # Perform stratified splits
        if self.pf_dataset is not None:
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
        else:
            self.train_pf = self.val_pf = None
            self.train_pf_indices = self.val_pf_indices = np.array([], dtype=int)
            self.train_labels_pf = self.val_labels_pf = np.array([], dtype=int)

        if self.puppi_dataset is not None:
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
        else:
            self.train_puppi = self.val_puppi = None
            self.train_puppi_indices = self.val_puppi_indices = np.array([], dtype=int)
            self.train_labels_puppi = self.val_labels_puppi = np.array([], dtype=int)

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
            print("MODE 6: 2D Distribution Matching + Bkg→Signal Reweighting")
            print("=" * 60)
            self._reweight_to_pf_signal()

        elif match_mode == "match_pf_to_puppi_and_reweight":
            print("\n" + "=" * 60)
            print("MODE 7: Match PF Shape to PUPPI + Reweight Signal -> Background")
            print("=" * 60)

            # 1. Force PF to look like PUPPI
            # This modifies self.train_pf and self.val_pf to match self.train_puppi stats
            self._match_pf_to_puppi_shape()

            print("\nReweighting Signal to match Background kinematics...")
            self._reweight_signal_to_background(
                self.pf_dataset,
                self.train_pf,
                max_pt=self.reweight_max_pt,
                name="PF Train",
            )
            self._reweight_signal_to_background(
                self.puppi_dataset,
                self.train_puppi,
                max_pt=self.reweight_max_pt,
                name="PUPPI Train",
            )
            # Validation
            self._reweight_signal_to_background(
                self.pf_dataset, self.val_pf, max_pt=self.reweight_max_pt, name="PF Val"
            )
            self._reweight_signal_to_background(
                self.puppi_dataset,
                self.val_puppi,
                max_pt=self.reweight_max_pt,
                name="PUPPI Val",
            )
        elif match_mode == "reweight_to_puppi_bkg":
            print("\n" + "=" * 60)
            print("MODE 8: Match Sizes/Ratios + Reweight ALL to PUPPI Background")
            print("=" * 60)
            self._reweight_to_puppi_bkg()

        elif match_mode == "reweight_to_puppi_bkg_relative":
            print("\n" + "=" * 60)
            print("MODE 9: Mode 8 + Convert to Relative Features")
            print("=" * 60)
            self._reweight_to_puppi_bkg()
            self._convert_to_relative_features(self.pf_dataset)
            self._convert_to_relative_features(self.puppi_dataset)

        # Sync stored indices with the (possibly updated) Subset objects
        # so that get_pf_loaders / get_puppi_loaders return correct indices.
        if self.train_pf is not None:
            self.train_pf_indices = np.array(self.train_pf.indices)
            self.val_pf_indices = np.array(self.val_pf.indices)
        if self.train_puppi is not None:
            self.train_puppi_indices = np.array(self.train_puppi.indices)
            self.val_puppi_indices = np.array(self.val_puppi.indices)

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
        """Mode 6: 2D distribution matching (Mode 4) + reweight background
        to match signal within each dataset.

        Step 1: Mode 4 — match sizes, class ratios, and per-class (pT, eta)
                distributions across PF and PUPPI via acceptance-rejection.
                After this step signal↔signal and bkg↔bkg match across
                the two detector representations.
        Step 2: Reweight each dataset's background so that its (pT, eta)
                distribution matches the signal in the same dataset.
                Signal weights are kept at 1.0.
        """
        # Step 1: 2D distribution matching (same as Mode 4)
        self._match_distributions_2d()

        # Step 2: Per-dataset background → signal reweighting
        rw_kwargs = dict(
            n_bins_pt=self.n_bins_pt,
            n_bins_eta=self.n_bins_eta,
            max_pt=self.reweight_max_pt,
            clip_max=self.reweight_clip_max,
            smooth_sigma=self.smooth_sigma,
            verbose=self.verbose,
        )

        print("\nReweighting training backgrounds to their matched signals...")
        print("  PF training:")
        _reweight_bkg_to_signal(self.pf_dataset, self.train_pf, **rw_kwargs)
        print("  PUPPI training:")
        _reweight_bkg_to_signal(self.puppi_dataset, self.train_puppi, **rw_kwargs)

        print("\nReweighting validation backgrounds to their matched signals...")
        print("  PF validation:")
        _reweight_bkg_to_signal(self.pf_dataset, self.val_pf, **rw_kwargs)
        print("  PUPPI validation:")
        _reweight_bkg_to_signal(self.puppi_dataset, self.val_puppi, **rw_kwargs)

    def _match_pf_to_puppi_shape(self):
        """
        Matches the PF dataset to the PUPPI dataset's shape and size.
        PUPPI remains unchanged (Reference).
        PF is resampled to match PUPPI's (pT, eta) distribution per class.
        """
        print("\nMatching PF shape/size to PUPPI reference...")

        # We will update these
        new_train_pf_indices = []
        new_val_pf_indices = []

        # --- Training Sets ---
        # Get PUPPI features (Reference)
        puppi_train_indices = np.array(self.train_puppi.indices)
        puppi_train_labels = self.puppi_dataset.labels[puppi_train_indices]
        puppi_train_feats = _compute_jet_features(
            self.puppi_dataset, puppi_train_indices
        )

        # Get PF features (Source)
        pf_train_indices = np.array(self.train_pf.indices)
        pf_train_labels = self.pf_dataset.labels[pf_train_indices]
        pf_train_feats = _compute_jet_features(self.pf_dataset, pf_train_indices)

        for cls in [0, 1]:
            # Reference (PUPPI)
            cls_mask_ref = puppi_train_labels == cls
            ref_pt = puppi_train_feats["pt"][cls_mask_ref]
            ref_eta = puppi_train_feats["eta"][cls_mask_ref]
            target_n = len(ref_pt)  # Match PUPPI size

            # Source (PF)
            cls_mask_src = pf_train_labels == cls
            src_indices = pf_train_indices[cls_mask_src]
            src_pt = pf_train_feats["pt"][cls_mask_src]
            src_eta = pf_train_feats["eta"][cls_mask_src]

            if target_n == 0 or len(src_indices) == 0:
                continue

            # Build Target Histogram from PUPPI
            # Use dynamic binning or fixed binning
            pt_min = min(ref_pt.min(), src_pt.min())
            pt_max = max(ref_pt.max(), src_pt.max())
            eta_min = min(ref_eta.min(), src_eta.min())
            eta_max = max(ref_eta.max(), src_eta.max())

            bin_edges_pt = np.linspace(pt_min, pt_max, self.n_bins_pt + 1)
            bin_edges_eta = np.linspace(eta_min, eta_max, self.n_bins_eta + 1)

            # 1. Compute PUPPI Density (Target)
            target_hist_2d, _, _ = np.histogram2d(
                ref_pt, ref_eta, bins=[bin_edges_pt, bin_edges_eta], density=True
            )

            # # Smooth PUPPI target slightly to avoid jagged spikes
            # sigma = max(1, self.n_bins_pt // 50)
            # target_hist_2d = gaussian_filter(target_hist_2d, sigma=sigma)
            # target_hist_2d = np.maximum(target_hist_2d, 1e-12)  # Numerical stability

            # Normalize
            bin_areas = np.outer(np.diff(bin_edges_pt), np.diff(bin_edges_eta))
            target_hist_2d /= np.sum(target_hist_2d * bin_areas)

            # 2. Resample PF to match this Target
            # We reuse the existing helper _resample_with_target_hist_2d
            selected_indices = _resample_with_target_hist_2d(
                src_indices,
                src_pt,
                src_eta,
                target_hist_2d,
                bin_edges_pt,
                bin_edges_eta,
                target_n=target_n,
                random_state=self.random_state + cls,
            )
            new_train_pf_indices.append(selected_indices)

            print(
                f"  Train Class {cls}: PF resampled {len(src_indices)} -> {len(selected_indices)} (matched to PUPPI)"
            )

        # --- Validation Sets ---
        # Repeat logic for validation to ensure consistent evaluation
        puppi_val_indices = np.array(self.val_puppi.indices)
        puppi_val_labels = self.puppi_dataset.labels[puppi_val_indices]
        puppi_val_feats = _compute_jet_features(self.puppi_dataset, puppi_val_indices)

        pf_val_indices = np.array(self.val_pf.indices)
        pf_val_labels = self.pf_dataset.labels[pf_val_indices]
        pf_val_feats = _compute_jet_features(self.pf_dataset, pf_val_indices)

        for cls in [0, 1]:
            # Reference
            cls_mask_ref = puppi_val_labels == cls
            ref_pt = puppi_val_feats["pt"][cls_mask_ref]
            ref_eta = puppi_val_feats["eta"][cls_mask_ref]
            target_n = len(ref_pt)

            # Source
            cls_mask_src = pf_val_labels == cls
            src_indices = pf_val_indices[cls_mask_src]
            src_pt = pf_val_feats["pt"][cls_mask_src]
            src_eta = pf_val_feats["eta"][cls_mask_src]

            if target_n == 0 or len(src_indices) == 0:
                continue

            pt_min = min(ref_pt.min(), src_pt.min())
            pt_max = max(ref_pt.max(), src_pt.max())
            eta_min = min(ref_eta.min(), src_eta.min())
            eta_max = max(ref_eta.max(), src_eta.max())
            bin_edges_pt = np.linspace(pt_min, pt_max, self.n_bins_pt + 1)
            bin_edges_eta = np.linspace(eta_min, eta_max, self.n_bins_eta + 1)

            target_hist_2d, _, _ = np.histogram2d(
                ref_pt, ref_eta, bins=[bin_edges_pt, bin_edges_eta], density=True
            )
            target_hist_2d = gaussian_filter(
                target_hist_2d, sigma=max(1, self.n_bins_pt // 50)
            )
            target_hist_2d = np.maximum(target_hist_2d, 1e-12)
            bin_areas = np.outer(np.diff(bin_edges_pt), np.diff(bin_edges_eta))
            target_hist_2d /= np.sum(target_hist_2d * bin_areas)

            selected_indices = _resample_with_target_hist_2d(
                src_indices,
                src_pt,
                src_eta,
                target_hist_2d,
                bin_edges_pt,
                bin_edges_eta,
                target_n=target_n,
                random_state=self.random_state + cls + 100,
            )
            new_val_pf_indices.append(selected_indices)

        # Apply updates
        all_train_pf = np.concatenate(new_train_pf_indices)
        all_val_pf = np.concatenate(new_val_pf_indices)
        np.random.shuffle(all_train_pf)
        np.random.shuffle(all_val_pf)

        self.train_pf = Subset(self.pf_dataset, all_train_pf.tolist())
        self.val_pf = Subset(self.pf_dataset, all_val_pf.tolist())

        # Update labels for consistency
        self.train_labels_pf = self.pf_dataset.labels[all_train_pf]
        self.val_labels_pf = self.pf_dataset.labels[all_val_pf]

    def _reweight_signal_to_background(
        self,
        dataset: StratifiedJetDataset,
        subset: Subset,
        n_bins_pt: int = 200,
        n_bins_eta: int = 200,
        max_pt: float = 500.0,
        clip_max: float = 50.0,
        smooth_sigma: float = 1.0,
        name: str = "Dataset",
    ):
        """
        Calculates weights w = P(Bkg)/P(Sig) for Signal events, forcing Signal
        kinematics to look like Background kinematics.
        Background weights are fixed at 1.0.
        """

        indices = np.array(subset.indices)
        labels = dataset.labels[indices]

        sig_mask = labels == 1
        bkg_mask = labels == 0

        if np.sum(sig_mask) == 0 or np.sum(bkg_mask) == 0:
            return

        jet_feats = _compute_jet_features(dataset, indices)
        pt = jet_feats["pt"]
        eta = jet_feats["eta"]

        pt_sig, eta_sig = pt[sig_mask], eta[sig_mask]
        pt_bkg, eta_bkg = pt[bkg_mask], eta[bkg_mask]

        pt_bins = np.linspace(0, max_pt, n_bins_pt + 1)
        eta_bins = np.linspace(-2.0, 2.0, n_bins_eta + 1)

        H_sig, _, _ = np.histogram2d(
            pt_sig, eta_sig, bins=[pt_bins, eta_bins], density=True
        )
        H_bkg, _, _ = np.histogram2d(
            pt_bkg, eta_bkg, bins=[pt_bins, eta_bins], density=True
        )

        H_sig = np.maximum(H_sig, 1e-10)
        ratio_map = H_bkg / H_sig

        if smooth_sigma > 0:
            ratio_map = gaussian_filter(ratio_map, sigma=smooth_sigma)

        pt_idx = np.clip(np.searchsorted(pt_bins, pt_sig) - 1, 0, n_bins_pt - 1)
        eta_idx = np.clip(np.searchsorted(eta_bins, eta_sig) - 1, 0, n_bins_eta - 1)

        sig_weights = ratio_map[pt_idx, eta_idx]
        sig_weights = np.clip(sig_weights, 0.1, clip_max)

        # Update weights in the dataset tensor
        new_weights = torch.ones(len(indices), 1, dtype=torch.float32)
        new_weights[torch.where(torch.from_numpy(sig_mask))[0]] = (
            torch.from_numpy(sig_weights).float().unsqueeze(1)
        )
        dataset.weights[indices] = new_weights

        if self.verbose:
            print(f"  {name} Sig->Bkg Reweighting: Mean Sig W={sig_weights.mean():.3f}")

    def _reweight_to_puppi_bkg(self):
        """
        MODE 8: Match sizes and class ratios (Mode 2), then reweight
        every per-class subset so its (pT, eta) count distribution
        matches the PUPPI background.  PUPPI background weights stay
        at 1.0.

        Steps:
            1. Match sizes and class ratios between PF and PUPPI
               (same as Mode 2).
            2. Build PUPPI-background count histogram as reference.
            3. For PF signal, PF background, and PUPPI signal: compute
               ratio = ref_counts / subset_counts per (eta, pT) bin and
               write per-event weights in-place.
        """
        verbose = self.verbose

        # --- Step 1: match sizes and class ratios (Mode 2) ---
        print("\nStep 1: Matching sizes and class ratios (Mode 2)...")
        self._match_sizes_and_ratios()

        # --- Step 2 & 3: reweight to PUPPI background per split ---
        for split_name, puppi_subset, pf_subset in [
            ("Training", self.train_puppi, self.train_pf),
            ("Validation", self.val_puppi, self.val_pf),
        ]:
            print(f"\nReweighting {split_name} subsets to PUPPI background...")
            self._reweight_split_to_puppi_bkg(
                self.puppi_dataset,
                puppi_subset,
                self.pf_dataset,
                pf_subset,
                verbose=verbose,
            )

    def _reweight_split_to_puppi_bkg(
        self,
        puppi_dataset: "StratifiedJetDataset",
        puppi_subset: Subset,
        pf_dataset: "StratifiedJetDataset",
        pf_subset: Subset,
        verbose: bool = True,
    ) -> None:
        """
        For one train/val split: build the PUPPI-background count
        histogram, then compute per-event weights for PF signal,
        PF background, and PUPPI signal so they each match that
        reference.  PUPPI background gets weight = 1.0.
        """
        n_bins_pt = self.n_bins_pt
        n_bins_eta = self.n_bins_eta
        smooth_sigma = self.smooth_sigma

        # --- Indices and labels ---
        puppi_idx = np.array(puppi_subset.indices)
        pf_idx = np.array(pf_subset.indices)
        puppi_labels = puppi_dataset.labels[puppi_idx]
        pf_labels = pf_dataset.labels[pf_idx]

        puppi_bkg_idx = puppi_idx[puppi_labels == 0]
        puppi_sig_idx = puppi_idx[puppi_labels == 1]
        pf_bkg_idx = pf_idx[pf_labels == 0]
        pf_sig_idx = pf_idx[pf_labels == 1]

        # --- Jet features ---
        puppi_bkg_feat = _compute_jet_features(puppi_dataset, puppi_bkg_idx)
        puppi_sig_feat = _compute_jet_features(puppi_dataset, puppi_sig_idx)
        pf_bkg_feat = _compute_jet_features(pf_dataset, pf_bkg_idx)
        pf_sig_feat = _compute_jet_features(pf_dataset, pf_sig_idx)

        # --- Bin edges: use filtering range if finite, else derive from data ---
        all_pt = np.concatenate(
            [
                puppi_bkg_feat["pt"],
                puppi_sig_feat["pt"],
                pf_bkg_feat["pt"],
                pf_sig_feat["pt"],
            ]
        )
        all_eta = np.concatenate(
            [
                puppi_bkg_feat["eta"],
                puppi_sig_feat["eta"],
                pf_bkg_feat["eta"],
                pf_sig_feat["eta"],
            ]
        )
        pt_lo = self.pt_min if np.isfinite(self.pt_min) else float(all_pt.min())
        pt_hi = self.pt_max if np.isfinite(self.pt_max) else float(all_pt.max())
        eta_lo = self.eta_min if np.isfinite(self.eta_min) else float(all_eta.min())
        eta_hi = self.eta_max if np.isfinite(self.eta_max) else float(all_eta.max())

        pt_bins = np.linspace(pt_lo, pt_hi, n_bins_pt + 1)
        eta_bins = np.linspace(eta_lo, eta_hi, n_bins_eta + 1)

        # --- Reference: PUPPI background count histogram (eta × pT) ---
        ref_hist, _, _ = np.histogram2d(
            puppi_bkg_feat["eta"],
            puppi_bkg_feat["pt"],
            bins=(eta_bins, pt_bins),
            density=False,
        )

        # --- Helper: build per-class histograms and compute weights ---
        def _per_class_weights(feat: dict, ref: np.ndarray) -> np.ndarray:
            """Return per-event weight array = ref_counts / class_counts."""
            cls_hist, _, _ = np.histogram2d(
                feat["eta"],
                feat["pt"],
                bins=(eta_bins, pt_bins),
                density=False,
            )
            ratio = np.divide(
                ref,
                cls_hist,
                out=np.zeros_like(cls_hist, dtype=np.float64),
                where=cls_hist > 0,
            )
            if smooth_sigma > 0:
                ratio = gaussian_filter(ratio, sigma=smooth_sigma)
            eta_bin_idx = np.clip(
                np.digitize(feat["eta"], eta_bins) - 1, 0, n_bins_eta - 1
            )
            pt_bin_idx = np.clip(np.digitize(feat["pt"], pt_bins) - 1, 0, n_bins_pt - 1)
            return ratio[eta_bin_idx, pt_bin_idx].astype(np.float32)

        # --- Compute weights for three subsets ---
        puppi_sig_w = _per_class_weights(puppi_sig_feat, ref_hist)
        pf_sig_w = _per_class_weights(pf_sig_feat, ref_hist)
        pf_bkg_w = _per_class_weights(pf_bkg_feat, ref_hist)
        puppi_bkg_w = np.ones(len(puppi_bkg_idx), dtype=np.float32)

        # --- Write weights in-place ---
        puppi_dataset.weights[puppi_bkg_idx] = (
            torch.from_numpy(puppi_bkg_w).float().unsqueeze(1)
        )
        puppi_dataset.weights[puppi_sig_idx] = (
            torch.from_numpy(puppi_sig_w).float().unsqueeze(1)
        )
        pf_dataset.weights[pf_bkg_idx] = torch.from_numpy(pf_bkg_w).float().unsqueeze(1)
        pf_dataset.weights[pf_sig_idx] = torch.from_numpy(pf_sig_w).float().unsqueeze(1)

        if verbose:
            print(f"  PUPPI Bkg: {len(puppi_bkg_idx)} jets, weight=1.0 (reference)")
            print(
                f"  PUPPI Sig: {len(puppi_sig_idx)} jets, "
                f"mean_w={puppi_sig_w.mean():.3f}, max_w={puppi_sig_w.max():.3f}"
            )
            print(
                f"  PF Bkg:    {len(pf_bkg_idx)} jets, "
                f"mean_w={pf_bkg_w.mean():.3f}, max_w={pf_bkg_w.max():.3f}"
            )
            print(
                f"  PF Sig:    {len(pf_sig_idx)} jets, "
                f"mean_w={pf_sig_w.mean():.3f}, max_w={pf_sig_w.max():.3f}"
            )

    @staticmethod
    def _convert_to_relative_features(dataset: "StratifiedJetDataset") -> None:
        """
        Replace the first 4 constituent features (m, pT, eta, phi) with
        relative features (delta_m, rel_pT, delta_eta, delta_phi)
        computed w.r.t. the jet-level 4-vector.  Modifies dataset.X
        in-place.

        Feature mapping:
            0: m       → delta_m   = m_const  - m_jet
            1: pT      → rel_pT    = pT_const / pT_jet
            2: eta     → delta_eta = eta_const - eta_jet
            3: phi     → delta_phi = phi_const - phi_jet  (wrapped to [-π, π])
        """
        # Ensure X/mask are in memory (no-op for NPZ-loaded datasets)
        if dataset._lazy or dataset._mmap:
            dataset._materialize()

        X = dataset.X  # (N, n_const, n_feat)
        mask = dataset.mask.numpy()  # (N, n_const)

        c_m = X[:, :, 0].numpy()
        c_pt = X[:, :, 1].numpy()
        c_eta = X[:, :, 2].numpy()
        c_phi = X[:, :, 3].numpy()

        # Zero out masked constituents for the jet-level sum
        c_m_z = np.where(mask, c_m, 0)
        c_pt_z = np.where(mask, c_pt, 0)
        c_eta_z = np.where(mask, c_eta, 0)
        c_phi_z = np.where(mask, c_phi, 0)

        v = vector.array(
            {
                "pt": c_pt_z,
                "eta": c_eta_z,
                "phi": c_phi_z,
                "mass": c_m_z,
            }
        )
        j = v.sum(axis=1)  # jet-level 4-vector

        jet_m = np.array(j.mass)[:, None]  # (N, 1)
        jet_pt = np.array(j.pt)[:, None]
        jet_eta = np.array(j.eta)[:, None]
        jet_phi = np.array(j.phi)[:, None]

        delta_m = c_m - jet_m
        rel_pt = np.where(jet_pt > 0, c_pt / jet_pt, 0.0)
        delta_eta = c_eta - jet_eta
        delta_phi = np.arctan2(np.sin(c_phi - jet_phi), np.cos(c_phi - jet_phi))

        # Write back in-place
        X[:, :, 0] = torch.from_numpy(delta_m).float()
        X[:, :, 1] = torch.from_numpy(rel_pt).float()
        X[:, :, 2] = torch.from_numpy(delta_eta).float()
        X[:, :, 3] = torch.from_numpy(delta_phi).float()

        n_jets = len(X)
        print(
            f"  Converted {n_jets} jets to relative features "
            f"(delta_m, rel_pT, delta_eta, delta_phi)"
        )

    @staticmethod
    def _needs_weak_shuffle(subset) -> bool:
        """Return True if the underlying dataset uses lazy HDF5 reads.

        WeakShuffleSampler is only beneficial for HDF5 (chunk-aware access).
        For mmap .npy or eager in-memory data, standard PyTorch shuffle is
        faster and equally effective.
        """
        ds = subset.dataset if isinstance(subset, Subset) else subset
        return getattr(ds, "_lazy", False)

    def _make_sampler_and_shuffle(self, dataset, shuffle: bool):
        """Return (sampler, shuffle_flag) for a DataLoader.

        - HDF5 lazy mode  → WeakShuffleSampler, shuffle=False
        - mmap / eager     → sampler=None,       shuffle=<requested>
        """
        if shuffle and self._needs_weak_shuffle(dataset):
            print("Loaded WeakShuffleSampler for shuffling.")
            return WeakShuffleSampler(dataset, chunk_size=self.batch_size), False
        print("Using default PyTorch sampler for shuffling.")
        return None, shuffle

    def get_train_loaders(self, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        Get training DataLoaders for both datasets.
        """
        pf_sampler, pf_shuffle = self._make_sampler_and_shuffle(self.train_pf, shuffle)
        puppi_sampler, puppi_shuffle = self._make_sampler_and_shuffle(self.train_puppi, shuffle)

        pf_loader = DataLoader(
            self.train_pf,
            batch_size=self.batch_size,
            shuffle=pf_shuffle,
            sampler=pf_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=5 if self.num_workers > 0 else None,
            collate_fn=precollated_collate,
        )
        puppi_loader = DataLoader(
            self.train_puppi,
            batch_size=self.batch_size,
            shuffle=puppi_shuffle,
            sampler=puppi_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=5 if self.num_workers > 0 else None,
            collate_fn=precollated_collate,
        )
        return pf_loader, puppi_loader

    def get_val_loaders(self, shuffle: bool = False) -> Tuple[DataLoader, DataLoader]:
        """
        Get validation DataLoaders for both datasets.
        """
        # Validation usually doesn't shuffle, but we support it just in case
        pf_sampler, pf_shuffle = self._make_sampler_and_shuffle(self.val_pf, shuffle)
        puppi_sampler, puppi_shuffle = self._make_sampler_and_shuffle(self.val_puppi, shuffle)

        pf_loader = DataLoader(
            self.val_pf,
            batch_size=self.batch_size,
            shuffle=pf_shuffle,
            sampler=pf_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=5 if self.num_workers > 0 else None,
            collate_fn=precollated_collate,
        )
        puppi_loader = DataLoader(
            self.val_puppi,
            batch_size=self.batch_size,
            shuffle=puppi_shuffle,
            sampler=puppi_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=5 if self.num_workers > 0 else None,
            collate_fn=precollated_collate,
        )
        return pf_loader, puppi_loader

    def get_pf_loaders(self, shuffle: bool = True):
        """
        Get training and validation indices and DataLoaders for PF dataset.
        """
        train_sampler, train_shuffle = self._make_sampler_and_shuffle(self.train_pf, shuffle)

        pf_train_loader = DataLoader(
            self.train_pf,
            batch_size=self.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=5 if self.num_workers > 0 else None,
            collate_fn=precollated_collate,
        )
        pf_val_loader = DataLoader(
            self.val_pf,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=5 if self.num_workers > 0 else None,
            collate_fn=precollated_collate,
        )
        return (
            pf_train_loader,
            self.train_pf_indices,
            pf_val_loader,
            self.val_pf_indices,
            self.train_labels_pf,
            self.val_labels_pf,
        )

    def get_puppi_loaders(self, shuffle: bool = True):
        """
        Get training and validation DataLoaders and indices for PUPPI dataset.
        """
        train_sampler, train_shuffle = self._make_sampler_and_shuffle(self.train_puppi, shuffle)

        puppi_train_loader = DataLoader(
            self.train_puppi,
            batch_size=self.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=5 if self.num_workers > 0 else None,
            collate_fn=precollated_collate,
        )
        puppi_val_loader = DataLoader(
            self.val_puppi,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=5 if self.num_workers > 0 else None,
            collate_fn=precollated_collate,
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
