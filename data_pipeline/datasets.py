"""Dataset classes, collation helpers, and format-conversion utilities."""

import argparse
import os
import numpy as np
import torch
import torch.utils.data as torch_data
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Tuple, Optional, List, Dict
import h5py


# ── collation ────────────────────────────────────────────────────────

def precollated_collate(batch):
    """Collate function for use with ``StratifiedJetDataset.__getitems__``.

    ``__getitems__`` already returns a single pre-collated tuple wrapped
    in a length-1 list.  This collate simply unwraps it, avoiding the
    1024-tensor IPC + ``torch.stack`` overhead of ``default_collate``.
    """
    return batch[0]


# ── custom Subset ────────────────────────────────────────────────────

class Subset(torch_data.Subset):
    """Custom Subset that allows DataLoader to request a whole batch at once."""

    def __getitems__(self, indices):
        mapped_indices = [self.indices[i] for i in indices]
        # If the underlying dataset supports bulk fetching, use it!
        if hasattr(self.dataset, "__getitems__"):
            return self.dataset.__getitems__(mapped_indices)
        return [self.dataset[i] for i in mapped_indices]


# ── array I/O helpers ────────────────────────────────────────────────

def _load_arrays(filepath: str, skip_keys: set | None = None) -> dict:
    """Load arrays from an .npz or .h5/.hdf5 file into a dict of numpy arrays.

    Parameters
    ----------
    filepath : str
        Path to the data file.
    skip_keys : set or None
        Keys to skip (not loaded into memory).
    """
    skip = skip_keys or set()
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".h5", ".hdf5"):
        arrays = {}
        with h5py.File(filepath, "r") as hf:
            for key in hf.keys():
                if key not in skip:
                    arrays[key] = hf[key][:]  # read into memory
        return arrays
    else:
        data = np.load(filepath)
        return {k: data[k] for k in data.files if k not in skip}


def h5_to_npy(h5_path: str, out_dir: str | None = None) -> str:
    """Convert an HDF5 dataset to memory-mappable ``.npy`` files.

    Creates the following files in *out_dir* (defaults to the same directory
    as *h5_path*):

    * ``<stem>_x.npy``   – shape (N, C, F), float32
    * ``<stem>_mask.npy`` – shape (N, C), bool  (if present)
    * ``<stem>_meta.npz`` – all remaining 1-D arrays (y, weights, …)

    Returns the *out_dir* path so the caller knows where to look.
    """
    import time as _time

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(h5_path))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(h5_path))[0]

    print(f"Converting {h5_path} → .npy files in {out_dir}/")
    t0 = _time.time()
    meta = {}
    with h5py.File(h5_path, "r") as f:
        # Big arrays → individual .npy
        print("  Writing x ...")
        np.save(os.path.join(out_dir, f"{stem}_x.npy"), f["x"][:])
        if "mask" in f:
            print("  Writing mask ...")
            np.save(os.path.join(out_dir, f"{stem}_mask.npy"), f["mask"][:])
        # Everything else → meta.npz
        for key in f.keys():
            if key not in ("x", "mask"):
                meta[key] = f[key][:]
    print("  Writing metadata ...")
    np.savez(os.path.join(out_dir, f"{stem}_meta.npz"), **meta)
    print(f"  Done in {_time.time() - t0:.1f}s")
    return out_dir


def convert_npz_to_h5(
    npz_path: str,
    h5_path: str | None = None,
    chunk_size: int = 512,
) -> str:
    """Convert an .npz file to a chunked HDF5 file.

    Parameters
    ----------
    npz_path : str
        Path to the source .npz file.
    h5_path : str or None
        Path for the output .h5 file.  If *None*, the .npz extension is
        replaced with .h5.
    chunk_size : int
        Chunk size along the first (sample) axis.  Default 512.

    Returns
    -------
    str
        The path of the created .h5 file.
    """
    if h5_path is None:
        base, _ = os.path.splitext(npz_path)
        h5_path = base + ".h5"

    data = np.load(npz_path)
    keys = list(data.files)
    print(f"Converting {npz_path}  →  {h5_path}")
    print(f"  Keys found: {keys}")

    with h5py.File(h5_path, "w") as hf:
        for key in keys:
            arr = data[key]

            # Downcast float64 → float32
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
                print(f"  {key:>15s}: {str(arr.shape):>25s}  float64 → float32")
            else:
                print(f"  {key:>15s}: {str(arr.shape):>25s}  {arr.dtype}")

            # Build chunk shape: chunk_size along axis 0, full extent elsewhere
            cs = min(chunk_size, arr.shape[0])
            chunks = (cs,) + arr.shape[1:]

            hf.create_dataset(key, data=arr, chunks=chunks)

    size_mb = os.path.getsize(h5_path) / 1024**2
    print(f"  Saved {h5_path}  ({size_mb:.1f} MB)")
    return h5_path


# ── Dataset classes ──────────────────────────────────────────────────

class L1JetDataset(Dataset):
    def __init__(self, filepath):
        print(f"Loading data from {filepath}...")
        data = _load_arrays(filepath)
        # X: (N, n_constituents, Features)
        self.X = torch.from_numpy(data["x"]).float()
        self.y = torch.from_numpy(data["y"]).float().unsqueeze(1)

        # Load particle mask if available, otherwise infer from non-zero energy
        if "mask" in data:
            self.mask = torch.from_numpy(data["mask"]).bool()
            print("Loaded particle mask from dataset.")
        else:
            # Fallback: infer mask from non-zero particles (E != 0)
            self.mask = self.X[..., 0] != 0
            print("No particle_mask in dataset, inferring from non-zero energy")

        if "weights" in data:
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


class StratifiedJetDataset(Dataset):
    """
    A Dataset class for jet constituent data that supports stratified splitting
    while preserving underlying feature distributions.

    Supports three storage backends:

    1. **NPZ / eager** – everything loaded into RAM as Torch tensors.
    2. **HDF5 lazy** – ``x``/``mask`` read per-batch from HDF5 (slow on
       network filesystems).
    3. **mmap ``.npy``** – ``x``/``mask`` memory-mapped via
       ``np.load(mmap_mode='r')``.  The OS page cache handles I/O
       automatically; only accessed pages reside in physical RAM.
       To use this mode, pass a directory containing ``<stem>_x.npy``,
       ``<stem>_mask.npy``, and ``<stem>_meta.npz`` (created by
       ``h5_to_npy``).

    Attributes:
        X: Tensor of shape (N, n_constituents, n_features) - constituent features
           (``None`` in lazy-HDF5 / mmap mode until ``_materialize()`` is called)
        y: Tensor of shape (N, 1) - binary labels
        mask: Tensor of shape (N, n_constituents) - particle mask
              (``None`` in lazy-HDF5 / mmap mode)
        weights: Tensor of shape (N, 1) - sample weights
    """

    # ── constructor ──────────────────────────────────────────────────
    def __init__(self, filepath: str, mode: str = None, pt_regression: bool = False):
        """
        Load dataset from .npz, .h5/.hdf5 file, or a directory of .npy files.

        For HDF5 files, ``x`` and ``mask`` are loaded lazily (per-sample in
        ``__getitem__``), so the full constituent tensor is never held in
        memory.  All other arrays are read eagerly.

        For a **directory** path (created by ``h5_to_npy``), ``x`` and
        ``mask`` are memory-mapped (`np.load(mmap_mode='r')`) — the OS
        page cache serves data with zero Python I/O overhead.

        Args:
            filepath: Path to the .npz / .h5 file, **or** a directory
                      produced by ``h5_to_npy`` containing
                      ``<stem>_x.npy``, ``<stem>_mask.npy``, ``<stem>_meta.npz``.
            mode: Data loading mode of the dataloader. If mode="match_distributions", weights are set to 1.0
            pt_regression: Whether to compute jet-level pt for regression tasks (if not pre-computed)
        """
        print(f"Loading data from {filepath}...")
        self.pt_regression = pt_regression

        # ---- detect format & decide lazy vs eager vs mmap ----
        self._mmap = False          # True when using memory-mapped .npy
        self._mmap_x_path = None    # path to x.npy (opened lazily per-worker)
        self._mmap_mask_path = None # path to mask.npy (or None)
        self._mmap_x = None         # lazily opened mmap handle
        self._mmap_mask = None      # lazily opened mmap handle

        ext = os.path.splitext(filepath)[1].lower()
        is_dir = os.path.isdir(filepath)
        self._lazy = (not is_dir) and ext in (".h5", ".hdf5")
        self._h5_path = os.path.abspath(filepath) if self._lazy else None
        self._h5_handle = None  # opened lazily per-worker

        if is_dir:
            # ── mmap .npy mode ──────────────────────────────────────
            npy_files = [f for f in os.listdir(filepath) if f.endswith("_x.npy")]
            if not npy_files:
                raise FileNotFoundError(
                    f"No *_x.npy file found in {filepath}. "
                    "Use h5_to_npy() to create mmap-ready files first."
                )
            stem = npy_files[0].replace("_x.npy", "")
            x_path = os.path.abspath(os.path.join(filepath, f"{stem}_x.npy"))
            mask_path = os.path.abspath(os.path.join(filepath, f"{stem}_mask.npy"))
            meta_path = os.path.join(filepath, f"{stem}_meta.npz")

            self._mmap = True
            self._mmap_x_path = x_path
            # Peek at shape without keeping the handle (pickle-safe)
            tmp = np.load(x_path, mmap_mode="r")
            self._x_shape = tmp.shape
            del tmp
            if os.path.exists(mask_path):
                self._mmap_mask_path = mask_path
            data = dict(np.load(meta_path))   # small – fits in RAM
            self._indices = np.arange(self._x_shape[0], dtype=np.int32)
            self.X = None
            self.mask = None
            print(f"  mmap .npy mode: stem={stem}")

        elif self._lazy:
            # ── HDF5 lazy mode ──────────────────────────────────────
            with h5py.File(filepath, "r") as f:
                self._x_shape = f["x"].shape
                self._has_mask_on_disk = "mask" in f
            data = _load_arrays(filepath, skip_keys={"x", "mask"})
            self._indices = np.arange(self._x_shape[0], dtype=np.int32)
            self.X = None
            self.mask = None

        else:
            # ── eager (npz) mode ────────────────────────────────────
            data = _load_arrays(filepath)
            self._indices = None
            self.X = torch.from_numpy(data["x"]).float()
            self._x_shape = self.X.shape
            if "mask" in data:
                self.mask = torch.from_numpy(data["mask"]).bool()
            else:
                self.mask = self.X[..., 1] != 0
                print("No mask in dataset, inferring from non-zero second feature")

        # ---- 1-D arrays (always in memory) ----
        self.y = torch.from_numpy(data["y"]).float().unsqueeze(1)

        if "weights" in data:
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

        if "jet_pt" in data and "jet_eta" in data:
            self.jet_pt = torch.from_numpy(data["jet_pt"]).float()
            self.jet_eta = torch.from_numpy(data["jet_eta"]).float()
        else:
            print("Warning: Pre-calc jet_pt/eta not found.")
            print(
                "Setting jet_pt and jet_eta to ones (will be computed on-the-fly if needed)."
            )
            self.jet_pt = torch.ones_like(self.y)
            self.jet_eta = torch.ones_like(self.y)

        if "gen_pt" in data:
            self.gen_pt = torch.from_numpy(data["gen_pt"]).float()
        else:
            print("Warning: gen_pt not found in dataset, setting to ones.")
            self.gen_pt = torch.ones_like(self.y)

        n, nc, nf = self._x_shape
        if self._mmap:
            tag = " [mmap .npy — zero-copy, OS-cached]"
        elif self._lazy:
            tag = " [lazy HDF5 — x/mask not in memory]"
        else:
            tag = ""
        print(f"Data loaded: {n} samples, {nc} constituents, {nf} features{tag}")

        if "qcd_weights" in data:
            self.qcd_weights = (
                torch.from_numpy(data["qcd_weights"]).float().unsqueeze(1)
            )
            print("Loaded qcd_weights from dataset.")
        else:
            self.qcd_weights = torch.ones_like(self.y)
            print("No qcd_weights in dataset, using uniform weights of 1.0")

    # ── I/O helpers (lazily opened, one per process) ──────────────
    def _get_mmap_x(self):
        """Return a mmap handle for x.npy (lazily opened, one per worker)."""
        if self._mmap_x is None:
            self._mmap_x = np.load(self._mmap_x_path, mmap_mode="r")
        return self._mmap_x

    def _get_mmap_mask(self):
        """Return a mmap handle for mask.npy (lazily opened, one per worker)."""
        if self._mmap_mask is None and self._mmap_mask_path is not None:
            self._mmap_mask = np.load(self._mmap_mask_path, mmap_mode="r")
        return self._mmap_mask

    def _get_h5(self):
        """Return an open HDF5 file handle (lazily opened, one per process)."""
        if self._h5_handle is None:
            self._h5_handle = h5py.File(self._h5_path, "r")
        return self._h5_handle

    def _materialize(self):
        """Load ``x`` and ``mask`` fully into memory.

        Required for in-place operations like ``_convert_to_relative_features``.
        After this call the dataset behaves identically to an NPZ-loaded one.
        """
        if not self._lazy and not self._mmap:
            return

        if self._mmap:
            print("  Materializing X and mask from mmap .npy ...")
            mx = self._get_mmap_x()
            self.X = torch.from_numpy(
                np.array(mx[self._indices])
            ).float()
            mm = self._get_mmap_mask()
            if mm is not None:
                self.mask = torch.from_numpy(
                    np.array(mm[self._indices])
                ).bool()
            else:
                self.mask = self.X[..., 1] != 0
            self._mmap = False
            self._mmap_x = None
            self._mmap_mask = None
            self._indices = None
            print(f"  Materialized: {self.X.shape}")
            return

        # HDF5 path
        print(f"  Materializing X and mask from {self._h5_path}...")
        with h5py.File(self._h5_path, "r") as f:
            all_x = f["x"][:]
            self.X = torch.from_numpy(all_x[self._indices]).float()
            del all_x
            if self._has_mask_on_disk:
                all_mask = f["mask"][:]
                self.mask = torch.from_numpy(all_mask[self._indices]).bool()
                del all_mask
            else:
                self.mask = self.X[..., 1] != 0
        self._lazy = False
        self._indices = None
        # Close any cached handle
        if self._h5_handle is not None:
            self._h5_handle.close()
            self._h5_handle = None
        print(f"  Materialized: {self.X.shape}")

    def __del__(self):
        if getattr(self, "_h5_handle", None) is not None:
            self._h5_handle.close()

    # ── core Dataset interface ──────────────────────────────────────
    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self._mmap:
            physical_idx = int(self._indices[idx])
            mx = self._get_mmap_x()
            x = torch.from_numpy(np.array(mx[physical_idx])).float()
            mm = self._get_mmap_mask()
            if mm is not None:
                mask = torch.from_numpy(np.array(mm[physical_idx])).bool()
            else:
                mask = x[..., 1] != 0
        elif self._lazy:
            physical_idx = int(self._indices[idx])
            h5 = self._get_h5()
            x = torch.from_numpy(np.array(h5["x"][physical_idx])).float()
            if self._has_mask_on_disk:
                mask = torch.from_numpy(np.array(h5["mask"][physical_idx])).bool()
            else:
                mask = x[..., 1] != 0
        else:
            x = self.X[idx]
            mask = self.mask[idx]

        return (
            x,
            self.y[idx],
            mask,
            self.weights[idx],
            self.jet_pt[idx],
            self.jet_eta[idx],
            self.gen_pt[idx],
            self.qcd_weights[idx],
        )

    def __getitems__(self, indices: List[int]):
        if self._mmap:
            # ── mmap path: direct fancy-index (safe with random shuffle) ──
            # Unlike HDF5, mmap fancy indexing is cheap — each row triggers
            # a page fault (~4 KB) handled by the kernel.  The bounding-box
            # approach would copy the entire file when indices are randomly
            # spread across the dataset.
            physical_indices = self._indices[indices]

            mx = self._get_mmap_x()
            x_batch = torch.from_numpy(np.array(mx[physical_indices])).float()

            mm = self._get_mmap_mask()
            if mm is not None:
                mask_batch = torch.from_numpy(np.array(mm[physical_indices])).bool()
            else:
                mask_batch = x_batch[..., 1] != 0
        elif self._lazy:
            h5 = self._get_h5()
            # Get the physical HDF5 rows we need
            physical_indices = self._indices[indices]
            unique_phys, inverse_idx = np.unique(physical_indices, return_inverse=True)

            start_idx = int(unique_phys[0])
            end_idx = int(unique_phys[-1] + 1)

            # 1. Read the contiguous block encompassing all needed indices (Lightning fast)
            block_x = h5["x"][start_idx:end_idx]

            # 2. Slice out the specific indices from the NumPy array in RAM (Zero h5py overhead)
            ram_indices = unique_phys - start_idx
            x_bulk = block_x[ram_indices]

            if self._has_mask_on_disk:
                block_mask = h5["mask"][start_idx:end_idx]
                mask_bulk = block_mask[ram_indices]
            else:
                mask_bulk = x_bulk[..., 1] != 0

            # Reorder back to the requested batch order
            x_batch = x_bulk[inverse_idx]
            mask_batch = mask_bulk[inverse_idx]

            x_batch = torch.from_numpy(x_batch).float()
            mask_batch = torch.from_numpy(mask_batch).bool() if self._has_mask_on_disk else torch.from_numpy(mask_batch)
        else:
            x_batch = self.X[indices]
            mask_batch = self.mask[indices]

        # Eagerly loaded 1D arrays
        y_batch = self.y[indices]
        weights_batch = self.weights[indices]
        jet_pt_batch = self.jet_pt[indices]
        jet_eta_batch = self.jet_eta[indices]
        gen_pt_batch = self.gen_pt[indices]
        qcd_weights_batch = self.qcd_weights[indices]

        # Return a *single* pre-collated tuple wrapped in a length-1 list.
        # When paired with ``collate_fn=lambda batch: batch[0]`` in the
        # DataLoader, this avoids 1024 per-tensor IPC serialisations and
        # a redundant ``torch.stack`` inside ``default_collate``.
        return [(
            x_batch, y_batch, mask_batch, weights_batch,
            jet_pt_batch, jet_eta_batch, gen_pt_batch, qcd_weights_batch,
        )]
    # ── metadata helpers ────────────────────────────────────────────
    @property
    def labels(self) -> np.ndarray:
        """Get labels as numpy array for stratification."""
        return self.y.squeeze().numpy().astype(int)

    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of classes in the dataset."""
        labels = self.labels
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))

    def filter_indices(self, pt_min, pt_max, eta_min, eta_max):
        """Returns indices of jets strictly within the kinematic window."""
        mask = (
            (self.jet_pt >= pt_min)
            & (self.jet_pt <= pt_max)
            & (self.jet_eta >= eta_min)
            & (self.jet_eta <= eta_max)
        )
        return torch.nonzero(mask, as_tuple=True)[0].numpy()

    def apply_kinematic_filter(self, pt_min, pt_max, eta_min, eta_max):
        """Filter dataset **in-place**, keeping only jets within the window.

        Slices all in-memory tensors.  In lazy-HDF5 mode the physical
        index map (``_indices``) is updated instead of slicing ``X``/``mask``.
        """
        if not hasattr(self, "jet_pt") or not hasattr(self, "jet_eta"):
            print("  Skipping kinematic filter: jet_pt/jet_eta not available.")
            return

        valid = self.filter_indices(pt_min, pt_max, eta_min, eta_max)
        n_before = len(self)
        if len(valid) == n_before:
            print(
                f"  Kinematic filter: all {n_before} jets pass — no filtering needed."
            )
            return

        # 1-D arrays — always sliced in memory
        self.y = self.y[valid]
        self.weights = self.weights[valid]
        self.jet_pt = self.jet_pt[valid]
        self.jet_eta = self.jet_eta[valid]
        self.gen_pt = self.gen_pt[valid]
        self.qcd_weights = self.qcd_weights[valid]

        # X / mask — depends on mode
        if self._lazy or self._mmap:
            self._indices = self._indices[valid]
        else:
            self.X = self.X[valid]
            self.mask = self.mask[valid]

        print(f"  Kinematic filter: {n_before} → {len(self)} jets kept")


# ── sampler ──────────────────────────────────────────────────────────

class WeakShuffleSampler(Sampler):
    """
    A custom sampler that groups dataset indices by their physical layout
    in the HDF5 file, chunks them, and shuffles the chunks to allow for
    randomization without destroying HDF5 chunk caching.
    """

    def __init__(self, dataset, chunk_size=512):
        self.dataset = dataset
        self.chunk_size = chunk_size

        # 1. Get logical indices (0 to len(dataset)-1)
        logical_indices = np.arange(len(dataset))

        # 2. Resolve the underlying physical HDF5 indices
        if isinstance(dataset, Subset):
            orig_indices = np.array(dataset.indices)
            orig_dataset = dataset.dataset
        else:
            orig_indices = logical_indices
            orig_dataset = dataset

        if (
            hasattr(orig_dataset, "_indices")
            and orig_dataset._indices is not None
            and (getattr(orig_dataset, "_lazy", False) or getattr(orig_dataset, "_mmap", False))
        ):
            physical_indices = orig_dataset._indices[orig_indices]
        else:
            physical_indices = orig_indices

        # 3. Sort logical indices so they are ordered by physical HDF5 layout
        sorted_by_physical = logical_indices[np.argsort(physical_indices)]

        # 4. Group into chunks
        self.chunks = [
            sorted_by_physical[i : i + chunk_size]
            for i in range(0, len(sorted_by_physical), chunk_size)
        ]

    def __iter__(self):
        # Shuffle the order of the chunks
        np.random.shuffle(self.chunks)

        # Yield indices one by one (DataLoader will batch them)
        for chunk in self.chunks:
            # Shuffle inside the chunk to ensure intra-batch randomness
            np.random.shuffle(chunk)
            for idx in chunk:
                yield int(idx)

    def __len__(self):
        return len(self.dataset)


# ── CLI entry-point ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset format conversion utilities"
    )
    sub = parser.add_subparsers(dest="command")

    # npz-to-h5
    p1 = sub.add_parser("npz-to-h5", help="Convert .npz → chunked HDF5")
    p1.add_argument("--npz-path", type=str, help="Path to the .npz file")
    p1.add_argument("--h5-path", type=str, default=None, help="Output .h5 path (default: same name)")
    p1.add_argument("--chunk-size", type=int, default=512, help="HDF5 chunk size (default: 512)")

    # h5-to-npy
    p2 = sub.add_parser("h5-to-npy", help="Convert HDF5 → memory-mappable .npy files")
    p2.add_argument("--h5-path", type=str, help="Path to the .h5 file")
    p2.add_argument("--out-dir", type=str, default=None, help="Output directory (default: same as input)")

    args = parser.parse_args()
    if args.command == "npz-to-h5":
        convert_npz_to_h5(args.npz_path, args.h5_path, args.chunk_size)
    elif args.command == "h5-to-npy":
        h5_to_npy(args.h5_path, args.out_dir)
    else:
        parser.print_help()
