"""
Convert an .npz dataset to a chunked HDF5 (.h5) file.

- Downcasts all float64 arrays to float32.
- Stores each array as a chunked HDF5 dataset (chunk_size along axis 0).
- Preserves all keys present in the source .npz file.

Usage (CLI):
    python npz_to_h5.py --npz_path data/qcd_bkg_cls_reg/l1_puppi_16_flat.npz

Usage (importable):
    from npz_to_h5 import convert_npz_to_h5
    convert_npz_to_h5("input.npz", "output.h5", chunk_size=512)
"""

import argparse
import os
import numpy as np
import h5py


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


# ── CLI entry-point ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .npz dataset → chunked HDF5 (.h5)"
    )
    parser.add_argument("--npz_path", type=str, help="Path to the .npz file")
    parser.add_argument(
        "--output", type=str, default=None, help="Output .h5 path (default: same name)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512, help="HDF5 chunk size (default: 512)"
    )
    args = parser.parse_args()
    convert_npz_to_h5(args.npz_path, args.output, args.chunk_size)
