"""
GPU-specific utility functions for fftvis.

These mirror the CPU implementations in :mod:`fftvis.cpu.utils`, operating on
cupy arrays instead of numpy arrays.
"""

import numpy as np

try:  # pragma: no cover - import guard
    import cupy as cp

    HAVE_CUDA = True
except ImportError:  # pragma: no cover - import guard
    cp = None
    HAVE_CUDA = False


def _require_cuda():
    """Raise a helpful error if cupy is unavailable."""
    if not HAVE_CUDA:  # pragma: no cover - import guard
        raise ImportError(
            "The GPU backend requires cupy and cufinufft. Install them with "
            "`pip install fftvis[gpu]`."
        )


def inplace_rot(rot: np.ndarray, b) -> None:
    """
    Rotate coordinates in place on the GPU.

    Equivalent to :func:`fftvis.cpu.utils.inplace_rot`.

    Parameters
    ----------
    rot : np.ndarray
        3x3 rotation matrix. May be a numpy or cupy array.
    b : cp.ndarray
        Array of shape ``(3, n)`` containing coordinates to rotate. Modified in
        place.
    """
    _require_cuda()
    rot = cp.asarray(rot, dtype=b.dtype)
    # cp.matmul cannot write into an array that aliases its input, so the
    # product is formed first and then copied back.
    b[:] = rot @ b
