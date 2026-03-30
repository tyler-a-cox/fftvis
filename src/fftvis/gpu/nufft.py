"""
GPU-specific non-uniform FFT implementation for fftvis.

This module provides GPU-specific NUFFT functionality using the cufinufft library.
Requires cufinufft >= 2.2.0 for native Type 3 NUFFT support.
"""

import logging
import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)

try:
    import cufinufft
    HAVE_CUFINUFFT = True
except ImportError:
    HAVE_CUFINUFFT = False


def gpu_nufft2d(
    x: cp.ndarray,
    y: cp.ndarray,
    weights: cp.ndarray,
    u: cp.ndarray,
    v: cp.ndarray,
    eps: float,
    n_threads: int = 1,  # n_threads is not used by cufinufft, but keep for consistent signature
) -> cp.ndarray:
    """
    Perform a 2D non-uniform FFT on the GPU using cufinufft.

    Parameters
    ----------
    x : cp.ndarray
        X coordinates of source positions.
    y : cp.ndarray
        Y coordinates of source positions.
    weights : cp.ndarray
        Weights of sources (typically beam-weighted fluxes).
    u : cp.ndarray
        U coordinates for baselines.
    v : cp.ndarray
        V coordinates for baselines.
    eps : float
        Desired accuracy of the transform.
    n_threads : int
        Number of threads to use (ignored by cufinufft).

    Returns
    -------
    cp.ndarray
        Visibility data.
    """
    if not HAVE_CUFINUFFT:
        raise ImportError("cufinufft is required for GPU NUFFT operations")

    # Handle empty input case
    if len(x) == 0 or len(u) == 0:
        if weights.ndim == 1:
            return cp.zeros(len(u), dtype=cp.complex128)
        else:
            return cp.zeros((weights.shape[0], len(u)), dtype=cp.complex128)

    # Ensure all inputs are contiguous CuPy arrays
    x = cp.ascontiguousarray(x, dtype=cp.float64)
    y = cp.ascontiguousarray(y, dtype=cp.float64)
    u = cp.ascontiguousarray(u, dtype=cp.float64)
    v = cp.ascontiguousarray(v, dtype=cp.float64)
    weights = cp.ascontiguousarray(weights, dtype=cp.complex128)

    return cufinufft.nufft2d3(x, y, weights, u, v, eps=eps, modeord=0)


def gpu_nufft3d(
    x: cp.ndarray,
    y: cp.ndarray,
    z: cp.ndarray,
    weights: cp.ndarray,
    u: cp.ndarray,
    v: cp.ndarray,
    w: cp.ndarray,
    eps: float,
    n_threads: int = 1,  # n_threads is not used by cufinufft, but keep for consistent signature
) -> cp.ndarray:
    """
    Perform a 3D non-uniform FFT on the GPU using cufinufft.

    Parameters
    ----------
    x : cp.ndarray
        X coordinates of source positions.
    y : cp.ndarray
        Y coordinates of source positions.
    z : cp.ndarray
        Z coordinates of source positions.
    weights : cp.ndarray
        Weights of sources (typically beam-weighted fluxes).
    u : cp.ndarray
        U coordinates for baselines.
    v : cp.ndarray
        V coordinates for baselines.
    w : cp.ndarray
        W coordinates for baselines.
    eps : float
        Desired accuracy of the transform.
    n_threads : int
        Number of threads to use (ignored by cufinufft).

    Returns
    -------
    cp.ndarray
        Visibility data.
    """
    if not HAVE_CUFINUFFT:
        raise ImportError("cufinufft is required for GPU NUFFT operations")

    # Handle empty input case
    if len(x) == 0 or len(u) == 0:
        if weights.ndim == 1:
            return cp.zeros(len(u), dtype=cp.complex128)
        else:
            return cp.zeros((weights.shape[0], len(u)), dtype=cp.complex128)

    # Ensure all inputs are contiguous CuPy arrays
    x = cp.ascontiguousarray(x, dtype=cp.float64)
    y = cp.ascontiguousarray(y, dtype=cp.float64)
    z = cp.ascontiguousarray(z, dtype=cp.float64)
    u = cp.ascontiguousarray(u, dtype=cp.float64)
    v = cp.ascontiguousarray(v, dtype=cp.float64)
    w = cp.ascontiguousarray(w, dtype=cp.float64)
    weights = cp.ascontiguousarray(weights, dtype=cp.complex128)

    return cufinufft.nufft3d3(x, y, z, weights, u, v, w, eps=eps, modeord=0)


def gpu_nufft2d_batch(
    x: cp.ndarray,
    y: cp.ndarray,
    weights_batch: cp.ndarray,
    u_batch: cp.ndarray,
    v_batch: cp.ndarray,
    eps: float,
    n_threads: int = 1,  # n_threads is not used by cufinufft
) -> cp.ndarray:
    """
    Perform batched 2D non-uniform FFT on the GPU for multiple frequencies.

    This function processes multiple frequencies by calling gpu_nufft2d
    for each frequency in the batch.

    Parameters
    ----------
    x : cp.ndarray
        X coordinates of source positions, shape (nsrc,)
    y : cp.ndarray
        Y coordinates of source positions, shape (nsrc,)
    weights_batch : cp.ndarray
        Batch of weights for each frequency, shape (n_freq, nfeeds**2, nsrc) for polarized
        or (n_freq, nsrc) for unpolarized
    u_batch : cp.ndarray
        U coordinates for baselines at each frequency, shape (n_freq, nbls)
    v_batch : cp.ndarray
        V coordinates for baselines at each frequency, shape (n_freq, nbls)
    eps : float
        Desired accuracy of the transform
    n_threads : int
        Number of threads (ignored by cufinufft)

    Returns
    -------
    cp.ndarray
        Visibility data, shape (n_freq, nfeeds**2, nbls) for polarized
        or (n_freq, nbls) for unpolarized
    """
    if not HAVE_CUFINUFFT:
        raise ImportError("cufinufft is required for GPU NUFFT operations")

    # Ensure all inputs are contiguous CuPy arrays
    x = cp.ascontiguousarray(x, dtype=cp.float64)
    y = cp.ascontiguousarray(y, dtype=cp.float64)
    weights_batch = cp.ascontiguousarray(weights_batch, dtype=cp.complex128)
    u_batch = cp.ascontiguousarray(u_batch, dtype=cp.float64)
    v_batch = cp.ascontiguousarray(v_batch, dtype=cp.float64)

    n_freq = weights_batch.shape[0]

    # Handle polarized vs unpolarized case
    is_polarized = weights_batch.ndim == 3
    if is_polarized:
        n_pol = weights_batch.shape[1]

    results = []
    for freq_idx in range(n_freq):
        if is_polarized:
            freq_results = []
            for pol_idx in range(n_pol):
                vis = gpu_nufft2d(
                    x, y,
                    weights_batch[freq_idx, pol_idx, :],
                    u_batch[freq_idx, :],
                    v_batch[freq_idx, :],
                    eps=eps,
                )
                freq_results.append(vis)
            results.append(cp.stack(freq_results))
        else:
            vis = gpu_nufft2d(
                x, y,
                weights_batch[freq_idx, :],
                u_batch[freq_idx, :],
                v_batch[freq_idx, :],
                eps=eps,
            )
            results.append(vis)

    return cp.stack(results)


def gpu_nufft3d_batch(
    x: cp.ndarray,
    y: cp.ndarray,
    z: cp.ndarray,
    weights_batch: cp.ndarray,
    u_batch: cp.ndarray,
    v_batch: cp.ndarray,
    w_batch: cp.ndarray,
    eps: float,
    n_threads: int = 1,  # n_threads is not used by cufinufft
) -> cp.ndarray:
    """
    Perform batched 3D non-uniform FFT on the GPU for multiple frequencies.

    This function processes multiple frequencies by calling gpu_nufft3d
    for each frequency in the batch.

    Parameters
    ----------
    x : cp.ndarray
        X coordinates of source positions, shape (nsrc,)
    y : cp.ndarray
        Y coordinates of source positions, shape (nsrc,)
    z : cp.ndarray
        Z coordinates of source positions, shape (nsrc,)
    weights_batch : cp.ndarray
        Batch of weights for each frequency, shape (n_freq, nfeeds**2, nsrc) for polarized
        or (n_freq, nsrc) for unpolarized
    u_batch : cp.ndarray
        U coordinates for baselines at each frequency, shape (n_freq, nbls)
    v_batch : cp.ndarray
        V coordinates for baselines at each frequency, shape (n_freq, nbls)
    w_batch : cp.ndarray
        W coordinates for baselines at each frequency, shape (n_freq, nbls)
    eps : float
        Desired accuracy of the transform
    n_threads : int
        Number of threads (ignored by cufinufft)

    Returns
    -------
    cp.ndarray
        Visibility data, shape (n_freq, nfeeds**2, nbls) for polarized
        or (n_freq, nbls) for unpolarized
    """
    if not HAVE_CUFINUFFT:
        raise ImportError("cufinufft is required for GPU NUFFT operations")

    # Ensure all inputs are contiguous CuPy arrays
    x = cp.ascontiguousarray(x, dtype=cp.float64)
    y = cp.ascontiguousarray(y, dtype=cp.float64)
    z = cp.ascontiguousarray(z, dtype=cp.float64)
    weights_batch = cp.ascontiguousarray(weights_batch, dtype=cp.complex128)
    u_batch = cp.ascontiguousarray(u_batch, dtype=cp.float64)
    v_batch = cp.ascontiguousarray(v_batch, dtype=cp.float64)
    w_batch = cp.ascontiguousarray(w_batch, dtype=cp.float64)

    n_freq = weights_batch.shape[0]

    # Handle polarized vs unpolarized case
    is_polarized = weights_batch.ndim == 3
    if is_polarized:
        n_pol = weights_batch.shape[1]

    results = []
    for freq_idx in range(n_freq):
        if is_polarized:
            freq_results = []
            for pol_idx in range(n_pol):
                vis = gpu_nufft3d(
                    x, y, z,
                    weights_batch[freq_idx, pol_idx, :],
                    u_batch[freq_idx, :],
                    v_batch[freq_idx, :],
                    w_batch[freq_idx, :],
                    eps=eps,
                )
                freq_results.append(vis)
            results.append(cp.stack(freq_results))
        else:
            vis = gpu_nufft3d(
                x, y, z,
                weights_batch[freq_idx, :],
                u_batch[freq_idx, :],
                v_batch[freq_idx, :],
                w_batch[freq_idx, :],
                eps=eps,
            )
            results.append(vis)

    return cp.stack(results)
