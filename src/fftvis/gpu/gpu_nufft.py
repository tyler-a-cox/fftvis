"""
GPU-specific non-uniform FFT implementation for fftvis.

This module provides GPU-specific NUFFT functionality using the cufinufft library.
"""

import cupy as cp
import cufinufft
import numpy as np  # Keep numpy import for type hinting if needed


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
        X coordinates of source positions (on GPU).
    y : cp.ndarray
        Y coordinates of source positions (on GPU).
    weights : cp.ndarray
        Weights of sources (typically beam-weighted fluxes) (on GPU).
    u : cp.ndarray
        U coordinates for baselines (on GPU).
    v : cp.ndarray
        V coordinates for baselines (on GPU).
    eps : float
        Desired accuracy of the transform.
    n_threads : int
        Number of threads to use (ignored by cufinufft).

    Returns
    -------
    cp.ndarray
        Visibility data (on GPU).
    """
    # Note: In fftvis, we need type 3 transform (non-uniform to non-uniform)
    # The cufinufft API v2.2+ doesn't have simple nufft*d3 functions like finufft
    # We need to use the Plan interface for type 3 transforms

    # Determine an appropriate grid size for the intermediate uniform grid
    # This is a heuristic based on the maximum frequencies
    # Ensure minimum grid size and handle edge cases
    N1 = max(16, 2 * int(cp.max(cp.abs(u)).get()) + 5) if len(u) > 0 else 16
    N2 = max(16, 2 * int(cp.max(cp.abs(v)).get()) + 5) if len(v) > 0 else 16

    # Create plan for type 3 transform
    plan = cufinufft.Plan(
        nufft_type=3,  # Type 3: non-uniform to non-uniform
        n_modes=(N1, N2),  # Size of intermediate grid
        n_trans=weights.shape[0] if weights.ndim > 1 else 1,
        eps=eps,
        isign=-1,  # Match finufft sign convention
    )

    # Set source and target points
    plan.setpts(x=x, y=y, s=u, t=v)

    # Execute the transform
    return plan.execute(weights)


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
        X coordinates of source positions (on GPU).
    y : cp.ndarray
        Y coordinates of source positions (on GPU).
    z : cp.ndarray
        Z coordinates of source positions (on GPU).
    weights : cp.ndarray
        Weights of sources (typically beam-weighted fluxes) (on GPU).
    u : cp.ndarray
        U coordinates for baselines (on GPU).
    v : cp.ndarray
        V coordinates for baselines (on GPU).
    w : cp.ndarray
        W coordinates for baselines (on GPU).
    eps : float
        Desired accuracy of the transform.
    n_threads : int
        Number of threads to use (ignored by cufinufft).

    Returns
    -------
    cp.ndarray
        Visibility data (on GPU).
    """
    # Note: In fftvis, we need type 3 transform (non-uniform to non-uniform)
    # The cufinufft API v2.2+ doesn't have simple nufft*d3 functions like finufft
    # We need to use the Plan interface for type 3 transforms

    # Determine an appropriate grid size for the intermediate uniform grid
    # This is a heuristic based on the maximum frequencies
    # Ensure minimum grid size and handle edge cases
    N1 = max(16, 2 * int(cp.max(cp.abs(u)).get()) + 5) if len(u) > 0 else 16
    N2 = max(16, 2 * int(cp.max(cp.abs(v)).get()) + 5) if len(v) > 0 else 16
    N3 = max(16, 2 * int(cp.max(cp.abs(w)).get()) + 5) if len(w) > 0 else 16

    # Create plan for type 3 transform
    plan = cufinufft.Plan(
        nufft_type=3,  # Type 3: non-uniform to non-uniform
        n_modes=(N1, N2, N3),  # Size of intermediate grid
        n_trans=weights.shape[0] if weights.ndim > 1 else 1,
        eps=eps,
        isign=-1,  # Match finufft sign convention
    )

    # Set source and target points
    plan.setpts(x=x, y=y, z=z, s=u, t=v, u=w)

    # Execute the transform
    return plan.execute(weights)
