"""
CPU-specific non-uniform FFT implementation for fftvis.

This module provides CPU-specific NUFFT functionality using the finufft library.
"""

import numpy as np
import finufft


def cpu_nufft2d(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    eps: float,
    n_threads: int = 1,
) -> np.ndarray:
    """
    Perform a 2D non-uniform FFT on the CPU.

    Parameters
    ----------
    x : np.ndarray
        X coordinates of source positions.
    y : np.ndarray
        Y coordinates of source positions.
    weights : np.ndarray
        Weights of sources (typically beam-weighted fluxes).
    u : np.ndarray
        U coordinates for baselines.
    v : np.ndarray
        V coordinates for baselines.
    eps : float
        Desired accuracy of the transform.
    n_threads : int
        Number of threads to use.

    Returns
    -------
    np.ndarray
        Visibility data.
    """
    return finufft.nufft2d3(
        x,
        y,
        weights,
        np.ascontiguousarray(u),
        np.ascontiguousarray(v),
        modeord=0,
        eps=eps,
        nthreads=n_threads,
        showwarn=0,
    )


def cpu_nufft3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    weights: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    eps: float,
    n_threads: int = 1,
) -> np.ndarray:
    """
    Perform a 3D non-uniform FFT on the CPU.

    Parameters
    ----------
    x : np.ndarray
        X coordinates of source positions.
    y : np.ndarray
        Y coordinates of source positions.
    z : np.ndarray
        Z coordinates of source positions.
    weights : np.ndarray
        Weights of sources (typically beam-weighted fluxes).
    u : np.ndarray
        U coordinates for baselines.
    v : np.ndarray
        V coordinates for baselines.
    w : np.ndarray
        W coordinates for baselines.
    eps : float
        Desired accuracy of the transform.
    n_threads : int
        Number of threads to use.

    Returns
    -------
    np.ndarray
        Visibility data.
    """
    return finufft.nufft3d3(
        x,
        y,
        z,
        weights,
        np.ascontiguousarray(u),
        np.ascontiguousarray(v),
        np.ascontiguousarray(w),
        modeord=0,
        eps=eps,
        nthreads=n_threads,
        showwarn=0,
    )
