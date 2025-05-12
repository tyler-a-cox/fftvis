"""
GPU-specific non-uniform FFT implementation for fftvis.

This module will provide GPU-specific NUFFT functionality.
Currently it contains stub implementations.
"""

import numpy as np


def gpu_nufft2d(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    eps: float,
    n_threads: int = 1,
) -> np.ndarray:
    """
    Perform a 2D non-uniform FFT on the GPU.

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
        Number of threads to use (not used in GPU implementation).

    Returns
    -------
    np.ndarray
        Visibility data.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    raise NotImplementedError("GPU NUFFT2D not yet implemented")


def gpu_nufft3d(
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
    Perform a 3D non-uniform FFT on the GPU.

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
        Number of threads to use (not used in GPU implementation).

    Returns
    -------
    np.ndarray
        Visibility data.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    raise NotImplementedError("GPU NUFFT3D not yet implemented")
