"""
GPU-specific non-uniform FFT implementation for fftvis.

Thin wrappers around ``cufinufft`` that mirror the CPU wrappers in
:mod:`fftvis.cpu.nufft` one-for-one, so the two simulation engines can be
compared directly.

Notes
-----
Three differences from the CPU wrappers are worth knowing about:

* ``n_threads`` is accepted for signature compatibility and ignored.
* ``nthreads`` and ``showwarn`` are CPU-only finufft options and must not be
  forwarded to cufinufft.
* ``modeord`` has no effect on a type-3 transform, so it is only passed for the
  type-1 wrapper (where fftvis relies on FFT-style ordering to index modes with
  signed integers).

GPU type-3 transforms require ``finufft >= 2.4``.
"""

from typing import Literal

import numpy as np

try:  # pragma: no cover - import guard
    import cupy as cp
    import cufinufft

    HAVE_CUDA = True
except ImportError:  # pragma: no cover - import guard
    cp = None
    cufinufft = None
    HAVE_CUDA = False


def _require_cuda():
    """Raise a helpful error if the GPU stack is unavailable."""
    if not HAVE_CUDA:  # pragma: no cover - import guard
        raise ImportError(
            "The GPU backend requires cupy and cufinufft. Install them with "
            "`pip install fftvis[gpu-cuda12]` (or [gpu-cuda11]). GPU type-3 transforms need "
            "finufft >= 2.4."
        )


def _real_dtype(weights) -> np.dtype:
    """Real dtype matching the precision of a complex array."""
    return np.float32 if weights.dtype == np.complex64 else np.float64


def _coords(arrays, rdtype) -> list:
    """Coerce coordinate arrays to contiguous device arrays of ``rdtype``."""
    return [cp.ascontiguousarray(cp.asarray(a), dtype=rdtype) for a in arrays]


def gpu_nufft2d(
    x,
    y,
    weights,
    u,
    v,
    eps: float,
    n_threads: int = 1,
    upsample_factor: Literal[1.25, 2] = 2,
):
    """
    Perform a 2D type-3 non-uniform FFT on the GPU.

    Parameters
    ----------
    x : cp.ndarray
        X coordinates of source positions.
    y : cp.ndarray
        Y coordinates of source positions.
    weights : cp.ndarray
        Weights of sources (typically beam-weighted fluxes). Shape
        ``(n_trans, nsrc)`` or ``(nsrc,)``.
    u : cp.ndarray
        U coordinates for baselines.
    v : cp.ndarray
        V coordinates for baselines.
    eps : float
        Desired accuracy of the transform.
    n_threads : int
        Unused on the GPU; accepted for signature compatibility with
        :func:`fftvis.cpu.nufft.cpu_nufft2d`.
    upsample_factor : default = 2
        Upsampling factor for the non-uniform FFT.

    Returns
    -------
    cp.ndarray
        Visibility data.
    """
    _require_cuda()
    rdtype = _real_dtype(weights)
    gx, gy, gu, gv = _coords((x, y, u, v), rdtype)
    return cufinufft.nufft2d3(
        gx,
        gy,
        cp.ascontiguousarray(weights),
        gu,
        gv,
        eps=float(eps),
        upsampfac=float(upsample_factor),
    )


def gpu_nufft3d(
    x,
    y,
    z,
    weights,
    u,
    v,
    w,
    eps: float,
    upsample_factor: Literal[1.25, 2] = 2,
    n_threads: int = 1,
):
    """
    Perform a 3D type-3 non-uniform FFT on the GPU.

    Parameters
    ----------
    x, y, z : cp.ndarray
        Coordinates of source positions.
    weights : cp.ndarray
        Weights of sources (typically beam-weighted fluxes). Shape
        ``(n_trans, nsrc)`` or ``(nsrc,)``.
    u, v, w : cp.ndarray
        Baseline coordinates.
    eps : float
        Desired accuracy of the transform.
    upsample_factor : default = 2
        Upsampling factor for the non-uniform FFT.
    n_threads : int
        Unused on the GPU; accepted for signature compatibility with
        :func:`fftvis.cpu.nufft.cpu_nufft3d`.

    Returns
    -------
    cp.ndarray
        Visibility data.

    Notes
    -----
    finufft sizes the internal type-3 grid from the product of the source and
    target half-widths. For a non-coplanar array with long baselines this can be
    very large; :func:`fftvis.gpu.gpu_simulate.estimate_type3_grid` checks it
    before the transform is attempted.
    """
    _require_cuda()
    rdtype = _real_dtype(weights)
    gx, gy, gz, gu, gv, gw = _coords((x, y, z, u, v, w), rdtype)
    return cufinufft.nufft3d3(
        gx,
        gy,
        gz,
        cp.ascontiguousarray(weights),
        gu,
        gv,
        gw,
        eps=float(eps),
        upsampfac=float(upsample_factor),
    )


def gpu_nufft2d_type1(
    x,
    y,
    weights,
    n_modes: int,
    index,
    eps: float,
    upsample_factor: Literal[1.25, 2] = 2,
    n_threads: int = 1,
):
    """
    Perform a 2D type-1 non-uniform FFT on the GPU.

    Parameters
    ----------
    x, y : cp.ndarray
        Coordinates of source positions.
    weights : cp.ndarray
        Weights of sources (typically beam-weighted fluxes).
    n_modes : int
        Number of Fourier modes per axis returned by the type-1 transform. The
        model is of shape ``(..., n_modes, n_modes)`` prior to indexing.
    index : np.ndarray or cp.ndarray
        Integer array of shape ``(2, nbls)`` selecting modes from the model.
        Values may be negative; ``modeord=1`` puts the zero mode first so that
        signed indices wrap correctly.
    eps : float
        Desired accuracy of the transform.
    upsample_factor : default = 2
        Upsampling factor for the non-uniform FFT.
    n_threads : int
        Unused on the GPU; accepted for signature compatibility with
        :func:`fftvis.cpu.nufft.cpu_nufft2d_type1`.

    Returns
    -------
    cp.ndarray
        Visibility data.
    """
    _require_cuda()
    rdtype = _real_dtype(weights)
    gx, gy = _coords((x, y), rdtype)

    model = cufinufft.nufft2d1(
        gx,
        gy,
        cp.ascontiguousarray(weights),
        (int(n_modes), int(n_modes)),
        eps=float(eps),
        modeord=1,
        upsampfac=float(upsample_factor),
    )

    index = cp.asarray(index)
    return model[..., index[0], index[1]]
