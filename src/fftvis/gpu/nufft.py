"""
GPU-specific non-uniform FFT implementation for fftvis.

This module provides GPU-specific NUFFT functionality using the cufinufft library.
"""

import cupy as cp
import numpy as np
try:
    import cufinufft
    HAVE_CUFINUFFT = True

    def _test_native_type3_support():
        """Test if native Type 3 functions are available."""
        try:
            has_2d3 = hasattr(cufinufft, 'nufft2d3') and callable(getattr(cufinufft, 'nufft2d3', None))
            has_3d3 = hasattr(cufinufft, 'nufft3d3') and callable(getattr(cufinufft, 'nufft3d3', None))
            return has_2d3 and has_3d3
        except Exception:
            return False

    HAS_NATIVE_TYPE3_SUPPORT = _test_native_type3_support()

except ImportError:
    HAVE_CUFINUFFT = False
    HAS_NATIVE_TYPE3_SUPPORT = False
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

    # Ensure all inputs are contiguous CuPy arrays
    x = cp.ascontiguousarray(x, dtype=cp.float64)
    y = cp.ascontiguousarray(y, dtype=cp.float64)
    u = cp.ascontiguousarray(u, dtype=cp.float64)
    v = cp.ascontiguousarray(v, dtype=cp.float64)
    weights = cp.ascontiguousarray(weights, dtype=cp.complex128)

    # Handle different weight shapes (polarized vs unpolarized)
    if weights.ndim == 1:
        # Unpolarized case: weights shape is (nsrc,)
        n_trans = 1
        weights_reshaped = weights
    else:
        # Polarized case: weights shape is (nfeeds**2, nsrc)
        n_trans = weights.shape[0]
        weights_reshaped = weights

    # Use native Type 3 if available, otherwise use Plan-based fallback
    if HAS_NATIVE_TYPE3_SUPPORT:
        return _gpu_nufft2d_native_type3(x, y, weights_reshaped, u, v, eps, n_trans)
    else:
        return _gpu_nufft2d_plan_fallback(x, y, weights_reshaped, u, v, eps, n_trans)
def _gpu_nufft2d_native_type3(x, y, weights, u, v, eps, n_trans):
    """Native Type 3 NUFFT implementation using cufinufft.nufft2d3()."""
    try:
        # Validate inputs
        if len(x) == 0 or len(u) == 0:
            # Handle empty input case
            if n_trans == 1:
                return cp.zeros(len(u), dtype=cp.complex128)
            else:
                return cp.zeros((n_trans, len(u)), dtype=cp.complex128)

        if n_trans == 1:
            result = cufinufft.nufft2d3(x, y, weights, u, v, eps=eps, modeord=0)
        else:
            result = cufinufft.nufft2d3(x, y, weights, u, v, eps=eps, modeord=0)

        # Validate output
        if result is None or (hasattr(result, 'size') and result.size == 0):
            raise RuntimeError("cufinufft.nufft2d3 returned empty result")

        return result

    except (AttributeError, Exception):
        return _gpu_nufft2d_plan_fallback(x, y, weights, u, v, eps, n_trans)
def _gpu_nufft2d_plan_fallback(x, y, weights, u, v, eps, n_trans):
    """Plan-based fallback implementation for 2D Type 3 NUFFT."""
    # Handle empty input case
    if len(x) == 0 or len(u) == 0:
        if n_trans == 1:
            return cp.zeros(len(u), dtype=cp.complex128)
        else:
            return cp.zeros((n_trans, len(u)), dtype=cp.complex128)

    try:
        if n_trans > 1:
            results = []
            for i in range(n_trans):
                plan = cufinufft.Plan(
                    nufft_type=3,
                    n_modes=2,
                    n_trans=1,
                    eps=eps,
                    dtype='complex128',
                    modeord=0
                )
                plan.setpts(x, y, s=u, t=v)
                result = plan.execute(weights[i])
                results.append(result)
            result = cp.stack(results)
        else:
            plan = cufinufft.Plan(
                nufft_type=3,
                n_modes=2,
                n_trans=1,
                eps=eps,
                dtype='complex128',
                modeord=0
            )
            plan.setpts(x, y, s=u, t=v)
            result = plan.execute(weights)
        return result

    except Exception as e:
        raise RuntimeError(
            f"GPU 2D Type 3 NUFFT failed: {e}\n"
            "Please install the latest beta versions:\n"
            "  pip install cufinufft==2.4.0b1\n"
            "  pip install finufft==2.4.0rc1"
        )
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

    # Ensure all inputs are contiguous CuPy arrays
    x = cp.ascontiguousarray(x, dtype=cp.float64)
    y = cp.ascontiguousarray(y, dtype=cp.float64)
    z = cp.ascontiguousarray(z, dtype=cp.float64)
    u = cp.ascontiguousarray(u, dtype=cp.float64)
    v = cp.ascontiguousarray(v, dtype=cp.float64)
    w = cp.ascontiguousarray(w, dtype=cp.float64)
    weights = cp.ascontiguousarray(weights, dtype=cp.complex128)

    # Handle different weight shapes (polarized vs unpolarized)
    if weights.ndim == 1:
        # Unpolarized case: weights shape is (nsrc,)
        n_trans = 1
        weights_reshaped = weights
    else:
        # Polarized case: weights shape is (nfeeds**2, nsrc)
        n_trans = weights.shape[0]
        weights_reshaped = weights

    # Use native Type 3 if available, otherwise use Plan-based fallback
    if HAS_NATIVE_TYPE3_SUPPORT:
        return _gpu_nufft3d_native_type3(x, y, z, weights_reshaped, u, v, w, eps, n_trans)
    else:
        return _gpu_nufft3d_plan_fallback(x, y, z, weights_reshaped, u, v, w, eps, n_trans)
def _gpu_nufft3d_native_type3(x, y, z, weights, u, v, w, eps, n_trans):
    """Native Type 3 NUFFT implementation using cufinufft.nufft3d3()."""
    try:
        # Validate inputs
        if len(x) == 0 or len(u) == 0:
            # Handle empty input case
            if n_trans == 1:
                return cp.zeros(len(u), dtype=cp.complex128)
            else:
                return cp.zeros((n_trans, len(u)), dtype=cp.complex128)

        if n_trans == 1:
            result = cufinufft.nufft3d3(x, y, z, weights, u, v, w, eps=eps, modeord=0)
        else:
            result = cufinufft.nufft3d3(x, y, z, weights, u, v, w, eps=eps, modeord=0)

        # Validate output
        if result is None or (hasattr(result, 'size') and result.size == 0):
            raise RuntimeError("cufinufft.nufft3d3 returned empty result")

        return result

    except (AttributeError, Exception):
        return _gpu_nufft3d_plan_fallback(x, y, z, weights, u, v, w, eps, n_trans)
def _gpu_nufft3d_plan_fallback(x, y, z, weights, u, v, w, eps, n_trans):
    """Plan-based fallback implementation for 3D Type 3 NUFFT."""
    # Handle empty input case
    if len(x) == 0 or len(u) == 0:
        if n_trans == 1:
            return cp.zeros(len(u), dtype=cp.complex128)
        else:
            return cp.zeros((n_trans, len(u)), dtype=cp.complex128)

    try:
        if n_trans > 1:
            results = []
            for i in range(n_trans):
                plan = cufinufft.Plan(
                    nufft_type=3,
                    n_modes=3,
                    n_trans=1,
                    eps=eps,
                    dtype='complex128',
                    modeord=0
                )
                plan.setpts(x, y, z, u, v, w)
                result = plan.execute(weights[i])
                results.append(result)
            result = cp.stack(results)
        else:
            plan = cufinufft.Plan(
                nufft_type=3,
                n_modes=3,
                n_trans=1,
                eps=eps,
                dtype='complex128',
                modeord=0
            )
            plan.setpts(x, y, z, u, v, w)
            result = plan.execute(weights)
        return result

    except Exception as e:
        raise RuntimeError(
            f"GPU 3D Type 3 NUFFT failed: {e}\n"
            "Please install the latest beta versions:\n"
            "  pip install cufinufft==2.4.0b1\n"
            "  pip install finufft==2.4.0rc1"
        )


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
    
    This function processes multiple frequencies simultaneously using cufinufft's
    n_trans parameter for improved performance.
    
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
    n_bls = u_batch.shape[1]
    
    # Handle polarized vs unpolarized case
    if weights_batch.ndim == 3:
        # Polarized: (n_freq, nfeeds**2, nsrc)
        n_pol = weights_batch.shape[1]
        is_polarized = True
    else:
        # Unpolarized: (n_freq, nsrc)
        n_pol = 1
        is_polarized = False
    
    # Process all frequencies together
    results = []
    
    # For each frequency in the batch
    for freq_idx in range(n_freq):
        if is_polarized:
            # Process each polarization component
            freq_results = []
            for pol_idx in range(n_pol):
                vis = gpu_nufft2d(
                    x, y,
                    weights_batch[freq_idx, pol_idx, :],
                    u_batch[freq_idx, :],
                    v_batch[freq_idx, :],
                    eps=eps,
                    n_threads=n_threads
                )
                freq_results.append(vis)
            # Stack polarization results
            results.append(cp.stack(freq_results))
        else:
            # Unpolarized case
            vis = gpu_nufft2d(
                x, y,
                weights_batch[freq_idx, :],
                u_batch[freq_idx, :],
                v_batch[freq_idx, :],
                eps=eps,
                n_threads=n_threads
            )
            results.append(vis)
    
    # Stack frequency results
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
    
    This function processes multiple frequencies simultaneously using cufinufft's
    n_trans parameter for improved performance.
    
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
    n_bls = u_batch.shape[1]
    
    # Handle polarized vs unpolarized case
    if weights_batch.ndim == 3:
        # Polarized: (n_freq, nfeeds**2, nsrc)
        n_pol = weights_batch.shape[1]
        is_polarized = True
    else:
        # Unpolarized: (n_freq, nsrc)
        n_pol = 1
        is_polarized = False
    
    # Process all frequencies together
    results = []
    
    # For each frequency in the batch
    for freq_idx in range(n_freq):
        if is_polarized:
            # Process each polarization component
            freq_results = []
            for pol_idx in range(n_pol):
                vis = gpu_nufft3d(
                    x, y, z,
                    weights_batch[freq_idx, pol_idx, :],
                    u_batch[freq_idx, :],
                    v_batch[freq_idx, :],
                    w_batch[freq_idx, :],
                    eps=eps,
                    n_threads=n_threads
                )
                freq_results.append(vis)
            # Stack polarization results
            results.append(cp.stack(freq_results))
        else:
            # Unpolarized case
            vis = gpu_nufft3d(
                x, y, z,
                weights_batch[freq_idx, :],
                u_batch[freq_idx, :],
                v_batch[freq_idx, :],
                w_batch[freq_idx, :],
                eps=eps,
                n_threads=n_threads
            )
            results.append(vis)
    
    # Stack frequency results
    return cp.stack(results)
