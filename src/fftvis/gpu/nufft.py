"""
GPU-specific non-uniform FFT implementation for fftvis.

This module provides GPU-specific NUFFT functionality using the cufinufft library.
"""

import cupy as cp
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

    except Exception:
        return _gpu_nufft2d_type1_type2_fallback(x, y, weights, u, v, eps, n_trans)
def _gpu_nufft2d_type1_type2_fallback(x, y, weights, u, v, eps, n_trans):
    """Type 1+2 decomposition fallback for 2D Type 3 NUFFT."""
    try:
        import math

        if eps < 1e-10:
            base_size = 256
        elif eps < 1e-8:
            base_size = 128
        else:
            base_size = 64

        max_points = max(len(x), len(u))
        scale_factor = max(1.0, math.sqrt(max_points / 50.0))

        N1 = int(base_size * scale_factor)
        N2 = int(base_size * scale_factor)

        N1 = max(64, 2 ** int(math.ceil(math.log2(N1))))
        N2 = max(64, 2 ** int(math.ceil(math.log2(N2))))

        # Limit maximum size to avoid memory issues
        N1 = min(2048, N1)
        N2 = min(2048, N2)

        # Step 1: Create and execute Type 1 plan (nonuniform -> uniform)
        plan1 = cufinufft.Plan(
            nufft_type=1,
            n_modes=(N1, N2),
            n_trans=n_trans,
            eps=eps,
            dtype='complex128',
            isign=1,        # finufft default for Type 1
            modeord=0       # Match CPU setting
        )
        plan1.setpts(x, y)

        # Execute Type 1 transform
        grid = plan1.execute(weights)

        # Debug info

        # Step 2: Create and execute Type 2 plan (uniform -> nonuniform)
        plan2 = cufinufft.Plan(
            nufft_type=2,
            n_modes=(N1, N2),
            n_trans=n_trans,
            eps=eps,
            dtype='complex128',
            isign=-1,       # finufft default for Type 2
            modeord=0       # Match CPU setting
        )
        plan2.setpts(u, v)

        # Execute Type 2 transform
        result = plan2.execute(grid)

        return result

    except Exception as e:
        raise RuntimeError(f"GPU 2D Type 3 NUFFT failed: {e}")
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

    except Exception:
        return _gpu_nufft3d_plan_type1_type2_fallback(x, y, z, weights, u, v, w, eps, n_trans)
def _gpu_nufft3d_plan_type1_type2_fallback(x, y, z, weights, u, v, w, eps, n_trans):
    """Type 1+2 decomposition fallback for 3D Type 3 NUFFT."""
    try:
        import math

        if eps < 1e-10:
            base_size = 128
        elif eps < 1e-8:
            base_size = 96
        else:
            base_size = 64

        # Scale with number of points (more points need larger grids)
        max_points = max(len(x), len(u))
        scale_factor = max(1.0, math.sqrt(max_points / 50.0))

        # Calculate grid sizes
        N1 = int(base_size * scale_factor)
        N2 = int(base_size * scale_factor)
        N3 = int(base_size * scale_factor)

        # Make power of 2 for FFT efficiency and ensure minimum size
        N1 = max(64, 2 ** int(math.ceil(math.log2(N1))))
        N2 = max(64, 2 ** int(math.ceil(math.log2(N2))))
        N3 = max(64, 2 ** int(math.ceil(math.log2(N3))))

        # Limit maximum size to avoid memory issues (3D grids can get huge!)
        N1 = min(512, N1)
        N2 = min(512, N2)
        N3 = min(512, N3)

        # Step 1: Create and execute Type 1 plan (nonuniform -> uniform)
        # Match CPU parameters: modeord=0, appropriate isign
        plan1 = cufinufft.Plan(
            nufft_type=1,
            n_modes=(N1, N2, N3),
            n_trans=n_trans,
            eps=eps,
            dtype='complex128',
            isign=1,        # finufft default for Type 1
            modeord=0       # Match CPU setting
        )
        plan1.setpts(x, y, z)

        # Execute Type 1 transform
        grid = plan1.execute(weights)

        # Debug info

        # Step 2: Create and execute Type 2 plan (uniform -> nonuniform)
        # Match CPU parameters: modeord=0, appropriate isign
        plan2 = cufinufft.Plan(
            nufft_type=2,
            n_modes=(N1, N2, N3),
            n_trans=n_trans,
            eps=eps,
            dtype='complex128',
            isign=-1,       # finufft default for Type 2
            modeord=0       # Match CPU setting
        )
        plan2.setpts(u, v, w)

        # Execute Type 2 transform
        result = plan2.execute(grid)

        return result

    except Exception as e:
        raise RuntimeError(f"GPU 3D Type 3 NUFFT failed: {e}")
