import pytest
import numpy as np

# Skip all GPU tests if CuPy is not available
cupy = pytest.importorskip("cupy")
cufinufft = pytest.importorskip("cufinufft")

from fftvis.gpu.nufft import (
    gpu_nufft2d,
    gpu_nufft3d,
    gpu_nufft2d_batch,
    gpu_nufft3d_batch,
    HAVE_CUFINUFFT,
    HAS_NATIVE_TYPE3_SUPPORT,
    _gpu_nufft2d_plan_fallback,
    _gpu_nufft3d_plan_fallback,
)
from fftvis.gpu.gpu_simulate import GPUSimulationEngine


# ============================================================================
# Basic NUFFT Tests (2D)
# ============================================================================

@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft2d_basic():
    """Test that gpu_nufft2d works with basic inputs."""
    # Create simple test case on GPU
    x = cupy.array([0.0, 1.0])
    y = cupy.array([0.0, 0.0])
    weights = cupy.array([1.0 + 0j, 1.0 + 0j])
    u = cupy.array([0.0, 1.0])
    v = cupy.array([0.0, 0.0])
    eps = 1e-6

    # Test that it runs without error
    result = gpu_nufft2d(x, y, weights, u, v, eps=eps)

    # Check that result is a cupy array
    assert isinstance(result, cupy.ndarray)
    assert result.dtype == cupy.complex64 or result.dtype == cupy.complex128
    assert result.shape == (len(u),)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft2d_empty_sources():
    """Test gpu_nufft2d with empty source arrays."""
    x = cupy.array([], dtype=cupy.float64)
    y = cupy.array([], dtype=cupy.float64)
    weights = cupy.array([], dtype=cupy.complex128)
    u = cupy.array([0.0, 1.0])
    v = cupy.array([0.0, 0.0])
    eps = 1e-6

    result = gpu_nufft2d(x, y, weights, u, v, eps=eps)

    # Should return zeros for empty input
    assert isinstance(result, cupy.ndarray)
    assert result.shape == (len(u),)
    assert cupy.allclose(result, 0.0)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft2d_empty_baselines():
    """Test gpu_nufft2d with empty baseline arrays."""
    x = cupy.array([0.0, 1.0])
    y = cupy.array([0.0, 0.0])
    weights = cupy.array([1.0 + 0j, 1.0 + 0j])
    u = cupy.array([], dtype=cupy.float64)
    v = cupy.array([], dtype=cupy.float64)
    eps = 1e-6

    result = gpu_nufft2d(x, y, weights, u, v, eps=eps)

    # Should return empty array
    assert isinstance(result, cupy.ndarray)
    assert result.shape == (0,)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft2d_polarized_weights():
    """Test gpu_nufft2d with polarized weights (n_trans > 1)."""
    nsrc = 5
    nbls = 3
    nfeeds = 2
    n_trans = nfeeds ** 2  # 4 polarization components

    x = cupy.random.uniform(-1, 1, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-1, 1, nsrc).astype(cupy.float64)
    # Polarized weights have shape (n_trans, nsrc)
    weights = cupy.random.random((n_trans, nsrc)).astype(cupy.complex128)
    u = cupy.random.uniform(-1, 1, nbls).astype(cupy.float64)
    v = cupy.random.uniform(-1, 1, nbls).astype(cupy.float64)
    eps = 1e-6

    result = gpu_nufft2d(x, y, weights, u, v, eps=eps)

    assert isinstance(result, cupy.ndarray)
    # Result should have shape (n_trans, nbls) for polarized case
    assert result.shape == (n_trans, nbls)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft2d_larger_dataset():
    """Test gpu_nufft2d with a larger, more realistic dataset."""
    nsrc = 100
    nbls = 50

    x = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    weights = cupy.random.random(nsrc).astype(cupy.complex128) + 1j * cupy.random.random(nsrc)
    u = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    v = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    eps = 1e-6

    result = gpu_nufft2d(x, y, weights, u, v, eps=eps)

    assert isinstance(result, cupy.ndarray)
    assert result.shape == (nbls,)
    # Check result is finite
    assert cupy.all(cupy.isfinite(result))


# ============================================================================
# Basic NUFFT Tests (3D)
# ============================================================================

@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft3d_basic():
    """Test that gpu_nufft3d works with basic inputs."""
    # Create simple test case on GPU
    x = cupy.array([0.0, 1.0])
    y = cupy.array([0.0, 0.0])
    z = cupy.array([0.0, 0.0])
    weights = cupy.array([1.0 + 0j, 1.0 + 0j])
    u = cupy.array([0.0, 1.0])
    v = cupy.array([0.0, 0.0])
    w = cupy.array([0.0, 0.0])
    eps = 1e-6

    # Test that it runs without error
    result = gpu_nufft3d(x, y, z, weights, u, v, w, eps=eps)

    # Check that result is a cupy array
    assert isinstance(result, cupy.ndarray)
    assert result.dtype == cupy.complex64 or result.dtype == cupy.complex128
    assert result.shape == (len(u),)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft3d_empty_sources():
    """Test gpu_nufft3d with empty source arrays."""
    x = cupy.array([], dtype=cupy.float64)
    y = cupy.array([], dtype=cupy.float64)
    z = cupy.array([], dtype=cupy.float64)
    weights = cupy.array([], dtype=cupy.complex128)
    u = cupy.array([0.0, 1.0])
    v = cupy.array([0.0, 0.0])
    w = cupy.array([0.0, 0.0])
    eps = 1e-6

    result = gpu_nufft3d(x, y, z, weights, u, v, w, eps=eps)

    # Should return zeros for empty input
    assert isinstance(result, cupy.ndarray)
    assert result.shape == (len(u),)
    assert cupy.allclose(result, 0.0)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft3d_empty_baselines():
    """Test gpu_nufft3d with empty baseline arrays."""
    x = cupy.array([0.0, 1.0])
    y = cupy.array([0.0, 0.0])
    z = cupy.array([0.0, 0.0])
    weights = cupy.array([1.0 + 0j, 1.0 + 0j])
    u = cupy.array([], dtype=cupy.float64)
    v = cupy.array([], dtype=cupy.float64)
    w = cupy.array([], dtype=cupy.float64)
    eps = 1e-6

    result = gpu_nufft3d(x, y, z, weights, u, v, w, eps=eps)

    # Should return empty array
    assert isinstance(result, cupy.ndarray)
    assert result.shape == (0,)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft3d_polarized_weights():
    """Test gpu_nufft3d with polarized weights (n_trans > 1)."""
    nsrc = 5
    nbls = 3
    nfeeds = 2
    n_trans = nfeeds ** 2  # 4 polarization components

    x = cupy.random.uniform(-1, 1, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-1, 1, nsrc).astype(cupy.float64)
    z = cupy.random.uniform(-1, 1, nsrc).astype(cupy.float64)
    # Polarized weights have shape (n_trans, nsrc)
    weights = cupy.random.random((n_trans, nsrc)).astype(cupy.complex128)
    u = cupy.random.uniform(-1, 1, nbls).astype(cupy.float64)
    v = cupy.random.uniform(-1, 1, nbls).astype(cupy.float64)
    w = cupy.random.uniform(-1, 1, nbls).astype(cupy.float64)
    eps = 1e-6

    result = gpu_nufft3d(x, y, z, weights, u, v, w, eps=eps)

    assert isinstance(result, cupy.ndarray)
    # Result should have shape (n_trans, nbls) for polarized case
    assert result.shape == (n_trans, nbls)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft3d_larger_dataset():
    """Test gpu_nufft3d with a larger, more realistic dataset."""
    nsrc = 100
    nbls = 50

    x = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    z = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    weights = cupy.random.random(nsrc).astype(cupy.complex128) + 1j * cupy.random.random(nsrc)
    u = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    v = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    w = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    eps = 1e-6

    result = gpu_nufft3d(x, y, z, weights, u, v, w, eps=eps)

    assert isinstance(result, cupy.ndarray)
    assert result.shape == (nbls,)
    # Check result is finite
    assert cupy.all(cupy.isfinite(result))


# ============================================================================
# Plan-based Fallback Tests
# ============================================================================

@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft2d_plan_fallback_basic():
    """Test the plan-based fallback for 2D NUFFT directly."""
    nsrc = 10
    nbls = 5

    x = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    weights = cupy.random.random(nsrc).astype(cupy.complex128)
    u = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    v = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    eps = 1e-6
    n_trans = 1

    result = _gpu_nufft2d_plan_fallback(x, y, weights, u, v, eps, n_trans)

    assert isinstance(result, cupy.ndarray)
    assert result.shape == (nbls,)
    assert cupy.all(cupy.isfinite(result))


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft2d_plan_fallback_polarized():
    """Test the plan-based fallback for 2D NUFFT with polarized weights."""
    nsrc = 10
    nbls = 5
    n_trans = 4  # Polarized

    x = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    weights = cupy.random.random((n_trans, nsrc)).astype(cupy.complex128)
    u = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    v = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    eps = 1e-6

    result = _gpu_nufft2d_plan_fallback(x, y, weights, u, v, eps, n_trans)

    assert isinstance(result, cupy.ndarray)
    assert result.shape == (n_trans, nbls)
    assert cupy.all(cupy.isfinite(result))


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft2d_plan_fallback_empty():
    """Test the plan-based fallback for 2D NUFFT with empty inputs."""
    x = cupy.array([], dtype=cupy.float64)
    y = cupy.array([], dtype=cupy.float64)
    weights = cupy.array([], dtype=cupy.complex128)
    u = cupy.array([0.0, 1.0])
    v = cupy.array([0.0, 0.0])
    eps = 1e-6
    n_trans = 1

    result = _gpu_nufft2d_plan_fallback(x, y, weights, u, v, eps, n_trans)

    assert result.shape == (2,)
    assert cupy.allclose(result, 0.0)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft2d_plan_fallback_empty_polarized():
    """Test the plan-based fallback for 2D NUFFT with empty inputs (polarized)."""
    x = cupy.array([], dtype=cupy.float64)
    y = cupy.array([], dtype=cupy.float64)
    weights = cupy.zeros((4, 0), dtype=cupy.complex128)
    u = cupy.array([0.0, 1.0])
    v = cupy.array([0.0, 0.0])
    eps = 1e-6
    n_trans = 4

    result = _gpu_nufft2d_plan_fallback(x, y, weights, u, v, eps, n_trans)

    assert result.shape == (4, 2)
    assert cupy.allclose(result, 0.0)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft3d_plan_fallback_basic():
    """Test the plan-based fallback for 3D NUFFT directly."""
    nsrc = 10
    nbls = 5

    x = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    z = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    weights = cupy.random.random(nsrc).astype(cupy.complex128)
    u = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    v = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    w = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    eps = 1e-6
    n_trans = 1

    result = _gpu_nufft3d_plan_fallback(x, y, z, weights, u, v, w, eps, n_trans)

    assert isinstance(result, cupy.ndarray)
    assert result.shape == (nbls,)
    assert cupy.all(cupy.isfinite(result))


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft3d_plan_fallback_polarized():
    """Test the plan-based fallback for 3D NUFFT with polarized weights."""
    nsrc = 10
    nbls = 5
    n_trans = 4  # Polarized

    x = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    z = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    weights = cupy.random.random((n_trans, nsrc)).astype(cupy.complex128)
    u = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    v = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    w = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    eps = 1e-6

    result = _gpu_nufft3d_plan_fallback(x, y, z, weights, u, v, w, eps, n_trans)

    assert isinstance(result, cupy.ndarray)
    assert result.shape == (n_trans, nbls)
    assert cupy.all(cupy.isfinite(result))


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft3d_plan_fallback_empty():
    """Test the plan-based fallback for 3D NUFFT with empty inputs."""
    x = cupy.array([], dtype=cupy.float64)
    y = cupy.array([], dtype=cupy.float64)
    z = cupy.array([], dtype=cupy.float64)
    weights = cupy.array([], dtype=cupy.complex128)
    u = cupy.array([0.0, 1.0])
    v = cupy.array([0.0, 0.0])
    w = cupy.array([0.0, 0.0])
    eps = 1e-6
    n_trans = 1

    result = _gpu_nufft3d_plan_fallback(x, y, z, weights, u, v, w, eps, n_trans)

    assert result.shape == (2,)
    assert cupy.allclose(result, 0.0)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft3d_plan_fallback_empty_polarized():
    """Test the plan-based fallback for 3D NUFFT with empty inputs (polarized)."""
    x = cupy.array([], dtype=cupy.float64)
    y = cupy.array([], dtype=cupy.float64)
    z = cupy.array([], dtype=cupy.float64)
    weights = cupy.zeros((4, 0), dtype=cupy.complex128)
    u = cupy.array([0.0, 1.0])
    v = cupy.array([0.0, 0.0])
    w = cupy.array([0.0, 0.0])
    eps = 1e-6
    n_trans = 4

    result = _gpu_nufft3d_plan_fallback(x, y, z, weights, u, v, w, eps, n_trans)

    assert result.shape == (4, 2)
    assert cupy.allclose(result, 0.0)


# ============================================================================
# Batch NUFFT Tests (2D)
# ============================================================================

@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft2d_batch_unpolarized():
    """Test gpu_nufft2d_batch with unpolarized weights."""
    nsrc = 10
    nbls = 5
    n_freq = 3

    x = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    # Unpolarized weights: (n_freq, nsrc)
    weights_batch = cupy.random.random((n_freq, nsrc)).astype(cupy.complex128)
    # UV coordinates for each frequency: (n_freq, nbls)
    u_batch = cupy.random.uniform(-np.pi, np.pi, (n_freq, nbls)).astype(cupy.float64)
    v_batch = cupy.random.uniform(-np.pi, np.pi, (n_freq, nbls)).astype(cupy.float64)
    eps = 1e-6

    result = gpu_nufft2d_batch(x, y, weights_batch, u_batch, v_batch, eps=eps)

    assert isinstance(result, cupy.ndarray)
    # Unpolarized result: (n_freq, nbls)
    assert result.shape == (n_freq, nbls)
    assert cupy.all(cupy.isfinite(result))


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft2d_batch_polarized():
    """Test gpu_nufft2d_batch with polarized weights."""
    nsrc = 10
    nbls = 5
    n_freq = 3
    nfeeds = 2
    n_pol = nfeeds ** 2

    x = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    # Polarized weights: (n_freq, nfeeds**2, nsrc)
    weights_batch = cupy.random.random((n_freq, n_pol, nsrc)).astype(cupy.complex128)
    # UV coordinates for each frequency: (n_freq, nbls)
    u_batch = cupy.random.uniform(-np.pi, np.pi, (n_freq, nbls)).astype(cupy.float64)
    v_batch = cupy.random.uniform(-np.pi, np.pi, (n_freq, nbls)).astype(cupy.float64)
    eps = 1e-6

    result = gpu_nufft2d_batch(x, y, weights_batch, u_batch, v_batch, eps=eps)

    assert isinstance(result, cupy.ndarray)
    # Polarized result: (n_freq, nfeeds**2, nbls)
    assert result.shape == (n_freq, n_pol, nbls)
    assert cupy.all(cupy.isfinite(result))


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft2d_batch_single_frequency():
    """Test gpu_nufft2d_batch with a single frequency."""
    nsrc = 10
    nbls = 5
    n_freq = 1

    x = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    weights_batch = cupy.random.random((n_freq, nsrc)).astype(cupy.complex128)
    u_batch = cupy.random.uniform(-np.pi, np.pi, (n_freq, nbls)).astype(cupy.float64)
    v_batch = cupy.random.uniform(-np.pi, np.pi, (n_freq, nbls)).astype(cupy.float64)
    eps = 1e-6

    result = gpu_nufft2d_batch(x, y, weights_batch, u_batch, v_batch, eps=eps)

    assert result.shape == (n_freq, nbls)
    assert cupy.all(cupy.isfinite(result))


# ============================================================================
# Batch NUFFT Tests (3D)
# ============================================================================

@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft3d_batch_unpolarized():
    """Test gpu_nufft3d_batch with unpolarized weights."""
    nsrc = 10
    nbls = 5
    n_freq = 3

    x = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    z = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    # Unpolarized weights: (n_freq, nsrc)
    weights_batch = cupy.random.random((n_freq, nsrc)).astype(cupy.complex128)
    # UVW coordinates for each frequency: (n_freq, nbls)
    u_batch = cupy.random.uniform(-np.pi, np.pi, (n_freq, nbls)).astype(cupy.float64)
    v_batch = cupy.random.uniform(-np.pi, np.pi, (n_freq, nbls)).astype(cupy.float64)
    w_batch = cupy.random.uniform(-np.pi, np.pi, (n_freq, nbls)).astype(cupy.float64)
    eps = 1e-6

    result = gpu_nufft3d_batch(x, y, z, weights_batch, u_batch, v_batch, w_batch, eps=eps)

    assert isinstance(result, cupy.ndarray)
    # Unpolarized result: (n_freq, nbls)
    assert result.shape == (n_freq, nbls)
    assert cupy.all(cupy.isfinite(result))


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft3d_batch_polarized():
    """Test gpu_nufft3d_batch with polarized weights."""
    nsrc = 10
    nbls = 5
    n_freq = 3
    nfeeds = 2
    n_pol = nfeeds ** 2

    x = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    z = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    # Polarized weights: (n_freq, nfeeds**2, nsrc)
    weights_batch = cupy.random.random((n_freq, n_pol, nsrc)).astype(cupy.complex128)
    # UVW coordinates for each frequency: (n_freq, nbls)
    u_batch = cupy.random.uniform(-np.pi, np.pi, (n_freq, nbls)).astype(cupy.float64)
    v_batch = cupy.random.uniform(-np.pi, np.pi, (n_freq, nbls)).astype(cupy.float64)
    w_batch = cupy.random.uniform(-np.pi, np.pi, (n_freq, nbls)).astype(cupy.float64)
    eps = 1e-6

    result = gpu_nufft3d_batch(x, y, z, weights_batch, u_batch, v_batch, w_batch, eps=eps)

    assert isinstance(result, cupy.ndarray)
    # Polarized result: (n_freq, nfeeds**2, nbls)
    assert result.shape == (n_freq, n_pol, nbls)
    assert cupy.all(cupy.isfinite(result))


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft3d_batch_single_frequency():
    """Test gpu_nufft3d_batch with a single frequency."""
    nsrc = 10
    nbls = 5
    n_freq = 1

    x = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    z = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    weights_batch = cupy.random.random((n_freq, nsrc)).astype(cupy.complex128)
    u_batch = cupy.random.uniform(-np.pi, np.pi, (n_freq, nbls)).astype(cupy.float64)
    v_batch = cupy.random.uniform(-np.pi, np.pi, (n_freq, nbls)).astype(cupy.float64)
    w_batch = cupy.random.uniform(-np.pi, np.pi, (n_freq, nbls)).astype(cupy.float64)
    eps = 1e-6

    result = gpu_nufft3d_batch(x, y, z, weights_batch, u_batch, v_batch, w_batch, eps=eps)

    assert result.shape == (n_freq, nbls)
    assert cupy.all(cupy.isfinite(result))


# ============================================================================
# Module Constants and Flags Tests
# ============================================================================

def test_have_cufinufft_flag():
    """Test that HAVE_CUFINUFFT is set correctly."""
    # Since we importorskip cufinufft at the top, it should be True
    assert HAVE_CUFINUFFT is True


def test_native_type3_support_flag():
    """Test that HAS_NATIVE_TYPE3_SUPPORT is a boolean."""
    assert isinstance(HAS_NATIVE_TYPE3_SUPPORT, bool)


# ============================================================================
# GPU Simulation Engine Tests
# ============================================================================

def test_gpu_simulation_engine_init():
    """Test that GPUSimulationEngine initializes correctly."""
    # Create a GPU simulation engine
    engine = GPUSimulationEngine()

    # It should inherit from the abstract base class
    from fftvis.core.simulate import SimulationEngine
    assert isinstance(engine, SimulationEngine)


# ============================================================================
# Accuracy / Numerical Tests
# ============================================================================

@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft2d_accuracy_different_eps():
    """Test gpu_nufft2d with different accuracy levels."""
    nsrc = 20
    nbls = 10

    x = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    weights = cupy.random.random(nsrc).astype(cupy.complex128)
    u = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    v = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)

    # Test different accuracy levels
    result_high = gpu_nufft2d(x, y, weights, u, v, eps=1e-12)
    result_low = gpu_nufft2d(x, y, weights, u, v, eps=1e-3)

    # Both should be valid
    assert cupy.all(cupy.isfinite(result_high))
    assert cupy.all(cupy.isfinite(result_low))
    # Low accuracy may differ more significantly, but should be in same order of magnitude
    # The key test is that both produce valid, finite results
    assert result_high.shape == result_low.shape


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft3d_accuracy_different_eps():
    """Test gpu_nufft3d with different accuracy levels."""
    nsrc = 20
    nbls = 10

    x = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    z = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    weights = cupy.random.random(nsrc).astype(cupy.complex128)
    u = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    v = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    w = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)

    # Test different accuracy levels
    result_high = gpu_nufft3d(x, y, z, weights, u, v, w, eps=1e-12)
    result_low = gpu_nufft3d(x, y, z, weights, u, v, w, eps=1e-3)

    # Both should be valid
    assert cupy.all(cupy.isfinite(result_high))
    assert cupy.all(cupy.isfinite(result_low))
    # Low accuracy may differ more significantly, but should be in same order of magnitude
    assert result_high.shape == result_low.shape


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_nufft2d_vs_3d_flat_array():
    """Test that 2D and 3D NUFFT give same results for flat arrays (z=w=0)."""
    nsrc = 20
    nbls = 10

    x = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    y = cupy.random.uniform(-np.pi, np.pi, nsrc).astype(cupy.float64)
    z = cupy.zeros(nsrc, dtype=cupy.float64)
    weights = cupy.random.random(nsrc).astype(cupy.complex128)
    u = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    v = cupy.random.uniform(-np.pi, np.pi, nbls).astype(cupy.float64)
    w = cupy.zeros(nbls, dtype=cupy.float64)
    eps = 1e-6

    result_2d = gpu_nufft2d(x, y, weights, u, v, eps=eps)
    result_3d = gpu_nufft3d(x, y, z, weights, u, v, w, eps=eps)

    # Results should be very close for flat arrays
    assert cupy.allclose(result_2d, result_3d, rtol=1e-5)