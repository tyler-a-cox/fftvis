import pytest
import numpy as np

# Skip all GPU tests if CuPy is not available
cupy = pytest.importorskip("cupy")
cufinufft = pytest.importorskip("cufinufft")

from fftvis.gpu.gpu_nufft import gpu_nufft2d, gpu_nufft3d
from fftvis.gpu.gpu_simulate import GPUSimulationEngine


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


def test_gpu_simulation_engine_init():
    """Test that GPUSimulationEngine initializes correctly."""
    # Create a GPU simulation engine
    engine = GPUSimulationEngine()

    # It should inherit from the abstract base class
    from fftvis.core.simulate import SimulationEngine
    assert isinstance(engine, SimulationEngine)