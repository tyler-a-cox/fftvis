import pytest
import numpy as np
from fftvis.gpu.gpu_nufft import gpu_nufft2d, gpu_nufft3d
from fftvis.gpu.gpu_simulate import GPUSimulationEngine


def test_gpu_nufft2d_not_implemented():
    """Test that gpu_nufft2d raises NotImplementedError."""
    # Create sample inputs
    x = np.array([0.0])
    y = np.array([0.0])
    data = np.array([[1.0]])
    u = np.array([0.0])
    v = np.array([0.0])
    eps = 1e-12  # Accuracy
    
    # Test that it raises NotImplementedError
    with pytest.raises(NotImplementedError):
        gpu_nufft2d(x, y, data, u, v, eps=eps)


def test_gpu_nufft3d_not_implemented():
    """Test that gpu_nufft3d raises NotImplementedError."""
    # Create sample inputs
    x = np.array([0.0])
    y = np.array([0.0])
    z = np.array([0.0])
    data = np.array([[1.0]])
    u = np.array([0.0])
    v = np.array([0.0])
    w = np.array([0.0])
    eps = 1e-12  # Accuracy
    
    # Test that it raises NotImplementedError
    with pytest.raises(NotImplementedError):
        gpu_nufft3d(x, y, z, data, u, v, w, eps=eps)


def test_gpu_simulation_engine_init():
    """Test that GPUSimulationEngine initializes correctly."""
    # Create a GPU simulation engine
    engine = GPUSimulationEngine()
    
    # It should inherit from the abstract base class
    from fftvis.core.simulate import SimulationEngine
    assert isinstance(engine, SimulationEngine)


def test_gpu_simulation_engine_not_implemented():
    """Test that GPUSimulationEngine methods raise NotImplementedError."""
    # Create a GPU simulation engine
    engine = GPUSimulationEngine()
    
    # Test that simulate raises NotImplementedError
    with pytest.raises(NotImplementedError):
        engine.simulate(
            ants={},
            freqs=np.array([]),
            fluxes=np.array([]),
            beam=None,
            ra=np.array([]),
            dec=np.array([]),
            times=np.array([]),
            telescope_loc=None
        ) 