import pytest
import numpy as np
from fftvis.gpu.beams import GPUBeamEvaluator
from fftvis.gpu.utils import inplace_rot


def test_gpu_beam_evaluator_init():
    """Test that the GPUBeamEvaluator initializes correctly."""
    # Create a GPU evaluator
    evaluator = GPUBeamEvaluator()
    
    # Check initialization values
    assert evaluator.beam_list == []
    assert evaluator.beam_idx is None
    assert evaluator.polarized is False
    assert evaluator.nant == 0
    assert evaluator.freq == 0.0
    assert evaluator.nsrc == 0
    assert evaluator.precision == 2


def test_gpu_beam_evaluator_not_implemented():
    """Test that the GPUBeamEvaluator methods raise NotImplementedError."""
    # Create a GPU evaluator
    evaluator = GPUBeamEvaluator()
    
    # Test evaluate_beam
    with pytest.raises(NotImplementedError):
        evaluator.evaluate_beam(
            beam=None,
            az=np.array([0.0]),
            za=np.array([0.0]),
            polarized=False,
            freq=150e6,
        )
    
    # Test get_apparent_flux_polarized
    with pytest.raises(NotImplementedError):
        evaluator.get_apparent_flux_polarized(
            beam=np.array([[[1.0]]]),
            flux=np.array([1.0]),
        )


def test_gpu_inplace_rot_not_implemented():
    """Test that the GPU inplace_rot function raises NotImplementedError."""
    # Create sample inputs
    rot = np.eye(3)
    b = np.zeros((3, 10))
    
    # Test that it raises NotImplementedError
    with pytest.raises(NotImplementedError):
        inplace_rot(rot, b) 