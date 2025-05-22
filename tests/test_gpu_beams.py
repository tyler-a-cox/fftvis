import pytest
import numpy as np

# Skip all GPU tests if CuPy is not available
cupy = pytest.importorskip("cupy")

from fftvis.gpu.gpu_beams import GPUBeamEvaluator
from fftvis.gpu.gpu_utils import inplace_rot


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


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_beam_evaluator_get_apparent_flux_polarized():
    """Test that the GPUBeamEvaluator get_apparent_flux_polarized works."""
    # Create a GPU evaluator
    evaluator = GPUBeamEvaluator()

    # Create test data on GPU
    nax, nfd, nsrc = 2, 2, 3
    beam = cupy.random.random((nax, nfd, nsrc)) + 1j * cupy.random.random((nax, nfd, nsrc))
    flux = cupy.random.random(nsrc)

    # Make a copy to compare
    beam_original = beam.copy()

    # Test get_apparent_flux_polarized
    evaluator.get_apparent_flux_polarized(beam, flux)

    # Check that beam was modified (should be different from original)
    assert not cupy.allclose(beam, beam_original)

    # Check that the shape is preserved
    assert beam.shape == (nax, nfd, nsrc)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_gpu_inplace_rot():
    """Test that the GPU inplace_rot function works correctly."""
    # Create sample inputs on GPU
    rot = cupy.eye(3, dtype=cupy.float64)
    b = cupy.random.random((3, 10), dtype=cupy.float64)

    # Make a copy to compare
    b_original = b.copy()

    # Test inplace rotation with identity matrix (should not change b)
    inplace_rot(rot, b)

    # With identity matrix, b should be unchanged
    assert cupy.allclose(b, b_original)

    # Test with a 90-degree rotation around z-axis
    rot_z = cupy.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=cupy.float64)
    b_test = cupy.array([[1.0], [0.0], [0.0]], dtype=cupy.float64)

    inplace_rot(rot_z, b_test)

    # After 90-degree rotation around z, [1,0,0] should become [0,1,0]
    expected = cupy.array([[0.0], [1.0], [0.0]], dtype=cupy.float64)
    assert cupy.allclose(b_test, expected, atol=1e-10)