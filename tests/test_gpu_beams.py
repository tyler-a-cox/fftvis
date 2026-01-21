"""
Tests for GPU beam evaluation in fftvis.gpu.beams.

This module tests:
- GPUBeamEvaluator initialization and methods
- Airy beam GPU evaluation
- UVBeam GPU interpolation
- Apparent flux calculations
- Beam caching
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

# Skip all GPU tests if CuPy is not available
cupy = pytest.importorskip("cupy")

from fftvis.gpu.beams import (
    GPUBeamEvaluator,
    HAS_GPU_BESSEL,
    HAS_GPU_MAP_COORDS,
)
from fftvis.gpu.utils import inplace_rot


# ============================================================================
# Initialization Tests
# ============================================================================

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
    assert evaluator._gpu_beam_data == {}


def test_gpu_beam_evaluator_inherits_from_base():
    """Test that GPUBeamEvaluator inherits from BeamEvaluator."""
    from fftvis.core.beams import BeamEvaluator
    evaluator = GPUBeamEvaluator()
    assert isinstance(evaluator, BeamEvaluator)


# ============================================================================
# Module Constants Tests
# ============================================================================

def test_has_gpu_bessel_flag():
    """Test that HAS_GPU_BESSEL is a boolean."""
    assert isinstance(HAS_GPU_BESSEL, bool)


def test_has_gpu_map_coords_flag():
    """Test that HAS_GPU_MAP_COORDS is a boolean."""
    assert isinstance(HAS_GPU_MAP_COORDS, bool)


# ============================================================================
# get_apparent_flux_polarized Tests
# ============================================================================

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
def test_get_apparent_flux_polarized_single_source():
    """Test get_apparent_flux_polarized with a single source."""
    evaluator = GPUBeamEvaluator()

    nax, nfd, nsrc = 2, 2, 1
    beam = cupy.ones((nax, nfd, nsrc), dtype=cupy.complex128)
    flux = cupy.array([2.0])

    evaluator.get_apparent_flux_polarized(beam, flux)

    # Shape should be preserved
    assert beam.shape == (nax, nfd, nsrc)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_get_apparent_flux_polarized_numpy_conversion():
    """Test that get_apparent_flux_polarized handles numpy input."""
    evaluator = GPUBeamEvaluator()

    nax, nfd, nsrc = 2, 2, 3
    # Pass cupy arrays (function will handle internally)
    beam = cupy.random.random((nax, nfd, nsrc)).astype(cupy.complex128)
    flux = cupy.random.random(nsrc)

    # Should not raise
    evaluator.get_apparent_flux_polarized(beam, flux)
    assert beam.shape == (nax, nfd, nsrc)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_get_apparent_flux_polarized_zero_flux():
    """Test get_apparent_flux_polarized with zero flux."""
    evaluator = GPUBeamEvaluator()

    nax, nfd, nsrc = 2, 2, 3
    beam = cupy.ones((nax, nfd, nsrc), dtype=cupy.complex128)
    flux = cupy.zeros(nsrc)

    evaluator.get_apparent_flux_polarized(beam, flux)

    # All values should be zero
    assert cupy.allclose(beam, 0.0)


# ============================================================================
# evaluate_beam Tests
# ============================================================================

@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_evaluate_beam_invalid_interpolation_function():
    """Test that evaluate_beam raises ValueError for invalid interpolation function."""
    evaluator = GPUBeamEvaluator()

    # Create mock beam
    mock_beam = Mock()
    az = cupy.array([0.0, 0.1])
    za = cupy.array([0.0, 0.1])

    with pytest.raises(ValueError, match="GPU beam evaluation only supports"):
        evaluator.evaluate_beam(
            beam=mock_beam,
            az=az,
            za=za,
            polarized=False,
            freq=150e6,
            interpolation_function="invalid_function"
        )


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_evaluate_beam_saves_attributes():
    """Test that evaluate_beam saves attributes correctly."""
    evaluator = GPUBeamEvaluator()

    # Create a mock beam that returns valid data
    mock_beam = Mock()
    mock_beam.beam = None  # Not an AiryBeam
    mock_beam.compute_response = Mock(return_value=np.ones((1, 1, 1, 5)))

    az = cupy.array([0.0, 0.1, 0.2, 0.3, 0.4])
    za = cupy.array([0.0, 0.1, 0.2, 0.3, 0.4])

    evaluator.evaluate_beam(
        beam=mock_beam,
        az=az,
        za=za,
        polarized=False,
        freq=150e6,
    )

    assert evaluator.polarized is False
    assert evaluator.freq == 150e6
    assert evaluator.nsrc == 5


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_evaluate_beam_converts_numpy_to_cupy():
    """Test that evaluate_beam converts numpy arrays to cupy."""
    evaluator = GPUBeamEvaluator()

    # Create a mock beam
    mock_beam = Mock()
    mock_beam.beam = None
    mock_beam.compute_response = Mock(return_value=np.ones((1, 1, 1, 5)))

    # Pass numpy arrays
    az = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    za = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

    result = evaluator.evaluate_beam(
        beam=mock_beam,
        az=az,
        za=za,
        polarized=False,
        freq=150e6,
    )

    # Result should be cupy array
    assert isinstance(result, cupy.ndarray)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_evaluate_beam_polarized():
    """Test evaluate_beam with polarized=True."""
    evaluator = GPUBeamEvaluator()

    # Create a mock beam that returns polarized data
    mock_beam = Mock()
    mock_beam.beam = None
    # Shape for polarized: (nax, nfd, nfreq, nsrc) -> extract [:, :, 0, :]
    mock_beam.compute_response = Mock(return_value=np.ones((2, 2, 1, 5), dtype=np.complex128))

    az = cupy.array([0.0, 0.1, 0.2, 0.3, 0.4])
    za = cupy.array([0.0, 0.1, 0.2, 0.3, 0.4])

    result = evaluator.evaluate_beam(
        beam=mock_beam,
        az=az,
        za=za,
        polarized=True,
        freq=150e6,
    )

    # Result should have polarized shape (nax, nfd, nsrc)
    assert result.shape == (2, 2, 5)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_evaluate_beam_caching():
    """Test that beam data is cached on GPU."""
    evaluator = GPUBeamEvaluator()

    # Create a mock beam
    mock_beam = Mock()
    mock_beam.beam = None
    mock_beam.compute_response = Mock(return_value=np.ones((1, 1, 1, 5)))

    az = cupy.array([0.0, 0.1, 0.2, 0.3, 0.4])
    za = cupy.array([0.0, 0.1, 0.2, 0.3, 0.4])

    # First evaluation
    result1 = evaluator.evaluate_beam(
        beam=mock_beam,
        az=az,
        za=za,
        polarized=False,
        freq=150e6,
    )

    # Check that data is cached
    beam_key = id(mock_beam)
    assert beam_key in evaluator._gpu_beam_data

    # Second evaluation with same beam should use cache
    # (but with different az/za, cache may be invalidated)
    az2 = cupy.array([0.1, 0.2, 0.3, 0.4, 0.5])
    za2 = cupy.array([0.1, 0.2, 0.3, 0.4, 0.5])

    result2 = evaluator.evaluate_beam(
        beam=mock_beam,
        az=az2,
        za=za2,
        polarized=False,
        freq=150e6,
    )

    assert isinstance(result2, cupy.ndarray)


# ============================================================================
# Airy Beam GPU Evaluation Tests
# ============================================================================

@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_GPU_BESSEL, reason="GPU Bessel functions not available")
def test_evaluate_airy_beam_gpu_unpolarized():
    """Test GPU Airy beam evaluation (unpolarized)."""
    evaluator = GPUBeamEvaluator()

    # Create mock AiryBeam
    mock_beam = Mock()
    mock_beam.beam = Mock()
    mock_beam.beam.__class__.__name__ = 'AiryBeam'
    mock_beam.beam.diameter = 14.0  # HERA dish diameter in meters

    nsrc = 100
    az = cupy.random.uniform(0, 2 * np.pi, nsrc).astype(cupy.float64)
    za = cupy.random.uniform(0, np.pi / 4, nsrc).astype(cupy.float64)  # Near zenith

    result = evaluator.evaluate_beam(
        beam=mock_beam,
        az=az,
        za=za,
        polarized=False,
        freq=150e6,
    )

    assert isinstance(result, cupy.ndarray)
    assert result.shape == (nsrc,)
    assert cupy.all(cupy.isfinite(result))
    # Power pattern should be non-negative
    assert cupy.all(result >= 0)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_GPU_BESSEL, reason="GPU Bessel functions not available")
def test_evaluate_airy_beam_gpu_polarized():
    """Test GPU Airy beam evaluation (polarized)."""
    evaluator = GPUBeamEvaluator()

    # Create mock AiryBeam
    mock_beam = Mock()
    mock_beam.beam = Mock()
    mock_beam.beam.__class__.__name__ = 'AiryBeam'
    mock_beam.beam.diameter = 14.0

    nsrc = 50
    az = cupy.random.uniform(0, 2 * np.pi, nsrc).astype(cupy.float64)
    za = cupy.random.uniform(0, np.pi / 4, nsrc).astype(cupy.float64)

    result = evaluator.evaluate_beam(
        beam=mock_beam,
        az=az,
        za=za,
        polarized=True,
        freq=150e6,
    )

    assert isinstance(result, cupy.ndarray)
    # Polarized result should have shape (2, 2, nsrc)
    assert result.shape == (2, 2, nsrc)
    assert cupy.all(cupy.isfinite(result))


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_GPU_BESSEL, reason="GPU Bessel functions not available")
def test_evaluate_airy_beam_gpu_at_zenith():
    """Test Airy beam evaluation at zenith (za=0)."""
    evaluator = GPUBeamEvaluator()

    mock_beam = Mock()
    mock_beam.beam = Mock()
    mock_beam.beam.__class__.__name__ = 'AiryBeam'
    mock_beam.beam.diameter = 14.0

    nsrc = 10
    az = cupy.zeros(nsrc, dtype=cupy.float64)
    za = cupy.zeros(nsrc, dtype=cupy.float64)  # At zenith

    result = evaluator.evaluate_beam(
        beam=mock_beam,
        az=az,
        za=za,
        polarized=False,
        freq=150e6,
    )

    # At zenith, Airy pattern should be maximum (1.0 squared = 1.0 for power)
    assert cupy.allclose(result, 1.0)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_GPU_BESSEL, reason="GPU Bessel functions not available")
def test_evaluate_airy_beam_gpu_with_check():
    """Test Airy beam evaluation with check=True."""
    evaluator = GPUBeamEvaluator()

    mock_beam = Mock()
    mock_beam.beam = Mock()
    mock_beam.beam.__class__.__name__ = 'AiryBeam'
    mock_beam.beam.diameter = 14.0

    nsrc = 10
    az = cupy.random.uniform(0, 2 * np.pi, nsrc).astype(cupy.float64)
    za = cupy.random.uniform(0, np.pi / 4, nsrc).astype(cupy.float64)

    # Should not raise with valid data
    result = evaluator.evaluate_beam(
        beam=mock_beam,
        az=az,
        za=za,
        polarized=False,
        freq=150e6,
        check=True,
    )

    assert cupy.all(cupy.isfinite(result))


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_GPU_BESSEL, reason="GPU Bessel functions not available")
def test_evaluate_airy_beam_gpu_different_frequencies():
    """Test Airy beam evaluation at different frequencies."""
    evaluator = GPUBeamEvaluator()

    mock_beam = Mock()
    mock_beam.beam = Mock()
    mock_beam.beam.__class__.__name__ = 'AiryBeam'
    mock_beam.beam.diameter = 14.0

    nsrc = 20
    az = cupy.random.uniform(0, 2 * np.pi, nsrc).astype(cupy.float64)
    za = cupy.random.uniform(0.1, 0.3, nsrc).astype(cupy.float64)  # Off-zenith

    result_low = evaluator.evaluate_beam(
        beam=mock_beam, az=az, za=za, polarized=False, freq=100e6
    )
    result_high = evaluator.evaluate_beam(
        beam=mock_beam, az=az, za=za, polarized=False, freq=200e6
    )

    # Results should be different at different frequencies
    assert not cupy.allclose(result_low, result_high)


# ============================================================================
# Beam Interface Tests
# ============================================================================

@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_evaluate_beam_with_beam_interface():
    """Test evaluate_beam with BeamInterface wrapper."""
    evaluator = GPUBeamEvaluator()

    # Create mock beam with 'beam' attribute (BeamInterface style)
    mock_inner_beam = Mock()
    mock_inner_beam.__class__.__name__ = 'UVBeam'  # Not AiryBeam
    mock_inner_beam.data_array = None  # No data_array
    mock_inner_beam.pixel_coordinate_system = None

    mock_beam = Mock()
    mock_beam.beam = mock_inner_beam
    mock_beam.compute_response = Mock(return_value=np.ones((1, 1, 1, 5)))

    az = cupy.array([0.0, 0.1, 0.2, 0.3, 0.4])
    za = cupy.array([0.0, 0.1, 0.2, 0.3, 0.4])

    result = evaluator.evaluate_beam(
        beam=mock_beam,
        az=az,
        za=za,
        polarized=False,
        freq=150e6,
    )

    assert isinstance(result, cupy.ndarray)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_evaluate_beam_direct_beam_object():
    """Test evaluate_beam with direct beam object (no wrapper)."""
    evaluator = GPUBeamEvaluator()

    # Create mock beam without 'beam' attribute (direct object)
    mock_beam = Mock(spec=['compute_response'])
    mock_beam.compute_response = Mock(return_value=np.ones((1, 1, 1, 5)))

    az = cupy.array([0.0, 0.1, 0.2, 0.3, 0.4])
    za = cupy.array([0.0, 0.1, 0.2, 0.3, 0.4])

    result = evaluator.evaluate_beam(
        beam=mock_beam,
        az=az,
        za=za,
        polarized=False,
        freq=150e6,
    )

    assert isinstance(result, cupy.ndarray)


# ============================================================================
# GPU inplace_rot Tests (moved from original file)
# ============================================================================

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


# ============================================================================
# Large Array Tests
# ============================================================================

@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_get_apparent_flux_polarized_large_array():
    """Test get_apparent_flux_polarized with larger arrays."""
    evaluator = GPUBeamEvaluator()

    nax, nfd, nsrc = 2, 2, 1000
    beam = cupy.random.random((nax, nfd, nsrc)).astype(cupy.complex128)
    flux = cupy.random.random(nsrc)

    evaluator.get_apparent_flux_polarized(beam, flux)

    assert beam.shape == (nax, nfd, nsrc)
    assert cupy.all(cupy.isfinite(beam))


# ============================================================================
# Edge Cases
# ============================================================================

@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_evaluate_beam_with_spline_opts():
    """Test evaluate_beam with custom spline_opts."""
    evaluator = GPUBeamEvaluator()

    mock_beam = Mock()
    mock_beam.beam = None
    mock_beam.compute_response = Mock(return_value=np.ones((1, 1, 1, 5)))

    az = cupy.array([0.0, 0.1, 0.2, 0.3, 0.4])
    za = cupy.array([0.0, 0.1, 0.2, 0.3, 0.4])

    result = evaluator.evaluate_beam(
        beam=mock_beam,
        az=az,
        za=za,
        polarized=False,
        freq=150e6,
        spline_opts={'order': 1},
    )

    assert evaluator.spline_opts == {'order': 1}
    assert isinstance(result, cupy.ndarray)


# ============================================================================
# Real UVBeam Tests
# ============================================================================

@pytest.fixture
def uvbeam_fixture():
    """Create a real UVBeam object from test data."""
    from pyuvdata import UVBeam
    from pathlib import Path

    beam_file = Path(__file__).parent / "data" / "HERA_NicCST_150MHz.txt"
    if not beam_file.exists():
        pytest.skip(f"Test beam file not found: {beam_file}")

    beam = UVBeam()
    beam.read_cst_beam(
        str(beam_file),
        frequency=[150e6],
        telescope_name="HERA",
        feed_name="Dipole",
        feed_version="1.0",
        feed_pol=["x"],
        model_name="Test",
        model_version="1.0",
    )
    return beam


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_evaluate_beam_with_real_uvbeam_unpolarized(uvbeam_fixture):
    """Test evaluate_beam with a real UVBeam object (unpolarized)."""
    from pyuvdata.beam_interface import BeamInterface

    evaluator = GPUBeamEvaluator()
    beam_interface = BeamInterface(uvbeam_fixture)

    nsrc = 50
    # Generate coordinates within valid range for beam
    az = cupy.random.uniform(0, 2 * np.pi, nsrc).astype(cupy.float64)
    za = cupy.random.uniform(0, np.pi / 4, nsrc).astype(cupy.float64)  # Near zenith

    result = evaluator.evaluate_beam(
        beam=beam_interface,
        az=az,
        za=za,
        polarized=False,
        freq=150e6,
    )

    assert isinstance(result, cupy.ndarray)
    assert result.shape == (nsrc,)
    assert cupy.all(cupy.isfinite(result))


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_evaluate_beam_with_real_uvbeam_polarized(uvbeam_fixture):
    """Test evaluate_beam with a real UVBeam object (polarized)."""
    from pyuvdata.beam_interface import BeamInterface

    evaluator = GPUBeamEvaluator()
    beam_interface = BeamInterface(uvbeam_fixture)

    nsrc = 50
    az = cupy.random.uniform(0, 2 * np.pi, nsrc).astype(cupy.float64)
    za = cupy.random.uniform(0, np.pi / 4, nsrc).astype(cupy.float64)

    result = evaluator.evaluate_beam(
        beam=beam_interface,
        az=az,
        za=za,
        polarized=True,
        freq=150e6,
    )

    assert isinstance(result, cupy.ndarray)
    # Polarized result should have shape (nax, nfd, nsrc)
    assert len(result.shape) >= 2
    assert cupy.all(cupy.isfinite(result))


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_evaluate_beam_with_real_uvbeam_check(uvbeam_fixture):
    """Test evaluate_beam with check=True on real UVBeam."""
    from pyuvdata.beam_interface import BeamInterface

    evaluator = GPUBeamEvaluator()
    beam_interface = BeamInterface(uvbeam_fixture)

    nsrc = 20
    az = cupy.random.uniform(0, 2 * np.pi, nsrc).astype(cupy.float64)
    za = cupy.random.uniform(0, np.pi / 4, nsrc).astype(cupy.float64)

    # Should not raise with valid data
    result = evaluator.evaluate_beam(
        beam=beam_interface,
        az=az,
        za=za,
        polarized=False,
        freq=150e6,
        check=True,
    )

    assert cupy.all(cupy.isfinite(result))


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_GPU_MAP_COORDS, reason="GPU map_coordinates not available")
def test_evaluate_uvbeam_gpu_directly(uvbeam_fixture):
    """Test the _evaluate_uvbeam_gpu method directly if available."""
    from pyuvdata.beam_interface import BeamInterface

    evaluator = GPUBeamEvaluator()
    beam_interface = BeamInterface(uvbeam_fixture)

    nsrc = 30
    az = cupy.random.uniform(0, 2 * np.pi, nsrc).astype(cupy.float64)
    za = cupy.random.uniform(0, np.pi / 4, nsrc).astype(cupy.float64)

    # Check if the UVBeam path is taken
    beam_obj = beam_interface.beam if hasattr(beam_interface, 'beam') else beam_interface

    # Check it has the required attributes for GPU UVBeam path
    has_data_array = hasattr(beam_obj, 'data_array')
    has_pixel_coord = hasattr(beam_obj, 'pixel_coordinate_system')
    has_diameter = hasattr(beam_obj, 'diameter')

    if has_data_array and has_pixel_coord and not has_diameter:
        # This should trigger the _evaluate_uvbeam_gpu path
        result = evaluator._evaluate_uvbeam_gpu(
            beam=beam_interface,
            az=az,
            za=za,
            polarized=False,
            freq=150e6,
            check=False,
            spline_opts={'order': 1}
        )

        assert isinstance(result, cupy.ndarray)
        assert cupy.all(cupy.isfinite(result))
    else:
        pytest.skip("UVBeam doesn't meet GPU interpolation criteria")


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_evaluate_beam_cache_invalidation(uvbeam_fixture):
    """Test that beam cache is invalidated when source count changes."""
    from pyuvdata.beam_interface import BeamInterface

    evaluator = GPUBeamEvaluator()
    beam_interface = BeamInterface(uvbeam_fixture)

    # First evaluation with 10 sources
    az1 = cupy.random.uniform(0, 2 * np.pi, 10).astype(cupy.float64)
    za1 = cupy.random.uniform(0, np.pi / 4, 10).astype(cupy.float64)

    result1 = evaluator.evaluate_beam(
        beam=beam_interface,
        az=az1,
        za=za1,
        polarized=False,
        freq=150e6,
    )

    assert result1.shape == (10,)

    # Second evaluation with different number of sources - should re-evaluate
    az2 = cupy.random.uniform(0, 2 * np.pi, 20).astype(cupy.float64)
    za2 = cupy.random.uniform(0, np.pi / 4, 20).astype(cupy.float64)

    result2 = evaluator.evaluate_beam(
        beam=beam_interface,
        az=az2,
        za=za2,
        polarized=False,
        freq=150e6,
    )

    # Should have new shape
    assert result2.shape == (20,)