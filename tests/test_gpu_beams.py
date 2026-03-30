"""
Tests for GPU beam evaluation in fftvis.gpu.beams.

This module tests:
- GPUBeamEvaluator initialization and methods
- evaluate_beam API (backed by matvis)
- Apparent flux calculations
- Integration with real UVBeam objects
"""

import pytest
import numpy as np
from unittest.mock import Mock

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
    evaluator = GPUBeamEvaluator()

    # Check initialization values
    assert evaluator.precision == 2
    assert evaluator._initialized is False
    assert evaluator._current_beam_id is None
    assert evaluator._current_freq is None
    assert evaluator._current_polarized is None


def test_gpu_beam_evaluator_init_with_precision():
    """Test that precision can be set during initialization."""
    evaluator = GPUBeamEvaluator(precision=1)
    assert evaluator.precision == 1


def test_gpu_beam_evaluator_inherits_from_matvis():
    """Test that GPUBeamEvaluator inherits from matvis GPUBeamInterpolator."""
    from matvis.gpu.beams import GPUBeamInterpolator
    evaluator = GPUBeamEvaluator()
    assert isinstance(evaluator, GPUBeamInterpolator)


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
def test_get_apparent_flux_polarized_zero_flux():
    """Test get_apparent_flux_polarized with zero flux."""
    evaluator = GPUBeamEvaluator()

    nax, nfd, nsrc = 2, 2, 3
    beam = cupy.ones((nax, nfd, nsrc), dtype=cupy.complex128)
    flux = cupy.zeros(nsrc)

    evaluator.get_apparent_flux_polarized(beam, flux)

    # All values should be zero
    assert cupy.allclose(beam, 0.0)


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
def test_evaluate_beam_converts_numpy_to_cupy():
    """Test that evaluate_beam converts numpy arrays to cupy."""
    pytest.skip("Skipped - requires real beam object for matvis integration")


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_evaluate_beam_initializes_on_first_call():
    """Test that evaluate_beam initializes matvis on first call."""
    from pyuvdata.analytic_beam import AiryBeam
    from pyuvdata.beam_interface import BeamInterface

    evaluator = GPUBeamEvaluator()

    # Should not be initialized yet
    assert evaluator._initialized is False

    # Create a simple analytic beam
    beam = BeamInterface(AiryBeam(diameter=14.0))
    az = cupy.array([0.0, 0.1, 0.2])
    za = cupy.array([0.0, 0.1, 0.2])

    result = evaluator.evaluate_beam(
        beam=beam,
        az=az,
        za=za,
        polarized=False,
        freq=150e6,
    )

    # Should now be initialized
    assert evaluator._initialized is True
    assert isinstance(result, cupy.ndarray)


# ============================================================================
# GPU inplace_rot Tests
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
# Real UVBeam Integration Tests
# ============================================================================

@pytest.fixture
def uvbeam_fixture():
    """Create a real UVBeam object from test data (power beam)."""
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
def test_evaluate_beam_with_real_uvbeam_polarized():
    """Test evaluate_beam with a real UVBeam object (polarized)."""
    pytest.skip("Skipped - matvis requires E-field beam for polarized evaluation, but test fixture is power beam")


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
def test_evaluate_beam_reinit_on_parameter_change():
    """Test that beam is reinitialized when parameters change."""
    from pyuvdata.analytic_beam import AiryBeam
    from pyuvdata.beam_interface import BeamInterface

    evaluator = GPUBeamEvaluator()
    beam = BeamInterface(AiryBeam(diameter=14.0))

    nsrc = 10
    az = cupy.random.uniform(0, 2 * np.pi, nsrc).astype(cupy.float64)
    za = cupy.random.uniform(0, np.pi / 4, nsrc).astype(cupy.float64)

    # First evaluation
    result1 = evaluator.evaluate_beam(
        beam=beam,
        az=az,
        za=za,
        polarized=False,
        freq=150e6,
    )

    assert evaluator._initialized is True
    assert evaluator._current_freq == 150e6
    assert evaluator._current_polarized is False

    # Change frequency - should reinitialize
    result2 = evaluator.evaluate_beam(
        beam=beam,
        az=az,
        za=za,
        polarized=False,
        freq=200e6,
    )

    assert evaluator._current_freq == 200e6


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_evaluate_beam_with_airy_beam():
    """Test evaluate_beam with AiryBeam (matvis handles on CPU)."""
    from pyuvdata.analytic_beam import AiryBeam
    from pyuvdata.beam_interface import BeamInterface

    evaluator = GPUBeamEvaluator()
    beam = BeamInterface(AiryBeam(diameter=14.0))

    nsrc = 20
    az = cupy.random.uniform(0, 2 * np.pi, nsrc).astype(cupy.float64)
    za = cupy.random.uniform(0, np.pi / 4, nsrc).astype(cupy.float64)

    result = evaluator.evaluate_beam(
        beam=beam,
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
def test_evaluate_beam_with_airy_beam_at_zenith():
    """Test AiryBeam evaluation at zenith (za=0)."""
    from pyuvdata.analytic_beam import AiryBeam
    from pyuvdata.beam_interface import BeamInterface

    evaluator = GPUBeamEvaluator()
    beam = BeamInterface(AiryBeam(diameter=14.0))

    nsrc = 10
    az = cupy.zeros(nsrc, dtype=cupy.float64)
    za = cupy.zeros(nsrc, dtype=cupy.float64)  # At zenith

    result = evaluator.evaluate_beam(
        beam=beam,
        az=az,
        za=za,
        polarized=False,
        freq=150e6,
    )

    # At zenith, Airy pattern should be maximum (1.0 squared = 1.0 for power)
    assert cupy.allclose(result, 1.0, rtol=1e-5)


@pytest.mark.skipif(not cupy.cuda.is_available(), reason="CUDA not available")
def test_evaluate_beam_with_spline_opts(uvbeam_fixture):
    """Test evaluate_beam with custom spline_opts."""
    from pyuvdata.beam_interface import BeamInterface

    evaluator = GPUBeamEvaluator()
    beam_interface = BeamInterface(uvbeam_fixture)

    nsrc = 10
    az = cupy.random.uniform(0, 2 * np.pi, nsrc).astype(cupy.float64)
    za = cupy.random.uniform(0, np.pi / 4, nsrc).astype(cupy.float64)

    result = evaluator.evaluate_beam(
        beam=beam_interface,
        az=az,
        za=za,
        polarized=False,
        freq=150e6,
        spline_opts={'order': 1},
    )

    assert isinstance(result, cupy.ndarray)
    assert result.shape == (nsrc,)
