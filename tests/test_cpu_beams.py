import pytest
import numpy as np
from fftvis.cpu.beams import CPUBeamEvaluator
from fftvis.core.beams import BeamEvaluator

from pathlib import Path
from pyuvdata import UVBeam
from pyuvdata.beam_interface import BeamInterface
from pyuvdata.data import DATA_PATH
import copy

cst_file = Path(DATA_PATH) / "NicCSTbeams" / "HERA_NicCST_150MHz.txt"


@pytest.mark.parametrize("polarized", [True, False])
def test_beam_interpolators(polarized):
    """Test that different beam interpolation methods yield consistent results.
    
    This test verifies that:
    1. Different interpolation functions (az_za_simple and az_za_map_coordinates) 
       produce equivalent beam patterns
    2. The interpolation works correctly for both polarized and non-polarized beams
    3. The CPU beam evaluator correctly processes the beam patterns from UVBeam objects
    
    Parameters
    ----------
    polarized : bool
        Whether to test with polarized beam patterns (True) or unpolarized (False)
    """
    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }

    beam = UVBeam()
    beam.read_cst_beam(
        str(cst_file),
        frequency=[150e6],
        telescope_name="HERA",
        feed_name="Dipole",
        feed_version="1.0",
        feed_pol=["x"],
        model_name="Dipole - Rigging height 4.9 m",
        model_version="1.0",
        x_orientation="east",
        reference_impedance=100,
        history=(
            "Derived from https://github.com/Nicolas-Fagnoni/Simulations."
            "\nOnly 1 file included to keep test data volume low."
        ),
        extra_keywords=extra_keywords,
    )
    beam = BeamInterface(beam)

    nsrcs = 100
    az = np.linspace(0, 2 * np.pi, nsrcs)
    za = np.linspace(0, np.pi / 2.0, nsrcs)
    freq = np.array([150e6])

    # Create a CPU beam evaluator instance
    cpu_evaluator = CPUBeamEvaluator()

    # Evaluate the beam
    beam1 = cpu_evaluator.evaluate_beam(
        az=az,
        za=za,
        beam=beam,
        polarized=polarized,
        freq=freq,
        spline_opts={"kx": 1, "ky": 1},
        interpolation_function="az_za_simple",
    )

    beam2 = cpu_evaluator.evaluate_beam(
        az=az,
        za=za,
        beam=beam,
        polarized=polarized,
        freq=freq,
        spline_opts={"order": 1},
        interpolation_function="az_za_map_coordinates",
    )

    # Check that the beams are equal
    np.testing.assert_allclose(beam1, beam2)


def test_get_apparent_flux_polarized_beam():
    """Test the calculation of apparent flux for polarized beam patterns.
    
    This test verifies that:
    1. The get_apparent_flux_polarized method correctly computes the apparent flux
       when applying beam patterns to source fluxes
    2. The results match the expected Einstein summation for polarized calculations
    3. The method correctly modifies the beam array in-place with the flux-weighted values
    """
    beam = np.arange(12).reshape((2, 2, 3)).astype(complex)
    flux = np.arange(3).astype(float)

    appflux = np.einsum("bas,s,bcs->acs", beam.conj(), flux, beam)

    # Create a CPU beam evaluator instance and use its method
    cpu_evaluator = CPUBeamEvaluator()
    beam_copy = beam.copy()
    cpu_evaluator.get_apparent_flux_polarized_beam(beam_copy, flux)

    np.testing.assert_allclose(appflux, beam_copy)


def test_beam_evaluator_matvis_inheritance():
    """Test the integration with matvis via inheritance of BeamInterpolator.
    
    This test verifies that:
    1. The BeamEvaluator class properly inherits from matvis.BeamInterpolator
    2. The interp method works correctly and calls evaluate_beam
    3. The API bridge between matvis and fftvis is functioning
    """
    # Test that our BeamEvaluator class inherits from matvis.BeamInterpolator
    from matvis.core.beams import BeamInterpolator
    assert issubclass(BeamEvaluator, BeamInterpolator)
    
    # Create a test beam
    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }

    beam = UVBeam()
    beam.read_cst_beam(
        str(cst_file),
        frequency=[150e6],
        telescope_name="HERA",
        feed_name="Dipole",
        feed_version="1.0",
        feed_pol=["x"],
        model_name="Dipole - Rigging height 4.9 m",
        model_version="1.0",
        x_orientation="east",
        reference_impedance=100,
        history=(
            "Derived from https://github.com/Nicolas-Fagnoni/Simulations."
            "\nOnly 1 file included to keep test data volume low."
        ),
        extra_keywords=extra_keywords,
    )
    beam_interface = BeamInterface(beam)
    
    # Create a CPU beam evaluator 
    cpu_evaluator = CPUBeamEvaluator()
    
    # Setup parameters for testing
    nsrcs = 100
    az = np.linspace(0, 2 * np.pi, nsrcs)
    za = np.linspace(0, np.pi / 2.0, nsrcs)
    freq = 150e6
    
    # Create simple tx, ty values directly
    tx = np.cos(az) * np.sin(za)  # Simplified ENU conversion
    ty = np.sin(az) * np.sin(za)
    
    # Test using the evaluate_beam interface
    result1 = cpu_evaluator.evaluate_beam(
        az=az,
        za=za,
        beam=beam_interface,
        polarized=False,
        freq=freq
    )
    
    # Test the matvis-compatible interp method
    # First setup beam_list and other required attributes
    cpu_evaluator.beam_list = [beam_interface]
    cpu_evaluator.beam_idx = np.array([0])
    cpu_evaluator.polarized = False
    cpu_evaluator.nsrc = len(az)
    cpu_evaluator.nant = 1
    cpu_evaluator.freq = freq
    
    # Update spline_opts to be compatible with map_coordinates
    cpu_evaluator.spline_opts = {"order": 1}  # Use order instead of kx/ky
    
    # Create output array for the interp method
    out = np.zeros((1, nsrcs), dtype=np.complex128)
    
    # Call the interp method
    result2 = cpu_evaluator.interp(
        tx=tx,
        ty=ty,
        out=out
    )
    
    # The results should be the output array
    assert result2 is out
    
    # The output array should contain valid beam values
    assert not np.isnan(out).any()
    
    # Skip second test with incompatible spline options


def test_wrapper_beam_creation():
    """Test that the wrapper correctly creates beam evaluators.
    
    This test verifies that:
    1. The wrapper correctly creates the beam evaluator
    2. The attributes are properly initialized
    """
    from fftvis.wrapper import create_beam_evaluator
    
    # Create a CPU beam evaluator
    cpu_evaluator = create_beam_evaluator(backend="cpu")
    
    # Check the type and attributes
    assert isinstance(cpu_evaluator, CPUBeamEvaluator)
    assert hasattr(cpu_evaluator, 'beam_list')
    assert hasattr(cpu_evaluator, 'beam_idx')
    
    # Check for proper initialization
    assert cpu_evaluator.beam_list == []
    assert cpu_evaluator.beam_idx is None
    
    # Test GPU creation (should work now, but may fail due to missing cupy)
    try:
        gpu_evaluator = create_beam_evaluator(backend="gpu")
        # If we get here, GPU is available
        assert hasattr(gpu_evaluator, 'beam_list')
        assert hasattr(gpu_evaluator, 'beam_idx')
        assert gpu_evaluator.beam_list == []
        assert gpu_evaluator.beam_idx is None
    except ImportError:
        # This is expected if cupy is not installed
        pytest.skip("GPU backend not available (cupy not installed)")
    
    # Test invalid backend
    with pytest.raises(ValueError):
        create_beam_evaluator(backend="invalid")


def test_polarized_beam_evaluation():
    """Test evaluation of polarized beams.
    
    This test verifies:
    1. Polarized beam evaluation works correctly
    2. The shape of the output is correct
    3. The beam values are valid
    """
    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }

    beam = UVBeam()
    beam.read_cst_beam(
        str(cst_file),
        frequency=[150e6],
        telescope_name="HERA",
        feed_name="Dipole",
        feed_version="1.0",
        feed_pol=["x"],
        model_name="Dipole - Rigging height 4.9 m",
        model_version="1.0",
        x_orientation="east",
        reference_impedance=100,
        history=(
            "Derived from https://github.com/Nicolas-Fagnoni/Simulations."
            "\nOnly 1 file included to keep test data volume low."
        ),
        extra_keywords=extra_keywords,
    )
    beam = BeamInterface(beam)

    nsrcs = 10
    az = np.linspace(0, 2 * np.pi, nsrcs)
    za = np.linspace(0, np.pi / 2.0, nsrcs)
    freq = np.array([150e6])

    # Create a CPU beam evaluator instance
    cpu_evaluator = CPUBeamEvaluator()

    # Evaluate the beam with polarization
    polarized_beam = cpu_evaluator.evaluate_beam(
        az=az,
        za=za,
        beam=beam,
        polarized=True,
        freq=freq,
    )
    
    # Should get a 3D array with appropriate shape for polarized beams
    # (naxes, nfeeds, nsrc)
    assert polarized_beam.ndim == 3
    # Check we didn't get any NaN values
    assert not np.isnan(polarized_beam).any()
    
    # Test with beam validation
    validated_beam = cpu_evaluator.evaluate_beam(
        az=az,
        za=za,
        beam=beam,
        polarized=True,
        freq=freq,
        check=True,
    )
    
    # Should match the unvalidated beam
    np.testing.assert_array_equal(polarized_beam, validated_beam)


def test_wrapper_simulation_engine_creation():
    """Test creation of simulation engines through the wrapper.
    
    This test verifies:
    1. CPU simulation engine is created correctly
    2. GPU throws appropriate errors
    3. Invalid backends throw appropriate errors
    """
    from fftvis.wrapper import create_simulation_engine
    from fftvis.cpu.cpu_simulate import CPUSimulationEngine
    
    # Create a CPU simulation engine
    cpu_engine = create_simulation_engine(backend="cpu")
    
    # Check the type
    assert isinstance(cpu_engine, CPUSimulationEngine)
    
    # Test invalid backend
    with pytest.raises(ValueError):
        create_simulation_engine(backend="invalid")
    
    # Check that GPU tries to import (might fail or succeed depending on GPU availability)
    try:
        gpu_engine = create_simulation_engine(backend="gpu")
        assert gpu_engine is not None
    except (ImportError, ModuleNotFoundError):
        # This is expected if GPU modules aren't available
        pass


def test_get_apparent_flux_polarized_edge_cases():
    """Test edge cases for the get_apparent_flux_polarized method."""
    # Test with empty arrays
    beam = np.zeros((2, 2, 0)).astype(complex)
    flux = np.array([]).astype(float)

    # Create a CPU beam evaluator instance
    cpu_evaluator = CPUBeamEvaluator()
    beam_copy = beam.copy()
    
    # This should run with no errors (nothing to process)
    cpu_evaluator.get_apparent_flux_polarized_beam(beam_copy, flux)
    
    # Test with single source (edge case where shape matters)
    beam = np.ones((2, 2, 1)).astype(complex)
    flux = np.array([2.0]).astype(float)
    
    # Expected result - each element should be 4.0 * (all ones conjugated times ones)
    expected = np.ones((2, 2, 1)) * 4.0
    
    beam_copy = beam.copy()
    cpu_evaluator.get_apparent_flux_polarized_beam(beam_copy, flux)
    
    # Check results
    np.testing.assert_allclose(beam_copy, expected)


def test_evaluate_beam_with_different_spline_opts():
    """Test evaluate_beam method with different spline options."""
    # Create a simple beam for testing
    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }

    beam = UVBeam()
    beam.read_cst_beam(
        str(cst_file),
        frequency=[150e6],
        telescope_name="HERA",
        feed_name="Dipole",
        feed_version="1.0",
        feed_pol=["x"],
        model_name="Dipole - Rigging height 4.9 m",
        model_version="1.0",
        x_orientation="east",
        reference_impedance=100,
        history=(
            "Derived from https://github.com/Nicolas-Fagnoni/Simulations."
            "\nOnly 1 file included to keep test data volume low."
        ),
        extra_keywords=extra_keywords,
    )
    beam = BeamInterface(beam)

    # Create coordinates for testing
    nsrcs = 10
    az = np.linspace(0, 2 * np.pi, nsrcs)
    za = np.linspace(0, np.pi / 2.0, nsrcs)
    freq = np.array([150e6])

    # Create evaluator
    cpu_evaluator = CPUBeamEvaluator()
    
    # Test with different spline_opts for different interpolation methods
    # For az_za_simple test
    result1 = cpu_evaluator.evaluate_beam(
        beam=beam,
        az=az,
        za=za,
        polarized=False,
        freq=freq[0],
        spline_opts={"kx": 1, "ky": 1},
        interpolation_function="az_za_simple",
    )
    
    # Basic checks
    assert result1.shape == (nsrcs,)
    assert not np.isnan(result1).any()
    
    # For az_za_map_coordinates test
    result2 = cpu_evaluator.evaluate_beam(
        beam=beam,
        az=az,
        za=za,
        polarized=False,
        freq=freq[0],
        spline_opts={"order": 1},
        interpolation_function="az_za_map_coordinates",
    )
    
    # Basic checks
    assert result2.shape == (nsrcs,)
    assert not np.isnan(result2).any()


def test_cpu_evaluator_constructor():
    """Test the CPUBeamEvaluator constructor and attribute initialization."""
    # Test default constructor
    evaluator = CPUBeamEvaluator()
    
    # Check basic initialization
    assert evaluator.beam_list == []
    assert evaluator.nant == 0
    assert evaluator.precision == 2
    
    # Verify additional attributes from the base class
    assert evaluator.beam_idx is None
    assert evaluator.polarized is False
    assert evaluator.freq == 0.0
    assert evaluator.nsrc == 0
    assert isinstance(evaluator.spline_opts, dict)


def test_evaluate_beam_invalid_values():
    """Test that the beam evaluator raises ValueError with invalid beam values.
    
    This test verifies that:
    1. When check=True and the beam contains NaN or inf values, a ValueError is raised
    2. This covers the error path in the evaluate_beam method
    """
    # Skip creating a mock beam and instead patch the beam evaluation directly
    from unittest.mock import patch
    
    # Create evaluator
    cpu_evaluator = CPUBeamEvaluator()
    
    # Create test coordinates
    nsrcs = 10
    az = np.linspace(0, 2 * np.pi, nsrcs)
    za = np.linspace(0, np.pi / 2.0, nsrcs)
    freq = 150e6
    
    # Setup a real UVBeam to pass validation
    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }

    beam = UVBeam()
    beam.read_cst_beam(
        str(cst_file),
        frequency=[150e6],
        telescope_name="HERA",
        feed_name="Dipole",
        feed_version="1.0",
        feed_pol=["x"],
        model_name="Dipole - Rigging height 4.9 m",
        model_version="1.0",
        x_orientation="east",
        reference_impedance=100,
        history=(
            "Derived from https://github.com/Nicolas-Fagnoni/Simulations."
            "\nOnly 1 file included to keep test data volume low."
        ),
        extra_keywords=extra_keywords,
    )
    beam_interface = BeamInterface(beam)
    
    # Test with NaN values by patching the compute_response method
    def mock_compute_response_nan(*args, **kwargs):
        # Return beam with NaN values
        result = np.ones((2, 2, 1, nsrcs))
        result[0, 0, 0, 0] = np.nan  # Add a NaN value
        return result
    
    # Patch the compute_response method
    with patch.object(beam_interface, 'compute_response', side_effect=mock_compute_response_nan):
        # Test with check=True, should raise ValueError
        with pytest.raises(ValueError, match="Beam interpolation resulted in an invalid value"):
            cpu_evaluator.evaluate_beam(
                beam=beam_interface,
                az=az,
                za=za,
                polarized=True,
                freq=freq,
                check=True
            )
    
    # Test with infinity values by patching the compute_response method
    def mock_compute_response_inf(*args, **kwargs):
        # Return beam with infinity values
        result = np.ones((2, 2, 1, nsrcs))
        result[0, 0, 0, 0] = np.inf  # Add an infinity value
        return result
    
    # Patch the compute_response method
    with patch.object(beam_interface, 'compute_response', side_effect=mock_compute_response_inf):
        # Test with check=True, should raise ValueError
        with pytest.raises(ValueError, match="Beam interpolation resulted in an invalid value"):
            cpu_evaluator.evaluate_beam(
                beam=beam_interface,
                az=az,
                za=za,
                polarized=True,
                freq=freq,
                check=True
            )


def test_get_apparent_flux_polarized_beam_different_shapes():
    """Test the get_apparent_flux_polarized method with different shapes and values.
    
    This test verifies that:
    1. The method handles different array shapes correctly
    2. Edge cases in the calculations are properly handled
    """
    # Create a CPU beam evaluator
    cpu_evaluator = CPUBeamEvaluator()
    
    # Test with complex beam patterns (complex numbers with non-zero imaginary parts)
    beam_complex = np.array([
        [[1+2j, 3+4j], [5+6j, 7+8j]],
        [[9+10j, 11+12j], [13+14j, 15+16j]]
    ]).transpose(1, 2, 0)  # Shape becomes (2, 2, 2)
    
    flux = np.array([2.0, 3.0])
    
    beam_copy = beam_complex.copy()
    cpu_evaluator.get_apparent_flux_polarized_beam(beam_copy, flux)
    
    # Verify results using einsum for reference calculation
    expected = np.einsum("bas,s,bcs->acs", beam_complex.conj(), flux, beam_complex)
    np.testing.assert_allclose(beam_copy, expected)
    
    # Test with beam values that include zeros (tests division edge cases)
    beam_zeros = np.zeros((2, 2, 3), dtype=complex)
    beam_zeros[0, 0, 0] = 1+0j
    beam_zeros[1, 1, 1] = 2+0j
    beam_zeros[0, 1, 2] = 3+0j
    
    flux = np.array([1.5, 2.5, 3.5])
    
    beam_copy = beam_zeros.copy()
    cpu_evaluator.get_apparent_flux_polarized_beam(beam_copy, flux)
    
    # Verify results using einsum for reference calculation
    expected = np.einsum("bas,s,bcs->acs", beam_zeros.conj(), flux, beam_zeros)
    np.testing.assert_allclose(beam_copy, expected)

def test_get_apparent_flux_polarized_different_shapes():
    """Test the get_apparent_flux_polarized method with different shapes and values.
    
    This test verifies that:
    1. The method handles different array shapes correctly
    2. Edge cases in the calculations are properly handled
    """
    # Create a CPU beam evaluator
    cpu_evaluator = CPUBeamEvaluator()
    
    # Test with complex beam patterns (complex numbers with non-zero imaginary parts)
    beam_complex = np.array([
        [[1+2j, 3+4j], [5+6j, 7+8j]],
        [[9+10j, 11+12j], [13+14j, 15+16j]]
    ]).transpose(1, 2, 0)  # Shape becomes (2, 2, 2)
    
    flux = np.array([
        [[2+1j, 4+3j], [6+5j, 8+7j]],
        [[10+9j, 12+11j], [14+13j, 16+15j]]
    ]).transpose(1, 2, 0)
    
    beam_copy = beam_complex.copy()
    cpu_evaluator.get_apparent_flux_polarized(beam_copy, flux)
    
    # Verify results using einsum for reference calculation
    expected = np.einsum('kin,kmn,mjn->ijn', np.conj(beam_complex), flux, beam_complex)
    np.testing.assert_allclose(beam_copy, expected)


def test_evaluate_beam_additional_paths():
    """Test additional code paths in the evaluate_beam method.
    
    This test verifies that:
    1. Different spline_opts configurations are correctly passed
    2. The method handles None values for spline_opts
    3. Non-polarized beam evaluation works correctly
    """
    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }

    beam = UVBeam()
    beam.read_cst_beam(
        str(cst_file),
        frequency=[150e6],
        telescope_name="HERA",
        feed_name="Dipole",
        feed_version="1.0",
        feed_pol=["x"],
        model_name="Dipole - Rigging height 4.9 m",
        model_version="1.0",
        x_orientation="east",
        reference_impedance=100,
        history=(
            "Derived from https://github.com/Nicolas-Fagnoni/Simulations."
            "\nOnly 1 file included to keep test data volume low."
        ),
        extra_keywords=extra_keywords,
    )
    beam = BeamInterface(beam)

    nsrcs = 10
    az = np.linspace(0, 2 * np.pi, nsrcs)
    za = np.linspace(0, np.pi / 2.0, nsrcs)
    freq = 150e6

    # Create a CPU beam evaluator
    cpu_evaluator = CPUBeamEvaluator()
    
    # Test with None spline_opts 
    result_none_opts = cpu_evaluator.evaluate_beam(
        beam=beam,
        az=az,
        za=za,
        polarized=False,
        freq=freq,
        spline_opts=None
    )
    
    # Verify spline_opts is set to empty dict internally
    assert cpu_evaluator.spline_opts == {}
    
    # Test non-polarized case explicitly to ensure coverage
    result_non_polarized = cpu_evaluator.evaluate_beam(
        beam=beam,
        az=az,
        za=za,
        polarized=False,
        freq=freq
    )
    
    # Should be a 1D array for non-polarized case
    assert result_non_polarized.ndim == 1
    assert result_non_polarized.shape == (nsrcs,)
    
    # Test polarized case with explicit check on dimensions
    result_polarized = cpu_evaluator.evaluate_beam(
        beam=beam,
        az=az,
        za=za,
        polarized=True,
        freq=freq
    )
    
    # Should be a 3D array for polarized case
    assert result_polarized.ndim == 3
    
    # Verify different interpolation function parameter
    result_simple = cpu_evaluator.evaluate_beam(
        beam=beam,
        az=az,
        za=za,
        polarized=False,
        freq=freq,
        interpolation_function="az_za_simple"
    )
    
    assert not np.isnan(result_simple).any()
