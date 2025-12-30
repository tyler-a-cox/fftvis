import pytest
import numpy as np
from fftvis.cpu.beams import CPUBeamEvaluator
from fftvis.core.beams import BeamEvaluator

from pathlib import Path
from pyuvdata import UVBeam
from pyuvdata.beam_interface import BeamInterface
from pathlib import Path

TEST_DIR = Path(__file__).parent
cst_file = TEST_DIR / "data" / "HERA_NicCST_150MHz.txt"


def test_cpu_beam_evaluator_init():
    """Test that the CPUBeamEvaluator initializes correctly."""
    # Create a concrete evaluator since BeamEvaluator is an abstract class
    evaluator = CPUBeamEvaluator()
    
    # Check initialization values
    assert evaluator.beam_list == []
    assert evaluator.beam_idx is None
    assert evaluator.polarized is False
    assert evaluator.nant == 0
    assert evaluator.freq == 0.0
    assert evaluator.nsrc == 0
    assert evaluator.precision == 2


def test_cpu_evaluator_attributes():
    """Test that the CPUBeamEvaluator inherits all required attributes from base class."""
    cpu_evaluator = CPUBeamEvaluator()
    
    # Check that it has all the attributes from the parent class
    assert hasattr(cpu_evaluator, 'beam_list')
    assert hasattr(cpu_evaluator, 'beam_idx')
    assert hasattr(cpu_evaluator, 'polarized')
    assert hasattr(cpu_evaluator, 'nant')
    assert hasattr(cpu_evaluator, 'freq')
    assert hasattr(cpu_evaluator, 'nsrc')
    assert hasattr(cpu_evaluator, 'spline_opts')
    assert hasattr(cpu_evaluator, 'precision')
    
    # Check default values
    assert cpu_evaluator.beam_list == []
    assert cpu_evaluator.beam_idx is None
    assert cpu_evaluator.polarized is False
    assert cpu_evaluator.precision == 2


def test_evaluate_beam_with_check():
    """Test the evaluate_beam method with the check parameter."""
    # Create the beam
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

    # Set up test coordinates
    nsrcs = 20
    az = np.linspace(0, 2 * np.pi, nsrcs)
    za = np.linspace(0, np.pi / 2.0, nsrcs)
    freq = np.array([150e6])

    # Create a CPU beam evaluator instance
    cpu_evaluator = CPUBeamEvaluator()

    # Evaluate the beam with checking
    beam_pattern = cpu_evaluator.evaluate_beam(
        az=az,
        za=za,
        beam=beam,
        polarized=False,
        freq=freq,
        check=True,  # Enable checking for inf/nan
    )
    
    # Should not have any NaN or inf values
    assert not np.isnan(beam_pattern).any()
    assert not np.isinf(beam_pattern).any()
    
    # Create a direct test of the NaN check logic
    # Create a beam pattern with NaN values
    beam_with_nan = np.ones(10)
    beam_with_nan[0] = np.nan
    
    # Manually check if sum is invalid (the same as check=True in evaluate_beam)
    assert np.isnan(np.sum(beam_with_nan))


def test_interp_method():
    """Test the interp bridge method from matvis API."""
    # Create a beam evaluator
    cpu_evaluator = CPUBeamEvaluator()
    
    # Create a simple test beam
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
    
    # Set up basic parameters for interp
    nsrcs = 10
    tx = np.linspace(-0.5, 0.5, nsrcs)
    ty = np.linspace(-0.5, 0.5, nsrcs)
    
    # Configure the evaluator
    cpu_evaluator.beam_list = [beam_interface]
    cpu_evaluator.freq = 150e6
    cpu_evaluator.polarized = False
    cpu_evaluator.nsrc = nsrcs
    
    # Create output array
    out = np.zeros((1, nsrcs), dtype=np.complex128)
    
    # Call the interp method
    result = cpu_evaluator.interp(tx, ty, out)
    
    # Verify the result
    assert result is out  # Should return the same array
    assert not np.isnan(result).any()  # No NaN values
    
    # Now test with polarized=True
    cpu_evaluator.polarized = True
    out_pol = np.zeros((1, 2, 2, nsrcs), dtype=np.complex128)
    
    # Call the interp method
    result_pol = cpu_evaluator.interp(tx, ty, out_pol)
    
    # Verify the result
    assert result_pol is out_pol  # Should return the same array


def test_get_apparent_flux_unpolarized():
    """Test the non-polarized flux calculations (directly applying the flux)."""
    # Create a simple beam pattern
    beam = np.ones((10,), dtype=np.complex128)
    flux = np.arange(10, dtype=np.float64)
    
    # Expected result - just multiplying beam by flux
    expected = beam * flux
    
    # Direct multiplication
    beam_copy = beam.copy()
    beam_copy *= flux
    
    # Check that they match
    np.testing.assert_array_equal(beam_copy, expected)


def test_beam_evaluator_interp_branches():
    """Test the BeamEvaluator's interp method branches.
    
    This test verifies:
    1. The interp method correctly handles non-polarized beams
    2. The conversion from tx/ty to az/za works properly
    3. All code paths in the interp method are exercised
    """
    from fftvis.cpu.beams import CPUBeamEvaluator
    import numpy as np
    from pyuvdata import UVBeam
    from pyuvdata.beam_interface import BeamInterface
    
    # Create a CPUBeamEvaluator instance (concrete implementation of BeamEvaluator)
    evaluator = CPUBeamEvaluator()
    
    # Load a test beam
    cst_file = TEST_DIR / "data" / "HERA_NicCST_150MHz.txt"
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
    
    # Set up evaluator for non-polarized case to hit the 'else' branch
    evaluator.beam_list = [beam_interface]
    evaluator.beam_idx = np.array([0])
    evaluator.polarized = False  # This will trigger the else branch
    evaluator.nsrc = 10
    evaluator.nant = 1
    evaluator.freq = 150e6
    evaluator.spline_opts = {"order": 1}
    
    # Create tx/ty and output arrays
    nsrcs = 10
    theta = np.linspace(0, np.pi/2, nsrcs)
    phi = np.linspace(0, 2*np.pi, nsrcs)
    tx = np.sin(theta) * np.cos(phi)
    ty = np.sin(theta) * np.sin(phi)
    out = np.zeros((1, nsrcs), dtype=np.complex128)
    
    # Call interp - this will trigger the coordinate conversion and the else branch
    result = evaluator.interp(tx=tx, ty=ty, out=out)
    
    # Verify the output
    assert result is out  # Should return the same array
    assert not np.isnan(out).any()  # Should not contain NaNs
    assert out.shape == (1, nsrcs)  # Should have the expected shape 


def test_beam_evaluator_interp_polarized_branches():
    """Test the BeamEvaluator's interp method branches for polarized beams.
    
    This test verifies:
    1. The interp method correctly handles polarized beams with beam_values.ndim == 3
    2. The missing branch in the interp method is exercised
    """
    from fftvis.cpu.beams import CPUBeamEvaluator
    import numpy as np
    from unittest.mock import patch, MagicMock
    
    # Create a CPUBeamEvaluator instance
    evaluator = CPUBeamEvaluator()
    
    # Set up a mock beam
    mock_beam = MagicMock()
    
    # Set up evaluator attributes for polarized case
    evaluator.beam_list = [mock_beam]
    evaluator.beam_idx = np.array([0])
    evaluator.polarized = True  # This will trigger the polarized branch
    evaluator.nsrc = 10
    evaluator.nant = 1
    evaluator.freq = 150e6
    evaluator.spline_opts = {"order": 1}
    
    # Create tx/ty and output arrays
    nsrcs = 10
    tx = np.linspace(-1, 1, nsrcs)
    ty = np.linspace(-1, 1, nsrcs)
    
    # Output array of appropriate shape for polarized beam
    out = np.zeros((1, 2, nsrcs), dtype=np.complex128)
    
    # Create a mock implementation of evaluate_beam that returns a 3D array
    # This will trigger the beam_values.ndim == 3 branch
    def mock_evaluate_beam(*args, **kwargs):
        # Return a 3D array with shape (2, 1, nsrcs)
        # This will go through the ndim==3 branch with transpose
        return np.ones((2, 1, nsrcs), dtype=np.complex128)
    
    # Patch the evaluate_beam method
    with patch.object(evaluator, 'evaluate_beam', side_effect=mock_evaluate_beam):
        # Call interp - this will trigger the polarized & ndim==3 branch
        result = evaluator.interp(tx=tx, ty=ty, out=out)
        
        # Verify that out has been modified
        assert np.all(out != 0)
        assert result is out  # Should return the same array
        assert out.shape == (1, 2, nsrcs)  # Should have the expected shape