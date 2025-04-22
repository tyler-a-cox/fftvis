import pytest
import numpy as np
from fftvis.cpu.cpu_beams import CPUBeamEvaluator

from pathlib import Path
from pyuvdata import UVBeam
from pyuvdata.beam_interface import BeamInterface
from pyuvdata.data import DATA_PATH

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


def test_get_apparent_flux_polarized():
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
    cpu_evaluator.get_apparent_flux_polarized(beam_copy, flux)

    np.testing.assert_allclose(appflux, beam_copy)
