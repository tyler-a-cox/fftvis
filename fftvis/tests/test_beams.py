import pytest
import numpy as np
from fftvis import beams

from pathlib import Path
from pyuvdata import UVBeam
from pyuvdata.data import DATA_PATH
cst_file = Path(DATA_PATH) / "NicCSTbeams" / "HERA_NicCST_150MHz.txt"


@pytest.mark.parametrize("polarized", [True, False])
def test_beam_interpolators(polarized):
    """
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

    nsrcs = 100
    az = np.linspace(0, 2 * np.pi, nsrcs)
    za = np.linspace(0, np.pi / 2.0, nsrcs)
    freq = np.array([150e6])

    beam1 = np.zeros((2, 2, nsrcs)) if polarized else np.zeros((1, 1, nsrcs))
    beam2 = np.zeros((2, 2, nsrcs)) if polarized else np.zeros((1, 1, nsrcs))

    # Evaluate the beam
    beams._evaluate_beam(
        beam1, az=az, za=za, beam=beam, polarized=polarized, freq=freq, spline_opts={'kx': 1, 'ky': 1}, interpolation_function="az_za_simple"
    )

    beams._evaluate_beam(
        beam2, az=az, za=za, beam=beam, polarized=polarized, freq=freq, spline_opts={'order': 1}, interpolation_function="az_za_map_coordinates"
    )

    # Check that the beams are equal
    np.testing.assert_allclose(beam1, beam2)