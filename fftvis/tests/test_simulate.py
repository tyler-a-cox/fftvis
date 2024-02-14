import matvis
import pytest
import numpy as np
from fftvis import simulate
from pyuvsim.analyticbeam import AnalyticBeam


def test_simulate():
    """ """
    ntimes = 10
    nfreqs = 5
    nants = 3
    nsrcs = 20

    # Define frequency and time range
    freqs = np.linspace(100e6, 200e6, nfreqs)
    lsts = np.linspace(0, np.pi, ntimes)

    # Set up the antenna positions
    antpos = {k: np.array([k * 10, 0, 0]) for k in range(nants)}

    # Define a Gaussian beam
    beam = AnalyticBeam("gaussian", diameter=14.0)

    # Set sky model
    ra = np.linspace(0.0, 2.0 * np.pi, nsrcs)
    dec = np.linspace(-0.5 * np.pi, 0.5 * np.pi, nsrcs)
    sky_model = np.random.uniform(0, 1, size=(nsrcs, 1)) * (freqs[None] / 150e6) ** -2.5

    # Use matvis as a reference
    mvis = matvis.simulate_vis(
        antpos, sky_model, ra, dec, freqs, lsts, beams=[beam], precision=2
    )

    # Use fftvis to simulate visibilities
    fvis = simulate.simulate_vis(
        antpos, sky_model, ra, dec, freqs, lsts, beam, precision=2, accuracy=1e-10
    )

    for i in range(nants):
        for j in range(nants):
            if i == j:
                continue
            assert np.allclose(mvis[..., i, j].T, fvis[i, j], atol=1e-5)
