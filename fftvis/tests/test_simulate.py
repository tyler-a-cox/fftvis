import matvis
import pytest
import numpy as np
from fftvis import simulate
from pyuvsim.analyticbeam import AnalyticBeam


def test_simulate():
    """ """
    # Simulation parameters
    ntimes = 10
    nfreqs = 5
    nants = 3
    nsrcs = 20

    # Set random set
    np.random.seed(42)

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
        antpos, sky_model, ra, dec, freqs, lsts, beam, precision=2, eps=1e-10
    )

    # Should have shape (nfreqs, ntimes, nants, nants)
    assert fvis.shape == (nfreqs, ntimes, nants, nants)

    # Check that the results are the same
    assert np.allclose(mvis, fvis, atol=1e-5)

    # Test polarized visibilities
    # Use matvis as a reference
    mvis = matvis.simulate_vis(
        antpos,
        sky_model,
        ra,
        dec,
        freqs,
        lsts,
        beams=[beam],
        precision=2,
        polarized=True,
    )

    # Use fftvis to simulate visibilities
    fvis = simulate.simulate_vis(
        antpos,
        sky_model,
        ra,
        dec,
        freqs,
        lsts,
        beam,
        precision=2,
        eps=1e-10,
        polarized=True,
    )

    # Should have shape (nfreqs, ntimes, nfeeds, nfeeds, nants, nants)
    assert fvis.shape == (nfreqs, ntimes, 2, 2, nants, nants)

    # Check that the polarized results are the same
    assert np.allclose(mvis, fvis, atol=1e-5)
