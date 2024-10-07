import matvis
import pytest
import numpy as np
from fftvis import simulate
from pyuvsim.analyticbeam import AnalyticBeam
from . import get_standard_sim_params

@pytest.mark.parametrize('polarized', [False, True])
@pytest.mark.parametrize('precision', [2,1])
@pytest.mark.parametrize('use_analytic_beam', [True, False])
def test_simulate(polarized: bool, precision: int, use_analytic_beam: bool):

    (
        ants,
        flux,
        ra,
        dec,
        freqs,
        lsts,
        times,
        cpu_beams,
        location,
    ) = get_standard_sim_params(use_analytic_beam, polarized)
    
    # Simulate with specified baselines
    sim_baselines = np.array([[0, 1]])#, [0, 2], [1, 2]])

    # Use matvis as a reference
    mvis = matvis.simulate_vis(
        ants=ants,
        fluxes=flux,
        ra=ra,
        dec=dec,
        freqs=freqs,
        times=times,
        beams=cpu_beams,
        telescope_loc=location,
        polarized=polarized,
        precision=precision,
        antpairs=sim_baselines,
    )

    # Use fftvis to simulate visibilities
    fvis = simulate.simulate_vis(
        ants, flux, ra, dec, freqs, times.jd, cpu_beams[0], eps=1e-10 if precision==2 else 6e-8,
        baselines=sim_baselines, polarized=polarized, precision=precision,
        telescope_loc=location,
    )

    # Should have shape (nfreqs, ntimes, nants, nants)
    if polarized:
        assert fvis.shape == (len(freqs), len(lsts), 2, 2, len(sim_baselines))
    else:
        assert fvis.shape == (len(freqs), len(lsts), len(sim_baselines))

    # Check that the polarized results are the same
    for bi, bl in enumerate(sim_baselines):
        print(bl, mvis.shape)
        np.testing.assert_allclose(
            fvis[..., bi], mvis[:, :, bi], 
            atol=1e-5 if precision==2 else 1e-4
        )



def test_simulate_non_coplanar():
    # Simulation parameters
    ntimes = 10
    nfreqs = 5
    nants = 3
    nsrcs = 20

    # Set random set
    rng = np.random.default_rng(42)

    # Define frequency and time range
    freqs = np.linspace(100e6, 200e6, nfreqs)
    lsts = np.linspace(0, np.pi, ntimes)

    # Set up the antenna positions
    antpos = {k: np.array([k * 10, 0, k]) for k in range(nants)}
    antpos_flat = {k: np.array([k * 10, 0, 0]) for k in range(nants)}

    # Define a Gaussian beam
    beam = AnalyticBeam("gaussian", diameter=14.0)

    # Set sky model
    ra = np.linspace(0.0, 2.0 * np.pi, nsrcs)
    dec = np.linspace(-0.5 * np.pi, 0.5 * np.pi, nsrcs)
    sky_model = rng.uniform(0, 1, size=(nsrcs, 1)) * (freqs[None] / 150e6) ** -2.5

    # Use matvis as a reference
    mvis = matvis.simulate_vis(
        antpos, sky_model, ra, dec, freqs, lsts, beams=[beam], precision=2
    )
    mvis_flat = matvis.simulate_vis(
        antpos_flat, sky_model, ra, dec, freqs, lsts, beams=[beam], precision=2
    )

    # Use fftvis to simulate visibilities
    fvis = simulate.simulate_vis(
        antpos, sky_model, ra, dec, freqs, lsts, beam, precision=2, eps=1e-10,
        baselines=[(i, j) for i in range(nants) for j in range(nants)]
    )
    fvis.shape = (nfreqs, ntimes, nants, nants)

    # Check that the results are the same
    np.testing.assert_allclose(mvis, fvis, atol=1e-5)

    # Check that the results are different
    assert not np.allclose(mvis_flat, fvis, atol=1e-5)
