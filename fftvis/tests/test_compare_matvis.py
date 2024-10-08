import matvis
import pytest
import numpy as np
from fftvis import simulate
from pyuvsim.analyticbeam import AnalyticBeam
from . import get_standard_sim_params

@pytest.mark.parametrize('polarized', [False, True])
@pytest.mark.parametrize('precision', [2,1])
@pytest.mark.parametrize('use_analytic_beam', [True, False])
@pytest.mark.parametrize('tilt_array', [True, False])
@pytest.mark.parametrize('nprocesses', [1, 2])
def test_simulate(polarized: bool, precision: int, use_analytic_beam: bool, tilt_array: bool, nprocesses: int):

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
    
    if tilt_array:
        # Tilt the array
        ants = {ant: np.array(ants[ant]) + np.array([0, 0, ai * 5]) for ai, ant in enumerate(ants)}

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
        coord_method='CoordinateRotationERFA'
    )

    # Use fftvis to simulate visibilities
    fvis = simulate.simulate_vis(
        ants, flux, ra, dec, freqs, times.jd, cpu_beams[0], eps=1e-10 if precision==2 else 6e-8,
        baselines=sim_baselines, polarized=polarized, precision=precision,
        telescope_loc=location, nprocesses=nprocesses
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