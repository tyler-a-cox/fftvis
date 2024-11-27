import matvis
import pytest
import numpy as np
from fftvis import simulate
from matvis._test_utils import get_standard_sim_params
import sys

@pytest.mark.parametrize('polarized', [False, True])
@pytest.mark.parametrize('precision', [2,1])
@pytest.mark.parametrize('use_analytic_beam', [True, False])
@pytest.mark.parametrize('tilt_array', [True, False])
@pytest.mark.parametrize('nprocesses', [1, 2])
def test_simulate(polarized: bool, precision: int, use_analytic_beam: bool, tilt_array: bool, nprocesses: int):
    if sys.platform == "darwin" and nprocesses == 2:
        pytest.skip("Cannot use Ray multiprocessing on MacOS")
        
    params, *_ = get_standard_sim_params(
        use_analytic_beam=use_analytic_beam, polarized=polarized
    )
    ants = params.pop('ants')
    if tilt_array:
        # Tilt the array
        ants = {ant: np.array(ants[ant]) + np.array([0, 0, ai * 5]) for ai, ant in enumerate(ants)}
    
    # Simulate with specified baselines
    sim_baselines = np.array([[0, 1]])#, [0, 2], [1, 2]])

    # Use matvis as a reference
    mvis = matvis.simulate_vis(
        ants=ants,
        precision=precision,
        antpairs=sim_baselines,
        coord_method='CoordinateRotationERFA',
        source_buffer=0.75,
        **params
    )

    beam = params.pop("beams")[0]
    times = params.pop('times').jd

    # Use fftvis to simulate visibilities
    fvis = simulate.simulate_vis(
        ants, eps=1e-10 if precision==2 else 6e-8,
        baselines=sim_baselines, precision=precision,
        nprocesses=nprocesses,
        coord_method_params={"source_buffer": 0.75},
        trace_mem=False,
        beam=beam,
        times=times,
        **params
    )

    fvis_all_bls = simulate.simulate_vis(
        ants, eps=1e-10 if precision==2 else 6e-8,
        precision=precision,
        nprocesses=nprocesses,
        coord_method_params={"source_buffer": 0.75},
        trace_mem=False,
        beam=beam,
        times=times,
        **params
    )

    # Check that the results are the same
    np.testing.assert_allclose(
        fvis, fvis_all_bls[:, :, 1], 
        atol=1e-5 if precision==2 else 1e-4
    )

    freqs = params['freqs']
    
    # Should have shape (nfreqs, ntimes, nants, nants)
    if polarized:
        assert fvis.shape == (len(freqs), len(times), 2, 2, len(sim_baselines))
    else:
        assert fvis.shape == (len(freqs), len(times), len(sim_baselines))

    # Check that the polarized results are the same
    for bi, bl in enumerate(sim_baselines):
        print(bl, mvis.shape)
        np.testing.assert_allclose(
            fvis[..., bi], mvis[:, :, bi], 
            atol=1e-5 if precision==2 else 1e-4
        )