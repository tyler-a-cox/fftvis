import sys
import pytest
import numpy as np

# Monkey patch pyuvdata.telescopes before importing matvis
import pyuvdata.telescopes
if not hasattr(pyuvdata.telescopes, 'get_telescope'):
    from pyuvdata import Telescope
    
    def get_telescope(telescope_name, **kwargs):
        """Compatibility function to create a Telescope object from name."""
        return Telescope.from_known_telescopes(telescope_name, **kwargs)
    
    # Add the function to the module
    pyuvdata.telescopes.get_telescope = get_telescope

# Now import matvis after the patch
import matvis
from matvis._test_utils import get_standard_sim_params
from fftvis.wrapper import simulate_vis


@pytest.mark.parametrize("polarized", [False, True])
@pytest.mark.parametrize("precision", [2, 1])
@pytest.mark.parametrize("use_analytic_beam", [True, False])
@pytest.mark.parametrize("tilt_array", [True, False])
@pytest.mark.parametrize("nprocesses", [1, 2])
@pytest.mark.parametrize("backend", ["cpu"])  # Add GPU backend when implemented
def test_simulate(
    polarized: bool,
    precision: int,
    use_analytic_beam: bool,
    tilt_array: bool,
    nprocesses: int,
    backend: str,
):
    """Test the simulation of interferometric visibilities with the CPU backend.
    
    This test compares the visibilities simulated by fftvis with those from matvis
    (used as a reference). It verifies that:
    1. The simulation correctly handles various combinations of parameters
    2. Results match the reference implementation within precision tolerance
    3. Output shapes are correct for polarized and non-polarized cases
    4. Simulations work with both specific baselines and all baselines
    5. The implementation works correctly with multiple processes
    
    Parameters
    ----------
    polarized : bool
        Whether to use polarized beam patterns and calculate full polarization visibilities
    precision : int
        Numerical precision to use (1=single, 2=double)
    use_analytic_beam : bool
        Whether to use an analytic beam model (True) or a tabulated beam (False)
    tilt_array : bool
        Whether to add vertical offsets to create a non-coplanar array
    nprocesses : int
        Number of processes to use for parallelization
    backend : str
        Computation backend to use ('cpu' or 'gpu')
    """
    if sys.platform == "darwin" and nprocesses == 2:
        pytest.skip("Cannot use Ray multiprocessing on MacOS")

    params, *_ = get_standard_sim_params(
        use_analytic_beam=use_analytic_beam, polarized=polarized
    )
    ants = params.pop("ants")
    if tilt_array:
        # Tilt the array
        ants = {
            ant: np.array(ants[ant]) + np.array([0, 0, ai * 5])
            for ai, ant in enumerate(ants)
        }

    # Simulate with specified baselines
    sim_baselines = np.array([[0, 1]])  # , [0, 2], [1, 2]])

    # Use matvis as a reference
    mvis = matvis.simulate_vis(
        ants=ants,
        precision=precision,
        antpairs=sim_baselines,
        coord_method="CoordinateRotationERFA",
        source_buffer=0.75,
        **params,
    )

    beam = params.pop("beams")[0]
    times = params.pop("times").jd

    # Use fftvis to simulate visibilities
    fvis = simulate_vis(
        ants,
        eps=1e-10 if precision == 2 else 6e-8,
        baselines=sim_baselines,
        precision=precision,
        nprocesses=nprocesses,
        coord_method_params={"source_buffer": 0.75},
        trace_mem=False,
        beam=beam,
        times=times,
        backend=backend,
        **params,
    )

    fvis_all_bls = simulate_vis(
        ants,
        eps=1e-10 if precision == 2 else 6e-8,
        precision=precision,
        nprocesses=nprocesses,
        coord_method_params={"source_buffer": 0.75},
        trace_mem=False,
        beam=beam,
        times=times,
        backend=backend,
        **params,
    )

    # Check shape of result when no baselines are specified
    nbls = (len(ants) * (len(ants) - 1)) // 2 + 1
    freqs = params["freqs"]

    # Should have shape (nfreqs, ntimes, nants, nants)
    if polarized:
        assert fvis.shape == (len(freqs), len(times), 2, 2, len(sim_baselines))
        assert fvis_all_bls.shape == (len(params["freqs"]), len(times), 2, 2, nbls)
    else:
        assert fvis.shape == (len(freqs), len(times), len(sim_baselines))
        assert fvis_all_bls.shape == (len(params["freqs"]), len(times), nbls)

    # Check that the polarized results are the same
    for bi, bl in enumerate(sim_baselines):
        print(bl, mvis.shape)
        np.testing.assert_allclose(
            fvis[..., bi], mvis[:, :, bi], atol=1e-5 if precision == 2 else 1e-4
        )
