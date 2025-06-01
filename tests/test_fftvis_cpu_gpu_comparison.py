"""
Direct comparison test between fftvis CPU and GPU implementations.
"""

import pytest
import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy import units as un
from pyuvdata.analytic_beam import AiryBeam
from pyuvdata.beam_interface import BeamInterface

from fftvis.wrapper import simulate_vis

# Check GPU availability
try:
    import cupy as cp
    from fftvis.gpu.nufft import HAVE_CUFINUFFT
    GPU_AVAILABLE = cp.cuda.is_available() and HAVE_CUFINUFFT
except ImportError:
    GPU_AVAILABLE = False
    cp = None


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestFFTvisCPUvsGPU:
    """Direct comparison of fftvis CPU and GPU implementations."""
    
    @pytest.mark.parametrize("polarized", [False, True])
    @pytest.mark.parametrize("precision", [1, 2])
    @pytest.mark.parametrize("is_coplanar", [True, False])
    def test_cpu_vs_gpu_exact_match(self, polarized, precision, is_coplanar):
        """Test that CPU and GPU give exactly the same results."""
        
        # Setup
        ants = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([10.0, 0.0, 0.0 if is_coplanar else 1.0]),
            2: np.array([0.0, 10.0, 0.0 if is_coplanar else 0.5]),
        }
        
        freqs = np.array([150e6])
        
        # Sources
        nsrc = 5
        ra = np.array([0.0, 0.01, -0.01, 0.02, -0.02])
        dec = np.array([0.0, 0.01, -0.01, 0.0, 0.0])
        fluxes = np.ones((nsrc, 1))  # 2D for matvis compatibility
        
        times = Time(['2020-01-01 00:00:00'], scale='utc')
        
        telescope_loc = EarthLocation.from_geodetic(
            lat=-30.7215 * un.deg, lon=21.4283 * un.deg, height=1073 * un.m
        )
        
        # Use AiryBeam
        beam = BeamInterface(AiryBeam(diameter=14.0))
        
        # Common parameters
        baselines = [(0, 1), (0, 2), (1, 2)]
        eps = 1e-10 if precision == 2 else 6e-8
        
        common_params = {
            'ants': ants,
            'freqs': freqs,
            'fluxes': fluxes,
            'beam': beam,
            'ra': ra,
            'dec': dec,
            'times': times.jd,
            'telescope_loc': telescope_loc,
            'polarized': polarized,
            'precision': precision,
            'eps': eps,
            'baselines': baselines,
            'nprocesses': 1,
        }
        
        # Run CPU
        vis_cpu = simulate_vis(backend='cpu', **common_params)
        
        # Run GPU
        vis_gpu = simulate_vis(backend='gpu', **common_params)
        
        # Convert GPU result to CPU if needed
        if hasattr(vis_gpu, 'get'):
            vis_gpu = vis_gpu.get()
        elif isinstance(vis_gpu, cp.ndarray):
            vis_gpu = cp.asnumpy(vis_gpu)
        
        # Check shapes
        assert vis_cpu.shape == vis_gpu.shape, \
            f"Shape mismatch: CPU {vis_cpu.shape} vs GPU {vis_gpu.shape}"
        
        # Check values with tight tolerance
        rtol = 1e-10 if precision == 2 else 1e-6
        atol = 1e-12 if precision == 2 else 1e-8
        
        np.testing.assert_allclose(
            vis_cpu, vis_gpu,
            rtol=rtol, atol=atol,
            err_msg=f"CPU and GPU results differ for polarized={polarized}, "
                    f"precision={precision}, is_coplanar={is_coplanar}"
        )
        
        # Print max difference for debugging
        max_diff = np.max(np.abs(vis_cpu - vis_gpu))
        print(f"\nMax difference: {max_diff:.2e} (polarized={polarized}, "
              f"precision={precision}, is_coplanar={is_coplanar})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])