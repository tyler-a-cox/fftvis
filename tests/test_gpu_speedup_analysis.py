"""
Detailed GPU speedup analysis focusing on compute-intensive scenarios.
"""

import pytest
import numpy as np
import time
import logging
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy import units as un
from pyuvdata.analytic_beam import AiryBeam
from pyuvdata.beam_interface import BeamInterface

# Direct imports to measure core computation
from fftvis.cpu.cpu_simulate import CPUSimulationEngine
from fftvis import utils
from matvis.core.coords import CoordinateRotation
from astropy.coordinates import SkyCoord

# Check GPU availability
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
    
    if GPU_AVAILABLE:
        # Try importing GPU modules
        try:
            from fftvis.gpu.gpu_simulate import GPUSimulationEngine
            from fftvis.gpu.nufft import HAVE_CUFINUFFT
            GPU_AVAILABLE = GPU_AVAILABLE and HAVE_CUFINUFFT
        except ImportError:
            GPU_AVAILABLE = False
except ImportError:
    GPU_AVAILABLE = False
    cp = None

# Import tabulate only if available
try:
    from tabulate import tabulate
    HAVE_TABULATE = True
except ImportError:
    HAVE_TABULATE = False
    # Simple fallback for tabulate
    def tabulate(data, headers=None, tablefmt=None):
        if headers == 'keys' and isinstance(data, list) and len(data) > 0:
            headers = list(data[0].keys())
            rows = [list(d.values()) for d in data]
        else:
            rows = data
        
        # Simple text table  
        if headers:
            print("  ".join(str(h) for h in headers))
            print("-" * 60)
        for row in rows:
            print("  ".join(str(cell) for cell in row))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestGPUSpeedupAnalysis:
    """Detailed analysis of GPU speedup factors."""
    
    def test_core_nufft_performance(self):
        """Test raw NUFFT performance CPU vs GPU."""
        logger.info("\n" + "="*60)
        logger.info("CORE NUFFT PERFORMANCE COMPARISON")
        logger.info("="*60)
        
        from fftvis.cpu.nufft import cpu_nufft3d
        if GPU_AVAILABLE:
            from fftvis.gpu.nufft import gpu_nufft3d
        else:
            # This should never be reached due to skipif decorator
            return
        
        # Test different problem sizes
        test_cases = [
            (100, 100),      # 100 sources, 100 baselines
            (1000, 100),     # 1000 sources, 100 baselines  
            (10000, 100),    # 10k sources, 100 baselines
            (100000, 100),   # 100k sources, 100 baselines
            (1000, 1000),    # 1000 sources, 1000 baselines
            (10000, 1000),   # 10k sources, 1000 baselines
        ]
        
        results = []
        
        for nsrc, nbl in test_cases:
            logger.info(f"\nTesting NUFFT with {nsrc} sources, {nbl} baselines...")
            
            # Create test data
            # Source positions (l,m,n)
            x = np.random.uniform(-0.5, 0.5, nsrc).astype(np.float64)
            y = np.random.uniform(-0.5, 0.5, nsrc).astype(np.float64)
            z = np.sqrt(np.maximum(0, 1 - x**2 - y**2)) - 1  # n = sqrt(1-l^2-m^2) - 1
            
            # Baseline coordinates (u,v,w)
            u = np.random.uniform(-100, 100, nbl).astype(np.float64)
            v = np.random.uniform(-100, 100, nbl).astype(np.float64)
            w = np.random.uniform(-100, 100, nbl).astype(np.float64)
            
            # Weights (complex visibility contributions)
            weights = (np.random.randn(nsrc) + 1j * np.random.randn(nsrc)).astype(np.complex128)
            
            # CPU timing
            cpu_times = []
            for _ in range(3):
                start = time.perf_counter()
                vis_cpu = cpu_nufft3d(x, y, z, weights, u, v, w, eps=1e-6)
                cpu_time = time.perf_counter() - start
                cpu_times.append(cpu_time)
            cpu_mean = np.mean(cpu_times[1:])  # Skip first for warm-up
            
            # GPU timing
            x_gpu = cp.asarray(x)
            y_gpu = cp.asarray(y)
            z_gpu = cp.asarray(z)
            u_gpu = cp.asarray(u)
            v_gpu = cp.asarray(v)
            w_gpu = cp.asarray(w)
            weights_gpu = cp.asarray(weights)
            
            # Warm-up
            _ = gpu_nufft3d(x_gpu, y_gpu, z_gpu, weights_gpu, u_gpu, v_gpu, w_gpu, eps=1e-6)
            cp.cuda.Stream.null.synchronize()
            
            gpu_times = []
            for _ in range(3):
                start = time.perf_counter()
                vis_gpu = gpu_nufft3d(x_gpu, y_gpu, z_gpu, weights_gpu, u_gpu, v_gpu, w_gpu, eps=1e-6)
                cp.cuda.Stream.null.synchronize()
                gpu_time = time.perf_counter() - start
                gpu_times.append(gpu_time)
            gpu_mean = np.mean(gpu_times)
            
            speedup = cpu_mean / gpu_mean
            
            results.append({
                'Sources': nsrc,
                'Baselines': nbl,
                'CPU Time (ms)': f"{cpu_mean*1000:.1f}",
                'GPU Time (ms)': f"{gpu_mean*1000:.1f}",
                'Speedup': f"{speedup:.1f}x"
            })
            
            # Verify results match
            vis_gpu_cpu = cp.asnumpy(vis_gpu)
            max_diff = np.max(np.abs(vis_cpu - vis_gpu_cpu))
            logger.info(f"  Max difference: {max_diff:.2e}")
        
        # Display results
        print("\n" + tabulate(results, headers='keys', tablefmt='grid'))
    
    def test_full_simulation_breakdown(self):
        """Break down where time is spent in full simulation."""
        logger.info("\n" + "="*60)
        logger.info("FULL SIMULATION TIME BREAKDOWN")
        logger.info("="*60)
        
        # Create a moderate test case
        nsrc = 10000
        nants = 30
        ntime = 5
        
        # Setup
        ants = {}
        for i in range(nants):
            angle = 2 * np.pi * i / nants
            r = 10.0 * (i + 1)
            ants[i] = np.array([r * np.cos(angle), r * np.sin(angle), 0.0])
        
        freqs = np.array([150e6])
        ra = np.random.uniform(-0.2, 0.2, nsrc)
        dec = np.random.uniform(-0.1, 0.1, nsrc)
        fluxes = np.ones((nsrc, 1))
        
        start_time = Time('2020-01-01 00:00:00', scale='utc')
        times = start_time + np.arange(ntime) * un.minute
        
        telescope_loc = EarthLocation.from_geodetic(
            lat=-30.7215 * un.deg, lon=21.4283 * un.deg, height=1073 * un.m
        )
        
        beam = BeamInterface(AiryBeam(diameter=14.0))
        
        # Measure CPU simulation with timing
        logger.info(f"\nProblem size: {nsrc} sources, {nants} antennas, {ntime} times")
        logger.info(f"Total baselines: {nants*(nants-1)//2}")
        
        # Time full CPU simulation
        start_total = time.perf_counter()
        cpu_engine = CPUSimulationEngine()
        vis_cpu = cpu_engine.simulate(
            ants=ants,
            freqs=freqs,
            fluxes=fluxes,
            beam=beam,
            ra=ra,
            dec=dec,
            times=times.jd,
            telescope_loc=telescope_loc,
            precision=1,
            eps=1e-6,
            nprocesses=1,
        )
        cpu_total_time = time.perf_counter() - start_total
        
        # Time full GPU simulation
        start_total = time.perf_counter()
        gpu_engine = GPUSimulationEngine()
        vis_gpu = gpu_engine.simulate(
            ants=ants,
            freqs=freqs,
            fluxes=fluxes,
            beam=beam,
            ra=ra,
            dec=dec,
            times=times.jd,
            telescope_loc=telescope_loc,
            precision=1,
            eps=1e-6,
            nprocesses=1,
        )
        gpu_total_time = time.perf_counter() - start_total
        
        logger.info(f"\nTotal simulation time:")
        logger.info(f"  CPU: {cpu_total_time:.3f} seconds")
        logger.info(f"  GPU: {gpu_total_time:.3f} seconds")
        logger.info(f"  Speedup: {cpu_total_time/gpu_total_time:.1f}x")
        
        # Verify results match
        if isinstance(vis_gpu, cp.ndarray):
            vis_gpu = cp.asnumpy(vis_gpu)
        max_diff = np.max(np.abs(vis_cpu - vis_gpu))
        logger.info(f"\nMax difference between CPU and GPU: {max_diff:.2e}")
    
    def test_memory_bandwidth_limited(self):
        """Test scenarios that are memory bandwidth limited."""
        logger.info("\n" + "="*60)
        logger.info("MEMORY BANDWIDTH LIMITED SCENARIOS")
        logger.info("="*60)
        
        # Very large number of sources but few baselines
        # This tests memory transfer overhead
        test_cases = [
            (100000, 10, 1),    # Many sources, few baselines, 1 time
            (500000, 10, 1),    # Even more sources
            (10000, 100, 10),   # Balanced case
            (1000, 1000, 10),   # Many baselines
        ]
        
        results = []
        
        for nsrc, nants, ntime in test_cases:
            logger.info(f"\nTesting {nsrc} sources, {nants} antennas, {ntime} times...")
            
            # Create minimal setup
            ants = {i: np.array([10*i, 0, 0]) for i in range(nants)}
            freqs = np.array([150e6])
            ra = np.random.uniform(-0.1, 0.1, nsrc)
            dec = np.random.uniform(-0.1, 0.1, nsrc)
            fluxes = np.ones((nsrc, 1))
            times = Time(['2020-01-01 00:00:00'], scale='utc')
            if ntime > 1:
                times = Time('2020-01-01 00:00:00', scale='utc') + np.arange(ntime) * un.minute
            
            telescope_loc = EarthLocation.from_geodetic(
                lat=-30.7215 * un.deg, lon=21.4283 * un.deg, height=1073 * un.m
            )
            beam = BeamInterface(AiryBeam(diameter=14.0))
            
            # Time simulations
            try:
                # CPU
                start = time.perf_counter()
                cpu_engine = CPUSimulationEngine()
                vis_cpu = cpu_engine.simulate(
                    ants=ants, freqs=freqs, fluxes=fluxes, beam=beam,
                    ra=ra, dec=dec, times=times.jd if hasattr(times, 'jd') else times,
                    telescope_loc=telescope_loc, precision=1, eps=1e-6, nprocesses=1,
                )
                cpu_time = time.perf_counter() - start
                
                # GPU
                start = time.perf_counter()
                gpu_engine = GPUSimulationEngine()
                vis_gpu = gpu_engine.simulate(
                    ants=ants, freqs=freqs, fluxes=fluxes, beam=beam,
                    ra=ra, dec=dec, times=times.jd if hasattr(times, 'jd') else times,
                    telescope_loc=telescope_loc, precision=1, eps=1e-6, nprocesses=1,
                )
                gpu_time = time.perf_counter() - start
                
                speedup = cpu_time / gpu_time
                nbl = nants * (nants - 1) // 2
                
                results.append({
                    'Sources': nsrc,
                    'Antennas': nants,
                    'Baselines': nbl,
                    'Times': ntime,
                    'CPU (s)': f"{cpu_time:.3f}",
                    'GPU (s)': f"{gpu_time:.3f}",
                    'Speedup': f"{speedup:.1f}x"
                })
                
            except Exception as e:
                logger.warning(f"Failed: {e}")
                results.append({
                    'Sources': nsrc,
                    'Antennas': nants,
                    'Baselines': nants * (nants - 1) // 2,
                    'Times': ntime,
                    'CPU (s)': "N/A",
                    'GPU (s)': "Failed",
                    'Speedup': "N/A"
                })
        
        print("\n" + tabulate(results, headers='keys', tablefmt='grid'))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])