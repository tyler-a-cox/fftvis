"""
Performance benchmark comparing fftvis CPU vs GPU implementations.

This test measures execution time for various problem sizes to quantify
GPU speedup across different scenarios.
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
import matplotlib.pyplot as plt
from tabulate import tabulate

from fftvis.wrapper import simulate_vis

# Check GPU availability
try:
    import cupy as cp
    from fftvis.gpu.nufft import HAVE_CUFINUFFT
    GPU_AVAILABLE = cp.cuda.is_available() and HAVE_CUFINUFFT
except ImportError:
    GPU_AVAILABLE = False
    cp = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestPerformanceCPUvsGPU:
    """Benchmark CPU vs GPU performance across various scenarios."""
    
    def create_setup(self, nsrc, nants, ntime, nfreq=1, polarized=False, is_coplanar=True):
        """Create a test setup with given parameters."""
        
        # Antenna positions
        ants = {}
        for i in range(nants):
            angle = 2 * np.pi * i / nants
            r = 10.0 * (i + 1)  # Increasing baseline lengths
            z_offset = 0.0 if is_coplanar else np.random.uniform(-2, 2)
            ants[i] = np.array([r * np.cos(angle), r * np.sin(angle), z_offset])
        
        # Frequencies
        if nfreq == 1:
            freqs = np.array([150e6])
        else:
            freqs = np.linspace(140e6, 160e6, nfreq)
        
        # Times
        start_time = Time('2020-01-01 00:00:00', scale='utc')
        times = start_time + np.arange(ntime) * un.minute
        
        # Sources distributed across the sky
        # Ensure sources are above horizon
        ra = np.random.uniform(-0.3, 0.3, nsrc)
        dec = np.random.uniform(-0.2, 0.2, nsrc)
        # Add some sources near zenith
        ra[:5] = np.linspace(-0.05, 0.05, min(5, nsrc))
        dec[:5] = np.linspace(-0.05, 0.05, min(5, nsrc))
        
        # Fluxes - 2D array for compatibility
        fluxes = np.ones((nsrc, nfreq))
        
        # Telescope location (HERA)
        telescope_loc = EarthLocation.from_geodetic(
            lat=-30.7215 * un.deg, 
            lon=21.4283 * un.deg, 
            height=1073 * un.m
        )
        
        # Simple beam
        beam = BeamInterface(AiryBeam(diameter=14.0))
        
        return {
            'ants': ants,
            'freqs': freqs,
            'fluxes': fluxes,
            'ra': ra,
            'dec': dec,
            'times': times,
            'telescope_loc': telescope_loc,
            'beam': beam,
            'polarized': polarized,
        }
    
    def benchmark_simulation(self, setup, precision=1, n_runs=3):
        """Run benchmark for a given setup."""
        
        # Common parameters
        sim_params = setup.copy()
        times = sim_params.pop('times')
        beam = sim_params.pop('beam')
        
        common_params = {
            **sim_params,
            'times': times.jd,
            'beam': beam,
            'precision': precision,
            'eps': 6e-8 if precision == 1 else 1e-10,
            'nprocesses': 1,
        }
        
        # Warm-up run for GPU
        try:
            _ = simulate_vis(backend='gpu', **common_params)
            if hasattr(cp, 'cuda'):
                cp.cuda.Stream.null.synchronize()
        except Exception as e:
            logger.warning(f"GPU warm-up failed: {e}")
            return None, None, None
        
        # CPU timing
        cpu_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            vis_cpu = simulate_vis(backend='cpu', **common_params)
            cpu_time = time.perf_counter() - start
            cpu_times.append(cpu_time)
        
        # GPU timing
        gpu_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            try:
                vis_gpu = simulate_vis(backend='gpu', **common_params)
                if hasattr(cp, 'cuda'):
                    cp.cuda.Stream.null.synchronize()
                gpu_time = time.perf_counter() - start
                gpu_times.append(gpu_time)
            except Exception as e:
                logger.warning(f"GPU run failed: {e}")
                return np.mean(cpu_times), None, None
        
        # Calculate statistics
        cpu_mean = np.mean(cpu_times)
        gpu_mean = np.mean(gpu_times) if gpu_times else None
        speedup = cpu_mean / gpu_mean if gpu_mean else None
        
        return cpu_mean, gpu_mean, speedup
    
    def test_varying_sources(self):
        """Test performance with varying number of sources."""
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK: Varying Number of Sources")
        logger.info("="*60)
        
        # Fixed parameters
        nants = 50  # More antennas for larger workload
        ntime = 10  # More time samples
        nfreq = 1
        polarized = False
        
        # Test different numbers of sources
        source_counts = [1000, 5000, 10000, 50000, 100000, 200000]
        results = []
        
        for nsrc in source_counts:
            logger.info(f"\nTesting with {nsrc} sources...")
            setup = self.create_setup(nsrc, nants, ntime, nfreq, polarized)
            
            cpu_time, gpu_time, speedup = self.benchmark_simulation(setup, precision=1)
            
            if gpu_time is not None:
                results.append({
                    'Sources': nsrc,
                    'CPU Time (s)': f"{cpu_time:.3f}",
                    'GPU Time (s)': f"{gpu_time:.3f}",
                    'Speedup': f"{speedup:.1f}x"
                })
            else:
                results.append({
                    'Sources': nsrc,
                    'CPU Time (s)': f"{cpu_time:.3f}",
                    'GPU Time (s)': "Failed",
                    'Speedup': "N/A"
                })
        
        # Display results
        print("\n" + tabulate(results, headers='keys', tablefmt='grid'))
    
    def test_varying_antennas(self):
        """Test performance with varying number of antennas."""
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK: Varying Number of Antennas")
        logger.info("="*60)
        
        # Fixed parameters
        nsrc = 1000
        ntime = 5
        nfreq = 1
        polarized = False
        
        # Test different numbers of antennas
        antenna_counts = [5, 10, 20, 50, 100]
        results = []
        
        for nants in antenna_counts:
            logger.info(f"\nTesting with {nants} antennas ({nants*(nants-1)//2} baselines)...")
            setup = self.create_setup(nsrc, nants, ntime, nfreq, polarized)
            
            cpu_time, gpu_time, speedup = self.benchmark_simulation(setup, precision=1)
            
            nbaselines = nants * (nants - 1) // 2
            
            if gpu_time is not None:
                results.append({
                    'Antennas': nants,
                    'Baselines': nbaselines,
                    'CPU Time (s)': f"{cpu_time:.3f}",
                    'GPU Time (s)': f"{gpu_time:.3f}",
                    'Speedup': f"{speedup:.1f}x"
                })
            else:
                results.append({
                    'Antennas': nants,
                    'Baselines': nbaselines,
                    'CPU Time (s)': f"{cpu_time:.3f}",
                    'GPU Time (s)': "Failed",
                    'Speedup': "N/A"
                })
        
        # Display results
        print("\n" + tabulate(results, headers='keys', tablefmt='grid'))
    
    def test_varying_time_samples(self):
        """Test performance with varying number of time samples."""
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK: Varying Number of Time Samples")
        logger.info("="*60)
        
        # Fixed parameters
        nsrc = 1000
        nants = 10
        nfreq = 1
        polarized = False
        
        # Test different numbers of time samples
        time_counts = [1, 5, 10, 50, 100, 200]
        results = []
        
        for ntime in time_counts:
            logger.info(f"\nTesting with {ntime} time samples...")
            setup = self.create_setup(nsrc, nants, ntime, nfreq, polarized)
            
            cpu_time, gpu_time, speedup = self.benchmark_simulation(setup, precision=1)
            
            if gpu_time is not None:
                results.append({
                    'Time Samples': ntime,
                    'CPU Time (s)': f"{cpu_time:.3f}",
                    'GPU Time (s)': f"{gpu_time:.3f}",
                    'Speedup': f"{speedup:.1f}x"
                })
            else:
                results.append({
                    'Time Samples': ntime,
                    'CPU Time (s)': f"{cpu_time:.3f}",
                    'GPU Time (s)': "Failed",
                    'Speedup': "N/A"
                })
        
        # Display results
        print("\n" + tabulate(results, headers='keys', tablefmt='grid'))
    
    def test_varying_complexity(self):
        """Test performance with varying complexity (polarization, precision, coplanarity)."""
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK: Varying Complexity")
        logger.info("="*60)
        
        # Fixed parameters
        nsrc = 1000
        nants = 10
        ntime = 5
        nfreq = 1
        
        # Test different complexity scenarios
        scenarios = [
            ("Unpolarized, Single, Coplanar", False, 1, True),
            ("Unpolarized, Double, Coplanar", False, 2, True),
            ("Polarized, Single, Coplanar", True, 1, True),
            ("Polarized, Double, Coplanar", True, 2, True),
            ("Unpolarized, Single, Non-coplanar", False, 1, False),
            ("Polarized, Double, Non-coplanar", True, 2, False),
        ]
        
        results = []
        
        for scenario_name, polarized, precision, is_coplanar in scenarios:
            logger.info(f"\nTesting: {scenario_name}...")
            setup = self.create_setup(nsrc, nants, ntime, nfreq, polarized, is_coplanar)
            
            cpu_time, gpu_time, speedup = self.benchmark_simulation(setup, precision)
            
            if gpu_time is not None:
                results.append({
                    'Scenario': scenario_name,
                    'CPU Time (s)': f"{cpu_time:.3f}",
                    'GPU Time (s)': f"{gpu_time:.3f}",
                    'Speedup': f"{speedup:.1f}x"
                })
            else:
                results.append({
                    'Scenario': scenario_name,
                    'CPU Time (s)': f"{cpu_time:.3f}",
                    'GPU Time (s)': "Failed",
                    'Speedup': "N/A"
                })
        
        # Display results
        print("\n" + tabulate(results, headers='keys', tablefmt='grid'))
    
    def test_scaling_analysis(self):
        """Comprehensive scaling analysis."""
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE SCALING ANALYSIS")
        logger.info("="*60)
        
        # Test matrix
        test_cases = [
            # (nsrc, nants, ntime, description)
            (100, 5, 1, "Small"),
            (500, 10, 5, "Medium"),
            (1000, 20, 10, "Large"),
            (5000, 30, 20, "X-Large"),
            (10000, 50, 50, "XX-Large"),
        ]
        
        results = []
        
        for nsrc, nants, ntime, size_name in test_cases:
            logger.info(f"\nTesting {size_name} problem size...")
            logger.info(f"  Sources: {nsrc}, Antennas: {nants}, Times: {ntime}")
            
            setup = self.create_setup(nsrc, nants, ntime, nfreq=1, polarized=False)
            
            cpu_time, gpu_time, speedup = self.benchmark_simulation(setup, precision=1)
            
            nbaselines = nants * (nants - 1) // 2
            total_ops = nsrc * nbaselines * ntime  # Approximate operation count
            
            if gpu_time is not None:
                results.append({
                    'Size': size_name,
                    'Sources': nsrc,
                    'Antennas': nants,
                    'Times': ntime,
                    'Total Ops (M)': f"{total_ops/1e6:.1f}",
                    'CPU (s)': f"{cpu_time:.2f}",
                    'GPU (s)': f"{gpu_time:.2f}",
                    'Speedup': f"{speedup:.1f}x"
                })
            else:
                results.append({
                    'Size': size_name,
                    'Sources': nsrc,
                    'Antennas': nants,
                    'Times': ntime,
                    'Total Ops (M)': f"{total_ops/1e6:.1f}",
                    'CPU (s)': f"{cpu_time:.2f}",
                    'GPU (s)': "OOM",
                    'Speedup': "N/A"
                })
        
        # Display results
        print("\n" + tabulate(results, headers='keys', tablefmt='grid'))
        
        # Summary
        valid_speedups = [r['Speedup'] for r in results if r['Speedup'] != 'N/A']
        if valid_speedups:
            speedup_values = [float(s.replace('x', '')) for s in valid_speedups]
            avg_speedup = np.mean(speedup_values)
            max_speedup = np.max(speedup_values)
            min_speedup = np.min(speedup_values)
            
            logger.info(f"\nSPEEDUP SUMMARY:")
            logger.info(f"  Average: {avg_speedup:.1f}x")
            logger.info(f"  Maximum: {max_speedup:.1f}x")
            logger.info(f"  Minimum: {min_speedup:.1f}x")


if __name__ == "__main__":
    # Run specific tests or all
    import sys
    if len(sys.argv) > 1:
        pytest.main([__file__, f"::TestPerformanceCPUvsGPU::{sys.argv[1]}", "-v", "-s"])
    else:
        pytest.main([__file__, "-v", "-s"])f