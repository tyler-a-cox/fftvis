"""
Test suite for comparing CPU and GPU beam evaluations.

This module specifically tests the beam implementations in both CPU and GPU backends
to ensure they produce consistent results across different scenarios.
"""

import numpy as np
import pytest

from fftvis.cpu.cpu_beams import CPUBeamEvaluator

# Try to import GPU functions
try:
    import cupy as cp
    from fftvis.gpu.gpu_beams import GPUBeamEvaluator
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU support not available")
class TestBeamComparison:
    """Test CPU vs GPU beam evaluation consistency."""
    
    def setup_method(self):
        """Set up test data and beam objects."""
        np.random.seed(42)
        
        # Test parameters
        self.freq = 150e6  # 150 MHz
        self.n_sources = 100
        
        # Create test beam - using a simple Gaussian beam for testing
        self.beam = self._create_test_beam()
        
        # Create evaluators
        self.cpu_evaluator = CPUBeamEvaluator()
        self.gpu_evaluator = GPUBeamEvaluator()
        
    def _create_test_beam(self):
        """Create a simple test beam object using an analytic beam."""
        # Use AiryBeam which is simpler and doesn't require all the UVBeam setup
        from pyuvdata.analytic_beam import AiryBeam
        from pyuvdata.beam_interface import BeamInterface
        
        # Create an Airy beam with 14m diameter
        analytic_beam = AiryBeam(diameter=14.0)
        
        # Convert to UVBeam for compatibility with both CPU and GPU
        # This is similar to what's done in the GPU tutorial
        freq_array = np.array([self.freq])  # Use our test frequency
        naz, nza = 360, 180  # Angular resolution
        
        beam = analytic_beam.to_uvbeam(
            freq_array=freq_array,
            axis1_array=np.linspace(0, 2 * np.pi, naz + 1)[:-1],  # azimuth
            axis2_array=np.linspace(0, np.pi, nza + 1),  # zenith angle
        )
        
        # Wrap in BeamInterface for compatibility
        return BeamInterface(beam)
    
    def test_evaluate_beam_unpolarized(self):
        """Test unpolarized beam evaluation consistency between CPU and GPU."""
        # Generate random sky positions
        az = np.random.uniform(0, 2 * np.pi, self.n_sources)
        za = np.random.uniform(0, np.pi / 4, self.n_sources)  # Keep within reasonable range
        
        # CPU evaluation
        cpu_result = self.cpu_evaluator.evaluate_beam(
            self.beam, az, za, 
            polarized=False, 
            freq=self.freq,
            check=True
        )
        
        # GPU evaluation
        az_gpu = cp.asarray(az)
        za_gpu = cp.asarray(za)
        gpu_result = self.gpu_evaluator.evaluate_beam(
            self.beam, az_gpu, za_gpu,
            polarized=False,
            freq=self.freq,
            check=True
        )
        
        # Compare results
        gpu_result_cpu = cp.asnumpy(gpu_result)
        np.testing.assert_allclose(
            cpu_result, gpu_result_cpu,
            rtol=1e-6, atol=1e-8,
            err_msg="Unpolarized beam: CPU and GPU results differ"
        )
        
    def test_evaluate_beam_polarized(self):
        """Test polarized beam evaluation consistency between CPU and GPU."""
        # Generate random sky positions
        az = np.random.uniform(0, 2 * np.pi, self.n_sources)
        za = np.random.uniform(0, np.pi / 4, self.n_sources)
        
        # CPU evaluation
        cpu_result = self.cpu_evaluator.evaluate_beam(
            self.beam, az, za,
            polarized=True,
            freq=self.freq,
            check=True
        )
        
        # GPU evaluation
        az_gpu = cp.asarray(az)
        za_gpu = cp.asarray(za)
        gpu_result = self.gpu_evaluator.evaluate_beam(
            self.beam, az_gpu, za_gpu,
            polarized=True,
            freq=self.freq,
            check=True
        )
        
        # Compare results
        gpu_result_cpu = cp.asnumpy(gpu_result)
        np.testing.assert_allclose(
            cpu_result, gpu_result_cpu,
            rtol=1e-6, atol=1e-8,
            err_msg="Polarized beam: CPU and GPU results differ"
        )
        
        # Check shape
        assert cpu_result.shape == (2, 2, self.n_sources), "CPU polarized beam shape incorrect"
        assert gpu_result_cpu.shape == (2, 2, self.n_sources), "GPU polarized beam shape incorrect"
        
    def test_get_apparent_flux_polarized(self):
        """Test apparent flux calculation consistency between CPU and GPU."""
        # Generate test data
        nax, nfd, nsrc = 2, 2, 50
        
        # Create beam data with complex values
        beam_data_cpu = np.random.randn(nax, nfd, nsrc) + 1j * np.random.randn(nax, nfd, nsrc)
        beam_data_gpu = cp.asarray(beam_data_cpu.copy())
        
        # Create flux data
        flux_cpu = np.random.randn(nsrc) + 1j * np.random.randn(nsrc)
        flux_gpu = cp.asarray(flux_cpu)
        
        # CPU calculation (modifies beam_data_cpu in place)
        CPUBeamEvaluator.get_apparent_flux_polarized(beam_data_cpu, flux_cpu)
        
        # GPU calculation (modifies beam_data_gpu in place)
        self.gpu_evaluator.get_apparent_flux_polarized(beam_data_gpu, flux_gpu)
        
        # Compare results
        gpu_result_cpu = cp.asnumpy(beam_data_gpu)
        np.testing.assert_allclose(
            beam_data_cpu, gpu_result_cpu,
            rtol=1e-6, atol=1e-8,
            err_msg="Apparent flux calculation: CPU and GPU results differ"
        )
        
    def test_beam_at_zenith(self):
        """Test beam evaluation at zenith (za=0) for consistency."""
        # Single source at zenith
        az = np.array([0.0])
        za = np.array([0.0])
        
        # CPU evaluation
        cpu_result = self.cpu_evaluator.evaluate_beam(
            self.beam, az, za,
            polarized=True,
            freq=self.freq
        )
        
        # GPU evaluation
        gpu_result = self.gpu_evaluator.evaluate_beam(
            self.beam, cp.asarray(az), cp.asarray(za),
            polarized=True,
            freq=self.freq
        )
        
        # At zenith, beam should be maximum
        gpu_result_cpu = cp.asnumpy(gpu_result)
        np.testing.assert_allclose(
            cpu_result, gpu_result_cpu,
            rtol=1e-6, atol=1e-8,
            err_msg="Beam at zenith: CPU and GPU results differ"
        )
        
        # Check that values are reasonable (should be close to 1 for normalized beam)
        assert np.all(np.abs(cpu_result) <= 1.1), "CPU beam values at zenith seem too large"
        assert np.all(np.abs(gpu_result_cpu) <= 1.1), "GPU beam values at zenith seem too large"
        
    def test_beam_at_horizon(self):
        """Test beam evaluation near horizon for consistency."""
        # Sources near horizon
        n_horizon = 10
        az = np.linspace(0, 2 * np.pi, n_horizon)
        za = np.full(n_horizon, np.pi / 2 - 0.1)  # Just above horizon
        
        # CPU evaluation
        cpu_result = self.cpu_evaluator.evaluate_beam(
            self.beam, az, za,
            polarized=False,
            freq=self.freq
        )
        
        # GPU evaluation
        gpu_result = self.gpu_evaluator.evaluate_beam(
            self.beam, cp.asarray(az), cp.asarray(za),
            polarized=False,
            freq=self.freq
        )
        
        # Compare results
        gpu_result_cpu = cp.asnumpy(gpu_result)
        np.testing.assert_allclose(
            cpu_result, gpu_result_cpu,
            rtol=1e-6, atol=1e-8,
            err_msg="Beam at horizon: CPU and GPU results differ"
        )
        
        # Near horizon, beam should be very small
        assert np.all(np.abs(cpu_result) < 0.1), "CPU beam values at horizon seem too large"
        assert np.all(np.abs(gpu_result_cpu) < 0.1), "GPU beam values at horizon seem too large"
        
    def test_empty_source_list(self):
        """Test handling of empty source lists."""
        # Empty arrays
        az = np.array([], dtype=np.float64)
        za = np.array([], dtype=np.float64)
        
        # CPU evaluation
        cpu_result = self.cpu_evaluator.evaluate_beam(
            self.beam, az, za,
            polarized=True,
            freq=self.freq
        )
        
        # GPU evaluation
        gpu_result = self.gpu_evaluator.evaluate_beam(
            self.beam, cp.asarray(az), cp.asarray(za),
            polarized=True,
            freq=self.freq
        )
        
        # Both should return empty arrays with correct shape
        assert cpu_result.shape == (2, 2, 0), "CPU empty result shape incorrect"
        assert gpu_result.shape == (2, 2, 0), "GPU empty result shape incorrect"
        
    def test_single_source(self):
        """Test beam evaluation for a single source."""
        # Single source
        az = np.array([np.pi / 4])
        za = np.array([np.pi / 6])
        
        # CPU evaluation
        cpu_result = self.cpu_evaluator.evaluate_beam(
            self.beam, az, za,
            polarized=True,
            freq=self.freq
        )
        
        # GPU evaluation
        gpu_result = self.gpu_evaluator.evaluate_beam(
            self.beam, cp.asarray(az), cp.asarray(za),
            polarized=True,
            freq=self.freq
        )
        
        # Compare results
        gpu_result_cpu = cp.asnumpy(gpu_result)
        np.testing.assert_allclose(
            cpu_result, gpu_result_cpu,
            rtol=1e-6, atol=1e-8,
            err_msg="Single source beam: CPU and GPU results differ"
        )
        
        assert cpu_result.shape == (2, 2, 1), "CPU single source shape incorrect"
        assert gpu_result_cpu.shape == (2, 2, 1), "GPU single source shape incorrect"
        
    def test_spline_options(self):
        """Test beam evaluation with different spline options."""
        # Generate test positions
        az = np.random.uniform(0, 2 * np.pi, 20)
        za = np.random.uniform(0, np.pi / 4, 20)
        
        # Test with specific spline options
        spline_opts = {"order": 1}  # Linear interpolation
        
        # CPU evaluation
        cpu_result = self.cpu_evaluator.evaluate_beam(
            self.beam, az, za,
            polarized=False,
            freq=self.freq,
            spline_opts=spline_opts
        )
        
        # GPU evaluation
        gpu_result = self.gpu_evaluator.evaluate_beam(
            self.beam, cp.asarray(az), cp.asarray(za),
            polarized=False,
            freq=self.freq,
            spline_opts=spline_opts
        )
        
        # Compare results
        gpu_result_cpu = cp.asnumpy(gpu_result)
        np.testing.assert_allclose(
            cpu_result, gpu_result_cpu,
            rtol=1e-6, atol=1e-8,
            err_msg="Beam with spline options: CPU and GPU results differ"
        )


    def test_airy_beam_direct_gpu(self):
        """Test AiryBeam evaluation directly on GPU without conversion to UVBeam."""
        from pyuvdata.analytic_beam import AiryBeam
        from pyuvdata.beam_interface import BeamInterface
        
        # Create AiryBeam
        diameter = 14.0
        airy_beam = AiryBeam(diameter=diameter)
        beam_interface = BeamInterface(airy_beam)
        
        # Generate test positions
        az = np.random.uniform(0, 2 * np.pi, self.n_sources)
        za = np.random.uniform(0, np.pi / 4, self.n_sources)
        
        # CPU evaluation (AiryBeam works natively on CPU)
        cpu_result = self.cpu_evaluator.evaluate_beam(
            beam_interface, az, za,
            polarized=False,
            freq=self.freq
        )
        
        # GPU evaluation (should use new GPU implementation)
        gpu_result = self.gpu_evaluator.evaluate_beam(
            beam_interface, cp.asarray(az), cp.asarray(za),
            polarized=False,
            freq=self.freq
        )
        
        # Compare results
        gpu_result_cpu = cp.asnumpy(gpu_result)
        np.testing.assert_allclose(
            cpu_result, gpu_result_cpu,
            rtol=1e-6, atol=1e-8,
            err_msg="AiryBeam direct GPU: CPU and GPU results differ"
        )
        
    def test_airy_beam_direct_gpu_polarized(self):
        """Test polarized AiryBeam evaluation directly on GPU."""
        from pyuvdata.analytic_beam import AiryBeam
        from pyuvdata.beam_interface import BeamInterface
        
        # Create AiryBeam
        diameter = 14.0
        airy_beam = AiryBeam(diameter=diameter)
        beam_interface = BeamInterface(airy_beam)
        
        # Generate test positions
        az = np.random.uniform(0, 2 * np.pi, 50)
        za = np.random.uniform(0, np.pi / 6, 50)
        
        # CPU evaluation
        cpu_result = self.cpu_evaluator.evaluate_beam(
            beam_interface, az, za,
            polarized=True,
            freq=self.freq
        )
        
        # GPU evaluation
        gpu_result = self.gpu_evaluator.evaluate_beam(
            beam_interface, cp.asarray(az), cp.asarray(za),
            polarized=True,
            freq=self.freq
        )
        
        # Compare results
        gpu_result_cpu = cp.asnumpy(gpu_result)
        np.testing.assert_allclose(
            cpu_result, gpu_result_cpu,
            rtol=1e-6, atol=1e-8,
            err_msg="AiryBeam direct GPU polarized: CPU and GPU results differ"
        )
        
        # Check shape
        assert cpu_result.shape == (2, 2, 50), "CPU polarized AiryBeam shape incorrect"
        assert gpu_result_cpu.shape == (2, 2, 50), "GPU polarized AiryBeam shape incorrect"
        
    def test_airy_beam_performance_comparison(self):
        """Test and benchmark AiryBeam GPU vs CPU performance."""
        import time
        from pyuvdata.analytic_beam import AiryBeam
        from pyuvdata.beam_interface import BeamInterface
        
        # Create AiryBeam
        diameter = 14.0
        airy_beam = AiryBeam(diameter=diameter)
        beam_interface = BeamInterface(airy_beam)
        
        # Test with different numbers of sources
        n_sources_list = [100, 1000, 10000]
        
        print("\n" + "="*60)
        print("AiryBeam GPU vs CPU Performance Comparison")
        print("="*60)
        
        for n_sources in n_sources_list:
            # Generate coordinates
            np.random.seed(42)
            az = np.random.uniform(0, 2*np.pi, n_sources)
            za = np.random.uniform(0, np.pi/2, n_sources)
            
            # CPU timing
            start = time.perf_counter()
            cpu_result = self.cpu_evaluator.evaluate_beam(
                beam_interface, az, za,
                polarized=False,
                freq=self.freq
            )
            cpu_time = time.perf_counter() - start
            
            # GPU timing
            az_gpu = cp.asarray(az)
            za_gpu = cp.asarray(za)
            
            # Warm up
            _ = self.gpu_evaluator.evaluate_beam(
                beam_interface, az_gpu, za_gpu,
                polarized=False,
                freq=self.freq
            )
            
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            gpu_result = self.gpu_evaluator.evaluate_beam(
                beam_interface, az_gpu, za_gpu,
                polarized=False,
                freq=self.freq
            )
            cp.cuda.Stream.null.synchronize()
            gpu_time = time.perf_counter() - start
            
            # Verify results match
            gpu_result_cpu = cp.asnumpy(gpu_result)
            np.testing.assert_allclose(
                cpu_result, gpu_result_cpu,
                rtol=1e-5, atol=1e-7,
                err_msg=f"Results don't match for n_sources={n_sources}"
            )
            
            speedup = cpu_time / gpu_time
            print(f"n_sources={n_sources:6d}: CPU={cpu_time:.4f}s, GPU={gpu_time:.4f}s, Speedup={speedup:.1f}x")
        
        # Compare with UVBeam approach
        print("\nComparing direct AiryBeam vs UVBeam conversion on GPU:")
        
        # Convert to UVBeam
        uvbeam = airy_beam.to_uvbeam(
            freq_array=np.array([self.freq]),
            axis1_array=np.linspace(0, 2 * np.pi, 361)[:-1],
            axis2_array=np.linspace(0, np.pi, 181),
        )
        uvbeam_interface = BeamInterface(uvbeam)
        
        n_sources = 10000
        az = np.random.uniform(0, 2*np.pi, n_sources)
        za = np.random.uniform(0, np.pi/2, n_sources)
        az_gpu = cp.asarray(az)
        za_gpu = cp.asarray(za)
        
        # Time direct AiryBeam
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        gpu_airy = self.gpu_evaluator.evaluate_beam(
            beam_interface, az_gpu, za_gpu,
            polarized=False,
            freq=self.freq
        )
        cp.cuda.Stream.null.synchronize()
        airy_time = time.perf_counter() - start
        
        # Time UVBeam
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        gpu_uvbeam = self.gpu_evaluator.evaluate_beam(
            uvbeam_interface, az_gpu, za_gpu,
            polarized=False,
            freq=self.freq
        )
        cp.cuda.Stream.null.synchronize()
        uvbeam_time = time.perf_counter() - start
        
        print(f"Direct AiryBeam: {airy_time:.4f}s")
        print(f"UVBeam conversion: {uvbeam_time:.4f}s")
        print(f"Direct AiryBeam is {uvbeam_time/airy_time:.1f}x faster!")
        print("="*60)
            
    def test_uvbeam_gpu_interpolation(self):
        """Test that UVBeam GPU interpolation produces correct results."""
        # The setup already created a UVBeam from AiryBeam
        # self.beam is a BeamInterface wrapping a UVBeam
        
        # Generate test positions
        n_test = 500
        az = np.random.uniform(0, 2 * np.pi, n_test)
        za = np.random.uniform(0, np.pi / 4, n_test)
        
        # CPU evaluation
        cpu_result = self.cpu_evaluator.evaluate_beam(
            self.beam, az, za,
            polarized=False,
            freq=self.freq,
            spline_opts={'order': 1},
            interpolation_function='az_za_map_coordinates'
        )
        
        # GPU evaluation (should use new GPU interpolation)
        gpu_result = self.gpu_evaluator.evaluate_beam(
            self.beam, cp.asarray(az), cp.asarray(za),
            polarized=False,
            freq=self.freq,
            spline_opts={'order': 1},
            interpolation_function='az_za_map_coordinates'
        )
        
        # Compare results
        gpu_result_cpu = cp.asnumpy(gpu_result)
        np.testing.assert_allclose(
            cpu_result, gpu_result_cpu,
            rtol=1e-5, atol=1e-7,
            err_msg="UVBeam GPU interpolation: CPU and GPU results differ"
        )
        
        print("\n✅ UVBeam GPU interpolation test passed!")
        
    def test_uvbeam_gpu_performance(self):
        """Benchmark UVBeam GPU interpolation performance."""
        import time
        
        print("\n" + "="*60)
        print("UVBeam GPU Interpolation Performance Test")
        print("="*60)
        
        # Test with larger number of sources
        n_sources_list = [1000, 10000, 50000]
        
        for n_sources in n_sources_list:
            az = np.random.uniform(0, 2*np.pi, n_sources)
            za = np.random.uniform(0, np.pi/2, n_sources)
            
            # CPU timing
            start = time.perf_counter()
            cpu_result = self.cpu_evaluator.evaluate_beam(
                self.beam, az, za,
                polarized=False,
                freq=self.freq,
                interpolation_function='az_za_map_coordinates'
            )
            cpu_time = time.perf_counter() - start
            
            # GPU timing with new implementation
            az_gpu = cp.asarray(az)
            za_gpu = cp.asarray(za)
            
            # Warm up
            _ = self.gpu_evaluator.evaluate_beam(
                self.beam, az_gpu, za_gpu,
                polarized=False,
                freq=self.freq,
                interpolation_function='az_za_map_coordinates'
            )
            
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            gpu_result = self.gpu_evaluator.evaluate_beam(
                self.beam, az_gpu, za_gpu,
                polarized=False,
                freq=self.freq,
                interpolation_function='az_za_map_coordinates'
            )
            cp.cuda.Stream.null.synchronize()
            gpu_time = time.perf_counter() - start
            
            # Verify results match
            gpu_result_cpu = cp.asnumpy(gpu_result)
            matches = np.allclose(cpu_result, gpu_result_cpu, rtol=1e-5)
            
            speedup = cpu_time / gpu_time
            print(f"n_sources={n_sources:6d}: CPU={cpu_time:.4f}s, GPU={gpu_time:.4f}s, "
                  f"Speedup={speedup:.1f}x, Match={matches}")
        
        print("="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])