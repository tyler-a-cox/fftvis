"""
Test to isolate and compare beam interpolation differences between CPU and GPU.
This test keeps all other parameters constant to focus solely on beam evaluation.
"""

import numpy as np
import pytest
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import AiryBeam
from fftvis.cpu import CPUBeamEvaluator
from fftvis.utils import _use_gpu as check_gpu_available

# Skip all tests if GPU not available
pytestmark = pytest.mark.skipif(
    not check_gpu_available(), reason="GPU not available"
)

# Only import GPU modules if available
if check_gpu_available():
    from fftvis.gpu import GPUBeamEvaluator


class TestBeamInterpolationComparison:
    """Test beam interpolation differences between CPU and GPU."""
    
    def _to_numpy(self, arr):
        """Convert CuPy array to numpy if needed."""
        if hasattr(arr, 'get'):
            return arr.get()
        return arr
    
    def setup_method(self):
        """Set up test parameters."""
        # Fixed random seed for reproducibility
        np.random.seed(42)
        
        # Generate test sky positions
        # Use a grid of positions to test interpolation systematically
        n_az = 20
        n_za = 10
        self.az = np.linspace(0, 2*np.pi, n_az, endpoint=False)
        self.za = np.linspace(0, np.pi/4, n_za)  # Up to 45 degrees zenith angle
        
        # Create meshgrid and flatten
        az_grid, za_grid = np.meshgrid(self.az, self.za)
        self.az_flat = az_grid.flatten()
        self.za_flat = za_grid.flatten()
        self.n_sources = len(self.az_flat)
        
        # Fixed frequency (must be within beam file range)
        self.freq = 100e6  # 100 MHz
        
        # Fixed polarization array (using XX only for simplicity)
        self.polarized = True
        self.npol = 1
        
    
    def test_uvbeam_interpolation_order(self):
        """Test UVBeam interpolation with different spline orders."""
        from pyuvdata.beam_interface import BeamInterface
        # Load UVBeam
        import os
        beam_file = os.path.join(os.path.dirname(__file__), 
                                "../matvis/src/matvis/data/NF_HERA_Dipole_small.fits")
        uvbeam_raw = UVBeam.from_file(beam_file)
        uvbeam = BeamInterface(uvbeam_raw)
        
        # Test different interpolation orders
        for order in [1, 2, 3]:
            print(f"\nTesting interpolation order {order}")
            
            spline_opts = {'order': order}
            
            # Create evaluators
            cpu_evaluator = CPUBeamEvaluator()
            gpu_evaluator = GPUBeamEvaluator()
            
            # Evaluate on CPU
            cpu_result = cpu_evaluator.evaluate_beam(
                beam=uvbeam,
                az=self.az_flat,
                za=self.za_flat,
                freq=self.freq,
                polarized=self.polarized,
                interpolation_function="az_za_map_coordinates",
                spline_opts=spline_opts
            )
            
            # Evaluate on GPU
            gpu_result = gpu_evaluator.evaluate_beam(
                beam=uvbeam,
                az=self.az_flat,
                za=self.za_flat,
                freq=self.freq,
                polarized=self.polarized,
                interpolation_function="az_za_map_coordinates",
                spline_opts=spline_opts
            )
            
            # Convert to numpy and calculate differences
            gpu_result_np = self._to_numpy(gpu_result)
            abs_diff = np.abs(cpu_result - gpu_result_np)
            rel_diff = abs_diff / (np.abs(cpu_result) + 1e-10)
            
            print(f"  Max absolute difference: {np.max(abs_diff):.2e}")
            print(f"  Max relative difference: {np.max(rel_diff):.2e}")
            print(f"  Mean absolute difference: {np.mean(abs_diff):.2e}")
            print(f"  Sources with >1% difference: {np.sum(rel_diff > 0.01)}/{self.n_sources}")
            
            # Plot difference pattern if significant
            if np.max(rel_diff) > 0.01:
                try:
                    import matplotlib.pyplot as plt
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Reshape for plotting
                    abs_diff_grid = abs_diff[:, 0, 0].reshape(len(self.za), len(self.az))
                    
                    # Plot absolute difference
                    im1 = ax1.imshow(abs_diff_grid, extent=[0, 360, 0, 45], 
                                   aspect='auto', origin='lower')
                    ax1.set_xlabel('Azimuth (degrees)')
                    ax1.set_ylabel('Zenith angle (degrees)')
                    ax1.set_title(f'Absolute Difference (order={order})')
                    plt.colorbar(im1, ax=ax1)
                    
                    # Plot relative difference
                    rel_diff_grid = rel_diff[:, 0, 0].reshape(len(self.za), len(self.az))
                    im2 = ax2.imshow(rel_diff_grid, extent=[0, 360, 0, 45], 
                                   aspect='auto', origin='lower', vmax=0.1)
                    ax2.set_xlabel('Azimuth (degrees)')
                    ax2.set_ylabel('Zenith angle (degrees)')
                    ax2.set_title(f'Relative Difference (order={order})')
                    plt.colorbar(im2, ax=ax2, label='Relative diff')
                    
                    plt.tight_layout()
                    plt.savefig(f'beam_diff_order_{order}.png')
                    plt.close()
                except ImportError:
                    # Matplotlib not available in CI, skip plotting
                    pass
    
    def test_uvbeam_with_beaminterface(self):
        """Test UVBeam wrapped in BeamInterface to check for the bug."""
        from pyuvdata.beam_interface import BeamInterface
        
        # Load UVBeam
        import os
        beam_file = os.path.join(os.path.dirname(__file__), 
                                "../matvis/src/matvis/data/NF_HERA_Dipole_small.fits")
        uvbeam_raw = UVBeam.from_file(beam_file)
        
        # Wrap in BeamInterface
        beam_interface = BeamInterface(uvbeam_raw)
        
        # Test with explicit spline_opts to avoid default mismatch
        spline_opts = {'order': 3}
        
        # Create evaluators
        cpu_evaluator = CPUBeamEvaluator()
        gpu_evaluator = GPUBeamEvaluator()
        
        # Evaluate on CPU
        cpu_result = cpu_evaluator.evaluate_beam(
            beam=beam_interface,
            az=self.az_flat,
            za=self.za_flat,
            freq=self.freq,
            polarized=self.polarized,
            interpolation_function="az_za_map_coordinates",
            spline_opts=spline_opts
        )
        
        # Evaluate on GPU
        gpu_result = gpu_evaluator.evaluate_beam(
            beam=beam_interface,
            az=self.az_flat,
            za=self.za_flat,
            freq=self.freq,
            polarized=self.polarized,
            interpolation_function="az_za_map_coordinates",
            spline_opts=spline_opts
        )
        
        # Convert to numpy and calculate differences
        gpu_result_np = self._to_numpy(gpu_result)
        abs_diff = np.abs(cpu_result - gpu_result_np)
        rel_diff = abs_diff / (np.abs(cpu_result) + 1e-10)
        
        print("\nBeamInterface wrapped UVBeam:")
        print(f"  Max absolute difference: {np.max(abs_diff):.2e}")
        print(f"  Max relative difference: {np.max(rel_diff):.2e}")
        
        # Check if GPU is using native interpolation or falling back
        # This will help identify if the bug is occurring
        if np.max(rel_diff) > 0.1:
            print("  WARNING: Large differences suggest GPU may be falling back to CPU")
    
    def test_edge_effects(self):
        """Test beam interpolation at edges where differences are often larger."""
        from pyuvdata.beam_interface import BeamInterface
        # Create positions near beam edge
        n_edge = 50
        # Zenith angles from 30 to 60 degrees
        za_edge = np.linspace(np.pi/6, np.pi/3, n_edge)
        az_edge = np.random.uniform(0, 2*np.pi, n_edge)
        
        # Load UVBeam
        import os
        beam_file = os.path.join(os.path.dirname(__file__), 
                                "../matvis/src/matvis/data/NF_HERA_Dipole_small.fits")
        uvbeam_raw = UVBeam.from_file(beam_file)
        uvbeam = BeamInterface(uvbeam_raw)
        
        for order in [1, 3]:
            spline_opts = {'order': order}
            
            # Create evaluators
            cpu_evaluator = CPUBeamEvaluator()
            gpu_evaluator = GPUBeamEvaluator()
            
            # Evaluate
            cpu_result = cpu_evaluator.evaluate_beam(
                beam=uvbeam,
                az=az_edge,
                za=za_edge,
                freq=self.freq,
                polarized=self.polarized,
                interpolation_function="az_za_map_coordinates",
                spline_opts=spline_opts
            )
            
            gpu_result = gpu_evaluator.evaluate_beam(
                beam=uvbeam,
                az=az_edge,
                za=za_edge,
                freq=self.freq,
                polarized=self.polarized,
                interpolation_function="az_za_map_coordinates",
                spline_opts=spline_opts
            )
            
            # Convert to numpy and compare
            gpu_result_np = self._to_numpy(gpu_result)
            abs_diff = np.abs(cpu_result - gpu_result_np)
            rel_diff = abs_diff / (np.abs(cpu_result) + 1e-10)
            
            print(f"\nEdge effects (order={order}):")
            print(f"  Max absolute difference: {np.max(abs_diff):.2e}")
            print(f"  Max relative difference: {np.max(rel_diff):.2e}")
            print(f"  Points with >5% difference: {np.sum(rel_diff > 0.05)}/{n_edge}")


if __name__ == "__main__":
    # Run tests
    test = TestBeamInterpolationComparison()
    test.setup_method()
    
    print("=" * 60)
    print("Beam Interpolation CPU vs GPU Comparison")
    print("=" * 60)
    
    test.test_uvbeam_interpolation_order()
    test.test_uvbeam_with_beaminterface()
    test.test_edge_effects()