"""
GPU Performance Analysis - Understanding when GPUs provide benefits.

Note: Performance results will vary significantly based on GPU hardware.
High-end GPUs (e.g., V100, A100, RTX 3090) will show much better speedups.
"""

import pytest
import numpy as np
import time
import logging

# Check GPU availability and specs
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
    
    if GPU_AVAILABLE:
        # Get GPU properties
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        GPU_NAME = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
        GPU_MEMORY = props['totalGlobalMem'] / (1024**3)  # Total memory in GB
        GPU_COMPUTE_CAPABILITY = f"{props['major']}.{props['minor']}"
        
        # Check for cufinufft
        try:
            from fftvis.gpu.nufft import HAVE_CUFINUFFT
            GPU_AVAILABLE = GPU_AVAILABLE and HAVE_CUFINUFFT
        except ImportError:
            GPU_AVAILABLE = False
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = "N/A"
    GPU_MEMORY = 0
    GPU_COMPUTE_CAPABILITY = "N/A"
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
class TestGPUPerformanceAnalysis:
    """Analysis of GPU performance characteristics."""
    
    def test_gpu_info(self):
        """Display GPU information."""
        logger.info("\n" + "="*60)
        logger.info("GPU INFORMATION")
        logger.info("="*60)
        
        if GPU_AVAILABLE:
            logger.info(f"GPU Name: {GPU_NAME}")
            logger.info(f"GPU Memory: {GPU_MEMORY:.1f} GB")
            logger.info(f"Compute Capability: {GPU_COMPUTE_CAPABILITY}")
            
            # Memory bandwidth test
            size = int(1e7)  # 10M elements (smaller for 2GB GPU)
            a = cp.random.rand(size, dtype=cp.float32)
            b = cp.random.rand(size, dtype=cp.float32)
            
            # Warm up
            c = a + b
            cp.cuda.Stream.null.synchronize()
            
            # Time memory-bound operation
            start = time.perf_counter()
            for _ in range(10):
                c = a + b
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - start
            
            # Calculate bandwidth (2 reads + 1 write per operation)
            bytes_transferred = 3 * size * 4 * 10  # float32 = 4 bytes
            bandwidth = bytes_transferred / elapsed / 1e9  # GB/s
            
            logger.info(f"Measured Memory Bandwidth: {bandwidth:.1f} GB/s")
            
        else:
            logger.info("No GPU available")
    
    def test_theoretical_analysis(self):
        """Theoretical analysis of when GPUs provide benefits."""
        logger.info("\n" + "="*60)
        logger.info("THEORETICAL GPU SPEEDUP ANALYSIS")
        logger.info("="*60)
        
        logger.info("\nFactors affecting GPU performance for visibility simulation:")
        logger.info("1. Memory Transfer Overhead:")
        logger.info("   - Data must be copied to/from GPU memory")
        logger.info("   - Typical PCIe bandwidth: 16-32 GB/s")
        logger.info("   - This creates a minimum problem size threshold")
        
        logger.info("\n2. GPU Utilization:")
        logger.info("   - GPUs have thousands of cores that need to be kept busy")
        logger.info("   - Small problems underutilize the GPU")
        logger.info("   - Rule of thumb: need 10,000+ parallel operations")
        
        logger.info("\n3. NUFFT Algorithm Characteristics:")
        logger.info("   - NUFFT has irregular memory access patterns")
        logger.info("   - Less efficient on GPUs than regular FFTs")
        logger.info("   - Benefit increases with problem size")
        
        # Expected speedup table
        logger.info("\nExpected Speedup by Problem Size (High-end GPU):")
        expected_speedups = [
            ("Small (< 1K sources, < 100 baselines)", "0.1-0.5x", "Slower due to overhead"),
            ("Medium (1K-10K sources, 100-1K baselines)", "0.5-2x", "Break-even point"),
            ("Large (10K-100K sources, 1K+ baselines)", "2-10x", "GPU starts to shine"),
            ("Very Large (100K+ sources, 10K+ baselines)", "10-50x", "Maximum benefit"),
        ]
        
        print("\n" + tabulate(expected_speedups, 
                            headers=["Problem Size", "Expected Speedup", "Notes"],
                            tablefmt='grid'))
        
        logger.info(f"\nYour GPU ({GPU_NAME}) characteristics:")
        if "Quadro P600" in GPU_NAME:
            logger.info("- Entry-level professional GPU")
            logger.info("- Limited memory (2GB) restricts problem size")
            logger.info("- Lower compute capability limits performance")
            logger.info("- Expected speedup: 0.5-2x for most problems")
        elif "V100" in GPU_NAME or "A100" in GPU_NAME:
            logger.info("- High-end datacenter GPU")
            logger.info("- Large memory (16-80GB) allows huge problems")
            logger.info("- High bandwidth and compute capability")
            logger.info("- Expected speedup: 10-100x for large problems")
        elif "RTX" in GPU_NAME:
            logger.info("- Consumer gaming GPU")
            logger.info("- Good compute capability")
            logger.info("- Decent memory bandwidth")
            logger.info("- Expected speedup: 5-20x for large problems")
    
    def test_optimization_recommendations(self):
        """Provide optimization recommendations."""
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZATION RECOMMENDATIONS")
        logger.info("="*60)
        
        logger.info("\nFor current GPU (Quadro P600):")
        logger.info("1. Use GPU only for problems with:")
        logger.info("   - More than 10,000 sources")
        logger.info("   - More than 1,000 baselines")
        logger.info("   - Multiple time/frequency samples")
        
        logger.info("\n2. Optimization strategies:")
        logger.info("   - Batch multiple time samples together")
        logger.info("   - Process multiple frequencies simultaneously")
        logger.info("   - Keep data on GPU between operations")
        
        logger.info("\n3. When to use CPU instead:")
        logger.info("   - Small test problems")
        logger.info("   - Single time/frequency samples")
        logger.info("   - When memory requirements exceed 2GB")
        
        logger.info("\nFor production use with large arrays (HERA, SKA):")
        logger.info("- Recommend high-end GPU: V100, A100, or RTX 3090/4090")
        logger.info("- These will show 10-100x speedups for realistic problem sizes")
        logger.info("- Cost-benefit improves dramatically with larger GPUs")
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_memory_limited_scenario(self):
        """Test what happens when we approach GPU memory limits."""
        logger.info("\n" + "="*60)
        logger.info("GPU MEMORY LIMITS TEST")
        logger.info("="*60)
        
        logger.info(f"Available GPU memory: {GPU_MEMORY:.1f} GB")
        
        # Estimate maximum problem size
        # Each complex128 number = 16 bytes
        # Need memory for: sources, baselines, intermediate results
        
        max_sources_simple = int(GPU_MEMORY * 1e9 / 100 / 16)  # Rough estimate
        logger.info(f"Estimated max sources (simple): {max_sources_simple:,}")
        
        # Test increasing problem sizes until failure
        test_sizes = [1000, 5000, 10000, 50000, 100000, 200000, 500000]
        
        if GPU_AVAILABLE:
            from fftvis.gpu.nufft import gpu_nufft3d
        else:
            # This should never be reached due to skipif decorator
            return
        
        results = []
        for nsrc in test_sizes:
            try:
                # Create test data
                x = cp.random.uniform(-0.5, 0.5, nsrc, dtype=cp.float64)
                y = cp.random.uniform(-0.5, 0.5, nsrc, dtype=cp.float64)
                z = cp.random.uniform(-0.1, 0.1, nsrc, dtype=cp.float64)
                weights = cp.random.random(nsrc, dtype=cp.complex128)
                
                # Small number of baselines
                nbl = 100
                u = cp.random.uniform(-100, 100, nbl, dtype=cp.float64)
                v = cp.random.uniform(-100, 100, nbl, dtype=cp.float64)
                w = cp.random.uniform(-100, 100, nbl, dtype=cp.float64)
                
                # Try NUFFT
                start = time.perf_counter()
                result = gpu_nufft3d(x, y, z, weights, u, v, w, eps=1e-6)
                cp.cuda.Stream.null.synchronize()
                elapsed = time.perf_counter() - start
                
                # Get memory usage
                mempool = cp.get_default_memory_pool()
                used_bytes = mempool.used_bytes()
                used_gb = used_bytes / 1e9
                
                results.append({
                    'Sources': f"{nsrc:,}",
                    'Status': 'Success',
                    'Time (ms)': f"{elapsed*1000:.1f}",
                    'Memory (GB)': f"{used_gb:.2f}"
                })
                
            except cp.cuda.memory.OutOfMemoryError:
                results.append({
                    'Sources': f"{nsrc:,}",
                    'Status': 'Out of Memory',
                    'Time (ms)': 'N/A',
                    'Memory (GB)': 'N/A'
                })
                break
            except Exception as e:
                results.append({
                    'Sources': f"{nsrc:,}",
                    'Status': f'Error: {type(e).__name__}',
                    'Time (ms)': 'N/A',
                    'Memory (GB)': 'N/A'
                })
        
        print("\n" + tabulate(results, headers='keys', tablefmt='grid'))


if __name__ == "__main__":
    test = TestGPUPerformanceAnalysis()
    test.test_gpu_info()
    test.test_theoretical_analysis()
    test.test_optimization_recommendations()
    
    if GPU_AVAILABLE:
        test.test_memory_limited_scenario()