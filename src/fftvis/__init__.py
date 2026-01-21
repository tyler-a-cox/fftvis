"""FFT-based visibility simulator."""

import logging

# Import key components for beams
from .core.beams import BeamEvaluator
from .cpu.beams import CPUBeamEvaluator
from .wrapper import create_beam_evaluator

# Import simulation functionality
from .core.simulate import SimulationEngine
from .cpu.cpu_simulate import CPUSimulationEngine
from .wrapper import create_simulation_engine, simulate_vis

# Import utility modules
from . import utils, logutils

# Check GPU availability: dependencies (cupy + cufinufft) AND hardware
GPU_AVAILABLE = False
_gpu_error_msg = None

try:
    import cupy as cp

    if not cp.cuda.is_available():
        logging.warning("CuPy installed but no CUDA device found.")
        _gpu_error_msg = (
            "No CUDA device available. CuPy is installed but no GPU hardware "
            "was detected."
        )
    else:
        # Also check for cufinufft
        import cufinufft  # noqa: F401

        from .gpu.beams import GPUBeamEvaluator
        from .gpu.gpu_simulate import GPUSimulationEngine
        GPU_AVAILABLE = True

except ImportError as e:
    if "cufinufft" in str(e):
        _gpu_error_msg = (
            "cufinufft not installed. "
            "Install with: pip install cufinufft"
        )
    else:
        _gpu_error_msg = (
            "GPU dependencies not installed. "
            "Install with: pip install fftvis[gpu]"
        )

# Define dummy functions with helpful error messages if GPU not available
if not GPU_AVAILABLE:
    def GPUBeamEvaluator(*args, **kwargs):
        raise ImportError(_gpu_error_msg)

    def GPUSimulationEngine(*args, **kwargs):
        raise ImportError(_gpu_error_msg)

__all__ = [
    # Beam-related exports
    "BeamEvaluator",
    "CPUBeamEvaluator",
    "create_beam_evaluator",
    # Simulation-related exports
    "SimulationEngine",
    "CPUSimulationEngine",
    "create_simulation_engine",
    "simulate_vis",
    # GPU-related exports (always available, but may raise helpful errors)
    "GPUBeamEvaluator",
    "GPUSimulationEngine",
    "GPU_AVAILABLE",
]
