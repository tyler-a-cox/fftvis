"""FFT-based visibility simulator."""

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

# Try to import GPU implementations if available
try:
    from .gpu.beams import GPUBeamEvaluator
    from .gpu.gpu_simulate import GPUSimulationEngine
    _gpu_available = True
except ImportError:
    _gpu_available = False

    def GPUBeamEvaluator(*args, **kwargs):
        raise ImportError(
            "GPUBeamEvaluator requires GPU dependencies. "
            "Install with: pip install fftvis[gpu]"
        )

    def GPUSimulationEngine(*args, **kwargs):
        raise ImportError(
            "GPUSimulationEngine requires GPU dependencies. "
            "Install with: pip install fftvis[gpu]"
        )

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
]

# Add GPU exports if available
if _gpu_available:
    __all__.extend([
        "GPUBeamEvaluator",
        "GPUSimulationEngine",
    ])
