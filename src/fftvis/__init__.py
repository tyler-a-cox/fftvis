"""FFT-based visibility simulator."""

# Import key components for beams
from .core.beams import BeamEvaluator
from .cpu.cpu_beams import CPUBeamEvaluator
from .wrapper import create_beam_evaluator

# Import simulation functionality
from .core.simulate import SimulationEngine
from .cpu.cpu_simulate import CPUSimulationEngine
from .wrapper import create_simulation_engine, simulate_vis

# Import utility modules
from . import utils, logutils

# Try to import GPU implementations if available
try:
    from .gpu.gpu_beams import GPUBeamEvaluator
    from .gpu.gpu_simulate import GPUSimulationEngine
    _gpu_available = True
except ImportError:
    _gpu_available = False
    GPUBeamEvaluator = None
    GPUSimulationEngine = None

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
