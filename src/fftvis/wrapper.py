from typing import Literal, Optional

from .core.beams import BeamEvaluator
from .cpu.cpu_beams import CPUBeamEvaluator


def create_beam_evaluator(
    backend: Literal["cpu", "gpu"] = "cpu", **kwargs
) -> BeamEvaluator:
    """Create a beam evaluator for the specified backend.

    Parameters
    ----------
    backend
        The backend to use for beam evaluation.
        Currently supported: "cpu", "gpu".
    **kwargs
        Additional keyword arguments to pass to the beam evaluator constructor.

    Returns
    -------
    BeamEvaluator
        A beam evaluator instance for the specified backend.

    Raises
    ------
    ValueError
        If the specified backend is not supported.
    """
    if backend == "cpu":
        return CPUBeamEvaluator(**kwargs)
    elif backend == "gpu":
        raise NotImplementedError("GPU backend not yet implemented")
    else:
        raise ValueError(f"Unsupported backend: {backend}")
