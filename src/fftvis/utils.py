"""
FFTVis utility functions.

This module provides access to common utility functions used throughout FFTVis.
It imports from core, CPU, and GPU implementation modules.
"""

import logging

# Import common utilities from core
from .core.utils import (
    IDEALIZED_BL_TOL,
    speed_of_light,
    get_pos_reds,
    get_plane_to_xy_rotation_matrix,
    get_task_chunks,
)

# Cache the GPU availability to avoid repeated checks
_cached_use_gpu = None


def _use_gpu():
    """Check if GPU implementation should be used."""
    global _cached_use_gpu
    if _cached_use_gpu is not None:
        return _cached_use_gpu

    try:
        import cupy as cp

        # Check if a CUDA device is actually available
        _cached_use_gpu = cp.cuda.is_available()
        if not _cached_use_gpu:
            logging.warning("CuPy installed but no CUDA device found. Using CPU backend.")
        return _cached_use_gpu
    except ImportError:
        _cached_use_gpu = False
        return False


# Import implementation-specific functions
if _use_gpu():
    from .gpu.gpu_utils import inplace_rot
else:
    from .cpu.cpu_utils import inplace_rot

__all__ = [
    "IDEALIZED_BL_TOL",
    "speed_of_light",
    "get_pos_reds",
    "get_plane_to_xy_rotation_matrix",
    "get_task_chunks",
    "inplace_rot",
]
