"""
FFTVis utility functions.

This module provides access to common utility functions used throughout FFTVis.
It imports from core, CPU, and GPU implementation modules.
"""

# Import common utilities from core
from .core.utils import (
    IDEALIZED_BL_TOL,
    speed_of_light,
    get_pos_reds,
    get_plane_to_xy_rotation_matrix,
    get_task_chunks,
)

# This is a simple check to determine which implementation to use
# This could be enhanced to check for actual CUDA availability
def _use_gpu():
    """Check if GPU implementation should be used."""
    # This is a placeholder. In a real implementation, 
    # you might check for CUDA availability or a config setting
    try:
        import cupy
        return True
    except ImportError:
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
