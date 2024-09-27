import logging
import warnings
import numpy as np

try:
    import cupy as cp

    HAVE_CUDA = True

except ImportError:
    # if not installed, don't warn
    HAVE_CUDA = False
except Exception as e:  # pragma: no cover
    # if installed but having initialization issues
    # warn, but default back to non-gpu functionality
    warnings.warn(str(e), stacklevel=2)
    HAVE_CUDA = False

try:
    import cufinufft
except Exception as e:  # pragma: no cover
    # if installed but having initialization issues
    # warn, but default back to non-gpu functionality
    warnings.warn(str(e), stacklevel=2)
    HAVE_CUDA = False

from ..core.nufft import BaseTransform


class GPU_NUFFT(BaseTransform):
    """
    GPU implementation of the NUFFT transform.
    """
    def compute(self, tx=None, ty=None, tz=None, source_strength=None, out=None):
        """
        """
        pass

    def rotate_coordinates(self):
        """
        """
        self.rot_matrix = cp.zeros((3, 3), dtype=self.ctype)