import logging
import warnings
import numpy as np
import finufft

from ..core.nufft import BaseTransform


class CPU_NUFFT(BaseTransform):
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
        self.rot_matrix = np.zeros((3, 3), dtype=self.ctype)