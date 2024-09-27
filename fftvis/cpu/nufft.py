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