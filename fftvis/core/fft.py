import finufft
import numpy as np



class BaseTransform:
    def __init__(self, eps=None, **options_params):
        """
        Base class for the NUFFT transform.
        """
        self.eps = eps
        self.options = options_params