import finufft
import numpy as np
from abc import ABC, abstractmethod
from typing import Any

from matvis._utils import get_dtypes


class BaseTransform(ABC):
    def __init__(
            self, 
            antpos: dict, 
            nfeed: int,
            antpairs: np.ndarray | None = None,
            precision: int = 2,
            eps=None,
            flat_array_tol: float = 0.0,
            **options_params
        ):
        """
        Base class for the NUFFT transform.
        """
        self.antpos = antpos
        self.nants = len(antpos)
        self.antpairs = antpairs
        
        # Get the desired precision if not provided
        if not eps:
            eps = 1e-13 if precision == 2 else 6e-8    
        
        self.eps = eps
        self.options = options_params
        self.ctype = get_dtypes(precision)[1]

    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        """
        Perform the Non-Uniform Fast Fourier Transform.
        """

    def allocate_vis(self):
        """
        Allocate memory for the visibility array.
        """
        self.vis = np.full(
            (self.nchunks, self.npairs, self.nfeeds, self.nfeeds), 0.0, dtype=self.complex_dtype
        )

    def setup(self):
        """
        Setup the NUFFT transform.
        """
        self.allocate_vis()

    def __call__(self, source_strength, tx, ty, tz=None) -> Any:
        self.nufft(source_strength, tx=tx, ty=ty, tz=tz, out=self.vis)
        return self.vis[chunk]