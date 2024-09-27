import finufft
import numpy as np
from abc import ABC, abstractmethod
from typing import Any

from matvis._utils import get_dtypes

class BaseTransform(ABC):
    """
    Base class for the NUFFT transform.
    """
    def __init__(
            self, 
            antpos: dict, 
            nfeed: int,
            nchunks: int,
            antpairs: np.ndarray | None = None,
            precision: int = 2,
            eps=None,
            gpu: bool = False,
            flat_array_tol: float = 0.0,
            **options_params
        ):
        """
        Base class for the NUFFT transform.
        """
        self.antpos = antpos
        self.nants = len(antpos)
        self.antpairs = antpairs
        self.nfeeds = nfeed
        self.nchunks = nchunks
        
        # Get the desired precision if not provided
        # TODO: should probably remove this and just use the default precision
        if not eps:
            eps = 1e-13 if precision == 2 else 6e-8    
        
        self.eps = eps
        self.options = options_params
        self.rtype, self.ctype = get_dtypes(precision)

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
            (self.nchunks, self.npairs, self.nfeeds, self.nfeeds), 0.0, dtype=self.ctype
        )

    def _compute_rotation_matrix(self):
        """
        Compute the rotation matrix.
        """
        self.rot_matrix = np.zeros((3, 3), dtype=self.rtype)

    def setup(self):
        """
        Setup the NUFFT transform.
        """
        self.allocate_vis()

    def __call__(self, chunk, source_strength, tx, ty, tz=None) -> Any:
        self.nufft(source_strength, tx=tx, ty=ty, tz=tz, out=self.vis)
        return self.vis[chunk]