import numpy as np
from abc import ABC, abstractmethod
from pyuvdata import UVBeam

class BeamInterpolator(ABC):
    """
    Class
    """
    def __init__(
        self,
        beam: UVBeam,
        polarized: bool,
        nsrcs: int,
        freq: float,
        spline_opts: dict = None,
        precision: int = 1,
    ):
        self.polarized = polarized
        self.freq = freq
        self.spline_opts = spline_opts
        self.nsrcs = nsrcs

        if self.polarized:
            self.nax = 2
            self.nfeed = 2
        else:
            self.nax = 1
            self.nfeed = 1

        if precision == 1:
            self.complex_dtype = np.complex64
            self.real_dtype = np.float32
        elif precision == 2:
            self.complex_dtype = np.complex128
            self.real_dtype = np.float64
        else:
            raise ValueError("precision must be 1 or 2")
        
    def setup(self):
        """
        Method
        """
        self.interpolated_beam = np.zeros(
            (self.nax, self.nfeed, self.nsrcs),
            dtype=self.complex_dtype,
        )

    @abstractmethod
    def interp(self, tx: np.ndarray, ty: np.ndarray, out: np.ndarray) -> np.ndarray:
        """
        Method
        """
        pass

    def __call__(self, tx: np.ndarray, ty: np.ndarray, check: bool=True) -> np.ndarray:
        """
        Method
        """
        self.interp(tx, ty, self.interpolated_beam)

        if check:
            # Check for invalid beam values
            sm = self.interpolated_beam.sum()
            if np.isinf(sm) or np.isnan(sm):
                raise ValueError("Beam interpolation resulted in an invalid value")

        return self.interpolated_beam