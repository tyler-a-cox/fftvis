import numpy as np
from abc import ABC, abstractmethod
from pyuvdata.beam_interface import BeamInterface
from typing import Dict, Optional


class BeamEvaluator(ABC):
    """Abstract base class for beam evaluation.

    This class defines the interface for evaluating beams across different implementations
    (CPU, GPU).
    """

    @abstractmethod
    def evaluate_beam(
        self,
        beam: BeamInterface,
        az: np.ndarray,
        za: np.ndarray,
        polarized: bool,
        freq: float,
        check: bool = False,
        spline_opts: Optional[Dict] = None,
        interpolation_function: str = "az_za_map_coordinates",
    ) -> np.ndarray:
        """Evaluate the beam on the CPU. Simplified version of the `_evaluate_beam_cpu` function
        in matvis.

        This function will either interpolate the beam to the given coordinates tx, ty,
        or evaluate the beam there if it is an analytic beam.

        Parameters
        ----------
        A_s
            Array of shape (nax, nfeed, nsrcs_up) that will be filled with beam
            values.
        beam
            UVBeam object to evaluate.
        tx, ty
            Coordinates to evaluate the beam at, in sin-projection.
        polarized
            Whether to use beam polarization.
        freq
            Frequency to interpolate beam to.
        check
            Whether to check that the beam has no inf/nan values. Set to False if you are
            sure that the beam is valid, as it will be faster.
        spline_opts
            Extra options to pass to the RectBivariateSpline class when interpolating.
        interpolation_function
            The interpolation function to use when interpolating the beam. Can be either be
            'az_za_simple' or 'az_za_map_coordinates'. The former is slower but more accurate
            at the edges of the beam, while the latter is faster but less accurate
            for interpolation orders greater than linear.
        """
        pass

    @abstractmethod
    def get_apparent_flux_polarized(
        self, beam: np.ndarray, flux: np.ndarray
    ) -> np.ndarray:
        """Calculate apparent flux of the sources.

        Parameters
        ----------
        beam
            Array with beam values.
        flux
            Array with source flux values.

        Returns
        -------
        np.ndarray
            Array with modified beam values accounting for source flux.
        """
        pass
