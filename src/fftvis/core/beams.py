import numpy as np
from abc import abstractmethod
from pyuvdata.beam_interface import BeamInterface
from typing import Dict, Optional

# Import the matvis base class
from matvis.core.beams import BeamInterpolator


class BeamEvaluator(BeamInterpolator):
    """Abstract base class for beam evaluation that inherits from matvis.BeamInterpolator.
    
    This class defines the interface for evaluating beams across different implementations
    (CPU, GPU).
    """

    def __init__(self, **kwargs):
        """Initialize with default values to be compatible with matvis.BeamInterpolator."""
        # We'll set these properly when evaluate_beam is called
        self.beam_list = []
        self.beam_idx = None
        self.polarized = False
        self.nant = 0
        self.freq = 0.0
        self.nsrc = 0
        self.spline_opts = {}
        self.precision = 2
        
        # Initialize the base class with minimal required parameters
        # These will be overridden when evaluate_beam is called
        super().__init__(
            beam_list=self.beam_list,
            beam_idx=None,
            polarized=self.polarized,
            nant=self.nant,
            freq=self.freq,
            nsrc=0,
            spline_opts=self.spline_opts,
            precision=self.precision
        )

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
        beam
            UVBeam object to evaluate.
        az
            Azimuth coordinates to evaluate the beam at.
        za
            Zenith angle coordinates to evaluate the beam at.
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
    
    # Bridge method that implements matvis's interp using our evaluate_beam
    def interp(self, tx: np.ndarray, ty: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Implement the matvis interp interface using our evaluate_beam method.
        
        This bridges between the matvis API and our API.
        """
        # Convert tx/ty to az/za (similar to how matvis does it)
        from matvis.coordinates import enu_to_az_za
        az, za = enu_to_az_za(enu_e=tx, enu_n=ty, orientation="uvbeam")
        
        # Update nsrc attribute based on the number of sources
        self.nsrc = len(az)
        
        # Call our evaluate_beam method (for each beam in beam_list)
        for i, bm in enumerate(self.beam_list):
            beam_values = self.evaluate_beam(
                bm,
                az,
                za,
                self.polarized,
                self.freq,
                spline_opts=self.spline_opts,
                interpolation_function="az_za_map_coordinates",
            )
            
            # Format the output to match what matvis expects
            if self.polarized:
                if beam_values.ndim == 3:  # If shape is like (nax, nfeed, nsrc)
                    out[i] = beam_values.transpose((1, 0, 2))
                else:
                    out[i] = beam_values
            else:
                out[i] = beam_values
                
        return out
