import logging
import numpy as np
from abc import abstractmethod
from pyuvdata.beam_interface import BeamInterface
from typing import Dict, Optional

# Import the matvis base class
from matvis.core.beams import BeamInterpolator

logger = logging.getLogger(__name__)


def to_gridded_beam(
    beam,
    freqs,
    *,
    naz: int = 721,
    nza: int = 361,
    polarized: bool = True,
    za_max: float = np.pi / 2,
) -> BeamInterface:
    """
    Sample an analytic beam onto a regular az/za grid.

    The GPU backend interpolates gridded ``UVBeam`` objects on the device.
    Analytic beams have no grid to interpolate, so they are *evaluated*, and
    pyuvdata only does that on the host -- which makes beam evaluation the
    dominant cost of a GPU run. Sampling the analytic beam onto a grid once,
    up front, moves the per-chunk work onto the device.

    This is an approximation: the beam is now interpolated from samples rather
    than evaluated exactly. The error is controlled by ``naz``/``nza`` and by
    the interpolation order used at simulation time, and can be measured with
    :func:`fftvis.gpu.beam_interpolation_error`.

    Parameters
    ----------
    beam : AnalyticBeam or BeamInterface
        The analytic beam to sample. A gridded ``UVBeam`` is returned
        unchanged.
    freqs : array_like
        Frequencies (Hz) to sample at. Use the same frequencies you will
        simulate, so no frequency interpolation is needed later.
    naz, nza : int
        Grid size in azimuth and zenith angle. The defaults give 0.5 degree
        sampling, which is finer than most HERA-class beams need.
    polarized : bool
        Whether to produce an efield beam (True) or a power beam (False).
    za_max : float
        Maximum zenith angle in radians. The default covers the visible
        hemisphere; do not reduce it below the horizon or sources near the
        horizon will extrapolate.

    Returns
    -------
    BeamInterface
        A gridded beam suitable for the GPU interpolation path.

    Examples
    --------
    >>> from fftvis.core.beams import to_gridded_beam  # doctest: +SKIP
    >>> gbeam = to_gridded_beam(beam, freqs, naz=1441, nza=721)  # doctest: +SKIP
    """
    bi = beam if isinstance(beam, BeamInterface) else BeamInterface(beam)

    if getattr(bi, "_isuvbeam", False):
        logger.debug("Beam is already gridded; returning it unchanged.")
        return bi

    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    az = np.linspace(0.0, 2.0 * np.pi, naz)
    za = np.linspace(0.0, float(za_max), nza)

    # pixel_coordinate_system is deliberately not passed. pyuvdata 3.2.0's
    # UVBeam initializer validates `uvb.pixel_coordinate_system` -- which is
    # still None at that point -- instead of the argument, so supplying *any*
    # non-None value raises "pixel_coordinate_system must be one of [...]".
    # Omitting it takes the `pixel_coordinate_system or "az_za"` default, which
    # is what we want regardless.
    uvb = bi.beam.to_uvbeam(
        freq_array=freqs,
        beam_type="efield" if polarized else "power",
        axis1_array=az,
        axis2_array=za,
    )

    logger.info(
        "Sampled analytic beam onto a %d x %d az/za grid at %d frequencies "
        "(%.3g deg resolution).",
        naz, nza, freqs.size, np.rad2deg(za[1] - za[0]),
    )
    return BeamInterface(uvb)


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
    ) -> np.ndarray:  # pragma: no cover
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
    ) -> np.ndarray:  # pragma: no cover
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
