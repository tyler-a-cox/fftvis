import numpy as np
import numba as nb
from pyuvdata.beam_interface import BeamInterface
from typing import Dict, Optional

from ..core.beams import BeamEvaluator
from matvis.cpu.beams import UVBeamInterpolator
from matvis.coordinates import enu_to_az_za


class CPUBeamEvaluator(BeamEvaluator):
    """
    CPU beam evaluator using matvis for consistency with GPU.
    """

    def __init__(self, precision: int = 2, **kwargs):
        """
        Initialize evaluator.

        Parameters
        ----------
        precision : int
            Precision level (1 for single, 2 for double precision).
        """
        self.precision = precision
        self._initialized = False
        self._current_beam_id = None
        self._current_freq = None
        self._current_polarized = None
        self._interpolator = None

        # Attributes expected by base class interp() method
        self.beam_list = []
        self.polarized = False
        self.freq = 0.0
        self.nsrc = 0
        self.nant = 1
        self.spline_opts = {}
        self.beam_idx = None

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
        """Evaluate the beam on the CPU.
            Simplified version of the `_evaluate_beam_cpu` function
        in matvis.
            This function will either interpolate the beam to the given coordinates,
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
                Whether to check that the beam has no inf/nan values.
            spline_opts
                Extra options to pass to interpolation functions.
            interpolation_function
                interpolation function to use when interpolating the beam. Can be either be
            'az_za_simple' or 'az_za_map_coordinates'. The former is slower but more accurate
            at the edges of the beam, while the latter is faster but less accurate
            for interpolation orders greater than linear.

            Returns
            -------
            np.ndarray
                Interpolated beam values.
        """
        # Convert az/za to ENU (tx/ty) for matvis interface
        tx = np.sin(za) * np.cos(az)  # enu_e
        ty = np.sin(za) * np.sin(az)  # enu_n
        nsrc = len(az)

        # Check if we need to reinitialize (beam/freq/polarization changed)
        beam_id = id(beam)
        need_reinit = (
            not hasattr(self, '_initialized') or not self._initialized
            or beam_id != self._current_beam_id
            or freq != self._current_freq
            or polarized != self._current_polarized
        )

        if need_reinit:
            # Initialize matvis interpolator
            self._interpolator = UVBeamInterpolator(
                beam_list=[beam],
                beam_idx=None,
                polarized=polarized,
                nant=1,
                freq=freq,
                nsrc=nsrc,
                precision=self.precision,
                spline_opts=spline_opts or {},
            )
            self._interpolator.setup()
            self._initialized = True
            self._current_beam_id = beam_id
            self._current_freq = freq
            self._current_polarized = polarized

            # Update attributes expected by base class
            self.beam_list = [beam]
            self.polarized = polarized
            self.freq = freq
            self.nsrc = nsrc
            self.spline_opts = spline_opts or {}

        # Call matvis's interp method
        nfeed = 2 if polarized else 1
        nax = 2 if polarized else 1
        complex_dtype = np.complex64 if self.precision == 1 else np.complex128

        out = np.zeros((1, nfeed, nax, nsrc), dtype=complex_dtype)
        self._interpolator.interp(tx, ty, out)

        # Check for invalid values
        if check:
            if np.isinf(np.sum(out)) or np.isnan(np.sum(out)):
                raise ValueError("Beam interpolation resulted in an invalid value")

        # Return in expected format (remove beam dimension)
        if polarized:
            return out[0]  # Shape: (nfeed, nax, nsrc)
        else:
            # Return power for unpolarized: fftvis computes V = NUFFT(beam * flux)
            # so beam must be |E|^2. matvis returns E-field, so we square it.
            result = out[0, 0, 0, :]
            return np.abs(result) ** 2

    @staticmethod
    @nb.jit(nopython=True, parallel=False, nogil=False)
    def get_apparent_flux_polarized_beam(beam: np.ndarray, flux: np.ndarray):
        """Calculate apparent flux of the sources. """
        nax, nfd, nsrc = beam.shape

        for isrc in range(nsrc):
            c = np.conj(beam[:, :, isrc])

            i00 = c[0, 0] * beam[0, 0, isrc] + c[1, 0] * beam[1, 0, isrc]
            i01 = c[0, 0] * beam[0, 1, isrc] + c[1, 0] * beam[1, 1, isrc]

            i11 = c[0, 1] * beam[0, 1, isrc] + c[1, 1] * beam[1, 1, isrc]
            beam[0, 0, isrc] = i00 * flux[isrc]
            beam[0, 1, isrc] = i01 * flux[isrc]
            beam[1, 0, isrc] = np.conj(i01) * flux[isrc]
            beam[1, 1, isrc] = i11 * flux[isrc]

    @staticmethod
    @nb.jit(nopython=True, parallel=False, nogil=False)
    def get_apparent_flux_polarized(beam, coherency):
        """
        Calculate the apparent flux of the sources using the beam and coherency matrices.
        
        This function computes the product of the conjugate transpose of beam and
        the coherency matrix, and then multiplies it with beam.
        
        Parameters
        ----------
        beam : np.ndarray
            A 3D array of shape (2, 2, Nsources) where each A[:,:,i] is a 2x2 matrix.
        coherency : np.ndarray
            A 3D array of shape (2, 2, Nsources) where each C[:,:,i] is a 2x2 matrix.
        """
        nsources = beam.shape[2]
        for i in range(nsources):
            a00 = beam[0,0,i]
            a01 = beam[0,1,i]
            a10 = beam[1,0,i]
            a11 = beam[1,1,i]

            # A^H @ C
            tmp00 = np.conj(a00) * coherency[0,0,i] + np.conj(a10) * coherency[1,0,i]
            tmp01 = np.conj(a00) * coherency[0,1,i] + np.conj(a10) * coherency[1,1,i]
            tmp10 = np.conj(a01) * coherency[0,0,i] + np.conj(a11) * coherency[1,0,i]
            tmp11 = np.conj(a01) * coherency[0,1,i] + np.conj(a11) * coherency[1,1,i]

            # (A^H C) @ A
            beam[0,0,i] = tmp00*a00 + tmp01*a10
            beam[0,1,i] = tmp00*a01 + tmp01*a11
            beam[1,0,i] = tmp10*a00 + tmp11*a10
            beam[1,1,i] = tmp10*a01 + tmp11*a11