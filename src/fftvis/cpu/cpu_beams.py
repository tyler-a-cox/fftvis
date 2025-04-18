import numpy as np
import numba as nb
from pyuvdata.beam_interface import BeamInterface
from typing import Dict, Optional

from ..core.beams import BeamEvaluator


class CPUBeamEvaluator(BeamEvaluator):
    """CPU implementation of beam evaluation."""

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
            A_s
                Array of shape (nax, nfeed, nsrcs_up) that will be filled with beam
            values.
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
        # Primary beam pattern using direct interpolation of UVBeam object
        kw = {
            "reuse_spline": True,
            "check_azza_domain": False,
            "spline_opts": spline_opts,
            "interpolation_function": interpolation_function,
        }

        interp_beam = beam.compute_response(
            az_array=az,
            za_array=za,
            freq_array=np.atleast_1d(freq),
            **kw,
        )

        if polarized:
            interp_beam = interp_beam[:, :, 0, :]
        else:
            # Here we have already asserted that the beam is a power beam and
            # has only one polarization, so we just evaluate that one.
            interp_beam = interp_beam[0, 0, 0, :]

        # Check for invalid beam values
        if check:
            sm = np.sum(interp_beam)
            if np.isinf(sm) or np.isnan(sm):
                raise ValueError("Beam interpolation resulted in an invalid value")

        return interp_beam

    @staticmethod
    @nb.jit(nopython=True, parallel=False, nogil=False)
    def get_apparent_flux_polarized(beam: np.ndarray, flux: np.ndarray):
        """Calculate apparent flux of the sources."""
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
