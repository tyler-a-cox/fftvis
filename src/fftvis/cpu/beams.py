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
        # Save these for matvis compatibility
        self.polarized = polarized
        self.freq = freq
        self.spline_opts = spline_opts or {}
        
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
    def get_apparent_flux_polarized_beam(beam: np.ndarray, flux: np.ndarray):  # pragma: no cover
        """
        Calculate apparent flux of the sources. Here we assume that the
        beam is a 2x2 matrix for each source, and the flux is a 1D array
        of the same length as the number of sources. The beam is modified
        in place to store the apparent flux.
        """
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
    def get_apparent_flux_polarized(beam1, coherency, beam2):
        """
        Calculate the apparent flux of the sources using the beam and coherency matrices.
        This function computes the product of the conjugate transpose of beam1 and
        the coherency matrix, and then multiplies it with beam2. The result is stored
        back in beam1.
        
        Parameters
        ----------
        beam1 : np.ndarray
            A 3D array of shape (2, 2, Nsources) where each A[:,:,i] is a 2x2 matrix.
        coherency : np.ndarray
            A 3D array of shape (2, 2, Nsources) where each C[:,:,i] is a 2x2 matrix.
        beam2 : np.ndarray
            A 3D array of shape (2, 2, Nsources) where each B[:,:,i] is a 2x2 matrix.
        """
        nsources = beam1.shape[2]
        
        for i in range(nsources):
            # Unpack elements of A for the i-th slice.
            a00 = beam1[0, 0, i]
            a01 = beam1[0, 1, i]
            a10 = beam1[1, 0, i]
            a11 = beam1[1, 1, i]
            
            # Unpack elements of C for the i-th slice.
            c00 = coherency[0, 0, i]
            c01 = coherency[0, 1, i]
            c10 = coherency[1, 0, i]
            c11 = coherency[1, 1, i]
            
            # Unpack elements of B for the i-th slice.
            b00 = beam2[0, 0, i]
            b01 = beam2[0, 1, i]
            b10 = beam2[1, 0, i]
            b11 = beam2[1, 1, i]
            
            # Compute the intermediate product: T = A^H * C.
            # A^H is the conjugate transpose of A.
            t00 = np.conj(a00) * c00 + np.conj(a10) * c10
            t01 = np.conj(a00) * c01 + np.conj(a10) * c11
            t10 = np.conj(a01) * c00 + np.conj(a11) * c10
            t11 = np.conj(a01) * c01 + np.conj(a11) * c11
            
            # Compute the final product: out = T * B = A^H * C * B.
            beam1[0, 0, i] = t00 * b00 + t01 * b10
            beam1[0, 1, i] = t00 * b01 + t01 * b11
            beam1[1, 0, i] = t10 * b00 + t11 * b10
            beam1[1, 1, i] = t10 * b01 + t11 * b11