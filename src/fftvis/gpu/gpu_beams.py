"""
GPU-specific beam evaluation implementation for fftvis.

This module provides GPU-specific beam evaluation functionality.
"""

import cupy as cp
import numpy as np
from pyuvdata.beam_interface import BeamInterface
from typing import Dict, Optional

from ..core.beams import BeamEvaluator


class GPUBeamEvaluator(BeamEvaluator):
    """GPU implementation of the beam evaluator."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._gpu_beam_data = {}  # Cache beam data on GPU
        self.nsrc = 0

    def evaluate_beam(
        self,
        beam: BeamInterface,
        az: cp.ndarray,
        za: cp.ndarray,
        polarized: bool,
        freq: float,
        check: bool = False,
        spline_opts: Optional[Dict] = None,
        interpolation_function: str = "az_za_map_coordinates",
    ) -> cp.ndarray:
        """
        Evaluate the beam pattern at the given coordinates using GPU.

        Parameters
        ----------
        beam : BeamInterface
            Beam object to evaluate.
        az : cp.ndarray
            Azimuth coordinates in radians (on GPU).
        za : cp.ndarray
            Zenith angle coordinates in radians (on GPU).
        freq : float
            Frequency to evaluate the beam at in Hz.
        polarized : bool
            Whether to evaluate the polarized beam.
        check : bool, optional
            Whether to check for invalid beam values.
        spline_opts : dict, optional
            Options for spline interpolation (passed to map_coordinates).
        interpolation_function : str, optional
            The interpolation function to use ('az_za_map_coordinates' is expected).

        Returns
        -------
        cp.ndarray
            Beam values at the given coordinates (on GPU).

        Raises
        ------
        ValueError
            If interpolation_function is not 'az_za_map_coordinates'.
            If beam interpolation results in invalid values (and check=True).
        """
        # Save these for matvis compatibility and internal use
        self.polarized = polarized
        self.freq = freq
        self.spline_opts = spline_opts or {}
        self.nsrc = len(az)

        if interpolation_function != "az_za_map_coordinates":
            raise ValueError(
                "GPU beam evaluation only supports 'az_za_map_coordinates'"
            )

        # Ensure inputs are cupy arrays
        if not isinstance(az, cp.ndarray):
            az = cp.asarray(az)
        if not isinstance(za, cp.ndarray):
            za = cp.asarray(za)

        # Get beam data and transfer to GPU if not already there
        beam_key = id(beam)  # Use object ID as key
        if beam_key not in self._gpu_beam_data:
            # Get beam data from the beam object
            # Convert from CPU to GPU
            # Use the same method as CPU implementation for consistency
            kw = {
                "reuse_spline": True,
                "check_azza_domain": False,
                "spline_opts": spline_opts,
                "interpolation_function": interpolation_function,
            }

            beam_data_cpu = beam.compute_response(
                az_array=np.asarray(cp.asnumpy(az)),
                za_array=np.asarray(cp.asnumpy(za)),
                freq_array=np.atleast_1d(freq),
                **kw,
            )

            if polarized:
                beam_data_cpu = beam_data_cpu[:, :, 0, :]
            else:
                # Here we have already asserted that the beam is a power beam and
                # has only one polarization, so we just evaluate that one.
                beam_data_cpu = beam_data_cpu[0, 0, 0, :]

            beam_data_gpu = cp.asarray(beam_data_cpu)
            self._gpu_beam_data[beam_key] = beam_data_gpu

        # Get cached beam data or use what was just created
        beam_data = self._gpu_beam_data.get(beam_key)

        # If the beam evaluation used different az/za coordinates, we need to re-evaluate
        if beam_data is None or beam_data.shape[-1] != self.nsrc:
            # Previous cached data won't work - re-evaluate
            kw = {
                "reuse_spline": True,
                "check_azza_domain": False,
                "spline_opts": spline_opts,
                "interpolation_function": interpolation_function,
            }

            beam_data_cpu = beam.compute_response(
                az_array=np.asarray(cp.asnumpy(az)),
                za_array=np.asarray(cp.asnumpy(za)),
                freq_array=np.atleast_1d(freq),
                **kw,
            )

            if polarized:
                beam_data_cpu = beam_data_cpu[:, :, 0, :]
            else:
                # Here we have already asserted that the beam is a power beam and
                # has only one polarization, so we just evaluate that one.
                beam_data_cpu = beam_data_cpu[0, 0, 0, :]

            beam_data = cp.asarray(beam_data_cpu)
            self._gpu_beam_data[beam_key] = beam_data

        # Check for invalid beam values if requested
        if check:
            sm = cp.sum(beam_data)
            if cp.isinf(sm) or cp.isnan(sm):
                raise ValueError("GPU Beam interpolation resulted in an invalid value")

        return beam_data

    def get_apparent_flux_polarized(self, beam: cp.ndarray, flux: cp.ndarray) -> None:
        """
        Calculate apparent flux of the sources using GPU.

        This modifies the beam array in-place to contain the apparent flux values.

        Parameters
        ----------
        beam : cp.ndarray
            Array with beam values of shape (nax, nfd, nsrc)
        flux : cp.ndarray
            Array with source flux values of shape (nsrc,)
        """
        # Ensure inputs are cupy arrays
        if not isinstance(beam, cp.ndarray):
            beam = cp.asarray(beam)
        if not isinstance(flux, cp.ndarray):
            flux = cp.asarray(flux)

        # Get dimensions
        nax, nfd, nsrc = beam.shape

        # Extract E-field components for each feed
        E_f0 = beam[:, 0, :]  # (nax, nsrc) for feed 0
        E_f1 = beam[:, 1, :]  # (nax, nsrc) for feed 1

        # Calculate values needed for coherency matrix
        E_f0_abs_sq = cp.sum(cp.abs(E_f0) ** 2, axis=0)
        E_f1_abs_sq = cp.sum(cp.abs(E_f1) ** 2, axis=0)
        E_f0_conj_dot_E_f1 = cp.sum(cp.conj(E_f0) * E_f1, axis=0)

        # Calculate apparent flux (modify beam in-place)
        beam[0, 0, :] = E_f0_abs_sq * flux
        beam[0, 1, :] = E_f0_conj_dot_E_f1 * flux
        beam[1, 0, :] = cp.conj(E_f0_conj_dot_E_f1) * flux
        beam[1, 1, :] = E_f1_abs_sq * flux
