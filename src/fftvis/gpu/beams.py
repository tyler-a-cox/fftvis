"""
GPU-specific beam evaluation implementation for fftvis.

This module provides GPU-specific beam evaluation functionality by inheriting
from matvis's proven GPUBeamInterpolator implementation.
"""

import cupy as cp
import numpy as np
from pyuvdata.beam_interface import BeamInterface
from typing import Dict, Optional

# Check for GPU capabilities (exported for compatibility)
try:
    from cupyx.scipy.special import j1 as gpu_j1
    HAS_GPU_BESSEL = True
except ImportError:
    HAS_GPU_BESSEL = False

try:
    from cupyx.scipy.ndimage import map_coordinates as gpu_map_coordinates
    HAS_GPU_MAP_COORDS = True
except ImportError:
    HAS_GPU_MAP_COORDS = False

# Import matvis's GPU beam interpolator
from matvis.gpu.beams import GPUBeamInterpolator


class GPUBeamEvaluator(GPUBeamInterpolator):
    """
    GPU beam evaluator inheriting directly from matvis.

    Provides backward-compatible evaluate_beam() API while using
    matvis's setup()/interp() pattern internally.
    """

    def __init__(self, precision: int = 2, **kwargs):
        """
        Initialize with minimal state - full init happens in evaluate_beam().

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
        Evaluate beam (backward-compatible API).

        Internally uses matvis's setup()/interp() pattern.

        Parameters
        ----------
        beam : BeamInterface
            Beam object to evaluate.
        az : cp.ndarray
            Azimuth coordinates in radians (on GPU).
        za : cp.ndarray
            Zenith angle coordinates in radians (on GPU).
        polarized : bool
            Whether to evaluate the polarized beam.
        freq : float
            Frequency to evaluate the beam at in Hz.
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
        if interpolation_function != "az_za_map_coordinates":
            raise ValueError(
                "GPU beam evaluation only supports 'az_za_map_coordinates'"
            )

        # Convert to expected types
        az = cp.asarray(az)
        za = cp.asarray(za)
        nsrc = len(az)

        # Check if we need to reinitialize (beam/freq/polarization changed)
        beam_id = id(beam)
        need_reinit = (
            not self._initialized
            or beam_id != self._current_beam_id
            or freq != self._current_freq
            or polarized != self._current_polarized
        )

        if need_reinit:
            # Initialize GPUBeamInterpolator
            super().__init__(
                beam_list=[beam],
                beam_idx=None,
                polarized=polarized,
                nant=1,
                freq=freq,
                nsrc=nsrc,
                precision=self.precision,
                spline_opts=spline_opts or {},
            )
            self.setup()  # Transfer beam data to GPU once
            self._initialized = True
            self._current_beam_id = beam_id
            self._current_freq = freq
            self._current_polarized = polarized

        # Convert az/za to ENU (tx/ty) for matvis interface
        # Inverse of matvis's enu_to_az_za function for uvbeam orientation
        tx = cp.sin(za) * cp.cos(az)  # enu_e
        ty = cp.sin(za) * cp.sin(az)  # enu_n

        # Call matvis's interp method
        nfeed = 2 if polarized else 1
        nax = 2 if polarized else 1
        complex_dtype = cp.complex64 if self.precision == 1 else cp.complex128

        out = cp.zeros((1, nfeed, nax, nsrc), dtype=complex_dtype)
        self.interp(tx, ty, out)

        # Check for invalid values
        if check:
            if cp.isinf(cp.sum(out)) or cp.isnan(cp.sum(out)):
                raise ValueError("Beam interpolation resulted in invalid value")

        # Return in expected format (remove beam dimension)
        if polarized:
            return out[0]  # Shape: (nfeed, nax, nsrc)
        else:
            # Return power for unpolarized: fftvis computes V = NUFFT(beam * flux)
            # so beam must be |E|^2. matvis returns E-field, so we square it.
            result = out[0, 0, 0, :]
            return cp.abs(result) ** 2

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
        # Ensure inputs are cupy arrays (no-op if already cupy)
        beam = cp.asarray(beam)
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
