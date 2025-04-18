"""
GPU-specific beam evaluation implementation for fftvis.

This module provides GPU-specific beam evaluation functionality.
Currently it contains stub implementations.
"""

import numpy as np
from ..core.beams import BeamEvaluator


class GPUBeamEvaluator(BeamEvaluator):
    """GPU implementation of the beam evaluator (stub)."""

    def evaluate_beam(self, az, za, freq, polarized):
        """
        Evaluate the beam pattern at the given coordinates using GPU.

        Parameters
        ----------
        az : np.ndarray
            Azimuth coordinates in radians.
        za : np.ndarray
            Zenith angle coordinates in radians.
        freq : float
            Frequency to evaluate the beam at in Hz.
        polarized : bool
            Whether to evaluate the polarized beam.

        Returns
        -------
        np.ndarray
            Beam values at the given coordinates.

        Raises
        ------
        NotImplementedError
            GPU beam evaluation is not yet implemented.
        """
        raise NotImplementedError("GPU beam evaluation not yet implemented")