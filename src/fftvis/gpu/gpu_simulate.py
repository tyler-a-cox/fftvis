"""
GPU-specific simulation implementation for fftvis.

This module provides a concrete implementation of the simulation engine for GPU.
Currently it contains a stub implementation.
"""

import numpy as np
import logging
from typing import Literal, Union
from astropy.coordinates import EarthLocation
from astropy.time import Time
from matvis.core.coords import CoordinateRotation

from ..core.simulate import SimulationEngine

logger = logging.getLogger(__name__)


class GPUSimulationEngine(SimulationEngine):
    """GPU implementation of the simulation engine."""

    def simulate(
        self,
        ants: dict,
        freqs: np.ndarray,
        fluxes: np.ndarray,
        beam,
        ra: np.ndarray,
        dec: np.ndarray,
        times: Union[np.ndarray, Time],
        telescope_loc: EarthLocation,
        baselines: list[tuple] = None,
        precision: int = 2,
        polarized: bool = False,
        eps: float = None,
        beam_spline_opts: dict = None,
        flat_array_tol: float = 1e-6,
        interpolation_function: str = "az_za_map_coordinates",
        nprocesses: int | None = 1,
        nthreads: int | None = None,
        coord_method: Literal[
            "CoordinateRotationAstropy", "CoordinateRotationERFA"
        ] = "CoordinateRotationERFA",
        coord_method_params: dict | None = None,
        force_use_ray: bool = False,
        trace_mem: bool = False,
        enable_memory_monitor: bool = False,
    ) -> np.ndarray:
        """
        Simulate visibilities using GPU implementation.

        See base class for parameter descriptions.

        Raises
        ------
        NotImplementedError
            GPU simulation is not yet implemented.
        """
        raise NotImplementedError("GPU simulation not yet implemented")

    def _evaluate_vis_chunk(
        self,
        time_idx: slice,
        freq_idx: slice,
        beam,
        coord_mgr: CoordinateRotation,
        rotation_matrix: np.ndarray,
        bls: np.ndarray,
        freqs: np.ndarray,
        complex_dtype: np.dtype,
        nfeeds: int,
        polarized: bool = False,
        eps: float = None,
        beam_spline_opts: dict = None,
        interpolation_function: str = "az_za_map_coordinates",
        n_threads: int = 1,
        is_coplanar: bool = False,
        trace_mem: bool = False,
    ) -> np.ndarray:
        """
        Evaluate a chunk of visibility data using GPU.

        See base class for parameter descriptions.

        Raises
        ------
        NotImplementedError
            GPU chunk evaluation is not yet implemented.
        """
        raise NotImplementedError("GPU _evaluate_vis_chunk not yet implemented")
