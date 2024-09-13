"""GPU implementation of the simulator."""

from __future__ import annotations

import logging
import numpy as np
import psutil
import time
import warnings
from astropy.constants import c as speed_of_light
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from collections.abc import Sequence
from docstring_parser import combine_docstrings
from pyuvdata import UVBeam
from typing import Callable, Literal

from .._utils import get_desired_chunks, get_dtypes, log_progress, logdebug
from ..core import _validate_inputs
from ..core.coords import CoordinateRotation
from ..core.getz import ZMatrixCalc
from ..core.tau import TauCalculator
from ..cpu.cpu import simulate as simcpu
from . import beams
from . import matprod as mp

try:
    import cupy as cp

    HAVE_CUDA = True

except ImportError:
    # if not installed, don't warn
    HAVE_CUDA = False
except Exception as e:  # pragma: no cover
    # if installed but having initialization issues
    # warn, but default back to non-gpu functionality
    warnings.warn(str(e), stacklevel=2)
    HAVE_CUDA = False

@combine_docstrings(simcpu)
def simulate(
    *,
    antpos: np.ndarray,
    freq: float,
    times: Time,
    skycoords: SkyCoord,
    telescope_loc: EarthLocation,
    I_sky: np.ndarray,
    beam: UVBeam | Callable | None = None,
    polarized: bool = False,
    antpairs: np.ndarray | list[tuple[int, int]] | None = None,
    beam_idx: np.ndarray | None = None,
    max_memory: int = np.inf,
    min_chunks: int = 1,
    precision: int = 1,
    beam_spline_opts: dict | None = None,
    coord_method: Literal[
        "CoordinateRotationAstropy",
        "CoordinateRotationERFA",
        "GPUCoordinateRotationERFA",
    ] = "CoordinateRotationAstropy",
    source_buffer: float = 1.0,
    coord_method_params: dict | None = None,
) -> np.ndarray:
    pass