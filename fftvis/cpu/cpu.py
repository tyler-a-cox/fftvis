import logging
import numpy as np
import psutil
import time
import tracemalloc as tm
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from collections.abc import Sequence
from pyuvdata import UVBeam
from typing import Callable, Literal

def simulate(
    *
    antpos: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    skycoords: np.ndarray,
    I_sky: np.ndarray,
    beam: UVBeam | Callable | None,
    antpairs: np.ndarray | None = None,
    precision: int = 1,
    polarized: bool = False,
    beam_spline_opts: dict | None = None,
    max_progress_reports: int = 100,
    coord_method: Literal[
        "CoordinateRotationAstropy", "CoordinateRotationERFA"
    ] = "CoordinateRotationAstropy",
    max_memory: int | float = np.inf,
    min_chunks: int = 1,
    source_buffer: float = 1.0,
    coord_method_params: dict | None = None,
):
    pass