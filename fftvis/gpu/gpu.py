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

from matvis._utils import get_desired_chunks, get_dtypes, log_progress, logdebug
from matvis.core import _validate_inputs
from matvis.core.coords import CoordinateRotation
from matvis.gpu import beams
from . import beams

from ..cpu import simulate as simcpu
from .. import utils

logger = logging.getLogger(__name__)

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
    fluxes: np.ndarray,
    beam: UVBeam | Callable | None = None,
    polarized: bool = False,
    baselines: np.ndarray | list[tuple[int, int]] | None = None,
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
    init_time = time.time()

    if not tm.is_tracing() and logger.isENABLED_FOR(logging.INFO):
        tm.start()

    # Validate inputs
    nax, nfeed, nant, ntimes = _validate_inputs(
        precision, polarized, antpos, times, I_sky
    )

    # TODO: Add comment
    highest_peak = memtrace(0)

    # Get the desired precision
    rtype, ctype = get_dtypes(precision)

    # Get the desired number of chunks
    nchunks, npixc = utils.get_desired_chunks(
        min(max_memory, psutil.virtual_memory().available),
        min_chunks,
        beam,
        nax,
        nfeed,
        nant,
        len(fluxes),
        precision,
        source_buffer=source_buffer,
    )
    nsrc_alloc = int(npixc * source_buffer)

    coord_method = CoordinateRotation._methods[coord_method]
    coord_method_params = coord_method_params or {}
    coords = coord_method(
        flux=np.zeros_like(fluxes[0]), # Dummy flux for coordinate rotation
        times=times,
        telescope_loc=telescope_loc,
        skycoords=skycoords,
        chunk_size=npixc,
        precision=precision,
        source_buffer=source_buffer,
        gpu=True,
        **coord_method_params,
    )
    
    # Get the beam interpolation function
    bmfunc = beams.GPUBeamInterpolator(
        beam_list=[beam],
        beam_idx=cp.array([0]),
        polarized=polarized,
        nant=1,
        freq=freq,
        nsrc=nsrc_alloc,
        spline_opts=beam_spline_opts,
        precision=precision,
    )

    logger.debug("Starting GPU allocations...")

    init_mem = cp.cuda.Device().mem_info[0]
    logger.debug(f"Before GPU allocations, GPU mem avail is: {init_mem / 1024**3} GB")

    memnow = cp.cuda.Device().mem_info[0]
    if bmfunc.use_interp:
        logger.debug(f"After bmfunc, GPU mem avail is: {memnow / 1024**3} GB.")
    
    coords.setup()
    memnow = cp.cuda.Device().mem_info[0]
    logger.debug(f"After coords, GPU mem avail is: {memnow / 1024**3} GB.")

    for ti, time in enumerate(times):
        for c, (stream, event) in enumerate(zip(streams, events)):
            # Rotate the coordinates to the current time
            coords.rotate(ti)

            # Get the source coordinates for this time
            crdtop, flux_up, nsrcs_up = coords.select_chunk(c)
