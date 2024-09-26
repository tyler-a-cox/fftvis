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

from matvis.core import _validate_inputs
from matvis.core.coords import CoordinateRotation
from matvis.cpu.beams import UVBeamInterpolator
from matvis._utils import get_dtypes, memtrace, logdebug

from .. import utils

logger = logging.getLogger(__name__)


def simulate(
    *
    antpos: np.ndarray,
    freq: float,
    times: np.ndarray,
    skycoords: np.ndarray,
    fluxes: np.ndarray,
    beam: UVBeam | Callable | None,
    baselines: np.ndarray | None = None,
    telescope_loc: EarthLocation | None = None,
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
    """
    Calculate visibility from an input intensity map and beam model.

    Parameters
    ----------
    antpos : array_like
        Antenna position array. Shape=(NANT, 3).
    freq : float
        Frequency to evaluate the visibilities at [GHz].
    I_sky : array_like
        Intensity distribution of sources/pixels on the sky, assuming intensity
        (Stokes I) only. The Stokes I intensity will be split equally between
        the two linear polarization channels, resulting in a factor of 0.5 from
        the value inputted here. This is done even if only one polarization
        channel is simulated.
        Shape=(NSRCS,).
    beam : UVBeam or AnalyticBeam
        If specified, evaluate primary beam values directly using UVBeam
        objects instead of using pixelized beam maps. Only one of ``bm_cube`` and
        ``beam_list`` should be provided.Note that if `polarized` is True,
        these beams must be efield beams, and conversely if `polarized` is False they
        must be power beams with a single polarization (either XX or YY).
    antpairs : array_like, optional
        Either a 2D array, shape ``(Npairs, 2)``, or list of 2-tuples of ints, with
        the list of antenna-pairs to return as visibilities (all feed-pairs are always
        calculated). If None, all feed-pairs are returned.
    precision : int, optional
        Which precision level to use for floats and complex numbers.
        Allowed values:
        - 1: float32, complex64
        - 2: float64, complex128
    polarized : bool, optional
        Whether to simulate a full polarized response in terms of nn, ne, en,
        ee visibilities. See Eq. 6 of Kohn+ (arXiv:1802.04151) for notation.
        Default: False.
    beam_spline_opts : dict, optional
        Dictionary of options to pass to the beam interpolation function.
    max_progress_reports : int, optional
        Maximum number of progress reports to print to the screen (if logging level
        allows). Default is 100.
    coord_method : str, optional
        The method to use to transform coordinates from the equatorial to horizontal
        frame. The default is to use Astropy coordinate transforms. A faster option,
        which is accurate to within 6 mas, is to use "CoordinateTransformERFA".
    max_memory : int, optional
        The maximum memory (in bytes) to use for the visibility calculation. This is
        not a hard-set limit, but rather a guideline for how much memory to use. If the
        expected memory usage is more than this, the calculation will be broken up into
        chunks. Default is 512 MB.
    min_chunks : int, optional
        The minimum number of chunks to break the source axis into. Default is 1.
    source_buffer : float, optional
        The fraction of the total sources (per chunk) to pre-allocate memory for.
        Default is 0.55, which allows for 10% variance around half the sources
        (since half should be below the horizon). If you have a particular sky model in
        which you expect more or less sources to appear above the horizon at any time,
        set this to a different value.
    coord_method_params
        Parameters particular to the coordinate rotation method of choice. For example,
        for the CoordinateRotationERFA (and GPU version of the same) method, there
        is the parameter ``update_bcrs_every``, which should be a time in seconds, for
        which larger values speed up the computation.

    Returns
    -------
    vis : array_like
        Simulated visibilities. If `polarized = True`, the output will have
        shape (NTIMES, NBLS, NFEED, NFEED), otherwise it will have
        shape (NTIMES, NBLS).

    """
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

    # Set coordinate method
    coord_method = CoordinateRotation._methods[coord_method]
    coord_method_params = coord_method_params or {}

    coords = coord_method(
        flux=np.zeros_like(fluxes[0]),
        times=times,
        telescope_loc=telescope_loc,
        skycoords=skycoords,
        chunk_size=npixc,
        precision=precision,
        source_buffer=source_buffer,
        **coord_method_params,
    )

    # Get the beam interpolation function
    bmfunc = UVBeamInterpolator(
        beam_list=[beam],
        beam_idx=np.array([0]),
        nant=1,
        freq=freq,
        spline_opts=beam_spline_opts,
        polarized=polarized,
        precision=precision,
        nsrc=nsrc_alloc,
    )

    # Setup the coordinate rotation and beam interpolation functions
    bmfunc.setup()
    coords.setup()
    
    for ti, time in enumerate(times):
        # Rotate the coordinates to the current time
        coords.rotate(ti)
        
        for c in range(nchunks):
            # Get the sky coordinates for this chunk
            crd_top, flux_up, nsrcs_up = coords.get_chunk(c)

            beam_values = bmfunc(crd_top[0], crd_top[1], check=ti == 0)

            # Calculate the visibilities for this chunk
            # TODO: Need a function for calculating visibilities
            vis_chunk = _simulate_chunk(
                coords=coords,
                bmfunc=bmfunc,
                fluxes=fluxes,
                baselines=baselines,
                nfeed=nfeed,
                nant=nant,
                ntimes=ntimes,
                nfeed_alloc=nsrc_alloc,
                polarized=polarized,
                precision=precision,
            )
            # Update the visibilities array
            vis[ti, c * npixc : (c + 1) * npixc] = vis_chunk