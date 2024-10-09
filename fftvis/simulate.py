from __future__ import annotations
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Pool, cpu_count
from functools import partial
import ray
from threadpoolctl import threadpool_limits

import finufft
import numpy as np
from matvis import coordinates
from matvis.core.beams import prepare_beam_unpolarized
from matvis.cpu.coords import CoordinateRotationAstropy, CoordinateRotationERFA
from matvis.core.coords import CoordinateRotation
from typing import Callable, Literal

from astropy.coordinates import EarthLocation, SkyCoord
from astropy import units as un
from astropy.time import Time
import time
import psutil
from rich.progress import Progress
import logging
import tracemalloc as tm
from pyuvdata import UVBeam

from . import utils, beams, logutils

# Default accuracy for the non-uniform fast fourier transform based on precision
default_accuracy_dict = {
    1: 6e-8,
    2: 1e-13,
}

logger = logging.getLogger(__name__)


def simulate_vis(
    ants: dict,
    fluxes: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    beam,
    telescope_loc: EarthLocation,
    baselines: list[tuple] = None,
    precision: int = 2,
    polarized: bool = False,
    eps: float = None,
    beam_spline_opts: dict = None,
    use_feed: str = "x",
    flat_array_tol: float = 0.0,
    interpolation_function: str = "az_za_map_coordinates",
    nprocesses: int | None = 1,
    coord_method: Literal[
        "CoordinateRotationAstropy", "CoordinateRotationERFA"
    ] = "CoordinateRotationERFA",
    coord_method_params: dict | None = None,
):
    """
    Parameters:
    ----------
    ants : dict
        Dictionary of antenna positions
    fluxes : np.ndarray
        Intensity distribution of sources/pixels on the sky, assuming intensity
        (Stokes I) only. The Stokes I intensity will be split equally between
        the two linear polarization channels, resulting in a factor of 0.5 from
        the value inputted here. This is done even if only one polarization
        channel is simulated.
    ra, dec : array_like
        Arrays of source RA and Dec positions in radians. RA goes from [0, 2 pi]
        and Dec from [-pi, +pi].
    freqs : np.ndarray
        Frequencies to evaluate visibilities in Hz.
    lsts : np.ndarray
        Local sidereal time in radians. Range is [0, 2 pi].
    beam : UVBeam
        Beam object to use for the array. Per-antenna beams are not yet supported.
    baselines : list of tuples, default = None
        If provided, only the baselines within the list will be simulated and array of shape
        (nbls, nfreqs, ntimes) will be returned if polarized is False, and (nbls, nfreqs, ntimes, 2, 2) if polarized is True.
    precision : int, optional
       Which precision level to use for floats and complex numbers
       Allowed values:
       - 1: float32, complex64
       - 2: float64, complex128
    polarized : bool, optional
        Whether to simulate polarized visibilities. If True, the output will have
        shape (nfreqs, ntimes, 2, 2, nants, nants), and if False, the output will
        have shape (nfreqs, ntimes, nants, nants).
    latitude : float, optional
        Latitude of the array in radians. The default is the
        HERA latitude = -30.7215 * pi / 180.
    eps : float, default = None
        Desired accuracy of the non-uniform fast fourier transform. If None, the default accuracy
        for the given precision will be used. For precision 1, the default accuracy is 6e-8, and for
        precision 2, the default accuracy is 1e-12.
    beam_spline_opts : dict, optional
        Options to pass to :meth:`pyuvdata.uvbeam.UVBeam.interp` as `spline_opts`.
    flat_array_tol : float, default = 0.0
        Tolerance for checking if the array is flat in units of meters. If the
        z-coordinate of all baseline vectors is within this tolerance, the array
        is considered flat and the z-coordinate is set to zero. Default is 0.0.
    live_progress : bool, default = True
        Whether to show progress bar during simulation.
    interpolation_function : str, default = "az_za_simple"
        The interpolation function to use when interpolating the beam. Can be either be
        'az_za_simple' or 'az_za_map_coordinates'. The former is slower but more accurate
        at the edges of the beam, while the latter is faster but less accurate
        for interpolation orders greater than linear.

    Returns:
    -------
    vis : np.ndarray
        Array of shape (nfreqs, ntimes, nants, nants) if polarized is False, and
        (nfreqs, ntimes, nfeed, nfeed, nants, nants) if polarized is True.
    """
    # Get the accuracy for the given precision if not provided
    if eps is None:
        eps = default_accuracy_dict[precision]

    # Make sure antpos has the right format
    ants = {k: np.array(v) for k, v in ants.items()}

    # Prepare the beam
    if not polarized:
        beam = prepare_beam_unpolarized(beam, use_pol=use_feed*2)

    return simulate(
        ants=ants,
        freqs=freqs,
        fluxes=fluxes,
        beam=beam,
        ra=ra,
        dec=dec,
        times=times,
        telescope_loc=telescope_loc,
        baselines=baselines,
        precision=precision,
        polarized=polarized,
        eps=eps,
        beam_spline_opts=beam_spline_opts,
        flat_array_tol=flat_array_tol,
        interpolation_function=interpolation_function,
        nprocesses=nprocesses,
        coord_method=coord_method,
        coord_method_params=coord_method_params,
    )


def simulate(
    ants: dict,
    freqs: np.ndarray,
    fluxes: np.ndarray,
    beam,
    ra: np.ndarray,
    dec: np.ndarray,
    times: np.ndarray,
    telescope_loc: EarthLocation,
    baselines: list[tuple[int, int]] | None = None,
    precision: int = 2,
    polarized: bool = False,
    eps: float | None = None,
    beam_spline_opts: dict = None,
    flat_array_tol: float = 0.0,
    interpolation_function: str = "az_za_map_coordinates",
    nprocesses: int | None = 1,
    coord_method: Literal[
        "CoordinateRotationAstropy", "CoordinateRotationERFA"
    ] = "CoordinateRotationERFA",
    coord_method_params: dict | None = None,
):
    """
    Parameters:
    ----------
    ants : dict
        Dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
    freqs : np.ndarray
        Frequencies to evaluate visibilities at in Hz.
    fluxes : np.ndarray
        Intensity distribution of sources/pixels on the sky, assuming intensity
        (Stokes I) only. The Stokes I intensity will be split equally between
        the two linear polarization channels, resulting in a factor of 0.5 from
        the value inputted here. This is done even if only one polarization
        channel is simulated.
    beam : UVBeam
        Beam object to use for the array. Per-antenna beams are not yet supported.
    crd_eq : np.ndarray
        Cartesian unit vectors of sources in an ECI (Earth Centered
        Inertial) system, which has the Earth's center of mass at
        the origin, and is fixed with respect to the distant stars.
        The components of the ECI vector for each source are:
        (cos(RA) cos(Dec), sin(RA) cos(Dec), sin(Dec)).
        Shape=(3, NSRCS).
    eq2tops : np.ndarray
        Set of 3x3 transformation matrices to rotate the RA and Dec
        cosines in an ECI coordinate system (see `crd_eq`) to
        topocentric ENU (East-North-Up) unit vectors at each
        time/LST/hour angle in the dataset. Shape=(NTIMES, 3, 3).
    baselines : list of tuples, default = None
        If provided, only the baselines within the list will be simulated and array of shape
        (nbls, nfreqs, ntimes) will be returned
    precision : int, optional
        Which precision level to use for floats and complex numbers
        Allowed values:
        - 1: float32, complex64
        - 2: float64, complex128
    eps : float, default = 6e-8
        Desired accuracy of the non-uniform fast fourier transform.
    beam_spline_opts : dict, optional
        Options to pass to :meth:`pyuvdata.uvbeam.UVBeam.interp` as `spline_opts`.
    flat_array_tol : float, default = 0.0
        Tolerance for checking if the array is flat in units of meters. If the
        z-coordinate of all baseline vectors is within this tolerance, the array
        is considered flat and the z-coordinate is set to zero. Default is 0.0.
    interpolation_function : str, default = "az_za_simple"
        The interpolation function to use when interpolating the beam. Can be either be
        'az_za_simple' or 'az_za_map_coordinates'. The former is slower but more accurate
        at the edges of the beam, while the latter is faster but less accurate
        for interpolation orders greater than linear.
    nprocesses : int, optional
        The number of parallel processes to use. Computations are parallelized over
        integration times. Set to 1 to disable multiprocessing entirely, or set to 
        None to use all available processors.
        
    Returns:
    -------
    vis : np.ndarray
        Array of shape (nfreqs, ntimes, nants, nants) if polarized is False, and
        (nfreqs, ntimes, nfeed, nfeedd, nants, nants) if polarized is True.
    """
    
    if not tm.is_tracing() and logger.isEnabledFor(logging.INFO):
        tm.start()

    highest_peak = logutils.memtrace(0)

    # Get sizes of inputs
    nfreqs = np.size(freqs)
    ntimes = len(times)


    nax = nfeeds = 2 if polarized else 1

    if precision == 1:
        real_dtype = np.float32
        complex_dtype = np.complex64
    else:
        real_dtype = np.float64
        complex_dtype = np.complex128

    if eps is None:
        eps = default_accuracy_dict[precision]
    
    if ra.dtype != real_dtype:
        ra = ra.astype(real_dtype)
    if dec.dtype != real_dtype:
        dec = dec.astype(real_dtype)
    if freqs.dtype != real_dtype:
        freqs = freqs.astype(real_dtype)

    # Get the redundant groups - TODO handle this better
    if baselines is None:
        reds = utils.get_pos_reds(ants, include_autos=True)
        baselines = [red[0] for red in reds]

    # Get number of baselines
    nbls = len(baselines)

    if isinstance(beam, UVBeam):
        beam = beam.interp(freq_array=freqs, new_object=True, run_check=False)
    
    # Factor of 0.5 accounts for splitting Stokes between polarization channels
    Isky = 0.5 * fluxes

    # Flatten antenna positions
    antkey_to_idx = dict(zip(ants.keys(), range(len(ants))))
    antvecs = np.array([ants[ant] for ant in ants], dtype=real_dtype)

    # Rotate the array to the xy-plane
    rotation_matrix = utils.get_plane_to_xy_rotation_matrix(antvecs)
    rotation_matrix = rotation_matrix.astype(real_dtype)
    rotated_antvecs = np.dot(rotation_matrix.T, antvecs.T)
    rotated_ants = {ant: rotated_antvecs[:, antkey_to_idx[ant]] for ant in ants}

    # Compute baseline vectors
    bls = np.array([rotated_ants[bl[1]] - rotated_ants[bl[0]] for bl in baselines])[
        :, :
    ].T.astype(real_dtype)

    # Check if the array is flat within tolerance
    is_coplanar = np.all(np.less_equal(np.abs(bls[2]), flat_array_tol))
    
    bls /= utils.speed_of_light
    
    # Have up to 100 reports as it iterates through time.
    highest_peak = logutils.memtrace(highest_peak)

    # Get number of processes for multiprocessing
    if nprocesses is None:
        nprocesses = cpu_count()

    coord_method = CoordinateRotation._methods[coord_method]
    coord_method_params = coord_method_params or {}
    coord_mgr = coord_method(
        flux=Isky,
        times=Time(times, format='jd'),
        telescope_loc=telescope_loc,
        skycoords=SkyCoord(ra=ra * un.rad, dec=dec * un.rad, frame='icrs'),
        precision=precision,
        **coord_method_params,
    )

    nprocesses, freq_chunks, time_chunks, nf, nt = utils.get_task_chunks(nprocesses, nfreqs, ntimes)
    if nprocesses > 1:
        if not ray.is_initialized():
            ray.init()
        
        # Put data into shared-memory pool
        Isky = ray.put(Isky)
        bls = ray.put(bls) 
        times = ray.put(times)
        rotation_matrix = ray.put(rotation_matrix)
        freqs = ray.put(freqs)
        ra = ray.put(ra)
        dec = ray.put(dec)
        beam = ray.put(beam)
        coord_mgr = ray.put(coord_mgr)
         
    logger.info(
        f"Splitting calculation into {nprocesses} chunk(s) with {nf} frequencies and {nt} times each"
    )
    
    futures = []
    init_time = time.time()
    with threadpool_limits(limits=cpu_count() if nprocesses > 1 else 1, user_api='blas'):
        for fc, tc in zip(freq_chunks, time_chunks):
            kw = dict(
                time_idx=tc,
                freq_idx=fc,
                beam=beam,
                nax=nax,
                coord_mgr=coord_mgr,
                rotation_matrix=rotation_matrix,
                bls=bls,
                freqs=freqs,
                complex_dtype=complex_dtype,
                nfeeds=nfeeds,
                location=telescope_loc,
                polarized=polarized,
                eps=eps,
                beam_spline_opts=beam_spline_opts,
                interpolation_function=interpolation_function,
                n_threads=1 if nprocesses > 1 else 0,
                is_coplanar=is_coplanar,
            )
        
            if nprocesses > 1:
                futures.append(_evaluate_vis_chunk_remote.remote(**kw))
            else:
                futures.append(_evaluate_vis_chunk(**kw))
        
        if nprocesses > 1:
            futures = ray.get(futures)
    end_time = time.time()
    print("Main loop evaluation time: ", end_time - init_time)
    
    vis = np.zeros(dtype=complex_dtype, shape=(ntimes, nbls, nfeeds, nfeeds, nfreqs))
    for fc, tc, future in zip(freq_chunks, time_chunks, futures):
        vis[tc][..., fc] = future
                     
    return (
        np.transpose(vis, (4, 0, 2, 3, 1))
        if polarized
        else np.moveaxis(vis[..., 0, 0, :], 2, 0)
    )


def _evaluate_vis_chunk(
    time_idx: slice,
    freq_idx: slice,
    beam: UVBeam,
    nax: int,
    coord_mgr: CoordinateRotation,
    rotation_matrix: np.ndarray,
    bls: np.ndarray,
    freqs: np.ndarray,
    complex_dtype: np.dtype,
    nfeeds: int,
    location: EarthLocation,
    polarized: bool = False,
    eps: float | None = None,
    beam_spline_opts: dict = None,
    interpolation_function: str = "az_za_map_coordinates",
    n_threads: int = 1,
    is_coplanar: bool = False,    
):
    nbls = bls.shape[1]
    ntimes = len(coord_mgr.times)
    nfreqs = len(freqs)
    
    nt_here = len(coord_mgr.times[time_idx])
    nf_here = len(freqs[freq_idx])
    vis = np.zeros(dtype=complex_dtype, shape=(nt_here, nbls, nfeeds, nfeeds, nf_here))
    coord_mgr.setup()
    
    for time_index, ti in enumerate(range(ntimes)[time_idx]):
        coord_mgr.rotate(ti)
        topo, flux, nsim_sources = coord_mgr.select_chunk(0)
        
        if nsim_sources == 0:
            continue

        # Compute azimuth and zenith angles
        az, za = coordinates.enu_to_az_za(
            enu_e=topo[0, :nsim_sources], enu_n=topo[1, :nsim_sources], orientation="uvbeam"
        )
        
        # Rotate source coordinates with rotation matrix.
        topo = np.dot(rotation_matrix.T, topo[:, :nsim_sources])
        topo *= 2*np.pi

        for freqidx in range(nfreqs)[freq_idx]:
            freq = freqs[freqidx]
            uvw = bls * freq

            A_s = beams._evaluate_beam(
                beam,
                az,
                za,
                polarized,
                freq,
                spline_opts=beam_spline_opts,
                interpolation_function=interpolation_function,
            ).astype(complex_dtype)
            if polarized:
               beams.get_apparent_flux_polarized(A_s, flux[:nsim_sources, freqidx])            
            else:
                A_s *= flux[:nsim_sources, freqidx]

            A_s.shape = (nfeeds**2, nsim_sources)
            i_sky = A_s
            
            if i_sky.dtype != complex_dtype:
                i_sky = i_sky.astype(complex_dtype)
            
            # Compute visibilities w/ non-uniform FFT
            if is_coplanar:
                _vis_here = finufft.nufft2d3(
                    topo[0],
                    topo[1],
                    i_sky,
                    np.ascontiguousarray(uvw[0]),
                    np.ascontiguousarray(uvw[1]),
                    modeord=0,
                    eps=eps,
                    nthreads=n_threads,
                )
            else:
                _vis_here = finufft.nufft3d3(
                    topo[0],
                    topo[1],
                    topo[2],
                    i_sky,
                    np.ascontiguousarray(uvw[0]),
                    np.ascontiguousarray(uvw[1]),
                    np.ascontiguousarray(uvw[2]),
                    modeord=0,
                    eps=eps,
                    nthreads=n_threads,
                )
            vis[time_index, ..., freqidx] = np.swapaxes(_vis_here.reshape(nfeeds, nfeeds, nbls), 2, 0)
    return vis

_evaluate_vis_chunk_remote = ray.remote(_evaluate_vis_chunk)