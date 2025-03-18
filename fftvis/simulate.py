from __future__ import annotations
from multiprocessing import cpu_count
import ray
from threadpoolctl import threadpool_limits
import os
import memray

import finufft
import numpy as np
from matvis import coordinates
from matvis.core.beams import prepare_beam_unpolarized
from matvis.core.coords import CoordinateRotation
from typing import Literal
from pyuvdata.beam_interface import BeamInterface

from astropy.coordinates import EarthLocation, SkyCoord
from astropy import units as un
from astropy.time import Time
import time
import psutil
import logging
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
    beam: BeamInterface | list[BeamInterface],
    beam_idx: list[int],
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
    nthreads: int | None = None,
    coord_method: Literal[
        "CoordinateRotationAstropy", "CoordinateRotationERFA"
    ] = "CoordinateRotationERFA",
    coord_method_params: dict | None = None,
    force_use_ray: bool = False,
    trace_mem: bool = False,
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
    times : astropy.Time instance or array_like
        Times of the observation (can be a numpy array of Julian dates or astropy.Time object).
    beam : UVBeam
        BeamInterface object or list of BeamInterface objects to use for simulation.
    beam_idx: list of ints
        The indices of the beams to use for each antenna in the array.
    telescope_loc
        An EarthLocation object representing the center of the array.
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
    eps : float, default = None
        Desired accuracy of the non-uniform fast fourier transform. If None, the default accuracy
        for the given precision will be used. For precision 1, the default accuracy is 6e-8, and for
        precision 2, the default accuracy is 1e-12.
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
    nthreads : int, optional
        The number of threads to use for each process. If None, the number of threads
        will be set to the number of available CPUs divided by the number of processes.
    coord_method : str, optional
        The method to use for coordinate rotation. Can be either 'CoordinateRotationAstropy'
        or 'CoordinateRotationERFA'. The former uses the astropy.coordinates package for
        coordinate transformations, while the latter uses the ERFA library.
    coord_method_params : dict, optional
        Parameters particular to the coordinate rotation method of choice. For example,
        for the CoordinateRotationERFA method, there is the parameter ``update_bcrs_every``, 
        which should be a time in seconds, for which larger values speed up the computation.
        See the documentation for the CoordinateRotation classes in matvis for more information.
    force_use_ray: bool, default = False
        Whether to force the use of Ray for parallelization. If False, Ray will only be used
        if nprocesses > 1.
    trace_mem : bool, default = False
        Whether to trace memory usage during the simulation. If True, the memory usage
        will be recorded at various points in the simulation and saved to a file.
    live_progress : bool, default = True
        Whether to show progress bar during simulation.

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

    # Check if beam is a list of beams
    if isinstance(beam, list):
        beam = [BeamInterface(b) for b in beam]

        # Prepare the beams
        if not polarized:
            beam = [prepare_beam_unpolarized(b, use_feed=use_feed) for b in beam]
    
    else:
        beam = BeamInterface(beam)

        # Prepare the beam
        if not polarized:
            beam = prepare_beam_unpolarized(beam, use_feed=use_feed)


    return simulate(
        ants=ants,
        freqs=freqs,
        fluxes=fluxes,
        beam=beam,
        beam_idx=beam_idx,
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
        nthreads=nthreads,
        coord_method=coord_method,
        coord_method_params=coord_method_params,
        force_use_ray=force_use_ray,
        trace_mem=trace_mem,
    )


def simulate(
    ants: dict,
    freqs: np.ndarray,
    fluxes: np.ndarray,
    beam: UVBeam,
    beam_idx: list[int],
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
    nthreads: int | None = None,
    coord_method: Literal[
        "CoordinateRotationAstropy", "CoordinateRotationERFA"
    ] = "CoordinateRotationERFA",
    coord_method_params: dict | None = None,
    force_use_ray: bool = False,
    trace_mem: bool = False,
    enable_memory_monitor: bool = False,
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
        pyuvdata UVBeam object to use for all antennas in the array. Per-antenna 
        beams are not yet supported.
    beam_idx: list of ints
        The indices of the beams to use for each antenna in the array.
    ra, dec : array_like
        Arrays of source RA and Dec positions in radians. RA goes from [0, 2 pi]
        and Dec from [-pi/2, +pi/2].
    times : astropy.Time instance or array_like
        Times of the observation (can be a numpy array of Julian dates or astropy.Time object).
    telescope_loc
        An EarthLocation object representing the center of the array.
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
    nthreads : int, optional
        The number of threads to use for each process. If None, the number of threads
        will be set to the number of available CPUs divided by the number of processes.
    coord_method : str, optional
        The method to use for coordinate rotation. Can be either 'CoordinateRotationAstropy'
        or 'CoordinateRotationERFA'. The former uses the astropy.coordinates package for
        coordinate transformations, while the latter uses the ERFA library.
    coord_method_params : dict, optional
        Parameters particular to the coordinate rotation method of choice. For example,
        for the CoordinateRotationERFA method, there is the parameter ``update_bcrs_every``, 
        which should be a time in seconds, for which larger values speed up the computation.
        See the documentation for the CoordinateRotation classes in matvis for more information.
    force_use_ray : bool, default = False
        Whether to force the use of Ray for parallelization. If False, Ray will only be used
        if nprocesses > 1.
    trace_mem : bool, default = False
        Whether to trace memory usage during the simulation. If True, the memory usage
        will be recorded at various points in the simulation and saved to a file.
    enable_memory_monitor : bool, optional
        Turn on Ray memory monitoring (i.e. its ability to track memory usage and
        kill tasks that are putting too much memory pressure on). Generally, this is a
        bad idea for the homogenous calculations done here: if a task goes beyond 
        available memory, the whole simulation should OOM'd, to save CPU cycles.
        
    Returns:
    -------
    vis : np.ndarray
        Array of shape (nfreqs, ntimes, nants, nants) if polarized is False, and
        (nfreqs, ntimes, nfeed, nfeedd, nants, nants) if polarized is True.
    """
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
    else:
        beam = [b.interp(freq_array=freqs, new_object=True, run_check=False) for b in beam]
    
    # Factor of 0.5 accounts for splitting Stokes between polarization channels
    Isky = 0.5 * fluxes

    # Flatten antenna positions
    antkey_to_idx = dict(zip(ants.keys(), range(len(ants))))
    antvecs = np.array([ants[ant] for ant in ants], dtype=real_dtype)

    # Rotate the array to the xy-plane
    rotation_matrix = utils.get_plane_to_xy_rotation_matrix(antvecs)
    rotation_matrix = np.ascontiguousarray(rotation_matrix.astype(real_dtype).T)
    rotated_antvecs = np.dot(rotation_matrix, antvecs.T)
    rotated_ants = {ant: rotated_antvecs[:, antkey_to_idx[ant]] for ant in ants}

    # Compute baseline vectors
    bls = np.array([rotated_ants[bl[1]] - rotated_ants[bl[0]] for bl in baselines])[
        :, :
    ].T.astype(real_dtype)

    # Check if the array is flat within tolerance
    is_coplanar = np.all(np.less_equal(np.abs(bls[2]), flat_array_tol))
    
    bls /= utils.speed_of_light
    
    # Get number of processes for multiprocessing
    if nprocesses is None:
        nprocesses = cpu_count()

    # Check if the times array is a numpy array
    if isinstance(times, np.ndarray):
        times = Time(times, format='jd')

    coord_method = CoordinateRotation._methods[coord_method]
    coord_method_params = coord_method_params or {}
    coord_mgr = coord_method(
        flux=Isky,
        times=times,
        telescope_loc=telescope_loc,
        skycoords=SkyCoord(ra=ra * un.rad, dec=dec * un.rad, frame='icrs'),
        precision=precision,
        **coord_method_params,
    )

    if getattr(coord_mgr, "update_bcrs_every", 0) > (times[-1] - times[0]).to(un.s):
        # We don't need to ever update BCRS, so we get it now before sending
        # out the jobs to multiple processes.
        coord_mgr._set_bcrs(0)
        
    nprocesses, freq_chunks, time_chunks, nf, nt = utils.get_task_chunks(nprocesses, nfreqs, ntimes)
    use_ray = nprocesses > 1 or force_use_ray

    if use_ray:
        # Try to estimate how much shared memory will be required.
        required_shm = bls.nbytes + rotation_matrix.nbytes + freqs.nbytes
        
        for key, val in coord_mgr.__dict__.items():
            if isinstance(val, np.ndarray):
                required_shm += val.nbytes
            
        if isinstance(beam, UVBeam):
            required_shm += beam.data_array.nbytes
        
        # Add visibility memory
        required_shm += (ntimes * nfreqs * nbls * nax * nfeeds) * np.dtype(complex_dtype).itemsize
        
        logger.info(f"Initializing with {2*required_shm/1024**3:.2f} GB of shared memory")
        if not ray.is_initialized():
            if trace_mem:
                # Record which lines of code assign to shared memory, for debugging.
                os.environ['RAY_record_ref_creation_sites'] = "1"
                
            if not enable_memory_monitor:
                os.environ['RAY_memory_monitor_refresh_ms'] = "0"
                
            # Only spill shared memory objects to disk if the Store is totally full.
            # If we don't do this, then since we need a relatively small amount of
            # SHM, it starts writing to disk even though we never needed to.
            os.environ['RAY_object_spilling_threshold'] = "1.0"
            
            try:
                ray.init(
                    num_cpus=nprocesses,
                    object_store_memory = 2*required_shm, 
                    include_dashboard=False
                )
            except ValueError:
                # If there is a ray cluster already running, just connect to it.
                ray.init()    
                
        if trace_mem:
            os.system("ray memory --units MB > before-puts.txt")
        
        # Put data into shared-memory pool        
        bls = ray.put(bls) 
        rotation_matrix = ray.put(rotation_matrix)
        freqs = ray.put(freqs)
        beam = ray.put(beam)
        coord_mgr = ray.put(coord_mgr)
        if trace_mem:
            os.system("ray memory --units MB > after-puts.txt")
    
    ncpus = nthreads or cpu_count()
    nthreads_per_proc = [
        ncpus // nprocesses + (i < ncpus % nprocesses) for i in range(nprocesses)
    ]
    _nbig = nthreads_per_proc.count(nthreads_per_proc[0])
    if _nbig != nprocesses:
        logger.info(
            f"Splitting calculation into {nprocesses} processes. {_nbig} processes will"
            f" use {nthreads_per_proc[0]} threads each, and {nprocesses-_nbig} will use"
            f" {nthreads_per_proc[-1]} threads. Each process will compute "
            f"{nf} frequencies and {nt} times."
        )
    else:
        logger.info(
            f"Splitting calculation into {nprocesses} processes with "
            f"{nthreads_per_proc[0]} threads per-process. Each process will compute "
            f"{nf} frequencies and {nt} times."
        )
    
    
    futures = []
    init_time = time.time()
    if use_ray:
        fnc = _evaluate_vis_chunk_remote.remote
    else:
        fnc = _evaluate_vis_chunk
    
    # logutils.printmem(psutil.Process(), 'before loop')
    for (nthi, fc, tc) in zip(nthreads_per_proc, freq_chunks, time_chunks):        
        futures.append(
            fnc(
                time_idx=tc,
                freq_idx=fc,
                beam=beam,
                coord_mgr=coord_mgr,
                rotation_matrix=rotation_matrix,
                bls=bls,
                freqs=freqs,
                complex_dtype=complex_dtype,
                nfeeds=nfeeds,
                polarized=polarized,
                eps=eps,
                beam_spline_opts=beam_spline_opts,
                interpolation_function=interpolation_function,
                n_threads=nthi,
                is_coplanar=is_coplanar,
                trace_mem=(nprocesses > 1 or force_use_ray) and trace_mem
            )
        )
        if trace_mem:
            os.system("ray memory --units MB > after-futures.txt")

    # logutils.printmem(psutil.Process(), 'while getting futures')
    if use_ray:
        futures = ray.get(futures)
        if trace_mem:
            os.system("ray memory --units MB > got-all.txt")
        
    end_time = time.time()
    logger.info(f"Main loop evaluation time: {end_time - init_time}")
    
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
    coord_mgr: CoordinateRotation,
    rotation_matrix: np.ndarray,
    bls: np.ndarray,
    freqs: np.ndarray,
    complex_dtype: np.dtype,
    nfeeds: int,
    polarized: bool = False,
    eps: float | None = None,
    beam_spline_opts: dict = None,
    interpolation_function: str = "az_za_map_coordinates",
    n_threads: int = 1,
    is_coplanar: bool = False,
    trace_mem: bool = False,
):
    pid = os.getpid()
    pr = psutil.Process(pid)

    if trace_mem:
        memray.Tracker(
            f"memray-{time.time()}_{pid}.bin"
        ).__enter__()

    nbls = bls.shape[1]
    ntimes = len(coord_mgr.times)
    nfreqs = len(freqs)
    
    nt_here = len(coord_mgr.times[time_idx])
    nf_here = len(freqs[freq_idx])
    vis = np.zeros(dtype=complex_dtype, shape=(nt_here, nbls, nfeeds, nfeeds, nf_here))

    coord_mgr.setup()
    

    with threadpool_limits(limits=n_threads, user_api='blas'):
        for time_index, ti in enumerate(range(ntimes)[time_idx]):
            coord_mgr.rotate(ti)
            topo, flux, nsim_sources = coord_mgr.select_chunk(0)
            #logutils.printmem(pr, f"[{time_index+1}/{nt_here}] After Select Chunk")
            
            # truncate to nsim_sources
            topo = topo[:, :nsim_sources]
            flux = flux[:nsim_sources]

            if nsim_sources == 0:
                continue

            # Compute azimuth and zenith angles
            az, za = coordinates.enu_to_az_za(
                enu_e=topo[0], enu_n=topo[1], orientation="uvbeam"
            )
            
            # Rotate source coordinates with rotation matrix.
            utils.inplace_rot(rotation_matrix, topo)     
            topo *= 2*np.pi
            # logutils.printmem(pr, f"[{time_index+1}/{nt_here}] After Az/Za")
            
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
                        A_s,
                        np.ascontiguousarray(uvw[0]),
                        np.ascontiguousarray(uvw[1]),
                        modeord=0,
                        eps=eps,
                        nthreads=n_threads,
                        showwarn=0,
                    )
                else:
                    _vis_here = finufft.nufft3d3(
                        topo[0],
                        topo[1],
                        topo[2],
                        A_s,
                        np.ascontiguousarray(uvw[0]),
                        np.ascontiguousarray(uvw[1]),
                        np.ascontiguousarray(uvw[2]),
                        modeord=0,
                        eps=eps,
                        nthreads=n_threads,
                        showwarn=0,
                    )

                vis[time_index, ..., freqidx] = np.swapaxes(_vis_here.reshape(nfeeds, nfeeds, nbls), 2, 0)

    return vis


_evaluate_vis_chunk_remote = ray.remote(_evaluate_vis_chunk)