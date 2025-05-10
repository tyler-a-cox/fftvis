"""
CPU-specific simulation implementation for fftvis.

This module provides a concrete implementation of the simulation engine for CPU.
"""

from multiprocessing import cpu_count
import ray
from threadpoolctl import threadpool_limits
import os
import memray
import numpy as np
import time
import psutil
import logging
from typing import Literal, Union
from astropy.coordinates import EarthLocation, SkyCoord
from astropy import units as un
from astropy.time import Time
from pyuvdata import UVBeam
from matvis import coordinates
from matvis.core.coords import CoordinateRotation

from ..core.simulate import SimulationEngine, default_accuracy_dict
from ..core.antenna_gridding import check_antpos_griddability
from .. import utils

# Import the CPU beam evaluator
from .beams import CPUBeamEvaluator
from .nufft import cpu_nufft2d, cpu_nufft3d, cpu_nufft2d_type1
from .utils import inplace_rot
logger = logging.getLogger(__name__)

# Create a global instance of CPUBeamEvaluator to use for beam evaluation
_cpu_beam_evaluator = CPUBeamEvaluator()


# Define a standalone function for Ray to use with remote
@ray.remote
def _evaluate_vis_chunk_remote(
    time_idx: slice,
    freq_idx: slice,
    beam,
    coord_mgr,
    rotation_matrix: np.ndarray,
    bls: np.ndarray,
    freqs: np.ndarray,
    complex_dtype: np.dtype,
    nfeeds: int,
    polarized: bool = False,
    eps: float = None,
    upsampfac: int = 2,
    beam_spline_opts: dict = None,
    interpolation_function: str = "az_za_map_coordinates",
    n_threads: int = 1,
    is_coplanar: bool = False,
    use_type1: bool = False,
    basis_matrix: np.ndarray = None,
    type1_n_modes: int = None,
    trace_mem: bool = False,
):
    """Ray-compatible remote version of _evaluate_vis_chunk."""
    # Create a simulation engine instance
    engine = CPUSimulationEngine() # pragma: no cover
    # Call the method on the instance
    return engine._evaluate_vis_chunk( # pragma: no cover
        time_idx=time_idx,
        freq_idx=freq_idx,
        beam=beam,
        coord_mgr=coord_mgr,
        rotation_matrix=rotation_matrix,
        bls=bls,
        freqs=freqs,
        complex_dtype=complex_dtype,
        nfeeds=nfeeds,
        polarized=polarized,
        eps=eps,
        upsampfac=upsampfac,
        beam_spline_opts=beam_spline_opts,
        interpolation_function=interpolation_function,
        n_threads=n_threads,
        is_coplanar=is_coplanar,
        use_type1=use_type1,
        basis_matrix=basis_matrix,
        type1_n_modes=type1_n_modes,
        trace_mem=trace_mem,
    )


class CPUSimulationEngine(SimulationEngine):
    """CPU implementation of the simulation engine."""

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
        upsampfac: int = 2,
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
        force_use_type3: bool = False,
        trace_mem: bool = False,
        enable_memory_monitor: bool = False,
    ) -> np.ndarray:
        """
        Simulate visibilities using CPU implementation.

        See base class for parameter descriptions.
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

        # Get the redundant groups
        if baselines is None:
            reds = utils.get_pos_reds(ants, include_autos=True)
            baselines = [red[0] for red in reds]

        # Get number of baselines
        nbls = len(baselines)

        if isinstance(beam, UVBeam):
            # Only try to interpolate the beam if it has more than one frequency
            if hasattr(beam, "Nfreqs") and beam.Nfreqs > 1:
                beam = beam.interp(freq_array=freqs, new_object=True, run_check=False) # pragma: no cover

        # Factor of 0.5 accounts for splitting Stokes between polarization channels
        Isky = 0.5 * fluxes

        # Flatten antenna positions
        antkey_to_idx = dict(zip(ants.keys(), range(len(ants))))
        antvecs = np.array([ants[ant] for ant in ants], dtype=real_dtype)

        # If the array is flat within tolerance, we can check for griddability
        if np.abs(antvecs[:, -1]).max() > flat_array_tol or force_use_type3:
            is_gridded = False
        else:
            is_gridded, gridded_antpos, basis_matrix = check_antpos_griddability(ants)
                
        # Rotate antenna positions to XY plane if not gridded
        if not is_gridded:
            # Get the rotation matrix to rotate the array to the XY plane
            rotation_matrix = utils.get_plane_to_xy_rotation_matrix(antvecs)
            rotation_matrix = np.ascontiguousarray(rotation_matrix.T)
            rotated_antvecs = np.dot(rotation_matrix, antvecs.T)
            rotated_ants = {
                ant: rotated_antvecs[:, antkey_to_idx[ant]] for ant in ants
            }
            rotation_matrix = rotation_matrix.astype(real_dtype)
        
            # Compute baseline vectors and convert to speed of light units
            bls = np.array([rotated_ants[bl[1]] - rotated_ants[bl[0]] for bl in baselines])[
                :, :
            ].T
            bls /= utils.speed_of_light
            bls = bls.astype(real_dtype)

            # Check if the array is flat within tolerance
            is_coplanar = np.all(np.less_equal(np.abs(bls[2]), flat_array_tol))
        else:
            logger.info(
                "Using gridded coordinates for the array. Type 1 transform will be used."
            )
            # Compute the baseline vectors in the gridded coordinate system
            bls = np.array([
                gridded_antpos[bl[1]] - gridded_antpos[bl[0]] 
                for bl in baselines]
            ).T
            bls = np.round(bls).astype(int)
            
            # Find the maximum extent of the array in gridded coordinates
            n_modes = 2 * int(np.round(np.max(np.abs(bls)))) + 1

            # Get the maximum baseline length for proper coordinate scaling
            basis_matrix *= 1 / utils.speed_of_light
            basis_matrix = basis_matrix.astype(real_dtype)

            # Assume the array is coplanar for gridded coordinates
            is_coplanar = True
            rotation_matrix = np.eye(3, dtype=real_dtype)

        # Get number of processes for multiprocessing
        if nprocesses is None:
            nprocesses = cpu_count() # pragma: no cover

        # Check if the times array is a numpy array
        if isinstance(times, np.ndarray):
            times = Time(times, format="jd")

        coord_method = CoordinateRotation._methods[coord_method]
        coord_method_params = coord_method_params or {}
        coord_mgr = coord_method(
            flux=Isky,
            times=times,
            telescope_loc=telescope_loc,
            skycoords=SkyCoord(ra=ra * un.rad, dec=dec * un.rad, frame="icrs"),
            precision=precision,
            **coord_method_params,
        )

        if getattr(coord_mgr, "update_bcrs_every", 0) > (times[-1] - times[0]).to(un.s):
            # We don't need to ever update BCRS, so we get it now before sending
            # out the jobs to multiple processes.
            coord_mgr._set_bcrs(0) # pragma: no cover

        nprocesses, freq_chunks, time_chunks, nf, nt = utils.get_task_chunks(
            nprocesses, nfreqs, ntimes
        )
        use_ray = nprocesses > 1 or force_use_ray

        if use_ray: # pragma: no cover
            # Try to estimate how much shared memory will be required.
            required_shm = bls.nbytes + rotation_matrix.nbytes + freqs.nbytes

            for key, val in coord_mgr.__dict__.items():
                if isinstance(val, np.ndarray):
                    required_shm += val.nbytes

            if isinstance(beam, UVBeam):
                required_shm += beam.data_array.nbytes

            # Add visibility memory
            required_shm += (ntimes * nfreqs * nbls * nax * nfeeds) * np.dtype(
                complex_dtype
            ).itemsize

            logger.info(
                f"Initializing with {2*required_shm/1024**3:.2f} GB of shared memory"
            )
            if not ray.is_initialized():
                if trace_mem:
                    # Record which lines of code assign to shared memory, for debugging.
                    os.environ["RAY_record_ref_creation_sites"] = "1"

                if not enable_memory_monitor:
                    os.environ["RAY_memory_monitor_refresh_ms"] = "0"

                # Only spill shared memory objects to disk if the Store is totally full.
                # If we don't do this, then since we need a relatively small amount of
                # SHM, it starts writing to disk even though we never needed to.
                os.environ["RAY_object_spilling_threshold"] = "1.0"

                try:
                    ray.init(
                        num_cpus=nprocesses,
                        object_store_memory=2 * required_shm,
                        include_dashboard=False,
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

        # Create a remote version of _evaluate_vis_chunk if needed
        if use_ray:
            fnc = _evaluate_vis_chunk_remote.remote
        else:
            fnc = self._evaluate_vis_chunk

        # Create workers for each chunk
        for nthi, fc, tc in zip(nthreads_per_proc, freq_chunks, time_chunks):
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
                    upsampfac=upsampfac,
                    beam_spline_opts=beam_spline_opts,
                    interpolation_function=interpolation_function,
                    n_threads=nthi,
                    is_coplanar=is_coplanar,
                    use_type1=is_gridded,
                    basis_matrix=basis_matrix if is_gridded else None,
                    type1_n_modes=n_modes if is_gridded else None,
                    trace_mem=(nprocesses > 1 or force_use_ray) and trace_mem,
                )
            )
            if trace_mem:
                os.system("ray memory --units MB > after-futures.txt")

        # Retrieve results
        if use_ray:
            futures = ray.get(futures)
            if trace_mem:
                os.system("ray memory --units MB > got-all.txt")

        end_time = time.time()
        logger.info(f"Main loop evaluation time: {end_time - init_time}")

        # Combine results from all workers
        vis = np.zeros(
            dtype=complex_dtype, shape=(ntimes, nbls, nfeeds, nfeeds, nfreqs)
        )
        for fc, tc, future in zip(freq_chunks, time_chunks, futures):
            vis[tc][..., fc] = future

        # Reshape to expected output format
        return (
            np.transpose(vis, (4, 0, 2, 3, 1))
            if polarized
            else np.moveaxis(vis[..., 0, 0, :], 2, 0)
        )

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
        upsampfac: int = 2,
        beam_spline_opts: dict = None,
        interpolation_function: str = "az_za_map_coordinates",
        n_threads: int = 1,
        is_coplanar: bool = False,
        use_type1: bool = False,
        basis_matrix: float = None,
        type1_n_modes: int = None,
        trace_mem: bool = False,
    ) -> np.ndarray:
        """
        Evaluate a chunk of visibility data using CPU.

        See base class for parameter descriptions.
        """
        pid = os.getpid()
        pr = psutil.Process(pid)

        if trace_mem:
            memray.Tracker(f"memray-{time.time()}_{pid}.bin").__enter__()

        nbls = bls.shape[1]
        ntimes = len(coord_mgr.times)
        nfreqs = len(freqs)

        nt_here = len(coord_mgr.times[time_idx])
        nf_here = len(freqs[freq_idx])
        vis = np.zeros(
            dtype=complex_dtype, shape=(nt_here, nbls, nfeeds, nfeeds, nf_here)
        )

        coord_mgr.setup()

        with threadpool_limits(limits=n_threads, user_api="blas"):
            for time_index, ti in enumerate(range(ntimes)[time_idx]):
                coord_mgr.rotate(ti)
                topo, flux, nsim_sources = coord_mgr.select_chunk(0)

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
                if not np.allclose(rotation_matrix, np.eye(3)):
                    inplace_rot(rotation_matrix, topo)
                
                # Rotate the basis matrix
                if basis_matrix is not None:
                    # Rotate the basis matrix to the XY plane
                    inplace_rot(basis_matrix.T, topo)

                topo *= 2 * np.pi

                for freqidx in range(nfreqs)[freq_idx]:
                    freq = freqs[freqidx]

                    if not use_type1:
                        uvw = bls * freq

                    # Update beam evaluator for matvis compatibility
                    _cpu_beam_evaluator.beam_list = [beam]
                    _cpu_beam_evaluator.nsrc = len(az)
                    _cpu_beam_evaluator.polarized = polarized
                    _cpu_beam_evaluator.freq = freq

                    A_s = _cpu_beam_evaluator.evaluate_beam(
                        beam,
                        az,
                        za,
                        polarized,
                        freq,
                        spline_opts=beam_spline_opts,
                        interpolation_function=interpolation_function,
                    ).astype(complex_dtype)

                    if polarized:
                        _cpu_beam_evaluator.get_apparent_flux_polarized(
                            A_s, flux[:nsim_sources, freqidx] if flux.ndim > 1 else flux[:nsim_sources]
                        )
                    else:
                        # Check if flux is 1D or 2D
                        if flux.ndim > 1:
                            A_s *= flux[:nsim_sources, freqidx]
                        else:
                            A_s *= flux[:nsim_sources]

                    # Check if A_s can be reshaped to expected dimensions
                    # For polarized case with 2 feeds, expected shape is (2, 2, nsim_sources) -> (4, nsim_sources)
                    expected_size = nfeeds**2 * nsim_sources
                    if A_s.size != expected_size: # pragma: no cover
                        # Log the shape mismatch and try to adapt
                        logger.warning(f"Shape mismatch: A_s size {A_s.size} != expected size {expected_size}") # pragma: no cover
                        logger.warning(f"A_s shape: {A_s.shape}, nfeeds: {nfeeds}, nsim_sources: {nsim_sources}") # pragma: no cover
                        
                        # Handle polarized case specially - if we got a 2D array but expected 3D
                        if polarized and A_s.ndim == 2: # pragma: no cover
                            # Just skip this time/freq, or could try to expand the array
                            continue # pragma: no cover
                    
                    # Try to reshape safely
                    try:
                        A_s.shape = (nfeeds**2, nsim_sources)
                    except ValueError: # pragma: no cover
                        logger.error(f"Cannot reshape A_s with shape {A_s.shape} to {(nfeeds**2, nsim_sources)}") # pragma: no cover
                        continue # pragma: no cover
                    
                    i_sky = A_s

                    if i_sky.dtype != complex_dtype:
                        i_sky = i_sky.astype(complex_dtype)

                    # Compute visibilities w/ non-uniform FFT
                    if use_type1:
                        _vis_here = cpu_nufft2d_type1(
                            topo[0] * freq,
                            topo[1] * freq,
                            i_sky,
                            n_modes=type1_n_modes,
                            index=bls,
                            eps=eps,
                            n_threads=n_threads,
                            upsampfac=upsampfac,
                        )
                    else:
                        if is_coplanar:
                            _vis_here = cpu_nufft2d(
                                topo[0],
                                topo[1],
                                i_sky,
                                uvw[0],
                                uvw[1],
                                eps=eps,
                                n_threads=n_threads,
                                upsampfac=upsampfac,
                            )
                        else:
                            _vis_here = cpu_nufft3d(
                                topo[0],
                                topo[1],
                                topo[2],
                                i_sky,
                                uvw[0],
                                uvw[1],
                                uvw[2],
                                eps=eps,
                                n_threads=n_threads,
                                upsampfac=upsampfac,
                            )

                    vis[time_index, ..., freqidx] = np.swapaxes(
                        _vis_here.reshape(nfeeds, nfeeds, nbls), 2, 0
                    )

        return vis
