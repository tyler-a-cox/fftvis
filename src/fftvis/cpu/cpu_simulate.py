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
from pyuvdata.beam_interface import BeamInterface
from matvis import coordinates
from matvis.core.coords import CoordinateRotation

from ..core.simulate import SimulationEngine, default_accuracy_dict
from .. import utils, beams, logutils
from .cpu_nufft import cpu_nufft2d, cpu_nufft3d

logger = logging.getLogger(__name__)


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
            beam = beam.interp(freq_array=freqs, new_object=True, run_check=False)

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
            coord_mgr._set_bcrs(0)

        nprocesses, freq_chunks, time_chunks, nf, nt = utils.get_task_chunks(
            nprocesses, nfreqs, ntimes
        )
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
            _evaluate_vis_chunk_remote = ray.remote(self._evaluate_vis_chunk)
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
                    beam_spline_opts=beam_spline_opts,
                    interpolation_function=interpolation_function,
                    n_threads=nthi,
                    is_coplanar=is_coplanar,
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
        vis = np.zeros(dtype=complex_dtype, shape=(ntimes, nbls, nfeeds, nfeeds, nfreqs))
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
        beam_spline_opts: dict = None,
        interpolation_function: str = "az_za_map_coordinates",
        n_threads: int = 1,
        is_coplanar: bool = False,
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
        vis = np.zeros(dtype=complex_dtype, shape=(nt_here, nbls, nfeeds, nfeeds, nf_here))
        
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
                utils.inplace_rot(rotation_matrix, topo)
                topo *= 2 * np.pi

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
                        _vis_here = cpu_nufft2d(
                            topo[0],
                            topo[1],
                            A_s,
                            uvw[0],
                            uvw[1],
                            eps=eps,
                            n_threads=n_threads,
                        )
                    else:
                        _vis_here = cpu_nufft3d(
                            topo[0],
                            topo[1],
                            topo[2],
                            A_s,
                            uvw[0],
                            uvw[1],
                            uvw[2],
                            eps=eps,
                            n_threads=n_threads,
                        )

                    vis[time_index, ..., freqidx] = np.swapaxes(
                        _vis_here.reshape(nfeeds, nfeeds, nbls), 2, 0
                    )

        return vis