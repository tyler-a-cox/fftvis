"""
GPU-specific simulation implementation for fftvis.

This module provides a concrete implementation of the simulation engine for GPU.
"""

import ray
import os
import numpy as np
import cupy as cp
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
from .. import utils
from . import utils as gpu_utils

# Import the GPU beam evaluator and NUFFT
from .beams import GPUBeamEvaluator
from .nufft import gpu_nufft2d, gpu_nufft3d, gpu_nufft2d_batch, gpu_nufft3d_batch
from .memory_manager import GPUMemoryManager

logger = logging.getLogger(__name__)

# Create a global instance of GPUBeamEvaluator to use for beam evaluation
_gpu_beam_evaluator = GPUBeamEvaluator()


# Define a standalone function for Ray to use with remote
@ray.remote
def _evaluate_vis_chunk_remote_gpu(
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
    beam_spline_opts: dict = None,
    interpolation_function: str = "az_za_map_coordinates",
    n_threads: int = 1,
    is_coplanar: bool = False,
    trace_mem: bool = False,
    freq_batch_size: int = 16,
):
    """Ray-compatible remote version of _evaluate_vis_chunk for GPU."""
    # Create a simulation engine instance in the remote process
    engine = GPUSimulationEngine()  # pragma: no cover
    # Call the method on the instance
    return engine._evaluate_vis_chunk(  # pragma: no cover
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
        beam_spline_opts=beam_spline_opts,
        interpolation_function=interpolation_function,
        n_threads=n_threads,
        is_coplanar=is_coplanar,
        trace_mem=trace_mem,
        freq_batch_size=freq_batch_size,
    )


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
        """
        # --- Initial Setup (mostly on CPU) ---
        if interpolation_function != "az_za_map_coordinates":
            logger.warning(
                "GPU backend only supports 'az_za_map_coordinates' for beam interpolation. Ignoring provided interpolation_function."
            )
            interpolation_function = "az_za_map_coordinates"

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

        # Ensure numpy arrays have correct dtype before potential transfer
        if ra.dtype != real_dtype:
            ra = ra.astype(real_dtype)
        if dec.dtype != real_dtype:
            dec = dec.astype(real_dtype)
        if freqs.dtype != real_dtype:
            freqs = freqs.astype(real_dtype)
        if fluxes.dtype != real_dtype:  # Fluxes can be 1D or 2D
            fluxes = fluxes.astype(real_dtype)

        # Get the redundant groups
        if baselines is None:
            reds = utils.get_pos_reds(ants, include_autos=True)
            baselines = [red[0] for red in reds]

        nbls = len(baselines)

        # Handle beam frequency interpolation on CPU before transferring to GPU
        if isinstance(beam, UVBeam):
            # Only try to interpolate the beam if it has more than one frequency
            if hasattr(beam, "Nfreqs") and beam.Nfreqs > 1:
                beam = beam.interp(
                    freq_array=freqs, new_object=True, run_check=False
                )  # pragma: no cover

        # Factor of 0.5 accounts for splitting Stokes between polarization channels
        # Apply this factor on CPU before transferring flux
        Isky = 0.5 * fluxes

        # Flatten antenna positions
        antkey_to_idx = dict(zip(ants.keys(), range(len(ants))))
        antvecs = np.array([ants[ant] for ant in ants], dtype=real_dtype)

        # Rotate the array to the xy-plane (on CPU)
        rotation_matrix_cpu = utils.get_plane_to_xy_rotation_matrix(antvecs)
        rotation_matrix_cpu = np.ascontiguousarray(
            rotation_matrix_cpu.astype(real_dtype).T
        )
        rotated_antvecs = np.dot(rotation_matrix_cpu, antvecs.T)
        rotated_ants = {ant: rotated_antvecs[:, antkey_to_idx[ant]] for ant in ants}

        # Compute baseline vectors (on CPU)
        bls_cpu = np.array(
            [rotated_ants[bl[1]] - rotated_ants[bl[0]] for bl in baselines]
        )[:, :].T.astype(real_dtype)

        # Check if the array is flat within tolerance (on CPU)
        is_coplanar = np.all(np.less_equal(np.abs(bls_cpu[2]), flat_array_tol))

        # Scale baselines by 1/c (on CPU)
        bls_cpu /= utils.speed_of_light

        # Get number of processes for multiprocessing (Ray)
        if nprocesses is None:
            nprocesses = min(
                cp.cuda.runtime.getDeviceCount(), 1
            )  # Use number of available GPUs if Ray is used

        # Check if the times array is a numpy array
        if isinstance(times, np.ndarray):
            times = Time(times, format="jd")

        # Instantiate CoordinateRotation (can be CPU or GPU compatible)
        coord_method_cls = CoordinateRotation._methods[coord_method]
        coord_method_params = coord_method_params or {}
        # Pass gpu=True to the coord_method constructor if it supports it
        coord_mgr = coord_method_cls(
            flux=Isky,  # Pass CPU flux here, coord_mgr handles transfer if needed
            times=times,
            telescope_loc=telescope_loc,
            skycoords=SkyCoord(ra=ra * un.rad, dec=dec * un.rad, frame="icrs"),
            precision=precision,
            **coord_method_params,
        )

        # --- Calculate optimal memory allocation ---
        # Get total number of sources
        total_sources = len(ra)
        
        # Initialize memory manager
        # Use 0.8 safety factor since we have better memory estimates now
        mem_manager = GPUMemoryManager(safety_factor=0.8)
        
        # Check available memory first
        available_memory = mem_manager.get_available_memory()
        device_free = mem_manager.device.mem_info[0]
        device_total = mem_manager.device.mem_info[1]
        
        logger.info(
            f"GPU Memory: {device_free/1e9:.2f}GB free / {device_total/1e9:.2f}GB total "
            f"(using {available_memory/1e9:.2f}GB)"
        )
        
        # Estimate maximum feasible sources for this GPU
        # First try with freq_batch_size=1 (process frequencies sequentially)
        # This gives the absolute maximum number of sources possible
        max_feasible_sources_seq = mem_manager.estimate_max_sources(
            nfreqs=nfreqs,
            ntimes=ntimes,
            nbls=nbls,
            nfeeds=nfeeds,
            precision=precision,
            polarized=polarized,
            freq_batch_size=1  # Sequential processing
        )
        
        # Also estimate with a more typical batch size for reference
        max_feasible_sources_batch = mem_manager.estimate_max_sources(
            nfreqs=nfreqs,
            ntimes=ntimes,
            nbls=nbls,
            nfeeds=nfeeds,
            precision=precision,
            polarized=polarized,
            freq_batch_size=min(nfreqs, 8)  # Batched processing
        )
        
        logger.info(
            f"Maximum feasible sources for this GPU: "
            f"~{max_feasible_sources_seq:,} (sequential frequencies) "
            f"to ~{max_feasible_sources_batch:,} (batched frequencies)"
        )
        
        # Use the sequential estimate for the hard limit check
        # optimize_chunking will find the best configuration
        # Be more conservative for large datasets to prevent kernel crashes
        if total_sources > max_feasible_sources_seq * 1.2:  # 20% margin
            logger.error(
                f"Dataset with {total_sources:,} sources exceeds GPU capacity "
                f"(max feasible: ~{max_feasible_sources_seq:,} sources). "
                f"Please use CPU backend."
            )
            raise ValueError(
                f"Dataset too large for GPU memory. Use backend='cpu' for {total_sources:,} sources."
            )
        
        # Additional safety check for borderline cases
        # CUDA errors during plan creation can crash the kernel
        # Be extra conservative when memory is tight to prevent crashes
        safety_factor = 0.6  # Use only 60% of estimated capacity when memory is low
        
        if device_free < 1.0e9:  # Less than 1GB free
            # When memory is very limited, be extra conservative
            safe_max_sources = int(max_feasible_sources_seq * safety_factor)
            
            if total_sources > safe_max_sources:
                logger.warning(
                    f"Low GPU memory ({device_free/1e9:.2f}GB free) with large dataset ({total_sources:,} sources). "
                    f"Safe limit estimated at ~{safe_max_sources:,} sources to prevent kernel crashes."
                )
                raise ValueError(
                    f"Dataset with {total_sources:,} sources exceeds safe limit (~{safe_max_sources:,}) "
                    f"with only {device_free/1e9:.2f}GB free GPU memory. "
                    f"Use backend='cpu' to avoid kernel crashes."
                )
        
        # Optimize source chunk size and frequency batch size together
        optimal_source_chunk, optimal_freq_batch = mem_manager.optimize_chunking(
            nsources_total=total_sources,
            nfreqs_total=nfreqs,
            ntimes_total=ntimes,
            nbls=nbls,
            nfeeds=nfeeds,
            precision=precision,
            polarized=polarized,
            min_chunk_size=1000,
            max_freq_batch=nfreqs  # Allow processing all frequencies if memory permits
        )
        
        # Update coord_mgr chunk size if needed
        if hasattr(coord_mgr, 'chunk_size'):
            if coord_mgr.chunk_size is None or coord_mgr.chunk_size > optimal_source_chunk:
                coord_mgr.chunk_size = optimal_source_chunk
                logger.info(f"Updated coordinate manager chunk size to {optimal_source_chunk:,}")
        
        # Use the optimized frequency batch size
        optimal_batch_size = optimal_freq_batch
        
        # --- Data Transfer to GPU (if not handled by coord_mgr) ---
        # Transfer data needed by _evaluate_vis_chunk to GPU *before* chunking/Ray
        # This avoids transferring the same data multiple times if Ray is used
        rotation_matrix_gpu = cp.asarray(rotation_matrix_cpu)
        bls_gpu = cp.asarray(bls_cpu)
        freqs_gpu = cp.asarray(freqs)
        # Beam data transfer is handled by GPUBeamEvaluator internally

        # --- Chunking and Parallelization (Ray) ---
        # Determine chunking based on available resources
        nchunks, nf, nt = utils.get_task_chunks(nprocesses, nfreqs, ntimes)[
            :3
        ]  # Only need nchunks, nf, nt

        # Re-calculate freq_chunks and time_chunks based on the actual nchunks
        # This is done inside utils.get_task_chunks, let's call it again with the determined nchunks
        nprocesses, freq_chunks, time_chunks, nf, nt = utils.get_task_chunks(
            nchunks, nfreqs, ntimes
        )

        use_ray = nprocesses > 1 or force_use_ray

        if use_ray:  # pragma: no cover
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                # Ray initialization with GPU support
                try:
                    # Attempt to connect to an existing Ray instance first
                    ray.init(address="auto", ignore_reinit_error=True)
                    logger.info("Connected to existing Ray cluster.")
                except ConnectionError:
                    # If no cluster is found, start a new one
                    logger.info(f"Starting new Ray instance with {nprocesses} GPUs.")
                    # Configure Ray to use GPUs
                    ray.init(
                        num_gpus=nprocesses,
                        include_dashboard=False,
                        ignore_reinit_error=True,
                    )

            # Put data into Ray's object store
            # Ray can handle transferring cupy arrays
            bls_gpu_ref = ray.put(bls_gpu)
            rotation_matrix_gpu_ref = ray.put(rotation_matrix_gpu)
            freqs_gpu_ref = ray.put(freqs_gpu)
            beam_ref = ray.put(
                beam
            )  # Ray needs to be able to serialize UVBeam/BeamInterface
            coord_mgr_ref = ray.put(
                coord_mgr
            )  # Ray needs to be able to serialize CoordinateRotation

            # Define the remote function to use
            fnc = _evaluate_vis_chunk_remote_gpu.remote
        else:
            # If not using Ray, call the method directly on the engine instance
            fnc = self._evaluate_vis_chunk
            # Ensure data is on GPU for the direct call
            bls_gpu_ref = bls_gpu
            rotation_matrix_gpu_ref = rotation_matrix_gpu
            freqs_gpu_ref = freqs_gpu
            beam_ref = beam
            coord_mgr_ref = coord_mgr

        # --- Launch Tasks / Run Directly ---
        futures = []
        init_time = time.time()

        # nthreads is ignored for GPU NUFFT, but keep the loop structure
        nthreads_per_proc = [
            1
        ] * nprocesses  # Each GPU process typically uses 1 thread for NUFFT

        for nthi, fc, tc in zip(nthreads_per_proc, freq_chunks, time_chunks):
            futures.append(
                fnc(
                    time_idx=tc,
                    freq_idx=fc,
                    beam=beam_ref,  # Pass Ray object ref or direct object
                    coord_mgr=coord_mgr_ref,  # Pass Ray object ref or direct object
                    rotation_matrix=rotation_matrix_gpu_ref,  # Pass Ray object ref or direct object
                    bls=bls_gpu_ref,  # Pass Ray object ref or direct object
                    freqs=freqs_gpu_ref,  # Pass Ray object ref or direct object
                    complex_dtype=complex_dtype,
                    nfeeds=nfeeds,
                    polarized=polarized,
                    eps=eps,
                    beam_spline_opts=beam_spline_opts,
                    interpolation_function=interpolation_function,
                    n_threads=nthi,  # Ignored by GPU NUFFT
                    is_coplanar=is_coplanar,
                    trace_mem=trace_mem,
                    freq_batch_size=optimal_batch_size,  # Dynamic batch size for frequency processing
                )
            )

        # --- Retrieve Results ---
        if use_ray:  # pragma: no cover
            # Ray returns futures, get the results
            results = ray.get(futures)
        else:
            # Direct call returns the result directly
            results = futures  # futures is already the list of results

        end_time = time.time()
        logger.info(f"Main loop evaluation time: {end_time - init_time}")

        # --- Combine Results ---
        # Results are cupy arrays from each chunk
        # Combine them into a single cupy array on the GPU
        vis_gpu = cp.zeros(
            dtype=complex_dtype, shape=(ntimes, nbls, nfeeds, nfeeds, nfreqs)
        )
        for fc, tc, result_chunk_gpu in zip(freq_chunks, time_chunks, results):
            # Copy data from the chunk result (on GPU) to the main vis_gpu array (on GPU)
            vis_gpu[tc][..., fc] = result_chunk_gpu

        # --- Transfer Final Result to CPU ---
        vis_cpu = cp.asnumpy(vis_gpu)  # Transfer the final result back to CPU

        # Reshape to expected output format (on CPU)
        return (
            np.transpose(vis_cpu, (4, 0, 2, 3, 1))
            if polarized
            else np.moveaxis(vis_cpu[..., 0, 0, :], 2, 0)
        )

    def _evaluate_vis_chunk(
        self,
        time_idx: slice,
        freq_idx: slice,
        beam,  # UVBeam or BeamInterface (on CPU, needs data transfer)
        coord_mgr: CoordinateRotation,  # CoordinateRotation instance
        rotation_matrix: cp.ndarray,  # Expect cupy array
        bls: cp.ndarray,  # Expect cupy array
        freqs: cp.ndarray,  # Expect cupy array
        complex_dtype: np.dtype,
        nfeeds: int,
        polarized: bool = False,
        eps: float = None,
        beam_spline_opts: dict = None,
        interpolation_function: str = "az_za_map_coordinates",
        n_threads: int = 1,  # Ignored for GPU NUFFT
        is_coplanar: bool = False,
        trace_mem: bool = False,
        freq_batch_size: int = 16,  # New parameter for frequency batching
    ) -> cp.ndarray:  # Return cupy array
        """
        Evaluate a chunk of visibility data using GPU with frequency batching.

        This implementation processes multiple frequencies simultaneously to
        reduce kernel launch overhead and improve GPU utilization.
        """
        pid = os.getpid()
        pr = psutil.Process(pid)

        # Ensure rotation_matrix, bls, and freqs are on GPU
        if not isinstance(rotation_matrix, cp.ndarray):
            rotation_matrix = cp.asarray(rotation_matrix)
        if not isinstance(bls, cp.ndarray):
            bls = cp.asarray(bls)
        if not isinstance(freqs, cp.ndarray):
            freqs = cp.asarray(freqs)

        nbls = bls.shape[1]
        ntimes_total = len(coord_mgr.times)
        nfreqs_total = len(freqs)

        # Handle slice(None) case
        time_start = time_idx.start if time_idx.start is not None else 0
        time_stop = time_idx.stop if time_idx.stop is not None else ntimes_total
        freq_start = freq_idx.start if freq_idx.start is not None else 0
        freq_stop = freq_idx.stop if freq_idx.stop is not None else nfreqs_total
        
        nt_here = time_stop - time_start
        nf_here = freq_stop - freq_start

        # Allocate output visibility array for this chunk on the GPU
        vis_chunk_gpu = cp.zeros(
            dtype=complex_dtype, shape=(nt_here, nbls, nfeeds, nfeeds, nf_here)
        )

        # Use the global GPU beam evaluator instance
        beam_evaluator = _gpu_beam_evaluator

        # Initialize the coordinate manager
        coord_mgr.setup()

        # Get frequency indices for this chunk
        freq_indices = list(range(nfreqs_total)[freq_idx])
        
        # Loop over time samples in this chunk
        for time_chunk_idx, time_total_idx in enumerate(range(ntimes_total)[time_idx]):
            # Rotate coordinates for the current time
            coord_mgr.rotate(time_total_idx)

            # Process frequencies in batches
            for batch_start in range(0, len(freq_indices), freq_batch_size):
                batch_end = min(batch_start + freq_batch_size, len(freq_indices))
                freq_batch_indices = freq_indices[batch_start:batch_end]
                batch_size = len(freq_batch_indices)
                
                # Get frequencies for this batch
                freq_batch = freqs[freq_batch_indices[0] - freq_start:freq_batch_indices[-1] - freq_start + 1]
                
                # Allocate batch arrays for visibilities
                vis_batch_gpu = cp.zeros(
                    dtype=complex_dtype, shape=(batch_size, nbls, nfeeds, nfeeds)
                )
                
                # Loop over source chunks
                n_source_chunks = coord_mgr.nsrc // coord_mgr.chunk_size + (
                    coord_mgr.nsrc % coord_mgr.chunk_size > 0
                )

                # Initialize batch tracking variables before the source loop
                # to avoid UnboundLocalError if all sources are below horizon
                current_batch_size = batch_size

                for source_chunk_idx in range(n_source_chunks):
                    # Select a chunk of sources above the horizon
                    topo, flux_sqrt, nsim_sources = coord_mgr.select_chunk(source_chunk_idx, time_total_idx)

                    # Ensure topo and flux_sqrt are on GPU
                    # Convert to GPU arrays with error handling
                    try:
                        if not isinstance(topo, cp.ndarray):
                            topo = cp.asarray(topo)
                        if not isinstance(flux_sqrt, cp.ndarray):
                            flux_sqrt = cp.asarray(flux_sqrt)
                    except cp.cuda.memory.OutOfMemoryError as e:
                        logger.error(f"GPU out of memory while copying arrays: {e}")
                        raise MemoryError(
                            f"GPU out of memory. Dataset too large for available GPU memory.\n"
                            f"Consider using backend='cpu' or reducing the problem size."
                        )

                    if nsim_sources == 0:
                        continue

                    # Truncate arrays to actual number of sources above horizon
                    topo = topo[:, :nsim_sources]
                    flux_sqrt = flux_sqrt[:nsim_sources]

                    # Compute azimuth and zenith angles (on GPU)
                    az, za = coordinates.enu_to_az_za(
                        enu_e=topo[0], enu_n=topo[1], orientation="uvbeam"
                    )

                    # Ensure az and za are on GPU
                    if not isinstance(az, cp.ndarray):
                        az = cp.asarray(az)
                    if not isinstance(za, cp.ndarray):
                        za = cp.asarray(za)

                    # Rotate source coordinates with rotation matrix (on GPU)
                    gpu_utils.inplace_rot(rotation_matrix, topo)

                    # Scale topo by 2*pi (on GPU)
                    topo *= 2 * np.pi

                    # Prepare batch arrays for UVW coordinates and weights
                    uvw_batch = cp.empty((batch_size, 3, nbls), dtype=cp.float64)
                    weights_batch = cp.empty((batch_size, nfeeds**2, nsim_sources), dtype=complex_dtype)
                    
                    # Process each frequency in the batch
                    for batch_idx, freq_idx in enumerate(freq_batch_indices):
                        freq_chunk_idx = freq_idx - freq_start
                        freq = freqs[freq_chunk_idx]
                        
                        # Compute uvw for this frequency
                        uvw_batch[batch_idx] = bls * freq
                        
                        # Get frequency value for beam evaluation
                        freq_value = freq.get().item() if isinstance(freq, cp.ndarray) else freq
                        
                        # Evaluate beam
                        A_s = beam_evaluator.evaluate_beam(
                            beam,
                            az,
                            za,
                            polarized,
                            freq_value,
                            spline_opts=beam_spline_opts,
                            interpolation_function=interpolation_function,
                        )
                        
                        # Apply flux
                        if polarized:
                            if flux_sqrt.ndim > 1:
                                flux_slice = flux_sqrt[:nsim_sources, freq_idx]
                            else:
                                flux_slice = flux_sqrt[:nsim_sources]
                            beam_evaluator.get_apparent_flux_polarized(A_s, flux_slice)
                        else:
                            if flux_sqrt.ndim > 1:
                                flux_slice = flux_sqrt[:nsim_sources, freq_idx]
                            else:
                                flux_slice = flux_sqrt[:nsim_sources]
                            A_s *= flux_slice
                        
                        # Reshape A_s
                        try:
                            A_s_reshaped = A_s.reshape(nfeeds**2, nsim_sources)
                        except ValueError:
                            logger.warning(f"Cannot reshape A_s with shape {A_s.shape} to {(nfeeds**2, nsim_sources)}")
                            continue
                        
                        # Ensure correct dtype
                        if A_s_reshaped.dtype != complex_dtype:
                            A_s_reshaped = A_s_reshaped.astype(complex_dtype)
                        
                        weights_batch[batch_idx] = A_s_reshaped
                    
                    # Compute visibilities for the batch using batch NUFFT
                    # Use helper function with automatic retry on memory errors
                    current_batch_size = batch_size
                    vis_batch_here = None

                    # Define the NUFFT call as a lambda for the retry helper
                    if is_coplanar:
                        def nufft_call():
                            return gpu_nufft2d_batch(
                                topo[0],  # x
                                topo[1],  # y
                                weights_batch[:current_batch_size],
                                uvw_batch[:current_batch_size, 0, :],  # u
                                uvw_batch[:current_batch_size, 1, :],  # v
                                eps=eps,
                                n_threads=n_threads,
                            )
                        context = "2D NUFFT batch"
                    else:
                        def nufft_call():
                            return gpu_nufft3d_batch(
                                topo[0],  # x
                                topo[1],  # y
                                topo[2],  # z
                                weights_batch[:current_batch_size],
                                uvw_batch[:current_batch_size, 0, :],  # u
                                uvw_batch[:current_batch_size, 1, :],  # v
                                uvw_batch[:current_batch_size, 2, :],  # w
                                eps=eps,
                                n_threads=n_threads,
                            )
                        context = "3D NUFFT batch"

                    # Execute with retry logic
                    vis_batch_here, current_batch_size = gpu_utils.execute_with_retry(
                        nufft_func=nufft_call,
                        func_args=(),
                        func_kwargs={},
                        initial_batch_size=batch_size,
                        max_retries=3,
                        logger_context=context
                    )

                    # Update batch indices if batch size was reduced during retry
                    if current_batch_size < batch_size:
                        freq_batch_indices = freq_batch_indices[:current_batch_size]
                        batch_size = current_batch_size
                    
                    # vis_batch_here shape: (current_batch_size, nfeeds**2, nbls)
                    # Reshape and transpose to (current_batch_size, nbls, nfeeds, nfeeds)
                    if vis_batch_here is not None:
                        for batch_idx in range(current_batch_size):
                            vis_single = vis_batch_here[batch_idx].reshape(nfeeds, nfeeds, nbls)
                            vis_single = cp.swapaxes(vis_single, 2, 0)  # (nbls, nfeeds, nfeeds)
                            vis_batch_gpu[batch_idx] += vis_single
                
                # Store the batch results in the main output array
                for batch_idx in range(current_batch_size):
                    freq_idx = freq_batch_indices[batch_idx]
                    freq_chunk_idx = freq_idx - freq_start
                    vis_chunk_gpu[time_chunk_idx, :, :, :, freq_chunk_idx] = vis_batch_gpu[batch_idx]
                
                # Periodic memory cleanup for large datasets
                if coord_mgr.nsrc > 200000 and batch_start % 10 == 0:
                    try:
                        mempool = cp.get_default_memory_pool()
                        mempool.free_all_blocks()
                    except:
                        pass

        # Return the chunk of visibility data (on GPU)
        return vis_chunk_gpu
