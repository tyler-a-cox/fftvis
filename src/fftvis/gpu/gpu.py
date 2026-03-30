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
from pyuvdata.beam_interface import BeamInterface
from matvis import coordinates
from matvis.core.coords import CoordinateRotation

from ..core.simulate import SimulationEngine, default_accuracy_dict, prepare_simulation_inputs
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
    engine = GPUSimulationEngine()
    # Call the method on the instance
    return engine._evaluate_vis_chunk(
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
        # Common setup shared with CPU backend
        ctx = prepare_simulation_inputs(
            ants=ants, freqs=freqs, ra=ra, dec=dec, precision=precision,
            eps=eps, baselines=baselines, flat_array_tol=flat_array_tol,
            times=times,
        )
        nfreqs = ctx["nfreqs"]
        ntimes = ctx["ntimes"]
        real_dtype = ctx["real_dtype"]
        complex_dtype = ctx["complex_dtype"]
        eps = ctx["eps"]
        ra = ctx["ra"]
        dec = ctx["dec"]
        freqs = ctx["freqs"]
        baselines = ctx["baselines"]
        nbls = ctx["nbls"]
        rotation_matrix_cpu = ctx["rotation_matrix"]
        bls_cpu = ctx["bls"]
        is_coplanar = ctx["is_coplanar"]
        times = ctx["times"]

        nax = nfeeds = 2 if polarized else 1

        if fluxes.dtype != real_dtype:
            fluxes = fluxes.astype(real_dtype)

        # Handle beam frequency interpolation on CPU before transferring to GPU
        if isinstance(beam, BeamInterface) and beam._isuvbeam:
            if hasattr(beam.beam, "Nfreqs") and beam.beam.Nfreqs > 1:
                beam.beam = beam.beam.interp(
                    freq_array=freqs, new_object=True, run_check=False
                )
        elif isinstance(beam, UVBeam):
            if hasattr(beam, "Nfreqs") and beam.Nfreqs > 1:
                beam = beam.interp(
                    freq_array=freqs, new_object=True, run_check=False
                )
            beam = BeamInterface(beam)

        # Factor of 0.5 accounts for splitting Stokes between polarization channels
        Isky = 0.5 * fluxes

        # Get number of processes for multiprocessing (Ray)
        if nprocesses is None:
            nprocesses = min(
                cp.cuda.runtime.getDeviceCount(), 1
            )

        # --- Calculate optimal memory allocation ---
        total_sources = len(ra)
        mem_manager = GPUMemoryManager(safety_factor=0.8)

        device_free, device_total = mem_manager.device.mem_info
        logger.info(
            f"GPU Memory: {device_free/1e9:.2f}GB free / {device_total/1e9:.2f}GB total"
        )

        # Find optimal chunk sizes that fit in GPU memory
        optimal_source_chunk, optimal_freq_batch = mem_manager.optimize_chunking(
            nsources_total=total_sources,
            nfreqs_total=nfreqs,
            ntimes_total=ntimes,
            nbls=nbls,
            nfeeds=nfeeds,
            precision=precision,
            polarized=polarized,
            min_chunk_size=100,
            max_freq_batch=nfreqs,
        )

        logger.info(
            f"Processing {total_sources:,} sources in chunks of {optimal_source_chunk:,}, "
            f"frequency batch size: {optimal_freq_batch}"
        )

        optimal_batch_size = optimal_freq_batch

        # Instantiate CoordinateRotation with GPU support and correct chunk_size
        coord_method_cls = CoordinateRotation._methods[coord_method]
        coord_method_params = coord_method_params or {}
        coord_mgr = coord_method_cls(
            flux=Isky,
            times=times,
            telescope_loc=telescope_loc,
            skycoords=SkyCoord(ra=ra * un.rad, dec=dec * un.rad, frame="icrs"),
            precision=precision,
            chunk_size=optimal_source_chunk,
            gpu=True,
            **coord_method_params,
        )
        
        # Transfer simulation data to GPU once before task distribution.
        # TODO: Move these into components (coord_mgr or a dedicated GPU data
        # holder) so each component manages its own GPU state, following the
        # matvis pattern where components handle transfers in their setup().
        rotation_matrix_gpu = cp.asarray(rotation_matrix_cpu)
        bls_gpu = cp.asarray(bls_cpu)
        freqs_gpu = cp.asarray(freqs)

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

        if use_ray:
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

        # Each GPU process typically uses 1 thread for NUFFT
        nthreads_per_proc = [1] * nprocesses

        for nthi, fc, tc in zip(nthreads_per_proc, freq_chunks, time_chunks):
            # Arguments are Ray object refs when use_ray=True, direct objects otherwise
            futures.append(
                fnc(
                    time_idx=tc,
                    freq_idx=fc,
                    beam=beam_ref,
                    coord_mgr=coord_mgr_ref,
                    rotation_matrix=rotation_matrix_gpu_ref,
                    bls=bls_gpu_ref,
                    freqs=freqs_gpu_ref,
                    complex_dtype=complex_dtype,
                    nfeeds=nfeeds,
                    polarized=polarized,
                    eps=eps,
                    beam_spline_opts=beam_spline_opts,
                    interpolation_function=interpolation_function,
                    n_threads=nthi,
                    is_coplanar=is_coplanar,
                    trace_mem=trace_mem,
                    freq_batch_size=optimal_batch_size,
                )
            )

        # --- Retrieve Results ---
        if use_ray:
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
        beam: BeamInterface,
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

                    if nsim_sources == 0:
                        continue

                    # Truncate arrays to actual number of sources above horizon
                    topo = topo[:, :nsim_sources]
                    flux_sqrt = flux_sqrt[:nsim_sources]

                    # Compute azimuth and zenith angles (on GPU)
                    az, za = coordinates.enu_to_az_za(
                        enu_e=topo[0], enu_n=topo[1], orientation="uvbeam"
                    )

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
                        
                        # Compute flux slice
                        if flux_sqrt.ndim > 1:
                            flux_slice = flux_sqrt[:nsim_sources, freq_idx]
                        else:
                            flux_slice = flux_sqrt[:nsim_sources]

                        # Apply flux
                        if polarized:
                            beam_evaluator.get_apparent_flux_polarized(A_s, flux_slice)
                        else:
                            A_s *= flux_slice
                        
                        # Reshape A_s
                        A_s_reshaped = A_s.reshape(nfeeds**2, nsim_sources)
                        
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

        return vis_chunk_gpu
