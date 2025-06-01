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

# Import the GPU beam evaluator and NUFFT
from .beams import GPUBeamEvaluator
from .nufft import gpu_nufft2d, gpu_nufft3d

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
    ) -> cp.ndarray:  # Return cupy array
        """
        Evaluate a chunk of visibility data using GPU.

        See base class for parameter descriptions.
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
        ntimes_total = len(coord_mgr.times)  # Total number of times
        nfreqs_total = len(freqs)  # Total number of frequencies

        # Handle slice(None) case - when start/stop are None, use full range
        time_start = time_idx.start if time_idx.start is not None else 0
        time_stop = time_idx.stop if time_idx.stop is not None else ntimes_total
        freq_start = freq_idx.start if freq_idx.start is not None else 0
        freq_stop = freq_idx.stop if freq_idx.stop is not None else nfreqs_total
        
        nt_here = time_stop - time_start  # Number of times in this chunk
        nf_here = freq_stop - freq_start  # Number of frequencies in this chunk

        # Allocate output visibility array for this chunk on the GPU
        vis_chunk_gpu = cp.zeros(
            dtype=complex_dtype, shape=(nt_here, nbls, nfeeds, nfeeds, nf_here)
        )

        # Use the global GPU beam evaluator instance
        beam_evaluator = _gpu_beam_evaluator

        # Initialize the coordinate manager
        coord_mgr.setup()

        # Loop over time samples in this chunk
        for time_chunk_idx, time_total_idx in enumerate(range(ntimes_total)[time_idx]):
            # Rotate coordinates for the current time
            coord_mgr.rotate(
                time_total_idx
            )  # This updates coord_mgr's internal coordinates

            # Loop over frequency indices in this chunk
            for freq_chunk_idx, freq_total_idx in enumerate(
                range(nfreqs_total)[freq_idx]
            ):
                # Allocate temporary array for visibilities from source chunks for this time/freq chunk
                vis_time_freq_chunk_gpu = cp.zeros(
                    dtype=complex_dtype, shape=(nbls, nfeeds, nfeeds)
                )
                
                # Loop over source chunks (within the time and freq loops)
                n_source_chunks = coord_mgr.nsrc // coord_mgr.chunk_size + (
                    coord_mgr.nsrc % coord_mgr.chunk_size > 0
                )

                for source_chunk_idx in range(n_source_chunks):
                    # Select a chunk of sources above the horizon
                    topo, flux_sqrt, nsim_sources = coord_mgr.select_chunk(source_chunk_idx)

                    # Ensure topo and flux_sqrt are on GPU
                    if not isinstance(topo, cp.ndarray):
                        topo = cp.asarray(topo)
                    if not isinstance(flux_sqrt, cp.ndarray):
                        flux_sqrt = cp.asarray(flux_sqrt)

                    if nsim_sources == 0:
                        continue  # Skip if no sources in this chunk are above horizon

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
                    utils.inplace_rot(rotation_matrix, topo)

                    # Scale topo by 2*pi (on GPU)
                    topo *= 2 * np.pi

                    # Get frequency value
                    freq = freqs[freq_chunk_idx]  # Get frequency value (on GPU)

                    # Compute uvw coordinates (on GPU)
                    uvw = (
                        bls * freq
                    )  # bls is (3, nbls), freq is scalar -> uvw is (3, nbls)

                    # Evaluate beam (on GPU)
                    # We need to get the actual scalar value for freq to pass to evaluate_beam
                    freq_value = (
                        freq.get().item() if isinstance(freq, cp.ndarray) else freq
                    )
                    A_s = beam_evaluator.evaluate_beam(
                        beam,
                        az,
                        za,
                        polarized,
                        freq_value,
                        spline_opts=beam_spline_opts,
                        interpolation_function=interpolation_function,
                    )  # A_s shape: (nax, nfeed, nsim_sources) or (nsim_sources,)

                    # Apply flux (on GPU)
                    if polarized:
                        # For polarized case, get the correct flux slice
                        if flux_sqrt.ndim > 1:
                            flux_slice = flux_sqrt[:nsim_sources, freq_total_idx]
                        else:
                            flux_slice = flux_sqrt[:nsim_sources]

                        # beam_evaluator.get_apparent_flux_polarized modifies A_s in-place
                        beam_evaluator.get_apparent_flux_polarized(A_s, flux_slice)
                        # A_s is now the apparent flux, shape (nax, nfeed, nsim_sources)
                    else:
                        # For unpolarized case, handle flux correctly
                        if flux_sqrt.ndim > 1:
                            # flux_sqrt is (nsources, nfreqs), get the right frequency slice
                            flux_slice = flux_sqrt[:nsim_sources, freq_total_idx]
                        else:
                            # flux_sqrt is (nsources,), use directly
                            flux_slice = flux_sqrt[:nsim_sources]

                        # Multiply by flux_sqrt
                        A_s *= flux_slice
                        # A_s is now the apparent flux, shape (nsim_sources,)

                    # Check if A_s can be reshaped to expected dimensions (like CPU implementation)
                    expected_size = nfeeds**2 * nsim_sources
                    if A_s.size != expected_size:
                        # Log the shape mismatch and try to adapt
                        print(f"Warning: Shape mismatch: A_s size {A_s.size} != expected size {expected_size}")
                        print(f"A_s shape: {A_s.shape}, nfeeds: {nfeeds}, nsim_sources: {nsim_sources}")

                        # Handle polarized case specially - if we got a 2D array but expected 3D
                        if polarized and A_s.ndim == 2:
                            # Skip this time/freq
                            continue

                    # Try to reshape safely (like CPU implementation)
                    try:
                        A_s = A_s.reshape(nfeeds**2, nsim_sources)
                    except ValueError:
                        print(f"Error: Cannot reshape A_s with shape {A_s.shape} to {(nfeeds**2, nsim_sources)}")
                        continue

                    i_sky = A_s

                    # Ensure weights have correct complex dtype
                    if i_sky.dtype != complex_dtype:
                        i_sky = i_sky.astype(complex_dtype)

                    # Compute visibilities w/ non-uniform FFT (on GPU)
                    if is_coplanar:
                        _vis_here = gpu_nufft2d(
                            topo[0],  # x
                            topo[1],  # y
                            i_sky,  # weights
                            uvw[0],  # u
                            uvw[1],  # v
                            eps=eps,
                            n_threads=n_threads,  # Ignored by GPU NUFFT
                        )
                    else:
                        _vis_here = gpu_nufft3d(
                            topo[0],  # x
                            topo[1],  # y
                            topo[2],  # z
                            i_sky,  # weights
                            uvw[0],  # u
                            uvw[1],  # v
                            uvw[2],  # w
                            eps=eps,
                            n_threads=n_threads,  # Ignored by GPU NUFFT
                        )

                    # Reshape and transpose _vis_here to match expected shape (like CPU implementation)
                    # _vis_here should have shape (nfeeds**2, nbls) -> reshape to (nfeeds, nfeeds, nbls) -> transpose to (nbls, nfeeds, nfeeds)
                    _vis_here = _vis_here.reshape(nfeeds, nfeeds, nbls)
                    _vis_here = cp.swapaxes(_vis_here, 2, 0)  # Shape (nbls, nfeeds, nfeeds)

                    # Add contribution from this source chunk to the total for this time/freq chunk
                    vis_time_freq_chunk_gpu += _vis_here

                # Store the summed visibility for this time/freq chunk in the main output array
                vis_chunk_gpu[time_chunk_idx, :, :, :, freq_chunk_idx] = vis_time_freq_chunk_gpu

        # Return the chunk of visibility data (on GPU)
        return vis_chunk_gpu
