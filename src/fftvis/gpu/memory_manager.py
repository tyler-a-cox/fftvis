"""
GPU Memory Manager for fftvis.

This module provides dynamic memory allocation strategies for GPU simulations,
optimizing both source chunking and frequency batching based on actual memory requirements.
"""

import numpy as np
import cupy as cp
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """Manages GPU memory allocation for fftvis simulations."""
    
    def __init__(self, safety_factor: float = 0.7):
        """
        Initialize the GPU memory manager.
        
        Parameters
        ----------
        safety_factor : float
            Fraction of available GPU memory to use (0-1).
            Default is 0.7 to leave room for other operations.
        """
        self.safety_factor = safety_factor
        self.device = cp.cuda.Device()
        
    def get_available_memory(self) -> int:
        """Get available GPU memory in bytes."""
        free_mem, total_mem = self.device.mem_info
        return int(free_mem * self.safety_factor)
    
    def calculate_memory_requirements(
        self,
        nsources: int,
        nfreqs: int,
        ntimes: int,
        nbls: int,
        nfeeds: int,
        precision: int = 2,
        polarized: bool = False
    ) -> Dict[str, int]:
        """
        Calculate memory requirements for different components.
        
        Parameters
        ----------
        nsources : int
            Number of sources
        nfreqs : int
            Number of frequencies
        ntimes : int
            Number of time samples
        nbls : int
            Number of baselines
        nfeeds : int
            Number of feeds (1 or 2)
        precision : int
            1 for single precision, 2 for double precision
        polarized : bool
            Whether using polarized calculations
            
        Returns
        -------
        dict
            Memory requirements in bytes for each component
        """
        # Data type sizes
        real_size = 4 if precision == 1 else 8
        complex_size = 8 if precision == 1 else 16
        
        # Per-source memory (constant across frequencies/times)
        source_positions = 3 * nsources * real_size  # ra, dec, flux positions
        
        # Per-source-frequency memory
        if polarized:
            source_flux_per_freq = nsources * 4 * complex_size  # 2x2 coherency matrix
        else:
            source_flux_per_freq = nsources * real_size  # scalar flux
            
        # Per-time memory
        rotation_matrix = 9 * real_size  # 3x3 matrix
        coord_rotation_overhead = nsources * 3 * real_size  # temporary arrays
        
        # Per-frequency memory for NUFFT
        uvw_per_freq = 3 * nbls * real_size  # u, v, w coordinates
        weights_per_freq = nfeeds**2 * nsources * complex_size  # beam-weighted fluxes
        vis_output_per_freq = nfeeds**2 * nbls * complex_size  # visibility output
        
        # NUFFT internal buffers (estimated based on cufinufft behavior)
        # Type 3 NUFFT needs buffers for both non-uniform and uniform grids
        # Note: source positions are already counted in source_positions above
        # Note: baseline uvw coordinates are already counted in uvw_per_freq
        # Updated based on actual cufinufft memory usage patterns
        nufft_overhead_per_freq = (
            # Internal buffers for NUFFT algorithm
            # cufinufft needs temporary arrays for the transform
            # For Type 3, this is roughly the size of input + output
            (nsources + nbls * nfeeds**2) * complex_size +
            # Small workspace for algorithm internals (sorting, etc.)
            # This is typically much smaller than the data arrays
            min(nsources, nbls) * complex_size // 2
        )
        
        return {
            'source_positions': source_positions,
            'source_flux_per_freq': source_flux_per_freq,
            'rotation_per_time': rotation_matrix + coord_rotation_overhead,
            'uvw_per_freq': uvw_per_freq,
            'weights_per_freq': weights_per_freq,
            'vis_output_per_freq': vis_output_per_freq,
            'nufft_overhead_per_freq': nufft_overhead_per_freq,
            'total_per_freq': (
                source_flux_per_freq + uvw_per_freq + weights_per_freq + 
                vis_output_per_freq + nufft_overhead_per_freq
            ),
            'total_per_time': rotation_matrix + coord_rotation_overhead,
        }
    
    def optimize_chunking(
        self,
        nsources_total: int,
        nfreqs_total: int,
        ntimes_total: int,
        nbls: int,
        nfeeds: int,
        precision: int = 2,
        polarized: bool = False,
        min_chunk_size: int = 1000,
        max_freq_batch: int = 1024  # Reasonable upper limit for memory allocation
    ) -> Tuple[int, int]:
        """
        Optimize source chunk size and frequency batch size based on available memory.
        
        Parameters
        ----------
        nsources_total : int
            Total number of sources
        nfreqs_total : int
            Total number of frequencies
        ntimes_total : int
            Total number of time samples
        nbls : int
            Number of baselines
        nfeeds : int
            Number of feeds
        precision : int
            Precision level (1 or 2)
        polarized : bool
            Whether using polarized calculations
        min_chunk_size : int
            Minimum number of sources per chunk
        max_freq_batch : int
            Maximum number of frequencies to process at once
            
        Returns
        -------
        source_chunk_size : int
            Optimal number of sources per chunk
        freq_batch_size : int
            Optimal number of frequencies per batch
        """
        available_memory = self.get_available_memory()
        
        # Start with full dataset and reduce until it fits
        source_chunk_size = nsources_total
        freq_batch_size = min(nfreqs_total, max_freq_batch)
        
        # Binary search for optimal source chunk size
        low_sources = min_chunk_size
        high_sources = nsources_total
        
        while low_sources < high_sources:
            mid_sources = (low_sources + high_sources + 1) // 2
            
            # Try different frequency batch sizes
            # Start from max_freq_batch (or nfreqs_total) and work down in powers of 2
            freq_batch_candidates = []
            test_batch = min(max_freq_batch, nfreqs_total)
            while test_batch >= 1:
                freq_batch_candidates.append(test_batch)
                test_batch = test_batch // 2
            
            for test_freq_batch in freq_batch_candidates:
                    
                mem_req = self.calculate_memory_requirements(
                    mid_sources, test_freq_batch, 1,  # Check for 1 time slice
                    nbls, nfeeds, precision, polarized
                )
                
                # Total memory needed for this configuration
                total_memory = (
                    mem_req['source_positions'] +
                    mem_req['total_per_time'] +
                    mem_req['total_per_freq'] * test_freq_batch
                )
                
                if total_memory <= available_memory:
                    source_chunk_size = mid_sources
                    freq_batch_size = test_freq_batch
                    low_sources = mid_sources
                    break
            else:
                # Couldn't fit even with freq_batch=1, reduce sources
                high_sources = mid_sources - 1
        
        # Ensure we found a valid configuration
        if source_chunk_size < min_chunk_size:
            source_chunk_size = min_chunk_size
            freq_batch_size = 1
            
        # Log the decision
        mem_req = self.calculate_memory_requirements(
            source_chunk_size, freq_batch_size, 1,
            nbls, nfeeds, precision, polarized
        )
        total_memory = (
            mem_req['source_positions'] +
            mem_req['total_per_time'] +
            mem_req['total_per_freq'] * freq_batch_size
        )
        
        logger.info(
            f"GPU Memory optimization: {available_memory/1e9:.2f}GB available\n"
            f"  Source chunk: {source_chunk_size:,}/{nsources_total:,} sources\n"
            f"  Frequency batch: {freq_batch_size}/{nfreqs_total} frequencies\n"
            f"  Estimated usage: {total_memory/1e9:.2f}GB ({100*total_memory/available_memory:.1f}%)"
        )
        
        return source_chunk_size, freq_batch_size
    
    def estimate_max_sources(
        self,
        nfreqs: int,
        ntimes: int,
        nbls: int,
        nfeeds: int,
        precision: int = 2,
        polarized: bool = False,
        freq_batch_size: int = None
    ) -> int:
        """
        Estimate maximum number of sources that can fit in GPU memory.
        
        Parameters
        ----------
        nfreqs : int
            Number of frequencies
        ntimes : int
            Number of time samples
        nbls : int
            Number of baselines  
        nfeeds : int
            Number of feeds
        precision : int
            Precision level
        polarized : bool
            Whether using polarized calculations
        freq_batch_size : int, optional
            If provided, use this batch size. Otherwise, determine automatically.
            
        Returns
        -------
        int
            Maximum number of sources that can fit
        """
        available_memory = self.get_available_memory()
        
        if freq_batch_size is None:
            freq_batch_size = min(nfreqs, 8)  # Default conservative batch
            
        # Binary search for maximum sources
        low = 1000
        high = 10_000_000  # 10 million sources as upper bound
        
        while low < high:
            mid = (low + high + 1) // 2
            
            mem_req = self.calculate_memory_requirements(
                mid, freq_batch_size, 1,
                nbls, nfeeds, precision, polarized
            )
            
            total_memory = (
                mem_req['source_positions'] +
                mem_req['total_per_time'] +
                mem_req['total_per_freq'] * freq_batch_size
            )
            
            if total_memory <= available_memory:
                low = mid
            else:
                high = mid - 1
                
        return low