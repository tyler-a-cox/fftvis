"""
GPU-specific beam evaluation implementation for fftvis.

This module provides GPU-specific beam evaluation functionality.
"""

import cupy as cp
import numpy as np
from pyuvdata.beam_interface import BeamInterface
from typing import Dict, Optional
try:
    from cupyx.scipy.special import j1 as gpu_j1
    HAS_GPU_BESSEL = True
except ImportError:
    HAS_GPU_BESSEL = False

try:
    from cupyx.scipy.ndimage import map_coordinates as gpu_map_coordinates
    HAS_GPU_MAP_COORDS = True
except ImportError:
    HAS_GPU_MAP_COORDS = False

from ..core.beams import BeamEvaluator


class GPUBeamEvaluator(BeamEvaluator):
    """GPU implementation of the beam evaluator."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._gpu_beam_data = {}  # Cache beam data on GPU
        self.nsrc = 0

    def evaluate_beam(
        self,
        beam: BeamInterface,
        az: cp.ndarray,
        za: cp.ndarray,
        polarized: bool,
        freq: float,
        check: bool = False,
        spline_opts: Optional[Dict] = None,
        interpolation_function: str = "az_za_map_coordinates",
    ) -> cp.ndarray:
        """
        Evaluate the beam pattern at the given coordinates using GPU.

        Parameters
        ----------
        beam : BeamInterface
            Beam object to evaluate.
        az : cp.ndarray
            Azimuth coordinates in radians (on GPU).
        za : cp.ndarray
            Zenith angle coordinates in radians (on GPU).
        freq : float
            Frequency to evaluate the beam at in Hz.
        polarized : bool
            Whether to evaluate the polarized beam.
        check : bool, optional
            Whether to check for invalid beam values.
        spline_opts : dict, optional
            Options for spline interpolation (passed to map_coordinates).
        interpolation_function : str, optional
            The interpolation function to use ('az_za_map_coordinates' is expected).

        Returns
        -------
        cp.ndarray
            Beam values at the given coordinates (on GPU).

        Raises
        ------
        ValueError
            If interpolation_function is not 'az_za_map_coordinates'.
            If beam interpolation results in invalid values (and check=True).
        """
        # Save these for matvis compatibility and internal use
        self.polarized = polarized
        self.freq = freq
        self.spline_opts = spline_opts or {}
        self.nsrc = len(az)

        if interpolation_function != "az_za_map_coordinates":
            raise ValueError(
                "GPU beam evaluation only supports 'az_za_map_coordinates'"
            )

        # Ensure inputs are cupy arrays
        if not isinstance(az, cp.ndarray):
            az = cp.asarray(az)
        if not isinstance(za, cp.ndarray):
            za = cp.asarray(za)

        # Check if it's an AnalyticBeam that we can handle on GPU
        beam_obj = None
        is_beam_interface = hasattr(beam, 'beam')
        if is_beam_interface:  # BeamInterface wrapper
            beam_obj = beam.beam
        else:  # Direct beam object
            beam_obj = beam
            
        # Handle AiryBeam directly on GPU if supported
        if (beam_obj is not None and 
            beam_obj.__class__.__name__ == 'AiryBeam' and 
            HAS_GPU_BESSEL):
            return self._evaluate_airy_beam_gpu(
                beam_obj.diameter, az, za, polarized, freq, check
            )

        # Check if we can do GPU interpolation for UVBeam
        # Make sure it's actually a UVBeam (not an AnalyticBeam)
        # Check using attributes rather than class name to handle BeamInterface wrapper
        if (HAS_GPU_MAP_COORDS and 
            interpolation_function == "az_za_map_coordinates" and
            beam_obj is not None and 
            hasattr(beam_obj, 'data_array') and  # UVBeam has data_array
            hasattr(beam_obj, 'pixel_coordinate_system') and  # UVBeam has this
            not hasattr(beam_obj, 'diameter')):  # AnalyticBeams like AiryBeam have diameter
            return self._evaluate_uvbeam_gpu(
                beam, az, za, polarized, freq, check, spline_opts
            )

        # Fall back to old implementation (CPU evaluation + transfer)
        # Get beam data and transfer to GPU if not already there
        beam_key = id(beam)  # Use object ID as key
        if beam_key not in self._gpu_beam_data:
            # Get beam data from the beam object
            # Convert from CPU to GPU
            # Use the same method as CPU implementation for consistency
            kw = {
                "reuse_spline": True,
                "check_azza_domain": False,
                "spline_opts": spline_opts,
                "interpolation_function": interpolation_function,
            }

            beam_data_cpu = beam.compute_response(
                az_array=np.asarray(cp.asnumpy(az)),
                za_array=np.asarray(cp.asnumpy(za)),
                freq_array=np.atleast_1d(freq),
                **kw,
            )

            if polarized:
                beam_data_cpu = beam_data_cpu[:, :, 0, :]
            else:
                # Here we have already asserted that the beam is a power beam and
                # has only one polarization, so we just evaluate that one.
                beam_data_cpu = beam_data_cpu[0, 0, 0, :]

            beam_data_gpu = cp.asarray(beam_data_cpu)
            self._gpu_beam_data[beam_key] = beam_data_gpu

        # Get cached beam data or use what was just created
        beam_data = self._gpu_beam_data.get(beam_key)

        # If the beam evaluation used different az/za coordinates, we need to re-evaluate
        if beam_data is None or beam_data.shape[-1] != self.nsrc:
            # Previous cached data won't work - re-evaluate
            kw = {
                "reuse_spline": True,
                "check_azza_domain": False,
                "spline_opts": spline_opts,
                "interpolation_function": interpolation_function,
            }

            beam_data_cpu = beam.compute_response(
                az_array=np.asarray(cp.asnumpy(az)),
                za_array=np.asarray(cp.asnumpy(za)),
                freq_array=np.atleast_1d(freq),
                **kw,
            )

            if polarized:
                beam_data_cpu = beam_data_cpu[:, :, 0, :]
            else:
                # Here we have already asserted that the beam is a power beam and
                # has only one polarization, so we just evaluate that one.
                beam_data_cpu = beam_data_cpu[0, 0, 0, :]

            beam_data = cp.asarray(beam_data_cpu)
            self._gpu_beam_data[beam_key] = beam_data

        # Check for invalid beam values if requested
        if check:
            sm = cp.sum(beam_data)
            if cp.isinf(sm) or cp.isnan(sm):
                raise ValueError("GPU Beam interpolation resulted in an invalid value")

        return beam_data

    def _evaluate_airy_beam_gpu(
        self, 
        diameter: float, 
        az: cp.ndarray, 
        za: cp.ndarray, 
        polarized: bool, 
        freq: float,
        check: bool = False
    ) -> cp.ndarray:
        """
        GPU-accelerated Airy beam evaluation.
        
        Implements the Airy disk pattern: 2*J₁(x)/x
        where x = (π * D * sin(θ)) / λ
        
        Parameters
        ----------
        diameter : float
            Dish diameter in meters
        az : cp.ndarray
            Azimuth angles in radians (on GPU)
        za : cp.ndarray  
            Zenith angles in radians (on GPU)
        polarized : bool
            Whether to return polarized beam
        freq : float
            Frequency in Hz
        check : bool
            Whether to check for invalid values
            
        Returns
        -------
        cp.ndarray
            Beam response values on GPU
        """
        c = 299792458.0  # speed of light in m/s
        
        # Calculate wave number: k = 2π*f/c
        kvals = (2.0 * cp.pi) * freq / c
        
        # Calculate x = (radius * sin(za) * k)
        radius = diameter / 2.0
        xvals = radius * cp.sin(za) * kvals
        
        # Initialize output array
        values = cp.zeros_like(xvals, dtype=cp.float64)
        
        # Create masks for zero and non-zero values
        nz = xvals != 0.0
        ze = xvals == 0.0
        
        # Evaluate Airy pattern: 2*J₁(x)/x
        values[nz] = 2.0 * gpu_j1(xvals[nz]) / xvals[nz]
        values[ze] = 1.0  # limit as x→0
        
        # Check for invalid values if requested
        if check:
            sm = cp.sum(values)
            if cp.isinf(sm) or cp.isnan(sm):
                raise ValueError("GPU Airy beam evaluation resulted in an invalid value")
        
        if polarized:
            # For polarized beam, create proper output shape
            nsrc = len(az)
            result = cp.zeros((2, 2, nsrc), dtype=cp.complex128)
            
            # Equal power splitting between polarizations (unpolarized beam)
            # Divide by sqrt(2) for E-field as in original implementation
            result[0, 0, :] = values / cp.sqrt(2.0)
            result[1, 1, :] = values / cp.sqrt(2.0)
            
            return result
        else:
            # For unpolarized, return power pattern (squared E-field)
            return values ** 2

    def _evaluate_uvbeam_gpu(
        self,
        beam: BeamInterface,
        az: cp.ndarray,
        za: cp.ndarray,
        polarized: bool,
        freq: float,
        check: bool = False,
        spline_opts: Optional[Dict] = None
    ) -> cp.ndarray:
        """
        GPU-accelerated UVBeam interpolation using map_coordinates.
        
        Parameters
        ----------
        beam : BeamInterface
            Beam object to evaluate (must wrap a UVBeam)
        az : cp.ndarray
            Azimuth angles in radians (on GPU)
        za : cp.ndarray
            Zenith angles in radians (on GPU)
        polarized : bool
            Whether to return polarized beam
        freq : float
            Frequency in Hz
        check : bool
            Whether to check for invalid values
        spline_opts : dict
            Options for interpolation (e.g., {'order': 1})
            
        Returns
        -------
        cp.ndarray
            Beam response values on GPU
        """
        # Get the actual beam object
        if hasattr(beam, 'beam'):
            uvbeam = beam.beam
        else:
            uvbeam = beam
            
        # Default spline options
        if spline_opts is None:
            spline_opts = {'order': 1}  # Linear interpolation
            
        # Check if it's actually a power beam
        is_power_beam = uvbeam.beam_type == 'power'
        
        # Get beam parameters
        freq_idx = np.argmin(np.abs(uvbeam.freq_array - freq))
        
        # Get beam data array and transfer to GPU if not cached
        beam_key = (id(beam), freq_idx, polarized)
        if beam_key not in self._gpu_beam_data:
            # Get the beam data for this frequency
            if polarized:
                # Shape: (Naxes_vec, Nfeeds, Nfreqs, Naxes2, Naxes1)
                # Extract for specific frequency: (Naxes_vec, Nfeeds, Naxes2, Naxes1)
                beam_data_cpu = uvbeam.data_array[:, :, freq_idx, :, :]
            else:
                # For unpolarized, get the appropriate data
                if is_power_beam:
                    # For power beams, data is already power
                    beam_data_cpu = uvbeam.data_array[0, 0, freq_idx, :, :]
                else:
                    # For E-field beams, get first component directly
                    # The CPU code does interp_beam[0, 0, 0, :] which is the E-field
                    beam_data_cpu = uvbeam.data_array[0, 0, freq_idx, :, :]
            
            # Transfer to GPU
            self._gpu_beam_data[beam_key] = cp.asarray(beam_data_cpu)
            
        beam_data_gpu = self._gpu_beam_data[beam_key]
        
        # Map az/za coordinates to grid indices
        # Match pyuvdata's exact normalization
        az_axis = uvbeam.axis1_array  # Azimuth axis (phi)
        za_axis = uvbeam.axis2_array  # Zenith angle axis (theta)
        
        # Check if azimuth wraps around (covers 2π)
        az_range = az_axis.max() - az_axis.min()
        wraps_around = np.isclose(az_range, 2 * np.pi, atol=np.diff(az_axis).max())
        
        # Transfer axes to GPU
        az_axis_gpu = cp.asarray(az_axis)
        za_axis_gpu = cp.asarray(za_axis)
        
        # Calculate grid indices using pyuvdata's normalization
        # For azimuth (phi)
        az_idx = az - az_axis_gpu.min()
        az_idx *= (len(az_axis) - 1) / (az_axis_gpu.max() - az_axis_gpu.min())
        
        # For zenith angle (theta)
        za_idx = za - za_axis_gpu.min()
        za_idx *= (len(za_axis) - 1) / (za_axis_gpu.max() - za_axis_gpu.min())
        
        # Handle azimuth wrapping if needed
        # Note: CuPy's map_coordinates doesn't support different modes per axis
        # So we use 'wrap' for both if azimuth wraps, 'constant' otherwise
        if wraps_around:
            mode = 'wrap'
        else:
            mode = 'constant'
            
        if polarized:
            # Interpolate for each component and feed
            naxes_vec, nfeeds = beam_data_gpu.shape[:2]
            nsrc = len(az)
            result = cp.zeros((naxes_vec, nfeeds, nsrc), dtype=cp.complex128)
            
            # Create coordinate array for map_coordinates
            coords = cp.array([za_idx, az_idx])  # Note: za is first axis
            
            for i in range(naxes_vec):
                for j in range(nfeeds):
                    # Interpolate this component
                    result[i, j, :] = gpu_map_coordinates(
                        beam_data_gpu[i, j, :, :],
                        coords,
                        order=spline_opts.get('order', 1),
                        mode=mode,
                        cval=0.0
                    )
            
            # Reshape to expected output format
            result = result[:, :, :]  # Already in correct shape
            
        else:
            # Unpolarized interpolation
            coords = cp.array([za_idx, az_idx])  # Note: za is first axis
            result = gpu_map_coordinates(
                beam_data_gpu,
                coords,
                order=spline_opts.get('order', 1),
                mode=mode,
                cval=0.0
            )
        
        # Check for invalid values if requested
        if check:
            sm = cp.sum(result)
            if cp.isinf(sm) or cp.isnan(sm):
                raise ValueError("GPU UVBeam interpolation resulted in an invalid value")
        
        return result

    def get_apparent_flux_polarized(self, beam: cp.ndarray, flux: cp.ndarray) -> None:
        """
        Calculate apparent flux of the sources using GPU.

        This modifies the beam array in-place to contain the apparent flux values.

        Parameters
        ----------
        beam : cp.ndarray
            Array with beam values of shape (nax, nfd, nsrc)
        flux : cp.ndarray
            Array with source flux values of shape (nsrc,)
        """
        # Ensure inputs are cupy arrays
        if not isinstance(beam, cp.ndarray):
            beam = cp.asarray(beam)
        if not isinstance(flux, cp.ndarray):
            flux = cp.asarray(flux)

        # Get dimensions
        nax, nfd, nsrc = beam.shape

        # Extract E-field components for each feed
        E_f0 = beam[:, 0, :]  # (nax, nsrc) for feed 0
        E_f1 = beam[:, 1, :]  # (nax, nsrc) for feed 1

        # Calculate values needed for coherency matrix
        E_f0_abs_sq = cp.sum(cp.abs(E_f0) ** 2, axis=0)
        E_f1_abs_sq = cp.sum(cp.abs(E_f1) ** 2, axis=0)
        E_f0_conj_dot_E_f1 = cp.sum(cp.conj(E_f0) * E_f1, axis=0)

        # Calculate apparent flux (modify beam in-place)
        beam[0, 0, :] = E_f0_abs_sq * flux
        beam[0, 1, :] = E_f0_conj_dot_E_f1 * flux
        beam[1, 0, :] = cp.conj(E_f0_conj_dot_E_f1) * flux
        beam[1, 1, :] = E_f1_abs_sq * flux
