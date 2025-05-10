"""
Core simulation functionality for fftvis.

This module defines the base classes and interfaces for visibility simulation,
independent of the specific backend implementation.
"""

from abc import ABC, abstractmethod
from typing import Literal, Union
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from matvis.core.coords import CoordinateRotation

# Default accuracy for the non-uniform fast fourier transform based on precision
default_accuracy_dict = {
    1: 6e-8,
    2: 1e-13,
}


class SimulationEngine(ABC):
    """Base class for visibility simulation engines."""

    @abstractmethod
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
        trace_mem: bool = False,
        enable_memory_monitor: bool = False,
    ) -> np.ndarray:
        """
        Simulate visibilities using the engine's implementation.

        Parameters
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
        polarized : bool, optional
            Whether to simulate polarized visibilities. If True, the output will have
            shape (nfreqs, ntimes, 2, 2, nants, nants), and if False, the output will
            have shape (nfreqs, ntimes, nants, nants).
        eps : float, default = None
            Desired accuracy of the non-uniform fast fourier transform. If None, the default accuracy
            for the given precision will be used. For precision 1, the default accuracy is 6e-8, and for
            precision 2, the default accuracy is 1e-12.
        upsampfac : int, default = 2
            Upsampling factor for the non-uniform fast fourier transform. This is the factor by which the
            intermediate grid is upsampled. Only values of 1.25 or 2 are allowed. Can be useful for decreasing
            the computation time and memory requirement for large arrays at the expensive of some accuracy. 
            The default value is 2.
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

        Returns
        -------
        np.ndarray
            Array of shape (nfreqs, ntimes, nants, nants) if polarized is False, and
            (nfreqs, ntimes, nfeed, nfeedd, nants, nants) if polarized is True.
        """
        pass

    @abstractmethod
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
        basis_matrix: np.ndarray = None,
        type1_n_modes: int = None,
        trace_mem: bool = False,
    ) -> np.ndarray:
        """
        Evaluate a chunk of visibility data.

        Parameters
        ----------
        time_idx : slice
            Time indices to process.
        freq_idx : slice
            Frequency indices to process.
        beam : UVBeam
            Beam object to use.
        coord_mgr : CoordinateRotation
            Coordinate manager object.
        rotation_matrix : np.ndarray
            Rotation matrix for antenna positions.
        bls : np.ndarray
            Baseline vectors.
        freqs : np.ndarray
            Frequencies to evaluate at.
        complex_dtype : np.dtype
            Complex data type to use.
        nfeeds : int
            Number of feeds.
        polarized : bool
            Whether to simulate polarized visibilities.
        eps : float, default = None
            Desired accuracy of the non-uniform fast fourier transform.
        upsampfac : int
            Upsampling factor for the non-uniform FFT.
        beam_spline_opts : dict, default = None
            Options for beam interpolation.
        interpolation_function : str
            The interpolation function to use for beam interpolation.
        n_threads : int
            Number of threads to use.
        is_coplanar : bool
            Whether the array is coplanar.
        basis_matrix : np.ndarray
            Lattice basis matrix used to grid baselines and sources for a type 1
            non-uniform FFT.
        type1_n_modes : int
            Number of modes for type 1 non-uniform FFT.
        trace_mem : bool
            Whether to trace memory usage.

        Returns
        -------
        np.ndarray
            Chunk of visibility data.
        """
        pass
