from typing import Literal, Union
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from pyuvdata.beam_interface import BeamInterface
from pyuvdata import UVBeam
from matvis.core.beams import prepare_beam_unpolarized

from .core.beams import BeamEvaluator
from .cpu.beams import CPUBeamEvaluator
from .core.simulate import SimulationEngine, default_accuracy_dict
from .cpu.simulate import CPUSimulationEngine


def create_beam_evaluator(
    backend: Literal["cpu", "gpu"] = "cpu", **kwargs
) -> BeamEvaluator:
    """Create a beam evaluator for the specified backend.

    Parameters
    ----------
    backend
        The backend to use for beam evaluation.
        Currently supported: "cpu", "gpu".
    **kwargs
        Additional keyword arguments to pass to the beam evaluator constructor.

    Returns
    -------
    BeamEvaluator
        A beam evaluator instance for the specified backend.

    Raises
    ------
    ValueError
        If the specified backend is not supported.
    """
    if backend == "cpu":
        evaluator = CPUBeamEvaluator(**kwargs)
        # Ensure the beam_list is properly initialized since this is required by matvis
        evaluator.beam_list = []
        evaluator.beam_idx = None
        return evaluator
    elif backend == "gpu":
        raise NotImplementedError("GPU backend not yet implemented")
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def create_simulation_engine(
    backend: Literal["cpu", "gpu"] = "cpu", **kwargs
) -> SimulationEngine:
    """Create a simulation engine for the specified backend.

    Parameters
    ----------
    backend
        The backend to use for simulation.
        Currently supported: "cpu".
        "gpu" is defined but not yet implemented.
    **kwargs
        Additional keyword arguments to pass to the simulation engine constructor.

    Returns
    -------
    SimulationEngine
        A simulation engine instance for the specified backend.

    Raises
    ------
    ValueError
        If the specified backend is not supported.
    """
    if backend == "cpu":
        return CPUSimulationEngine(**kwargs)
    elif backend == "gpu":
        from .gpu.simulate import GPUSimulationEngine

        return GPUSimulationEngine(**kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def simulate_vis(
    ants: dict,
    fluxes: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    freqs: np.ndarray,
    times: Union[np.ndarray, Time],
    beam,
    telescope_loc: EarthLocation,
    baselines: list[tuple] = None,
    precision: int = 2,
    polarized: bool = False,
    eps: float = None,
    upsampfac: int = 2,
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
    force_use_type3: bool = False,
    force_use_ray: bool = False,
    trace_mem: bool = False,
    backend: Literal["cpu", "gpu"] = "cpu",
) -> np.ndarray:
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
        Beam object to use for the array. Per-antenna beams are not yet supported.
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
    upsampfac : int, default = 2
        Upsampling factor for the non-uniform fast fourier transform. This is the factor by which the
        intermediate grid is upsampled. Only values of 1.25 or 2 are allowed. Can be useful for decreasing
        the computation time and memory requirement for large arrays at the expensive of some accuracy. 
        The default value is 2.
    beam_spline_opts : dict, optional
        Options to pass to :meth:`pyuvdata.uvbeam.UVBeam.interp` as `spline_opts`.
    use_feed : str, default = "x"
        Feed to use for unpolarized simulations.
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
    force_use_type3: bool, default = False
        Whether to force the use of type 3 NUFFT. If False, type 3 will only be used
        if the array cannot be defined in a regular grid.
    force_use_ray: bool, default = False
        Whether to force the use of Ray for parallelization. If False, Ray will only be used
        if nprocesses > 1.
    trace_mem : bool, default = False
        Whether to trace memory usage during the simulation. If True, the memory usage
        will be recorded at various points in the simulation and saved to a file.
    live_progress : bool, default = True
        Whether to show progress bar during simulation.
    backend : str
        Backend to use for simulation ("cpu" or "gpu").

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

    # Interpolate the beam to the desired frequencies to avoid redundant
    # interpolation in the simulation engine
    if isinstance(beam, UVBeam):
        if hasattr(beam, "Nfreqs") and beam.Nfreqs > 1:
            beam = beam.interp(freq_array=freqs, new_object=True, run_check=False) # pragma: no cover
    elif isinstance(beam, BeamInterface) and beam._isuvbeam:
        if hasattr(beam.beam, "Nfreqs") and beam.beam.Nfreqs > 1:
            beam.beam = beam.beam.interp(freq_array=freqs, new_object=True, run_check=False)

    beam = BeamInterface(beam)

    # Prepare the beam
    if not polarized:
        beam = prepare_beam_unpolarized(beam, use_feed=use_feed)

    # Create the simulation engine for the desired backend
    engine = create_simulation_engine(backend=backend)

    # Run the simulation
    return engine.simulate(
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
        upsampfac=upsampfac,
        beam_spline_opts=beam_spline_opts,
        flat_array_tol=flat_array_tol,
        interpolation_function=interpolation_function,
        nprocesses=nprocesses,
        nthreads=nthreads,
        coord_method=coord_method,
        coord_method_params=coord_method_params,
        force_use_type3=force_use_type3,
        force_use_ray=force_use_ray,
        trace_mem=trace_mem,
    )
