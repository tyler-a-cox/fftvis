import numpy as np
from pyuvdata import UVBeam
from astropy import units as un
from astropy.coordinates import SkyCoord, EarthLocation
from matvis.core.beams import prepare_beam_unpolarized
from .cpu.cpu import simulate as cpu_simulate_vis
# from .gpu.gpu import simulate as gpu_simulate_vis

# Default accuracy for the non-uniform fast fourier transform based on precision
default_accuracy_dict = {
    1: 6e-8,
    2: 1e-13,
}


def simulate_vis(
    ants: dict,
    fluxes: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    beam: UVBeam,
    telescope_loc: EarthLocation,
    baselines: list[tuple] = None,
    precision: int = 2,
    polarized: bool = False,
    eps: float = None,
    beam_spline_opts: dict = None,
    flat_array_tol: float = 0.0,
    use_gpu: bool = False,
    max_progress_reports: int = 100,
    coord_method: str = "CoordinateRotationERFA",
    max_memory: int = np.inf,
    min_chunks: int = 1,
    source_buffer: float = 1.0,
    coord_method_params: dict = {"update_bcrs_every": 1.0},
):
    """
    TODO: Add description and update docstring.

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
    times : astropy.time.Time
        Array of times to evaluate visibilties for.
    beam : UVBeam
        Beam object to use for the array. Per-antenna beams are not yet supported.
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
    latitude : float, optional
        Latitude of the array in radians. The default is the
        HERA latitude = -30.7215 * pi / 180.
    eps : float, default = None
        Desired accuracy of the non-uniform fast fourier transform. If None, the default accuracy
        for the given precision will be used. For precision 1, the default accuracy is 6e-8, and for
        precision 2, the default accuracy is 1e-12.
    beam_spline_opts : dict, optional
        Options to pass to :meth:`pyuvdata.uvbeam.UVBeam.interp` as `spline_opts`.
    flat_array_tol : float, default = 0.0
        Tolerance for checking if the array is flat in units of meters. If the
        z-coordinate of all baseline vectors is within this tolerance, the array
        is considered flat and the z-coordinate is set to zero. Default is 0.0.
    live_progress : bool, default = True
        Whether to show progress bar during simulation.
    interpolation_function : str, default = "az_za_simple"
        The interpolation function to use when interpolating the beam. Can be either be
        'az_za_simple' or 'az_za_map_coordinates'. The former is slower but more accurate
        at the edges of the beam, while the latter is faster but less accurate
        for interpolation orders greater than linear.
    Returns:
    -------
    vis : np.ndarray
        Array of shape (nfreqs, ntimes, nants, nants) if polarized is False, and
        (nfreqs, ntimes, nfeed, nfeed, nants, nants) if polarized is True.
    """
    function = cpu_simulate_vis if not use_gpu else gpu_simulate_vis

    # Get the accuracy for the given precision if not provided
    if eps is None:
        eps = default_accuracy_dict[precision]

    # Make sure antpos has the right format
    ants = {k: np.array(v) for k, v in ants.items()}

    # Get the baselines
    skycoords = SkyCoord(ra=ra * un.rad, dec=dec * un.rad, frame="icrs")

    # Get the baselines
    vis = np.zeros(
        (len(times), len(freqs), len(ants), len(ants), 2, 2), dtype=np.complex128
    )

    # Prepare the beam
    beam = prepare_beam_unpolarized(beam)

    for fi, freq in enumerate(freqs):
        vis[..., fi] = function(
            antpos=ants,
            freq=freq,
            times=times,
            skycoords=skycoords,
            fluxes=fluxes[:, fi],
            beam=beam,
            baselines=baselines,
            precision=precision,
            polarized=polarized,
            eps=eps,
            telescope_loc=telescope_loc,
            beam_spline_opts=beam_spline_opts,
            flat_array_tol=flat_array_tol,
            # interpolation_function=interpolation_function,
            max_progress_reports=max_progress_reports,
            max_memory=max_memory,
            coord_method=coord_method,
            min_chunks=min_chunks,
            source_buffer=source_buffer,
            coord_method_params=coord_method_params,
        )
    return vis
