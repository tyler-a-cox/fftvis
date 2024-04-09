from __future__ import annotations

import finufft
import numpy as np
from matvis import conversions

from . import utils, beams

# Default accuracy for the non-uniform fast fourier transform based on precision
default_accuracy_dict = {
    1: 6e-8,
    2: 1e-13,
}


def simulate_vis(
    antpos: dict,
    sources: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    freqs: np.ndarray,
    lsts: np.ndarray,
    beam,
    baselines: list[tuple] = None,
    precision: int = 1,
    polarized: bool = False,
    latitude: float = -0.5361913261514378,
    eps: float = None,
    use_feed: str = "x",
):
    """
    Parameters:
    ----------
    antpos : dict
        Dictionary of antenna positions
    sources : np.ndarray
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
    lsts : np.ndarray
        Local sidereal time in radians. Range is [0, 2 pi].
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

    Returns:
    -------
    vis : np.ndarray
        Array of shape (nfreqs, ntimes, nants, nants) if polarized is False, and
        (nfreqs, ntimes, 2, 2, nants, nants) if polarized is True.
    """
    # Get the accuracy for the given precision if not provided
    if eps is None:
        eps = default_accuracy_dict[precision]

    # Source coordinate transform, from equatorial to Cartesian
    crd_eq = conversions.point_source_crd_eq(ra, dec)

    # Make sure antpos has the right format
    antpos = {k: np.array(v) for k, v in antpos.items()}

    # Get coordinate transforms as a function of LST
    eq2tops = np.array([conversions.eci_to_enu_matrix(lst, latitude) for lst in lsts])

    return simulate(
        antpos=antpos,
        freqs=freqs,
        sources=sources,
        beam=beam,
        crd_eq=crd_eq,
        eq2tops=eq2tops,
        baselines=baselines,
        precision=precision,
        polarized=polarized,
        eps=eps,
        use_feed=use_feed,
    )


def simulate(
    antpos: dict,
    freqs: np.ndarray,
    sources: np.ndarray,
    beam,
    crd_eq: np.ndarray,
    eq2tops: np.ndarray,
    baselines: list[tuple] = None,
    precision: int = 1,
    polarized: bool = False,
    eps: float = 6e-8,
    use_feed: str = "x",
):
    """
    Parameters:
    ----------
    antpos : dict
        Dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
    freqs : np.ndarray
        Frequencies to evaluate visibilities at in Hz.
    sources : np.ndarray
        Intensity distribution of sources/pixels on the sky, assuming intensity
        (Stokes I) only. The Stokes I intensity will be split equally between
        the two linear polarization channels, resulting in a factor of 0.5 from
        the value inputted here. This is done even if only one polarization
        channel is simulated.
    beam : UVBeam
        Beam object to use for the array. Per-antenna beams are not yet supported.
    crd_eq : np.ndarray
        Cartesian unit vectors of sources in an ECI (Earth Centered
        Inertial) system, which has the Earth's center of mass at
        the origin, and is fixed with respect to the distant stars.
        The components of the ECI vector for each source are:
        (cos(RA) cos(Dec), sin(RA) cos(Dec), sin(Dec)).
        Shape=(3, NSRCS).
    eq2tops : np.ndarray
        Set of 3x3 transformation matrices to rotate the RA and Dec
        cosines in an ECI coordinate system (see `crd_eq`) to
        topocentric ENU (East-North-Up) unit vectors at each
        time/LST/hour angle in the dataset. Shape=(NTIMES, 3, 3).
    baselines : list of tuples, default = None
        If provided, only the baselines within the list will be simulated and array of shape
        (nbls, nfreqs, ntimes) will be returned
    precision : int, optional
        Which precision level to use for floats and complex numbers
        Allowed values:
        - 1: float32, complex64
        - 2: float64, complex128
    eps : float, default = 6e-8
        Desired accuracy of the non-uniform fast fourier transform.


    Returns:
    -------
    vis : np.ndarray
        Array of shape (nfreqs, ntimes, nants, nants) if polarized is False, and
        (nfreqs, ntimes, 2, 2, nants, nants) if polarized is True.
    """
    # Get sizes of inputs
    nfreqs = np.size(freqs)
    nants = len(antpos)
    ntimes = len(eq2tops)

    if polarized:
        nax = nfeeds = 2
    else:
        nax = nfeeds = 1

    if precision == 1:
        real_dtype = np.float32
        complex_dtype = np.complex64
    else:
        real_dtype = np.float64
        complex_dtype = np.complex128

    # Get the redundant groups - TODO handle this better
    if not baselines:
        reds = utils.get_pos_reds(antpos, include_autos=True)
        baselines = [red[0] for red in reds]
        nbls = len(baselines)
        bl_to_red_map = {red[0]: np.array(red) for red in reds}
        expand_vis = True
    else:
        nbls = len(baselines)
        expand_vis = False

    # Prepare the beam
    beam = conversions.prepare_beam(beam, polarized=polarized, use_feed=use_feed)

    # Check if the beam is complex
    beam_values, _ = beam.interp(
        az_array=np.array([0]),
        za_array=np.array([0]),
        freq_array=np.array([freqs[0]]),
    )
    is_beam_complex = np.issubdtype(beam_values.dtype, np.complexfloating)

    # Convert to correct precision
    crd_eq = crd_eq.astype(real_dtype)
    eq2tops = eq2tops.astype(real_dtype)

    # Factor of 0.5 accounts for splitting Stokes between polarization channels
    Isky = (0.5 * sources).astype(complex_dtype)

    # Compute baseline vectors
    blx, bly = np.array([antpos[bl[1]] - antpos[bl[0]] for bl in baselines])[
        :, :2
    ].T.astype(real_dtype)

    # Generate visibility array
    if expand_vis:
        vis = np.zeros(
            (ntimes, nants, nants, nfeeds, nfeeds, nfreqs), dtype=complex_dtype
        )
    else:
        vis = np.zeros((ntimes, nbls, nfeeds, nfeeds, nfreqs), dtype=complex_dtype)

    # Loop over time samples
    for ti, eq2top in enumerate(eq2tops):
        # Convert to topocentric coordinates
        tx, ty, tz = np.dot(eq2top, crd_eq)

        # Only simulate above the horizon
        above_horizon = tz > 0
        tx = tx[above_horizon]
        ty = ty[above_horizon]

        # Number of above horizon points
        nsim_sources = above_horizon.sum()

        # Form the visibility array
        _vis = np.zeros((nfeeds, nfeeds, nbls, nfreqs), dtype=complex_dtype)

        if is_beam_complex:
            _vis_negatives = np.zeros(
                (nfeeds, nfeeds, nbls, nfreqs), dtype=complex_dtype
            )

        # Compute azimuth and zenith angles
        az, za = conversions.enu_to_az_za(enu_e=tx, enu_n=ty, orientation="uvbeam")

        for fi in range(nfreqs):
            # Compute uv coordinates
            u, v = (
                blx * freqs[fi] / utils.speed_of_light,
                bly * freqs[fi] / utils.speed_of_light,
            )

            # Compute beams - only single beam is supported
            A_s = np.zeros((nax, nfeeds, nsim_sources), dtype=complex_dtype)
            A_s = beams._evaluate_beam(A_s, beam, az, za, polarized, freqs[fi])
            A_s = A_s.transpose((1, 0, 2))
            beam_product = np.einsum("abs,cbs->acs", A_s.conj(), A_s)
            beam_product = beam_product.reshape(nax * nfeeds, nsim_sources)

            # Compute sky beam product
            i_sky = beam_product * Isky[above_horizon, fi]

            # Compute visibilities w/ non-uniform FFT
            _vis_here = finufft.nufft2d3(
                2 * np.pi * tx,
                2 * np.pi * ty,
                i_sky,
                u,
                v,
                modeord=0,
                eps=eps,
            )

            # Expand out the visibility array
            _vis[..., fi] = _vis_here.reshape(nfeeds, nfeeds, nbls)

            # If beam is complex, we need to compute the reverse negative frequencies
            # TODO: no way to store this in the loop
            if is_beam_complex:
                # Compute
                _vis_here_neg = finufft.nufft2d3(
                    2 * np.pi * tx,
                    2 * np.pi * ty,
                    i_sky,
                    -u,
                    -v,
                    modeord=0,
                    eps=eps,
                )
                _vis_negatives[..., fi] = _vis_here_neg.reshape(nfeeds, nfeeds, nbls)

        # Expand out the visibility array in antenna by antenna matrix
        if expand_vis:
            for bi, bls in enumerate(baselines):
                np.add.at(
                    vis,
                    (ti, bl_to_red_map[bls][:, 0], bl_to_red_map[bls][:, 1]),
                    _vis[..., bi, :],
                )

                # Add the conjugate, avoid auto baselines twice
                if bls[0] != bls[1]:
                    if is_beam_complex:
                        np.add.at(
                            vis,
                            (ti, bl_to_red_map[bls][:, 1], bl_to_red_map[bls][:, 0]),
                            _vis_negatives[..., bi, :],
                        )
                    else:
                        np.add.at(
                            vis,
                            (ti, bl_to_red_map[bls][:, 1], bl_to_red_map[bls][:, 0]),
                            _vis[..., bi, :].conj(),
                        )

        else:
            vis[ti] = np.swapaxes(_vis, 2, 0)

    if expand_vis:
        return (
            np.transpose(vis, (5, 0, 3, 4, 1, 2))
            if polarized
            else np.moveaxis(vis[..., 0, 0, :], 3, 0)
        )
    else:
        return (
            np.transpose(vis, (4, 0, 2, 3, 1))
            if polarized
            else np.moveaxis(vis[..., 0, 0, :], 2, 0)
        )
