from . import utils
import numpy as np
from typing import Callable
from matvis import conversions

from __future__ import annotations

import finufft
import numpy as np
from typing import Callable
from matvis import conversions


def simulate_vis(
    antpos: dict,
    sources: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    freqs: np.ndarray,
    lsts: np.ndarray,
    beam,
    precision: int = 1,
    polarized: bool = False,
    latitude: float = -0.5361913261514378,
    use_redundancy: bool = True,
    accuracy: float = 1e-6,
):
    """
    Parameters:
    ----------
    antpos : dict
        Dictionary of antenna positions
    freqs : np.ndarray
        Frequencies to evaluate visibilities at MHz.
    sources : np.ndarray
        asdf
    beam : UVBeam
        pass
    crd_eq : np.ndarray
        pass
    eq2tops : np.ndarray
        pass
    precision : int, optional
       Which precision level to use for floats and complex numbers
       Allowed values:
       - 1: float32, complex64
       - 2: float64, complex128
    use_redundancy : bool, default = True
        If True,
    check : bool, default = False
        If True, perform checks on the input data array prior to
    accuracy : float, default = 1e-6
        pass

    Returns:
    -------
    vis : np.ndarray

    """
    # Source coordinate transform, from equatorial to Cartesian
    crd_eq = conversions.point_source_crd_eq(ra, dec)

    # Get coordinate transforms as a function of LST
    eq2tops = np.array([conversions.eci_to_enu_matrix(lst, latitude) for lst in lsts])

    return simulate(
        antpos=antpos,
        freqs=freqs,
        sources=sources,
        beam=beam,
        crd_eq=crd_eq,
        eq2tops=eq2tops,
        precision=precision,
        polarized=polarized,
        use_redundancy=use_redundancy,
        accuracy=accuracy,
    )


def _evaluate_beam(
    beam,
    tx: np.ndarray,
    ty: np.ndarray,
    freqs: np.ndarray,
):
    kw = {
        "reuse_spline": True,
        "check_azza_domain": False,
    }
    # Primary beam pattern using direct interpolation of UVBeam object
    az, za = conversions.enu_to_az_za(enu_e=tx, enu_n=ty, orientation="uvbeam")
    # beam_vals = beam.interp(az_array=az, za_array=za, freq_array=freqs, **kw)[0][0, 0].T
    beam_vals = beam.interp(az_array=az, za_array=za, freq_array=freqs, **kw)[0][0, 1].T
    return beam_vals**2


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
    use_redundancy: bool = True,
    check: bool = False,
    accuracy: float = 1e-6,
):
    """
    Parameters:
    ----------
    antpos : dict
        Dictionary of antenna positions
    freqs : np.ndarray
        Frequencies to evaluate visibilities at MHz.
    sources : np.ndarray
        asdf
    beam : UVBeam
        pass
    crd_eq : np.ndarray
        pass
    eq2tops : np.ndarray
        pass
    precision : int, optional
       Which precision level to use for floats and complex numbers
       Allowed values:
       - 1: float32, complex64
       - 2: float64, complex128
    use_redundancy : bool, default = True
        If True,
    check : bool, default = False
        If True, perform checks on the input data array prior to
    accuracy : float, default = 1e-6
        pass

    Returns:
    -------
    vis : np.ndarray

    """
    # Check inputs are valid
    if check:
        nax, nfeeds, nants, ntimes = _validate_inputs(
            precision, polarized, antpos, eq2tops, crd_eq, sources
        )
    else:
        nax, nfeeds, nants, ntimes, nfreqs = (
            1,
            1,
            len(antpos),
            eq2tops.shape[0],
            freqs.shape[0],
        )

    if precision == 1:
        real_dtype = np.float32
        complex_dtype = np.complex64
    else:
        real_dtype = np.float64
        complex_dtype = np.complex128

    # Get the redundant - TODO handle this better
    if not baselines:
        reds = utils.get_pos_reds(antpos)
        baselines = [red[0] for red in reds]
        nbls = len(baselines)
        expand_vis = True
    else:
        nbls = len(baselines)
        expand_vis = False

    # prepare beam
    # TODO: uncomment and test this when simulating multiple polarizations
    # beam = conversions.prepare_beam(beam)

    if use_redundancy and expand_vis:
        bl_to_red_map = {red[0]: np.array(red) for red in reds}

    # Convert to correct precision
    crd_eq = crd_eq.astype(real_dtype)
    eq2tops = eq2tops.astype(real_dtype)
    Isky = (0.5 * sources).astype(complex_dtype)

    # Compute coordinates
    blx, bly = np.array([antpos[bl[1]] - antpos[bl[0]] for bl in baselines])[
        :, :2
    ].T.astype(real_dtype)

    # Generate visibility array
    if expand_vis:
        vis = np.full((nants, nants, ntimes, nfreqs), 0, dtype=complex_dtype)
    else:
        vis = np.full((nbls, ntimes, nfreqs), 0, dtype=complex_dtype)

    # Loop over time samples
    for ti, eq2top in enumerate(eq2tops):
        # Convert to topocentric coordinates
        tx, ty, tz = np.dot(eq2top, crd_eq)

        above_horizon = tz > 0
        tx = tx[above_horizon]
        ty = ty[above_horizon]

        # Compute the beam sky product
        i_sky = (Isky[above_horizon] * _evaluate_beam(beam, tx, ty, freqs)).astype(
            complex_dtype
        )

        _vis = np.full((nbls, nfreqs), 0, dtype=complex_dtype)

        # TODO: finufft2d3 is not vectorized over time
        # TODO: finufft2d3 gives me warning if I don't use ascontiguousarray
        for ni in range(nfreqs):
            u, v = (
                blx * freqs[ni] / utils.speed_of_light,
                bly * freqs[ni] / utils.speed_of_light,
            )
            _vis[:, ni] = finufft.nufft2d3(
                2 * np.pi * ty,
                2 * np.pi * tx,
                np.ascontiguousarray(i_sky[:, ni]),
                v,
                u,
                modeord=0,
                eps=accuracy,
            )

        if use_redundancy and expand_vis:
            for bi, bls in enumerate(baselines):
                np.add.at(
                    vis,
                    (bl_to_red_map[bls][:, 0], bl_to_red_map[bls][:, 1], ti),
                    _vis[bi],
                )
                np.add.at(
                    vis,
                    (bl_to_red_map[bls][:, 1], bl_to_red_map[bls][:, 0], ti),
                    _vis[bi].conj(),
                )
        else:
            vis[:, ti, :] = _vis

    return vis


def simulate_basis(
    antpos: dict,
    eta: np.ndarray,
    freqs: np.ndarray,
    sources: np.ndarray,
    crd_eq: np.ndarray,
    eq2tops: np.ndarray,
    precision: int = 1,
    polarized: bool = False,
    check: bool = False,
    accuracy: float = 1e-6,
    use_redundancy=True,
):
    """
    Simulate the sky using some basis as simulation

    precision : int, optional
        Which precision level to use for floats and complex numbers
        Allowed values:
        - 1: float32, complex64
        - 2: float64, complex128
    """
    # Check inputs are valid
    if check:
        nax, nfeeds, nants, ntimes = _validate_inputs(
            precision, polarized, antpos, eq2tops, crd_eq, sources
        )
    else:
        nax, nfeeds, ntimes, nfreqs = (
            1,
            1,
            eq2tops.shape[0],
            freqs.shape[-1],
        )

    if precision == 1:
        real_dtype = np.float32
        complex_dtype = np.complex64
    else:
        real_dtype = np.float64
        complex_dtype = np.complex128

    # Get the redundant - TODO handle this better
    reds = utils.get_pos_reds(antpos)
    baselines = [red[0] for red in reds]
    nbls = len(baselines)
    nants = len(antpos)

    # prepare beam
    # TODO: add this when using multiple polarizations
    # beam = conversions.prepare_beam(beam)

    if use_redundancy:
        bl_to_red_map = {red[0]: np.array(red) for red in reds}

    # Convert to correct precision
    crd_eq = crd_eq.astype(real_dtype)
    eq2tops = eq2tops.astype(real_dtype)
    Isky = (0.5 * sources).astype(complex_dtype)

    # Compute coordinates
    blx, bly = np.array([antpos[bl[1]] - antpos[bl[0]] for bl in baselines])[
        :, :2
    ].T.astype(real_dtype)

    # Baseline coordinates
    U, tF = np.meshgrid(blx / utils.speed_of_light, freqs)
    V, _ = np.meshgrid(bly / utils.speed_of_light, freqs)

    # zeroed visibility array
    vis = np.full((nants, nants, ntimes, nfreqs), 0, dtype=complex_dtype)

    # Loop over time samples
    for ti, eq2top in enumerate(eq2tops):
        # Convert to topocentric coordinates
        tx, ty, tz = np.dot(eq2top, crd_eq)
        above_horizon = tz > 0
        tx = tx[above_horizon]
        ty = ty[above_horizon]

        # Sky Coordinates
        tX, _eta = np.meshgrid(tx, eta)
        tY, _ = np.meshgrid(ty, eta)

        # Simulate
        _vis = finufft.nufft3d3(
            2 * np.pi * np.ravel(tY),
            2 * np.pi * np.ravel(tX),
            np.ravel(_eta),
            np.ravel(Isky[above_horizon]),
            np.ravel(V * tF),
            np.ravel(U * tF),
            np.ravel(tF),
            modeord=0,
            eps=accuracy,
        )
        _vis.shape = (nfreqs, nbls)

        if use_redundancy:
            for bi, bls in enumerate(baselines):
                np.add.at(
                    vis,
                    (bl_to_red_map[bls][:, 0], bl_to_red_map[bls][:, 1], ti),
                    _vis[:, bi],
                )
                np.add.at(
                    vis,
                    (bl_to_red_map[bls][:, 1], bl_to_red_map[bls][:, 0], ti),
                    _vis[:, bi].conj(),
                )

    return vis
