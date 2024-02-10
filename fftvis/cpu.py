from __future__ import annotations

import finufft
import numpy as np
from typing import Callable
from matvis import conversions

IDEALIZED_BL_TOL = 1e-8  # bl_error_tol for redcal.get_reds when using antenna positions calculated from reds


def reverse_bl(bl):
    """Reverses a (i,j) or (i,j,pol) baseline key to make (j,i)
    or (j,i,pol[::-1]), respectively."""
    i, j = bl[:2]
    return (j, i)


def get_pos_reds(antpos, bl_error_tol=1.0, include_autos=False):
    """Figure out and return list of lists of redundant baseline pairs. Ordered by length. All baselines
    in a group have the same orientation with a preference for positive b_y and, when b_y==0, positive
    b_x where b((i,j)) = pos(j) - pos(i).

    Args:
        antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}. 1D and 2D also OK.
        bl_error_tol: the largest allowable difference between baselines in a redundant group
            (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.
        include_autos: bool, optional
            if True, include autos in the list of pos_reds. default is False
    Returns:
        reds: list (sorted by baseline legnth) of lists of redundant tuples of antenna indices (no polarizations),
        sorted by index with the first index of the first baseline the lowest in the group.
    """
    keys = list(antpos.keys())
    reds = {}
    assert np.all(
        [len(pos) <= 3 for pos in antpos.values()]
    ), "Get_pos_reds only works in up to 3 dimensions."
    ap = {
        ant: np.pad(pos, (0, 3 - len(pos)), mode="constant")
        for ant, pos in antpos.items()
    }  # increase dimensionality
    array_is_flat = np.all(
        np.abs(
            np.array(list(ap.values()))[:, 2] - np.mean(list(ap.values()), axis=0)[2]
        )
        < bl_error_tol / 4.0
    )
    p_or_m = (0, -1, 1)
    if array_is_flat:
        epsilons = [[dx, dy, 0] for dx in p_or_m for dy in p_or_m]
    else:
        epsilons = [[dx, dy, dz] for dx in p_or_m for dy in p_or_m for dz in p_or_m]

    def check_neighbors(
        delta,
    ):  # Check to make sure reds doesn't have the key plus or minus rounding error
        for epsilon in epsilons:
            newKey = (
                delta[0] + epsilon[0],
                delta[1] + epsilon[1],
                delta[2] + epsilon[2],
            )
            if newKey in reds:
                return newKey
        return

    for i, ant1 in enumerate(keys):
        if include_autos:
            start_ind = i
        else:
            start_ind = i + 1
        for ant2 in keys[start_ind:]:
            bl_pair = (ant1, ant2)
            delta = tuple(
                np.round(
                    1.0 * (np.array(ap[ant2]) - np.array(ap[ant1])) / bl_error_tol
                ).astype(int)
            )
            new_key = check_neighbors(delta)
            if new_key is None:  # forward baseline has no matches
                new_key = check_neighbors(tuple([-d for d in delta]))
                if new_key is not None:  # reverse baseline does have a match
                    bl_pair = (ant2, ant1)
            if (
                new_key is not None
            ):  # either the forward or reverse baseline has a match
                reds[new_key].append(bl_pair)
            else:  # this baseline is entirely new
                if (
                    delta[0] <= 0
                    or (delta[0] == 0 and delta[1] <= 0)
                    or (delta[0] == 0 and delta[1] == 0 and delta[2] <= 0)
                ):
                    delta = tuple([-d for d in delta])
                    bl_pair = (ant2, ant1)
                reds[delta] = [bl_pair]

    # sort reds by length and each red to make sure the first antenna of the first bl in each group is the lowest antenna number
    orderedDeltas = [
        delta
        for (length, delta) in sorted(
            zip([np.linalg.norm(delta) for delta in reds.keys()], reds.keys())
        )
    ]
    return [
        (
            sorted(reds[delta])
            if sorted(reds[delta])[0][0] == np.min(reds[delta])
            else sorted([reverse_bl(bl) for bl in reds[delta]])
        )
        for delta in orderedDeltas
    ]


def _validate_inputs(
    precision: int,
    polarized: bool,
    antpos: dict,
    eq2tops: np.ndarray,
    crd_eq: np.ndarray,
    Isky: np.ndarray,
):
    pass


def _evaluate_beam(
    beam,
    tx: np.ndarray,
    ty: np.ndarray,
    freqs: np.ndarray,
):
    # Primary beam pattern using direct interpolation of UVBeam object
    az, za = conversions.enu_to_az_za(enu_e=tx, enu_n=ty, orientation="uvbeam")
    beam_vals = beam.interp(az, za, freqs)[0][0, 1].T
    return beam_vals**2


def simulate_cpu(
    antpos: dict,
    freqs: np.ndarray,
    sources: np.ndarray,
    beam,
    crd_eq: np.ndarray,
    eq2tops: np.ndarray,
    precision: int = 1,
    polarized: bool = False,
    use_redundancy: bool = True,
    vectorize_times: bool = True,
    check: bool = False,
):
    """
    antpos : dict
        Dictionary of antenna positions
    freqs : np.ndarray
        Frequencies to evaluate visibilities at MHz.
    precision : int, optional
       Which precision level to use for floats and complex numbers
       Allowed values:
       - 1: float32, complex64
       - 2: float64, complex128
    use_redundancy : bool, default = True
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
    reds = get_pos_reds(antpos)
    baselines = [red[0] for red in reds]
    nbls = len(baselines)

    if use_redundancy:
        bl_to_red_map = {red[0]: np.array(red) for red in reds}

    # Convert to correct precision
    crd_eq = crd_eq.astype(real_dtype)
    eq2tops = eq2tops.astype(real_dtype)
    Isky = (0.5 * sources).astype(complex_dtype)

    # Compute coordinates
    blx, bly = np.array([antpos[bl[1]] - antpos[bl[0]] / 2.998e8 for bl in baselines])[
        :, :2
    ].T.astype(real_dtype)

    # Zero arrays:
    vis = np.full((nants, nants, ntimes, nfreqs), 0, dtype=complex_dtype)

    # Loop over time samples
    for ti, eq2top in enumerate(eq2tops):
        # Convert to topocentric coordinates
        tx, ty, tz = crd_top = np.dot(eq2top, crd_eq)

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
            u, v = blx * freqs[ni] / 2.998e8, bly * freqs[ni] / 2.998e8
            _vis[:, ni] = finufft.nufft2d3(
                2 * np.pi * ty,
                2 * np.pi * tx,
                np.ascontiguousarray(i_sky[:, ni]),
                u,
                v,
                modeord=0,
                eps=1e-6,
            )

        if use_redundancy:
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

    return vis


def simulate_basis(
    antpos: dict,
    freqs: np.ndarray,
    basis_comps: np.ndarray,
    basis: np.ndarray,
    beam,
    ra: np.ndarray,
    dec: np.ndarray,
    precision: int = 1,
):
    """
    Simulate the sky using some basis as simulation

    precision : int, optional
        Which precision level to use for floats and complex numbers
        Allowed values:
        - 1: float32, complex64
        - 2: float64, complex128
    """
    pass
