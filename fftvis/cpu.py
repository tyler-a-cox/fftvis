from __future__ import annotations

import finufft
import numpy as np
from typing import Callable

from hera_cal import redcal


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
    beam: Callable,
    tx: np.ndarray,
    ty: np.ndarray,
    polarized: bool,
    freqs: np.ndarray,
):
    pass


def simulate(
    antpos: dict,
    freqs: np.ndarray,
    sources: np.ndarray,
    beam: Callable,
    crd_eq: np.ndarray,
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
        complex_dtype = np.complex128
    else:
        real_dtype = np.float64
        complex_dtype = np.complex128

    # Get the redundant - TODO handle this better
    reds = redcal.get_pos_reds(antpos)
    baselines = [red[0] for red in reds]
    nbls = len(baselines)

    if use_redundancy:
        bl_to_red_map = {red[0]: np.array(red) for red in reds}

    # Convert to correct precision
    crd_eq = crd_eq.astype(real_dtype)
    eq2tops = eq2tops.astype(real_dtype)
    Isky = (0.5 * sources).astype(real_dtype)

    # Compute coordinates
    blx, bly = np.array([antpos[bl[1]] - antpos[bl[0]] for bl in baselines])[:, :2].T

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
        i_sky = Isky[above_horizon] * _evaluate_beam(beam, tx, ty, freqs)

        _vis = np.full((nbls, nfreqs), 0, dtype=complex_dtype)
        for ni in enumerate(nfreqs):
            u, v = blx * freqs[ni] / 2.998e8, bly * freqs[ni] / 2.998e8
            _vis[:, ni] = finufft.nufft2d3(tx, ty, i_sky[:, ni], u, v)

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
    beam: Callable,
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
