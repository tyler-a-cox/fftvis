import numpy as np
from math import lcm
from fractions import Fraction
from typing import Any, Dict, Tuple


def find_integer_multiplier(
    arr: np.ndarray,
    max_denominator: int = 10**6
) -> int:
    """
    Return the smallest positive integer f such that f * arr[i] is integral
    for all entries (up to rational approximation). Zeros in arr are ignored.

    Parameters
    ----------
    arr : np.ndarray
        Array of values to check.
    max_denominator : int
        Maximum denominator for the fractions.

    Returns
    -------
    f : int
        The integer factor used for scaling.
    """
    fracs = [
        Fraction(val).limit_denominator(max_denominator)
        for val in np.ravel(arr)
        if val != 0
    ]
    if not fracs:
        return 1
    dens = (frac.denominator for frac in fracs)
    return lcm(*dens)


def can_scale_to_int(
    arr: np.ndarray,
    tol: float = 1e-9,
    max_denominator: int = 10**6,
    max_factor: int = None
) -> Tuple[bool, int]:
    """
    Check if there exists an integer factor f such that f * arr is
    approximately integers. Returns (is_griddable, factor).

    Parameters
    ----------
    arr : np.ndarray
        Array of values to check.
    tol : float
        Tolerance for checking if values are close to integers in units of meters.
    max_denominator : int
        Maximum denominator for the fractions.
    max_factor : int
        Maximum allowed factor for scaling.

    Returns
    -------
    is_griddable : bool
        True if the array can be scaled to integers, False otherwise.
    factor : int
        The integer factor used for scaling.
    """
    f = find_integer_multiplier(arr, max_denominator)
    if max_factor is not None and f > max_factor:
        return False, f
    scaled = f * np.array(arr, dtype=float)
    if np.allclose(scaled, np.round(scaled), atol=tol):
        return True, f
    return False, f

def find_lattice_basis(
    antpos: Dict[Any, np.ndarray],
    tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the lattice basis for the antenna positions. This function assumes that 
    the antenna positions are in a 2D plane (x, y) and that the z-coordinate is
    not relevant for the basis calculation. The function computes the pairwise
    differences between the antenna positions and selects two non-collinear
    vectors to form the basis. If all vectors are collinear, it returns a default
    basis.

    Parameters
    ----------
    antpos : dict
        Dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}
        where the antenna positinons are in units of meters.
    tol : float
        Tolerance for checking if values are close to integers in units of meters.
    max_denominator : int
        Maximum denominator for the fractions.

    Returns
    -------
    basis : np.ndarray
        The lattice basis.
    transform_matrix : np.ndarray
        The transformation matrix.
    """
    antvecs = np.array([antpos[ant][:2] for ant in antpos], dtype=float)

    # pairwise differences
    blvec = np.reshape(antvecs[:, None, :] - antvecs[None, :, :], (-1, 2))

    # filter out zeros
    norms = np.linalg.norm(blvec, axis=1)
    mask = norms > tol
    blvec = blvec[mask]
    norms = norms[mask]

    # sort by ascending length
    order = np.argsort(norms)
    blvec = blvec[order]

    # Pick the shortest non-zero baseline as a basis vector
    basis_vec_1 = blvec[0]

    # find second that’s not collinear (cross‑product ≠ 0)
    for v in blvec[1:]:
        # Compute the cross product of the x and y components
        cross_x = basis_vec_1[0]*v[1] - basis_vec_1[1]*v[0]
        if np.abs(cross_x) > tol:
            basis_vec_2 = v
            break
    else:
        # If all are collinear, use the shortest baseline
        return np.vstack([basis_vec_1, np.array([0, 1])])

    return np.column_stack([basis_vec_1, basis_vec_2])

def check_antpos_griddability(
    antpos: Dict[Any, np.ndarray],
    tol: float = 1e-9,
    max_denominator: int = 10**6,
    max_factor: int = 1000,
) -> Tuple[bool, Dict[Any, np.ndarray], np.ndarray]:
    """
    Check if the antenna positions can be gridded. This function checks if the
    antenna positions can be scaled to integers using a linear transformation.
    It first infers the 2D lattice basis from the antenna positions, then checks
    if the positions can be scaled to integers. If they can, it returns the
    modified antenna positions and the transformation matrix. If not, it returns
    the original antenna positions and an identity transformation matrix.

    Parameters
    ----------
    antpos : dict
        Dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}
        where the antenna positinons are in units of meters.
    tol : float
        Tolerance for checking if values are close to integers in units of meters.
    max_denominator : int
        Maximum denominator for the fractions.
    max_factor : int
        Maximum allowed factor for scaling. If the factor exceeds this value,
        the function will return False.

    Returns:
    -------
    is_griddable : bool
        True if the antenna positions can be gridded, False otherwise.
    modified_antpos : dict
        The modified antenna positions after scaling.
    transform_matrix : np.ndarray
        The transformation matrix used to scale the antenna positions.
        This is a 3x3 matrix that transforms the original coordinates
        to the grid coordinates.
    """
    # Map antenna keys to index and stack positions
    antkey_to_idx = dict(zip(antpos.keys(), range(len(antpos))))
    antvecs = np.array([antpos[ant] for ant in antpos], dtype=float)

    # Infer the 2D lattice basis from the antenna positions
    basis_2D = find_lattice_basis(
        antpos,
        tol=tol,
    )
    basis = np.zeros((3, 3))
    basis[:2, :2] = basis_2D
    basis[2, 2] = 1.0

    # Rotate the basis to align with the inferred grid
    modified_antvecs = np.linalg.solve(
        basis,
        (antvecs - antvecs[0]).T
    ).T

    # Scale the inferred lattice basis to integers
    is_griddable, factor = can_scale_to_int(
        np.ravel(modified_antvecs),
        tol=tol,
        max_denominator=max_denominator,
        max_factor=max_factor,
    )

    if is_griddable:
        # Scale the antenna positions, rounding to integers
        antpos = {
            ant: np.round(
                factor * modified_antvecs[antkey_to_idx[ant]]
            ).astype(int)
            for ant in antpos
        }
        return True, antpos, basis / factor
    else:
        # If not griddable, check 
        return False, antpos, np.eye(antvecs.shape[-1])