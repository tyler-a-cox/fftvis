import numpy as np
from math import lcm
from fractions import Fraction
from typing import Any, Dict, Tuple


# Hexagonal grid rotation matrix
def get_hex_rot_matrix():
    """
    Returns the rotation matrix for hexagonal grid.
    The rotation matrix is used to rotate the antenna positions
    to align with the hexagonal grid.
    """
    # Rotation matrix to grid a hexagonal array
    return np.array([
        [1.0, 0.5, 0.0],
        [0.0, np.sqrt(3) / 2, 0.0],
        [0.0, 0.0, 1.0]
    ])

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
        Tolerance for checking if values are close to integers.
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


def check_antpos_griddability(
    antpos: Dict[Any, np.ndarray],
    tol: float = 1e-9,
    max_denominator: int = 10**6,
    rotation_matrix: np.ndarray = None
) -> Tuple[bool, Dict[Any, np.ndarray], np.ndarray]:
    """
    Check if antenna positions lie on an integer grid (up to scaling)
    in native or hex-rotated basis.

    Parameters
    ----------
    antpos : dict
        Dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
    tol : float
        Tolerance for checking if values are close to integers.
    max_denominator : int
        Maximum denominator for the fractions.
    rotation_matrix : np.ndarray
        Rotation matrix to apply to the antenna positions.

    Returns (is_griddable, modified_antpos, transform_matrix).
    """
    # Map antenna keys to index and stack positions
    antkey_to_idx = dict(zip(antpos.keys(), range(len(antpos))))
    antvecs = np.array([antpos[ant] for ant in antpos], dtype=float)

    # Compute all baselines and their magnitudes
    blvec = np.array([
        antpos[j] - antpos[i]
        for i in antpos for j in antpos
    ])
    blmag = np.linalg.norm(blvec, axis=-1)
    nonzero = blmag[blmag > 0]
    minimum_bl_spacing = nonzero.min()
    max_bl_spacing = nonzero.max()

    # Normalize antenna vectors
    antvecs -= antvecs[0]
    antvecs /= minimum_bl_spacing

    # If requested, rotate the antenna positions 
    if rotation_matrix is not None:
        antvecs_rot = (rotation_matrix @ antvecs.T).T
        
        is_griddable, factor = can_scale_to_int(
            np.ravel(antvecs_rot),
            tol=tol,
            max_denominator=max_denominator,
            max_factor=int(max_bl_spacing)
        )
        if not is_griddable:
            raise ValueError(
                "Antenna positions are not griddable in the rotated basis."
            )
        else:   
            modified_antpos = {
                key: antvecs_rot[idx] * factor
                for key, idx in antkey_to_idx.items()
            }
            return True, modified_antpos, np.linalg.inv(M_hex)

    # Try native grid
    is_griddable, factor = can_scale_to_int(
        np.ravel(antvecs),
        tol=tol,
        max_denominator=max_denominator,
        max_factor=int(max_bl_spacing)
    )
    if is_griddable:
        modified_antpos = {
            key: antvecs[idx] * factor
            for key, idx in antkey_to_idx.items()
        }
        return True, modified_antpos, np.eye(antvecs.shape[-1])

    # Try hexagonal grid
    M_hex = get_hex_rot_matrix()
    antvecs_rot = (M_hex.T @ antvecs.T).T
    is_griddable, factor = can_scale_to_int(
        np.ravel(antvecs_rot),
        tol=tol,
        max_denominator=max_denominator,
        max_factor=int(max_bl_spacing)
    )
    if is_griddable:
        modified_antpos = {
            key: antvecs_rot[idx] * factor
            for key, idx in antkey_to_idx.items()
        }
        return True, modified_antpos, np.linalg.inv(M_hex)

    # Not griddable
    return False, antpos, np.eye(antvecs.shape[-1])