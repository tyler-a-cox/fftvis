import numpy as np
from scipy import linalg

from math import lcm
from fractions import Fraction
from typing import Any, Dict, Tuple

IDEALIZED_BL_TOL = 1e-8  # bl_error_tol for redcal.get_reds when using antenna positions calculated from reds
speed_of_light = 299792458.0  # m/s

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

def get_pos_reds(antpos, decimals=3, include_autos=True):
    """
    Figure out and return list of lists of redundant baseline pairs. This function is a modified version of the
    get_pos_reds function in redcal. It is used to calculate the redundant baseline groups from antenna positions
    rather than from a list of baselines. This is useful for simulating visibilities with fftvis.

    Parameters:
    ----------
        antpos: dict
            dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
        decimals: int, optional
            Number of decimal places to round to when determining redundant baselines. default is 3.
        include_autos: bool, optional
            if True, include autos in the list of pos_reds. default is False
    Returns:
    -------
        reds: list of lists of redundant tuples of antenna indices (no polarizations),
        sorted by index with the first index of the first baseline the lowest in the group.
    """
    # Create a dictionary of redundant baseline groups
    uv_to_red_key = {}
    reds = {}

    # Compute baseline lengths and round to specified precision
    baselines = np.round(
        [
            antpos[aj] - antpos[ai]
            for ai in antpos
            for aj in antpos
            if ai < aj or include_autos and ai == aj
        ],
        decimals,
    )

    ci = 0
    for ai in antpos:
        for aj in antpos:
            if ai < aj or include_autos and ai == aj:
                u, v, _ = baselines[ci]

                if (u, v) not in uv_to_red_key and (-u, -v) not in uv_to_red_key:
                    reds[(ai, aj)] = [(ai, aj)]
                    uv_to_red_key[(u, v)] = (ai, aj)
                elif (-u, -v) in uv_to_red_key:
                    reds[uv_to_red_key[(-u, -v)]].append((aj, ai))
                elif (u, v) in uv_to_red_key:
                    reds[uv_to_red_key[(u, v)]].append((ai, aj))

                ci += 1

    reds_list = []
    for k in reds:
        red = reds[k]
        ant1, ant2 = red[0]
        _, bly, _ = antpos[ant2] - antpos[ant1]
        if bly < 0:
            reds_list.append([(bl[1], bl[0]) for bl in red])
        else:
            reds_list.append(red)

    return reds_list


def get_plane_to_xy_rotation_matrix(antvecs):
    """
    Compute the rotation matrix that projects the antenna positions onto the xy-plane.
    This function is used to rotate the antenna positions so that they lie in the xy-plane.

    Parameters:
    ----------
        antvecs: np.array
            Array of antenna positions in the form (Nants, 3).

    Returns:
    -------
        rotation_matrix: np.array
            Rotation matrix that projects the antenna positions onto the xy-plane of shape (3, 3).
    """
    # Fit a plane to the antenna positions
    antx, anty, antz = antvecs.T
    basis = np.array([antx, anty, np.ones_like(antz)]).T
    plane, res, rank, s = linalg.lstsq(basis, antz)

    # Project the antenna positions onto the plane
    slope_x, slope_y, z_offset = plane

    # Plane is already approximately aligned with the xy-axes,
    # return identity rotation matrix
    if np.isclose(slope_x, 0) and np.isclose(slope_y, 0.0):
        return np.eye(3)

    # Normalize the normal vector
    normal = np.array([slope_x, slope_y, -1])
    normal = normal / np.linalg.norm(normal)

    # Compute the rotation axis
    axis = np.array([slope_y, -slope_x, 0])
    axis = axis / np.linalg.norm(axis)

    # Compute the rotation angle
    theta = np.arccos(-normal[2])

    # Compute the rotation matrix using Rodrigues' formula
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rotation_matrix = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    return rotation_matrix


def get_task_chunks(
    nprocesses: int, nfreqs: int, ntimes: int
) -> tuple[int, list[slice], list[slice], int, int]:
    """Compute the optimal set of task chunks in time and frequency.

    Prefer putting more frequencies in each chunk rather than times, but scale as
    necessary.

    Parameters
    ----------
    nprocesses : int
        The number of processes that can be used.
    nfreqs : int
        The number of frequency channels required.
    ntimes
        The number of time integrations required.

    Returns
    -------
    nprocesses : int
        The number of processes to actually use (typically the same as the input,
        but can be returned as 1 if sub-process threading is better)
    freq_chunks : list of slices
        A length-nprocesses list of lists of integers, where each sublist defines
        a set of frequency channels to compute.
    time_chunks : list of slices
        A length-nprocesses list of lists of integers, where each sublist defines
        a set of time integrations to compute.
    nf : int
        The number of frequency channels per chunk.
    nt : int
        The number of time integrations per chunk.
    """
    ntasks = ntimes * nfreqs

    if ntasks < 2 * nprocesses:
        # Prefer lower-latency parallelization of the component calculations
        return 1, [slice(None)], [slice(None)], nfreqs, ntimes

    if ntimes >= nprocesses:
        freq_chunks = [list(range(nfreqs))] * nprocesses

    nt = int(np.ceil(ntimes / nprocesses))
    nf = nfreqs
    nfc = 1
    size = nf * nt
    sizes = [size]

    while nf > 1 and (nprocesses * size) > ntasks:
        nfc += 1
        nf = int(np.ceil(nfreqs / nfc))
        nt = int(np.ceil(ntimes / (nprocesses / nfc)))
        size = nf * nt
        sizes.append(size)

    idx = np.argmin(sizes)
    nfc = 1 + idx
    nf = int(np.ceil(nfreqs / nfc))
    nt = int(np.ceil(ntimes / (nprocesses / nfc)))

    ntc = int(np.ceil(nprocesses / nfc))
    freq_chunks = [slice(nf * i, min(nfreqs, (i + 1) * nf)) for i in range(nfc)] * ntc
    time_chunks = sum(
        ([slice(i * nt, min(ntimes, (i + 1) * nt))] * nfc for i in range(ntc)), start=[]
    )
    return nprocesses, freq_chunks, time_chunks, nf, nt


def inplace_rot_base(rot, b):
    """
    Base implementation of in-place rotation of coordinates.
    
    This is a reference implementation that will be optimized by
    CPU and GPU specific implementations.

    Parameters:
    ----------
    rot : np.ndarray
        3x3 rotation matrix
    b : np.ndarray
        Array of shape (3, n) containing coordinates to rotate
    """
    nsrc = b.shape[1]
    out = np.zeros(3, dtype=b.dtype)

    for n in range(nsrc):
        out[0] = rot[0, 0] * b[0, n] + rot[0, 1] * b[1, n] + rot[0, 2] * b[2, n]
        out[1] = rot[1, 0] * b[0, n] + rot[1, 1] * b[1, n] + rot[1, 2] * b[2, n]
        out[2] = rot[2, 0] * b[0, n] + rot[2, 1] * b[1, n] + rot[2, 2] * b[2, n]
        b[:, n] = out


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


def check_scaling(
    antpos: Dict[Any, np.ndarray],
    tol: float = 1e-9,
    max_denominator: int = 10**6
) -> Tuple[bool, Dict[Any, np.ndarray], np.ndarray]:
    """
    Check if antenna positions lie on an integer grid (up to scaling)
    in native or hex-rotated basis.

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