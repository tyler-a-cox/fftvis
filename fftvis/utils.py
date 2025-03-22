import numpy as np
from scipy import linalg
IDEALIZED_BL_TOL = 1e-8  # bl_error_tol for redcal.get_reds when using antenna positions calculated from reds
speed_of_light = 299792458.0  # m/s
import numba as nb

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
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    rotation_matrix = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    
    return rotation_matrix

def get_task_chunks(nprocesses: int, nfreqs: int, ntimes: int) -> tuple[int, list[slice], list[slice]]:
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
    """
    ntasks = ntimes * nfreqs

    if ntasks < 2*nprocesses:
        # Prefer lower-latency parallelization of the component calculations
        return 1, [slice(None)], [slice(None)], nfreqs, ntimes
        
    
    if ntimes >= nprocesses:
        freq_chunks = [list(range(nfreqs))]*nprocesses
    
    nt = int(np.ceil(ntimes / nprocesses))
    nf = nfreqs
    nfc = 1
    size = nf*nt
    sizes = [size]
    
    while nf > 1 and (nprocesses*size) > ntasks:
        nfc += 1
        nf = int(np.ceil(nfreqs / nfc))
        nt = int(np.ceil(ntimes / (nprocesses/nfc)))
        size = nf*nt
        sizes.append(size)
        
    idx = np.argmin(sizes)
    nfc = 1 + idx
    nf = int(np.ceil(nfreqs / nfc))
    nt = int(np.ceil(ntimes / (nprocesses/nfc)))
    
    ntc = int(np.ceil(nprocesses/nfc))
    freq_chunks = [slice(nf*i, min(nfreqs, (i + 1)*nf)) for i in range(nfc)]*ntc
    time_chunks = sum(([slice(i*nt, min(ntimes, (i + 1)*nt))]*nfc for i in range(ntc)), start=[])
    return nprocesses, freq_chunks, time_chunks, nf, nt

def stokes_to_coherency(stokes: np.ndarray) -> np.ndarray:
    """
    Convert Stokes parameters to coherency matrix.

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters in the form [I, Q, U, V].
    
    Returns
    -------
    coherency : np.ndarray
        Coherency matrix.
    """
    coherency = 0.5 * np.array(
        [
            [stokes[..., 0] + stokes[..., 3], stokes[..., 1] + 1j * stokes[..., 2]],
            [stokes[..., 1] - 1j * stokes[..., 2], stokes[..., 0] - stokes[..., 3]]
        ]
    )
    return coherency

@nb.jit(nopython=True)
def inplace_rot(rot: np.ndarray, b: np.ndarray):  # pragma: no cover
    """In-place rotation of coordinates."""
    nsrc = b.shape[1]
    out = np.zeros(3, dtype=b.dtype)
    
    for n in range(nsrc):
        out[0] = rot[0, 0]*b[0, n] + rot[0, 1]*b[1, n] + rot[0, 2]*b[2, n]
        out[1] = rot[1, 0]*b[0, n] + rot[1, 1]*b[1, n] + rot[1, 2]*b[2, n]
        out[2] = rot[2, 0]*b[0, n] + rot[2, 1]*b[1, n] + rot[2, 2]*b[2, n]
        b[:, n] = out