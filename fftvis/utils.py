import numpy as np
from scipy import linalg
IDEALIZED_BL_TOL = 1e-8  # bl_error_tol for redcal.get_reds when using antenna positions calculated from reds
speed_of_light = 299792458.0  # m/s


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