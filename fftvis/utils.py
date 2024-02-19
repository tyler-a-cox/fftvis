import numpy as np

IDEALIZED_BL_TOL = 1e-8  # bl_error_tol for redcal.get_reds when using antenna positions calculated from reds
speed_of_light = 299792458.0  # m/s


def get_pos_reds(antpos, decimals=3, include_autos=True):
    """Figure out and return list of lists of redundant baseline pairs. Ordered by length. All baselines
    in a group have the same orientation with a preference for positive b_y and, when b_y==0, positive
    b_x where b((i,j)) = pos(j) - pos(i).

    Parameters:
    ----------
        antpos: dict
            dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}. 1D and 2D also OK.
        decimals: int, optional
            Number of decimal places to round to when determining redundant baselines. default is 3.
        include_autos: bool, optional
            if True, include autos in the list of pos_reds. default is False
    Returns:
    -------
        reds: list (sorted by baseline legnth) of lists of redundant tuples of antenna indices (no polarizations),
        sorted by index with the first index of the first baseline the lowest in the group.
    """

    uv_to_red_key = {}
    reds = {}

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

    return [reds[k] for k in reds]
