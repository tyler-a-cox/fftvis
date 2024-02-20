import numpy as np

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

    return [reds[k] for k in reds]
