import numpy as np
import numba as nb


@nb.jit(nopython=True)
def inplace_rot(rot: np.ndarray, b: np.ndarray):  # pragma: no cover
    """
    CPU implementation of in-place rotation of coordinates using Numba.

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
