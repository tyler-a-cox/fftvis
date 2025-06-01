import cupy as cp


def inplace_rot(rot: cp.ndarray, b: cp.ndarray):  # pragma: no cover
    """
    GPU implementation of in-place rotation of coordinates using CuPy.

    Parameters:
    ----------
    rot : cp.ndarray
        3x3 rotation matrix (on GPU)
    b : cp.ndarray
        Array of shape (3, n) containing coordinates to rotate (on GPU)
    """
    # Ensure inputs are cupy arrays
    if not isinstance(rot, cp.ndarray):
        rot = cp.asarray(rot)
    if not isinstance(b, cp.ndarray):
        b = cp.asarray(b)

    # Perform matrix multiplication in-place
    cp.matmul(rot, b, out=b)
