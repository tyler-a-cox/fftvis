import finufft
import numpy as np
from abc import ABC, abstractmethod
from typing import Any

from matvis._utils import get_dtypes

try:
    import cupy as cp

    HAVE_CUDA = True
except ImportError:
    HAVE_CUDA = False


class BaseTransform(ABC):
    """
    Base class for the NUFFT transform.
    """

    def __init__(
        self,
        antpos: dict,
        freq: float,
        nfeed: int,
        nchunks: int,
        antpairs: np.ndarray | None = None,
        precision: int = 2,
        eps=None,
        flat_array_tol: float = 0.0,
        **options_params,
    ):
        """
        Base class for the NUFFT transform.
        """
        self.antpos = antpos
        self.freq = freq
        self.nants = len(antpos)
        self.nfeeds = nfeed
        self.nchunks = nchunks
        self.flat_array_tol = flat_array_tol
        self.antkey_to_index = {ant: i for i, ant in enumerate(antpos)}

        if antpairs is not None:
            self.antpairs = antpairs
        else:
            self.antpairs = [(ant1, ant2) for ant1 in antpos for ant2 in antpos]

        self.npairs = len(self.antpairs)

        # Get the desired precision if not provided
        # TODO: should probably remove this and just use the default precision
        if not eps:
            eps = 1e-13 if precision == 2 else 6e-8

        self.eps = eps
        self.options = options_params
        self.rtype, self.ctype = get_dtypes(precision)

    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        """
        Perform the Non-Uniform Fast Fourier Transform.
        """

    def sum_chunks(self, out=None):
        """
        Sum the chunks into the output array.

        Parameters
        ----------
        out
            The output visibilities, with shape (Nfeed, Nfeed, Npairs).
        """
        if self.nchunks == 1:
            out[:] = self.vis[0]
        else:
            self.vis.sum(axis=0, out=out)

    def __call__(self, source_strength, tx, ty, tz, chunk) -> Any:
        self.compute(source_strength, tx=tx, ty=ty, tz=tz, out=self.vis[chunk])
        return self.vis[chunk]


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
    xp = cp if HAVE_CUDA else np

    # Fit a plane to the antenna positions
    antx, anty, antz = antvecs.T
    basis = xp.array([antx, anty, xp.ones_like(antz)]).T
    plane, _, _, _ = xp.linalg.lstsq(basis, antz)

    # Project the antenna positions onto the plane
    slope_x, slope_y, _ = plane

    # Plane is already approximately aligned with the xy-axes,
    # return identity rotation matrix
    if xp.isclose(slope_x, 0) and xp.isclose(slope_y, 0.0):
        return xp.eye(3)

    # Normalize the normal vector
    normal = xp.array([slope_x, slope_y, -1])
    normal = normal / xp.linalg.norm(normal)

    # Compute the rotation axis
    axis = xp.array([slope_y, -slope_x, 0])
    axis = axis / xp.linalg.norm(axis)

    # Compute the rotation angle
    theta = xp.arccos(-normal[2])

    # Compute the rotation matrix using Rodrigues' formula
    K = xp.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rotation_matrix = xp.eye(3) + xp.sin(theta) * K + (1 - xp.cos(theta)) * xp.dot(K, K)

    return rotation_matrix
