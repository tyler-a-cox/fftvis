"""Core abstract class for coordinate rotation."""

import numpy as np

from .._utils import get_dtypes

try:
    import cupy as cp

    HAVE_CUDA = True
except ImportError:
    HAVE_CUDA = False


class CoordinateRotation:
    """
    Abstract class for coordinate rotation.

    Parameters
    ----------
    flux
        Flux of each source. Shape=(Nsrc,).
    crd_eq
        Equatorial coordinates of each source. Shape=(3, Nsrc).
    eq2top
        Set of 3x3 transformation matrices to rotate the RA and Dec
        cosines in an ECI coordinate system (see `crd_eq`) to
        topocentric coordinates. Shape=(Nt, 3, 3).
    chunk_size
        Number of sources to rotate at a time.
    precision
        The precision of the data (1 or 2).
    """

    def __init__(
        self,
        flux: np.ndarray,
        crd_eq: np.ndarray,
        eq2top: np.ndarray,
        chunk_size: int | None = None,
        source_buffer: float = 0.55,
        precision: int = 1,
        gpu: bool = False,
    ):
        self.rtype, _ = get_dtypes(precision)
        self.flux = flux.astype(self.rtype)
        self.coords_eq = crd_eq.astype(self.rtype)
        self.eq2top = eq2top.astype(self.rtype)
        self.nsrc = len(flux)

        assert flux.ndim == 1
        assert crd_eq.ndim == 2
        assert eq2top.ndim == 3
        self.chunk_size = chunk_size or self.nsrc
        self.source_buffer = source_buffer
        self.nsrc_alloc = int(self.chunk_size * self.source_buffer)

        self.gpu = gpu
        if self.gpu and not HAVE_CUDA:
            raise ValueError("GPU requested but cupy not installed.")

        self.xp = cp if self.gpu else np

    def setup(self):
        """Allocate memory for the rotation."""
        self.all_coords_topo = self.xp.full((3, self.nsrc), 0.0, dtype=self.rtype)
        self.coords_above_horizon = self.xp.full(
            (3, self.nsrc_alloc), 0.0, dtype=self.rtype
        )
        self.flux_above_horizon = self.xp.full(
            (self.nsrc_alloc,), 0.0, dtype=self.rtype
        )

        self.eq2top = self.xp.asarray(self.eq2top)
        self.coords_eq = self.xp.asarray(self.coords_eq)
        self.flux = self.xp.asarray(self.flux)

    def select_chunk(self, chunk: int):
        """Set the chunk of coordinates to rotate."""
        # The last index can be larger than the actual size of the array without error.
        slc = slice(chunk * self.chunk_size, (chunk + 1) * self.chunk_size)

        topo = self.all_coords_topo[:, slc]
        flux = self.flux[slc]

        above_horizon = self.xp.where(topo[2] > 0)[0]
        n = len(above_horizon)
        if n > self.nsrc_alloc:
            raise ValueError(
                f"nsrc_alloc ({self.nsrc_alloc}) is too small for the number of "
                f"sources above horizon ({n}). Try increasing source_buffer."
            )

        self.coords_above_horizon[:, :n] = topo[:, above_horizon]
        self.flux_above_horizon[:n] = flux[above_horizon]
        self.flux_above_horizon[n:] = 0

        if self.gpu:
            self.xp.cuda.Device().synchronize()

        return self.coords_above_horizon, self.flux_above_horizon, n

    def rotate(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Rotate the given coordinates with the given 3x3 rotation matrix.

        Parameters
        ----------
        crd
            Coordinates to rotate. Shape=(3, Nsrc).
        rot
            Rotation matrix. Shape=(3, 3).

        Returns
        -------
        np.ndarray
            Rotated coordinates. Shape=(3, Nsrcs_above_horizon).
        np.ndarray
            Flux. Shape=(Nsrcs_above_horizon,).
        """
        self.xp.matmul(self.eq2top[t], self.coords_eq, out=self.all_coords_topo)

        if self.gpu:
            self.xp.cuda.Device().synchronize()