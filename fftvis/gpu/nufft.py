import logging
import warnings
import numpy as np
from ..utils import get_plane_to_xy_rotation_matrix
from astropy.constants import c

speed_of_light = c.value

try:
    import cupy as cp

    HAVE_CUDA = True

except ImportError:
    # if not installed, don't warn
    HAVE_CUDA = False
except Exception as e:  # pragma: no cover
    # if installed but having initialization issues
    # warn, but default back to non-gpu functionality
    warnings.warn(str(e), stacklevel=2)
    HAVE_CUDA = False

try:
    import cufinufft
except Exception as e:  # pragma: no cover
    # if installed but having initialization issues
    # warn, but default back to non-gpu functionality
    warnings.warn(str(e), stacklevel=2)
    HAVE_CUDA = False

from ..core.nufft import BaseTransform


class GPU_NUFFT(BaseTransform):
    """
    GPU implementation of the NUFFT transform.
    """

    def setup(self):
        """ """
        self.antvec = cp.array([self.antpos[ant] for ant in self.antpos])
        self.rotation_matrix = get_plane_to_xy_rotation_matrix(self.antvec)

        if cp.allclose(self.rotation_matrix, cp.eye(3)):
            self.rotate_array = False

        # Compute the rotated antenna vectors
        self.rotated_antvec = cp.dot(self.rotation_matrix.T, self.antvec.T).T
        blx, bly, blz = cp.array(
            [
                self.rotated_antvec[ai] - self.rotated_antvec[aj]
                for (ai, aj) in self.antpairs
            ]
        ).T

        # Determine if the array is coplanar
        self.is_coplanar = cp.all(cp.less_equal(cp.abs(blz), self.flat_array_tol))
        self.u, self.v, self.w = (
            self.freq * blx / speed_of_light,
            self.freq * bly / speed_of_light,
            self.freq * blz / speed_of_light,
        )

        # Allocate the output array
        self.vis = cp.zeros(
            (self.nchunks, self.nfeeds * self.nfeeds, self.npairs), dtype=self.ctype
        )

    def compute(self, source_strength, tx, ty, tz, out):
        """ """
        if self.rotate_array:
            tx, ty, tz = cp.dot(self.rot_matrix.T, cp.array([tx, ty, tz]))

        if self.is_coplanar:
            cufinufft.nufft2d3(
                x=2 * cp.pi * tx,
                y=2 * cp.pi * ty,
                c=source_strength,
                s=self.u,
                t=self.v,
                eps=self.eps,
                out=out,
            )
        else:
            cufinufft.nufft3d3(
                x=2 * cp.pi * tx,
                y=2 * cp.pi * ty,
                z=2 * cp.pi * tz,
                c=source_strength,
                s=self.u,
                t=self.v,
                u=self.w,
                eps=self.eps,
                out=out,
            )
