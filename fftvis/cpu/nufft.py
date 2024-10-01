import logging
import warnings
import numpy as np
import finufft

from ..core.nufft import BaseTransform
from ..utils import get_plane_to_xy_rotation_matrix
from astropy.constants import c

speed_of_light = c.value


class CPU_NUFFT(BaseTransform):
    """
    GPU implementation of the NUFFT transform.
    """

    def setup(self):
        """ """
        self.antvec = np.array([self.antpos[ant] for ant in self.antpos])
        self.rotation_matrix = get_plane_to_xy_rotation_matrix(self.antvec)

        if np.allclose(self.rotation_matrix, np.eye(3)):
            self.rotate_array = False

        # Compute the rotated antenna vectors
        self.rotated_antvec = np.dot(self.rotation_matrix.T, self.antvec.T).T
        blx, bly, blz = np.array(
            [
                self.rotated_antvec[ai] - self.rotated_antvec[aj]
                for (ai, aj) in self.antpairs
            ]
        ).T

        # Determine if the array is coplanar
        self.is_coplanar = np.all(np.less_equal(np.abs(blz), self.flat_array_tol))
        self.u, self.v, self.w = (
            self.freq * blx / speed_of_light,
            self.freq * bly / speed_of_light,
            self.freq * blz / speed_of_light,
        )

        # Allocate the output array
        self.vis = np.zeros(
            (self.nchunks, self.nfeeds * self.nfeeds, self.npairs), dtype=self.ctype
        )

    def compute(self, source_strength, tx, ty, tz, out):
        """ """
        if self.rotate_array:
            tx, ty, tz = np.dot(self.rot_matrix.T, np.array([tx, ty, tz]))

        if self.is_coplanar:
            finufft.nufft2d3(
                x=tx, y=ty, c=source_strength, s=self.u, t=self.v, eps=self.eps, out=out
            )
        else:
            finufft.nufft3d3(
                x=tx,
                y=ty,
                z=tz,
                c=source_strength,
                s=self.u,
                t=self.v,
                u=self.w,
                eps=self.eps,
                out=out,
            )
