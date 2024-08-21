import numpy as np
from pyuvdata import UVBeam

from ..coordinates import enu_to_az_za
from ..core.beams import BeamInterpolator

class CPUBeamInterpolator(BeamInterpolator):
    """"""
    def interp(self, tx: np.ndarray, ty: np.ndarray, out: np.ndarray) -> np.ndarray:
        """
        Method
        """
        # Interpolate beam
        az, za = enu_to_az_za(tx, ty, orientation="uvbeam")

        kw = (
            {
                "reuse_spline": True,
                "check_azza_domain": False,
                "spline_opts": self.spline_opts,
            }
            if isinstance(self.beam, UVBeam)
            else {}
        )

        if isinstance(self.beam, UVBeam) and not self.beam.future_array_shapes:
            self.beam.use_future_array_shapes()

        interp_beam = self.beam.interp(
            az_array=az,
            za_array=za,
            freq_array=np.atleast_1d(self.freq),
            **kw,
        )[0]

        if self.polarized:
            interp_beam = np.transpose(interp_beam[..., 0, :], (1, 0, 2))
        else:
            interp_beam = np.sqrt(interp_beam[0, 0, 0])

        out[:] = interp_beam