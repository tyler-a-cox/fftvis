import numpy as np
from pyuvdata import UVBeam
from fast_interp import interp2d


def _evaluate_beam(
    A_s: np.ndarray,
    beam: UVBeam,
    az: np.ndarray,
    za: np.ndarray,
    polarized: bool,
    freq: float,
    check: bool = False,
    spline_opts: dict = None,
    interpolation_function: str = "az_za_map_coordinates",
):
    """Evaluate the beam on the CPU. Simplified version of the `_evaluate_beam_cpu` function
    in matvis.

    This function will either interpolate the beam to the given coordinates tx, ty,
    or evaluate the beam there if it is an analytic beam.

    Parameters
    ----------
    A_s
        Array of shape (nax, nfeed, nsrcs_up) that will be filled with beam
        values.
    beam
        UVBeam object to evaluate.
    tx, ty
        Coordinates to evaluate the beam at, in sin-projection.
    polarized
        Whether to use beam polarization.
    freq
        Frequency to interpolate beam to.
    check
        Whether to check that the beam has no inf/nan values. Set to False if you are
        sure that the beam is valid, as it will be faster.
    spline_opts
        Extra options to pass to the RectBivariateSpline class when interpolating.
    interpolation_function
        The interpolation function to use when interpolating the beam. Can be either be
        'az_za_simple' or 'az_za_map_coordinates'. The former is slower but more accurate
        at the edges of the beam, while the latter is faster but less accurate
        for interpolation orders greater than linear. 
    """
    # Primary beam pattern using direct interpolation of UVBeam object
    kw = (
        {
            "reuse_spline": True,
            "check_azza_domain": False,
            "spline_opts": spline_opts,
            "interpolation_function": interpolation_function,
        }
        if isinstance(beam, UVBeam)
        else {}
    )
    if isinstance(beam, UVBeam) and not beam.future_array_shapes:
        beam.use_future_array_shapes()

    interp_beam = beam.interp(
        az_array=az,
        za_array=za,
        freq_array=np.atleast_1d(freq),
        **kw,
    )[0]

    if polarized:
        interp_beam = interp_beam[:, :, 0, :]
    else:
        # Here we have already asserted that the beam is a power beam and
        # has only one polarization, so we just evaluate that one.
        interp_beam = np.sqrt(interp_beam[0, 0, 0, :])

    A_s[:, :] = interp_beam

    # Check for invalid beam values
    if check:
        sm = np.sum(A_s)
        if np.isinf(sm) or np.isnan(sm):
            raise ValueError("Beam interpolation resulted in an invalid value")

    return A_s

class ParallelBeamInterpolator:
    def __init__(self, beam: UVBeam, freq_ind: int, order: int = 1):
        """
        Parameters
        ----------
        beam
        """
        self.data_array = beam.data_array[:, :, freq_ind]
        self.freq_array = beam.freq_array
        az_min, az_max = np.min(beam.axis1_array), np.max(beam.axis1_array)
        za_min, za_max = np.min(beam.axis2_array), np.max(beam.axis2_array)
        daz = np.diff(beam.axis1_array)[0]
        dza = np.diff(beam.axis2_array)[0]
        
        self.complex_beam = np.issubdtype(self.data_array.dtype, np.complexfloating)

        # Beam interpolation object
        self.interp_objs_real = [
            interp2d(
                a=(az_min, za_min),
                b=(az_max, za_max),
                h=(daz, dza),
                f=np.copy(self.data_array[ax_ind, feed_ind].real.T), # numba is very picky about this copy
                k=order,
            )
            for ax_ind in range(self.data_array.shape[0])
            for feed_ind in range(self.data_array.shape[1])
        ]

        if self.complex_beam:
            self.interp_objs_imag = [
                interp2d(
                    a=(az_min, za_min),
                    b=(az_max, za_max),
                    h=(daz, dza),
                    f=np.copy(self.data_array[ax_ind, feed_ind].imag.T), # numba is very picky about this copy
                    k=order,
                )
                for ax_ind in range(self.data_array.shape[0])
                for feed_ind in range(self.data_array.shape[1])
            ]

    def interp(self, A_s: np.ndarray, az: np.ndarray, za: np.ndarray, polarized: bool, check: bool = False):
        """
        Parameters
        ----------
        az
        za
        freq
        """
        #for i, interp_obj in enumerate(self.interp_objs_real):
        
        ci = 0
        for ax_ind in range(self.data_array.shape[0]):
            for feed_ind in range(self.data_array.shape[1]):
                interp_obj = self.interp_objs_real[ci]
                if self.complex_beam:
                    interp_obj_imag = self.interp_objs_imag[ci]
                    interp_beam = interp_obj(az, za) + 1j * interp_obj_imag(az, za)
                else:
                    interp_beam = interp_obj(az, za)

                if not polarized:
                    interp_beam = np.sqrt(interp_beam)

                A_s[ax_ind, feed_ind] = interp_beam
                ci += 1

        # Check for invalid beam values
        if check:
            sm = np.sum(A_s)
            if np.isinf(sm) or np.isnan(sm):
                raise ValueError("Beam interpolation resulted in an invalid value")

        return A_s