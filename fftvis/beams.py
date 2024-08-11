import numpy as np
from pyuvsim import AnalyticBeam
from pyuvdata import UVBeam


def _evaluate_beam(
    A_s: np.ndarray,
    beam: UVBeam | AnalyticBeam,
    az: np.ndarray,
    za: np.ndarray,
    polarized: bool,
    freq: float,
    check: bool = False,
    spline_opts: dict = None,
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
    """
    # Primary beam pattern using direct interpolation of UVBeam object
    kw = (
        {
            "reuse_spline": True,
            "check_azza_domain": False,
            "spline_opts": spline_opts,
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
