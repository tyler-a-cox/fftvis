from . import utils

import jax
import numpy as np
from pyuvdata import UVBeam
from jax import numpy as jnp


c_ms = 299792458.0  # Speed of light in meters per second


def diameter_to_sigma(diameter, freqs):
    """
    Find the sigma that gives a beam width similar to an Airy disk.

    Find the stddev of a gaussian with fwhm equal to that of
    an Airy disk's main lobe for a given diameter. Similar to the
    `AnalyticBeam` class in pyuvsim, but in jax.

    Parameters
    ----------
    diameter : float
        Antenna diameter in meters
    freqs : array
        Frequencies in Hz

    Returns
    -------
    sigma : float
        The standard deviation in zenith angle radians for a Gaussian beam
        with FWHM equal to that of an Airy disk's main lobe for an aperture
        with the given diameter.
    """
    wavelengths = c_ms / freqs

    scalar = 2.2150894  # Found by fitting a Gaussian to an Airy disk function

    return jnp.arcsin(scalar * wavelengths / (jnp.pi * diameter)) * 2 / 2.355


@jax.jit
def gaussian_beam(
    az_array,
    za_array,
    freqs,
    diameter,
    spectral_index: float = 0.0,
    ref_freq: float = 1e9,
):
    """
    A jax implementation of a gaussian beam.

    Parameters
    ----------
    az_array : jnp.ndarray
        Azimuth angle in radians.
    za_array : jnp.ndarray
        Zenith angle in radians.
    freqs : jnp.ndarray
        Frequencies in Hz.
    sigma : float
        The standard deviation of the Gaussian beam in radians.
    spectral_index : float
        The spectral index of the beam pattern. Defaults to 0.
    ref_freq : float
        The reference frequency in Hz. Defaults to 1 GHz.
    """
    # Calculate the sigma for each frequency
    sigmas = diameter_to_sigma(diameter, freqs)

    # Calculate the beam pattern
    return jnp.exp(
        -jnp.square(za_array[jnp.newaxis]) / (2 * jnp.square(sigmas[:, jnp.newaxis]))
    )


@jax.jit
def airy_beam(
    az_array,
    za_array,
    freqs,
    diameter,
    spectral_index: float = 0.0,
    ref_freq: float = 1e9,
):
    """
    A jax implementation of the airy beam function.

    Parameters
    ----------
    az_array : jnp.ndarray
        Azimuth angle in radians.
    za_array : jnp.ndarray
        Zenith angle in radians.
    freqs : jnp.ndarray
        Frequencies in Hz.
    diameter : float
        The diameter of the antenna in meters.
    spectral_index : float
        The spectral index of the beam pattern. Defaults to 0.
    ref_freq : float
        The reference frequency in Hz. Defaults to 1 GHz.
    """
    za_grid, f_grid = jnp.meshgrid(za_array, freqs)
    xvals = diameter / 2.0 * jnp.sin(za_grid) * 2.0 * np.pi * f_grid / c_ms
    return jnp.where(xvals != 0, 2.0 * utils.j1(xvals) / xvals, 1.0)


def beam_decomposition(
    az_array,
    za_array,
    freqs,
    diameter,
    spectral_index: float = 0.0,
    ref_freq: float = 1e9,
):
    """
    Generate a basis of beam patterns.
    """
    pass


def interpolate_beam(
    beam_vals,
    az_array,
    za_array,
    freqs,
    freq_interp_kind="cubic",
):
    """
    Interpolate the beam values to the given frequencies.

    Parameters
    ----------
    beam_vals : np.ndarray
    az_array : np.ndarray
        Azimuth angle in radians.
    za_array : np.ndarray
        Zenith angle in radians.
    freqs : np.ndarray
        Frequencies in Hz.

    Returns
    -------
    beam_vals : array
        The beam values at the given frequencies.
    """
    raise NotImplementedError("Interpolation not yet implemented.")


def _evaluate_beam(
    A_s: np.ndarray,
    beam_list: list[UVBeam],
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
        Array of shape (nax, nfeed, nbeam, nsrcs_up) that will be filled with beam
        values.
    beam_list
        List of unique beams.
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
    for i, bm in enumerate(beam_list):
        kw = (
            {
                "reuse_spline": True,
                "check_azza_domain": False,
                "spline_opts": spline_opts,
            }
            if isinstance(bm, UVBeam)
            else {}
        )
        if isinstance(bm, UVBeam) and not bm.future_array_shapes:
            bm.use_future_array_shapes()

        interp_beam = bm.interp(
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

        A_s[:, :, i] = interp_beam

        # Check for invalid beam values
        if check:
            sm = np.sum(A_s)
            if np.isinf(sm) or np.isnan(sm):
                raise ValueError("Beam interpolation resulted in an invalid value")

    return A_s
