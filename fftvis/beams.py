from . import utils

import jax
import numpy as np
from pyuvdata import UVBeam
from jax import numpy as jnp
from jax import lax
from jax._src.typing import Array, ArrayLike
from functools import partial

import jax_finufft

# Set the jax configuration to use 64-bit precision
jax.config.update("jax_enable_x64", True)

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


@partial(jax.jit, static_argnums=(2, 3, 4))
def gaussian_beam(
    za_array: ArrayLike,
    freqs: ArrayLike,
    diameter: float = 14.6,
    spectral_index: float = 0.0,
    ref_freq: float = 150e6,
) -> ArrayLike:
    """
    A jax implementation of a gaussian beam function. The beam pattern is assumed to be azimuthally symmetric.

    Parameters
    ----------
    za_array : jnp.ndarray
        Zenith angle in radians. This is the angle from the zenith to the point on the sky.
    freqs : jnp.ndarray
        Frequencies in Hz. The beam pattern will be calculated for each frequency.
    diameter : float, default=14.6
        The standard deviation of the Gaussian beam in radians.
    spectral_index : float, default=0.0
        The spectral index of the beam pattern. Defaults to 0.
    ref_freq : float, default=150e6
        The reference frequency in Hz. Defaults to 150e6 Hz.
    """
    # Calculate the sigma for each frequency
    sigmas = diameter_to_sigma(diameter, freqs)

    # Apply the spectral index
    sigmas = sigmas * (freqs / ref_freq) ** spectral_index

    # Calculate the beam pattern from the zenith angle
    return jnp.exp(
        -jnp.square(za_array[jnp.newaxis]) / (2 * jnp.square(sigmas[:, jnp.newaxis]))
    )


@partial(jax.jit, static_argnums=(2,))
def airy_beam(
    za_array: ArrayLike,
    freqs: ArrayLike,
    diameter: float = 14.6,
) -> ArrayLike:
    """
    A jax implementation of a gaussian beam function. The beam pattern is assumed to be azimuthally symmetric.

    Parameters
    ----------
    za_array : jnp.ndarray
        Zenith angle in radians. This is the angle from the zenith to the point on the sky.
    freqs : jnp.ndarray
        Frequencies in Hz. The beam pattern will be calculated for each frequency.
    diameter : float
        The diameter of the antenna in meters.

    Returns
    -------
    airy_beam : jnp.ndarray
        The beam pattern for the given antenna diameter and frequencies.
    """
    za_grid, f_grid = jnp.meshgrid(za_array, freqs)
    xvals = diameter / 2.0 * jnp.sin(za_grid) * 2.0 * np.pi * f_grid / c_ms
    return jnp.where(xvals != 0, 2.0 * utils.j1(xvals) / xvals, 1.0)


def uv_beam_interpolation(
    beam: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    topocentric_x: ArrayLike,
    topocentric_y: ArrayLike,
) -> ArrayLike:
    """
    Interpolate the beam values to a given set of sky coordinates, topocentric_x and topocentric_y,
    given a beam pattern in the uv-plane. This function uses the jax_finufft library to
    perform the interpolation from the uv-plane to the sky. Assumes that the beam is centered
    at the origin of the uv-plane.

    Parameters
    ----------
    beam : array
        The beam pattern in the uv-plane.
    u : array
        The u-coordinates in the uv-plane.
    v : array
        The v-coordinates in the uv-plane.
    topocentric_x : array
        Coordinate x of the beam pixel in topocentric coordinates.
    topocentric_y : array
        Coordinate y of the beam pixel in topocentric coordinates.

    Returns:
    --------
    sky_beam : array
        The beam pattern interpolated to the given sky coordinates.
    """
    # Get the maximum u and v values to scale the coordinates
    u_max = jnp.max(u)
    v_max = jnp.max(v)

    # Scale the coordinates to the range [-pi, pi]
    scaled_topocentric_x = topocentric_x * jnp.pi / u_max
    scaled_topocentric_y = topocentric_y * jnp.pi / v_max

    # Compute the 2D fourier transform from uv-space to pixel space
    sky_beam = jax_finufft.nufft2(
        u, v, beam, scaled_topocentric_x, scaled_topocentric_y
    )

    return sky_beam


@partial(jax.jit, static_argnums=(3, 4, 5))
def bessel_beam_decomposition(
    beam_vals: ArrayLike,
    az_array: ArrayLike,
    za_array: ArrayLike,
    n_radial_modes: int = 50,
    n_azimuthal_modes: int = 4,
    alpha: float = 1.01,
):
    """
    Compute the Bessel beam decomposition of the beam pattern according to
    the method outlined in Wilensky et al. (2024).

    Parameters
    ----------
    beam_vals : jnp.ndarray
        The beam values at the given frequencies. This is a 2D array of shape
        (nfreqs, npix).
    az_array : jnp.ndarray
        Azimuth angle in radians.
    za_array : jnp.ndarray
        Zenith angle in radians.
    n_radial_modes : int, default=50
        The number of radial modes to use in the decomposition.
    n_azimuthal_modes : int, default=4
        The number of azimuthal modes to use in the decomposition.
    alpha : float, default=1.01
        The scaling factor for the Bessel beam decomposition.

    Returns
    -------
    beam_components : jnp.ndarray
        The Bessel beam coefficients for the given beam pattern of shape (nfreqs, radial_modes, azimuthal_modes).
    """
    # Compute phi from the zenith angle
    phi = jnp.sqrt(1 - jnp.sin(za_array)) / alpha

    # Compute the Bessel beam decomposition
    # number of basis components will be assumed to be static across frequency
    u_n = utils.jn_zeros(0, n_radial_modes)
    q_n = jnp.sqrt(jnp.pi) * utils.j1(u_n)
    azimuthal_modes = jnp.cos(jnp.arange(n_azimuthal_modes) * az_array)

    bessel_basis = (
        utils.j0(phi * u_n[:, jnp.newaxis])
        / q_n[:, jnp.newaxis]
        * azimuthal_modes[jnp.newaxis, jnp.newaxis]
    )

    # Compute the Bessel basis coefficients for the radial modes
    beam_components = jnp.einsum("zra,fz->fra", bessel_basis, beam_vals)

    return beam_components


@partial(jax.jit, static_argnums=(3,))
def bessel_beam_interpolation(
    beam_components: ArrayLike,
    az_array,
    za_array,
    alpha: float = 1.01,
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
    beam: UVBeam,
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
