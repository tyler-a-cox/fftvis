import jax
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


def gaussian_beam(freqs, diameter, spectral_index, ref_freq):
    """
    A jax implementation of a gaussian beam.

    Parameters
    ----------
    type : str
        The type of beam pattern. Currently only 'gaussian' is supported.
    sigma : float
        The standard deviation of the Gaussian beam in radians.
    spectral_index : float
        The spectral index of the beam pattern. Defaults to 0.
    ref_freq : float
        The reference frequency in Hz. Defaults to 1 GHz.
    """
    raise NotImplementedError("Gaussian beam not yet implemented.")


def airy_beam(
    az_array,
    za_array,
    freqs,
    diameter,
    spectral_index,
    ref_freq,
):
    """
    A jax implementation of the airy beam function.

    Parameters
    ----------
    az_array : np.ndarray
        Azimuth angle in radians.
    za_array : np.ndarray
        Zenith angle in radians.
    freqs : np.ndarray
        Frequencies in Hz.
    ref_freq : float
        The reference frequency in Hz. Defaults to 1 GHz.
    """
    raise NotImplementedError("Airy beam not yet implemented.")


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
