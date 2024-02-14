import jax
from jax import numpy as jnp

c_ms = 299792458.0  # Speed of light in meters per second


def diameter_to_sigma(diameter, freqs):
    """
    Find the sigma that gives a beam width similar to an Airy disk.

    Find the stddev of a gaussian with fwhm equal to that of
    an Airy disk's main lobe for a given diameter.

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
def gaussian_beam(freqs, diameter, spectral_index, ref_freq):
    """
    A jax implementation of the AnalyticBeam class from pyuvsim.

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
    pass
