"""Tests."""

import numpy as np
from astropy import units as un
from astropy.coordinates import EarthLocation, Latitude, Longitude
from astropy.time import Time
from astropy.units import Quantity
from pathlib import Path
from pyradiosky import SkyModel
from pyuvdata import UVBeam
from pyuvsim import AnalyticBeam, simsetup
from pyuvsim.telescope import BeamList

from matvis import DATA_PATH, conversions

nfreq = 1
ntime = 5  # 20
nants = 3  # 4
nsource = 15  # 250
beam_file = DATA_PATH / "NF_HERA_Dipole_small.fits"


def get_standard_sim_params(
    use_analytic_beam: bool,
    polarized: bool,
    nants=nants,
    nfreq=nfreq,
    ntime=ntime,
    nsource=nsource,
    first_source_antizenith=False,
):
    """Create some standard random simulation parameters for use in tests."""
    hera_lat = -30.7215
    hera_lon = 21.4283
    hera_alt = 1073.0
    obstime = Time("2018-08-31T04:02:30.11", format="isot", scale="utc")

    # HERA location
    location = EarthLocation.from_geodetic(lat=hera_lat, lon=hera_lon, height=hera_alt)

    np.random.seed(1)

    # Beam model
    if use_analytic_beam:
        n_freq = nfreq
        beam = AnalyticBeam("gaussian", diameter=14.0)
    else:
        n_freq = min(nfreq, 2)
        # This is a peak-normalized e-field beam file at 100 and 101 MHz,
        # downsampled to roughly 4 square-degree resolution.
        beam = UVBeam()
        beam.read_beamfits(beam_file)
        if not polarized:
            uvsim_beam = beam.copy()
            beam.efield_to_power(calc_cross_pols=False, inplace=True)
            beam.select(polarizations=["xx"], inplace=True)

        # Now, the beam we have on file doesn't actually properly wrap around in azimuth.
        # This is fine -- the uvbeam.interp() handles the interpolation well. However, when
        # comparing to the GPU interpolation, which first has to interpolate to a regular
        # grid that ends right at 2pi, it's better to compare like for like, so we
        # interpolate to such a grid here.
        beam = beam.interp(
            az_array=np.linspace(0, 2 * np.pi, 181),
            za_array=np.linspace(0, np.pi / 2, 46),
            az_za_grid=True,
            new_object=True,
        )

    if polarized or use_analytic_beam:
        uvsim_beams = BeamList([beam] * nants)
    else:
        uvsim_beams = BeamList([uvsim_beam] * nants)

    # beams = [AnalyticBeam('uniform') for i in range(len(ants.keys()))]
    beam_dict = {str(i): i for i in range(nants)}

    # Random antenna locations
    x = np.random.random(nants) * 400.0  # Up to 400 metres
    y = np.random.random(nants) * 400.0
    z = np.random.random(nants) * 0.0
    ants = {i: (x[i], y[i], z[i]) for i in range(nants)}

    # Observing parameters in a UVData object
    uvdata = simsetup.initialize_uvdata_from_keywords(
        Nfreqs=n_freq,
        start_freq=100e6,
        channel_width=97.3e3,
        start_time=obstime.jd,
        integration_time=182.0,  # Just over 3 mins between time samples
        Ntimes=ntime,
        array_layout=ants,
        polarization_array=np.array(["XX", "YY", "XY", "YX"]),
        telescope_location=(hera_lat, hera_lon, hera_alt),
        telescope_name="test_array",
        phase_type="drift",
        vis_units="Jy",
        complete=True,
        write_files=False,
    )
    lsts = np.unique(uvdata.lst_array)

    # One fixed source plus random other sources
    sources = [
        [
            300 if first_source_antizenith else 125.7,
            -30.72,
            2,
            0,
        ],  # Fix a single source near zenith
    ]
    if nsource > 1:  # Add random other sources
        ra = np.random.uniform(low=0.0, high=360.0, size=nsource - 1)
        dec = -30.72 + np.random.random(nsource - 1) * 10.0
        flux = np.random.random(nsource - 1) * 4
        sources.extend([ra[i], dec[i], flux[i], 0] for i in range(nsource - 1))
    sources = np.array(sources)

    # Source locations and frequencies
    ra_dec = np.deg2rad(sources[:, :2])
    freqs = np.unique(uvdata.freq_array)

    # Correct source locations so that matvis uses the right frame
    ra_new, dec_new = conversions.equatorial_to_eci_coords(
        ra_dec[:, 0], ra_dec[:, 1], obstime, location, unit="rad", frame="icrs"
    )

    # Calculate source fluxes for matvis
    flux = ((freqs[:, np.newaxis] / freqs[0]) ** sources[:, 3].T * sources[:, 2].T).T

    # Stokes for the first frequency only. Stokes for other frequencies
    # are calculated later.
    stokes = np.zeros((4, 1, ra_dec.shape[0]))
    stokes[0, 0] = sources[:, 2]
    reference_frequency = np.full(len(ra_dec), freqs[0])

    # Set up sky model
    sky_model = SkyModel(
        name=[str(i) for i in range(len(ra_dec))],
        ra=Longitude(ra_dec[:, 0], "rad"),
        dec=Latitude(ra_dec[:, 1], "rad"),
        frame="icrs",
        spectral_type="spectral_index",
        spectral_index=sources[:, 3],
        stokes=stokes * un.Jy,
        reference_frequency=Quantity(reference_frequency, "Hz"),
    )

    # Calculate stokes at all the frequencies.
    sky_model.at_frequencies(Quantity(freqs, "Hz"), inplace=True)

    return (
        sky_model,
        ants,
        flux,
        ra_new,
        dec_new,
        freqs,
        lsts,
        [beam],
        uvsim_beams,
        beam_dict,
        hera_lat,
        uvdata,
    )
