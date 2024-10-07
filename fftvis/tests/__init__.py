"""Tests."""

import numpy as np
from astropy import units as un
from astropy.coordinates import EarthLocation, Latitude, Longitude
from astropy.time import Time
from astropy.units import Quantity
from pyradiosky import SkyModel
from pyuvdata import UVBeam
from pyuvdata.telescopes import Telescope
from pyuvsim import AnalyticBeam, simsetup
from pyuvsim.telescope import BeamList

from matvis import DATA_PATH, coordinates

nfreq = 1
ntime = 1  # 20
nants = 2  # 4
nsource = 1
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
    hera = Telescope.from_known_telescopes('hera')
    obstime = Time("2018-08-31T04:02:30.11", format="isot", scale="utc")

    # HERA location
    location = hera.location

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
        telescope_location=(
            float(hera.location.lat.deg),
            float(hera.location.lon.deg),
            float(hera.location.height.to_value("m")),
        ),
        telescope_name="HERA",
        phase_type="drift",
        vis_units="Jy",
        complete=False,
        write_files=False,
    )
    times = Time(np.unique(uvdata.time_array), format="jd")
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
    ra_new, dec_new = coordinates.equatorial_to_eci_coords(
        ra_dec[:, 0], ra_dec[:, 1], obstime, location, unit="rad", frame="icrs"
    )

    # Calculate source fluxes for matvis
    flux = ((freqs[:, np.newaxis] / freqs[0]) ** sources[:, 3].T * sources[:, 2].T).T

    # Stokes for the first frequency only. Stokes for other frequencies
    # are calculated later.
    stokes = np.zeros((4, 1, ra_dec.shape[0]))
    stokes[0, 0] = sources[:, 2]

    return (
        ants,
        flux,
        ra_new,
        dec_new,
        freqs,
        lsts,
        times,
        [beam],
        location,
    )
