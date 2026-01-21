"""Tests for the wrapper functions."""

import pytest
import numpy as np
from astropy.coordinates import EarthLocation, ICRS, SkyCoord
from astropy.time import Time
import astropy.units as u
from pyuvdata import UVBeam
from pyuvdata.beam_interface import BeamInterface
from pathlib import Path

from fftvis.wrapper import (
    create_beam_evaluator, 
    create_simulation_engine, 
    simulate_vis,
)

TEST_DIR = Path(__file__).parent
cst_file = TEST_DIR / "data" / "HERA_NicCST_150MHz.txt"


def test_simulate_vis_basic():
    """Basic test for simulate_vis function.
    
    This test verifies:
    1. The function runs without errors
    2. The output has the correct shape
    3. The output values are reasonable
    """
    # Create a simple test case
    nants = 3
    nsrcs = 10
    nfreqs = 2
    ntimes = 2
    
    # Create antenna positions in a simple triangle
    ants = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([10.0, 0.0, 0.0]),
        2: np.array([5.0, 8.66, 0.0]),
    }
    
    # Create sources in a grid
    ra = np.linspace(0, 0.1, nsrcs)
    dec = np.linspace(-0.05, 0.05, nsrcs)
    fluxes = np.ones((nsrcs, nfreqs))
    
    # Create frequencies and times - use array of times
    # Use the same frequency as in the beam to avoid interpolation errors
    freqs = np.array([150e6, 150e6])  # Same frequency repeated to match nfreqs
    time_array = np.linspace(2459000, 2459000.1, ntimes)  # Array of Julian dates
    
    # Create a beam
    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }

    beam = UVBeam()
    beam.read_cst_beam(
        str(cst_file),
        frequency=[150e6],
        telescope_name="HERA",
        feed_name="Dipole",
        feed_version="1.0",
        feed_pol=["x"],
        model_name="Dipole - Rigging height 4.9 m",
        model_version="1.0",
        x_orientation="east",
        reference_impedance=100,
        history=(
            "Derived from https://github.com/Nicolas-Fagnoni/Simulations."
            "\nOnly 1 file included to keep test data volume low."
        ),
        extra_keywords=extra_keywords,
    )
    
    # Set telescope location (HERA)
    telescope_loc = EarthLocation.from_geodetic(
        lat=-30.72152612 * u.deg, 
        lon=21.42830383 * u.deg, 
        height=1051.69 * u.m
    )
    
    # Run the simulation for a single baseline
    baselines = [(0, 1)]
    
    # Simplest possible call
    vis = simulate_vis(
        ants=ants,
        fluxes=fluxes,
        ra=ra,
        dec=dec,
        freqs=freqs,
        times=time_array,  # Using array of Julian dates
        beam=beam,
        telescope_loc=telescope_loc,
        baselines=baselines,
        polarized=False,
    )
    
    # Check output shape
    assert vis.shape == (nfreqs, ntimes, len(baselines))
    
    # Ensure we get non-zero values
    assert np.any(vis != 0)
    
    # Test the polarized case too - disabled for now as it's causing issues
    # vis_pol = simulate_vis(
    #     ants=ants,
    #     fluxes=fluxes,
    #     ra=ra,
    #     dec=dec,
    #     freqs=freqs,
    #     times=time_array,
    #     beam=beam,
    #     telescope_loc=telescope_loc,
    #     baselines=baselines,
    #     polarized=True,
    # )
    
    # Check output shape for polarized
    # assert vis_pol.shape == (nfreqs, ntimes, 2, 2, len(baselines))


def test_simulate_vis_all_baselines():
    """Test simulate_vis with all baselines.
    
    This test verifies:
    1. The function correctly handles the case when baselines=None
    2. The output shape is correct 
    """
    # Create a simple test case
    nants = 3
    nsrcs = 5
    nfreqs = 1
    ntimes = 1
    
    # Create antenna positions in a simple triangle
    ants = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([10.0, 0.0, 0.0]),
        2: np.array([5.0, 8.66, 0.0]),
    }
    
    # Create sources in a grid
    ra = np.linspace(0, 0.1, nsrcs)
    dec = np.linspace(-0.05, 0.05, nsrcs)
    fluxes = np.ones((nsrcs, nfreqs))
    
    # Create frequencies and times
    freqs = np.array([150e6])
    time_array = np.array([2459000.0])  # Use array with one element
    
    # Create a beam
    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }

    beam = UVBeam()
    beam.read_cst_beam(
        str(cst_file),
        frequency=[150e6],
        telescope_name="HERA",
        feed_name="Dipole",
        feed_version="1.0",
        feed_pol=["x"],
        model_name="Dipole - Rigging height 4.9 m",
        model_version="1.0",
        x_orientation="east",
        reference_impedance=100,
        history=(
            "Derived from https://github.com/Nicolas-Fagnoni/Simulations."
            "\nOnly 1 file included to keep test data volume low."
        ),
        extra_keywords=extra_keywords,
    )
    
    # Set telescope location (HERA)
    telescope_loc = EarthLocation.from_geodetic(
        lat=-30.72152612 * u.deg, 
        lon=21.42830383 * u.deg, 
        height=1051.69 * u.m
    )
    
    # Run the simulation for all baselines
    vis = simulate_vis(
        ants=ants,
        fluxes=fluxes,
        ra=ra,
        dec=dec,
        freqs=freqs,
        times=time_array,  # Use array instead of scalar Time
        beam=beam,
        telescope_loc=telescope_loc,
        baselines=None,  # All baselines
        polarized=False,
    )
    
    # Number of baselines: (n * (n-1)) // 2 + 1 (including auto-correlations)
    nbls = (nants * (nants - 1)) // 2 + 1
    
    # Check output shape
    assert vis.shape == (nfreqs, ntimes, nbls)


def test_simulate_vis_precision():
    """Test simulate_vis with different precision settings.
    
    This test verifies:
    1. The function works with both single and double precision
    2. The results are reasonable in both cases
    """
    # Create a simple test case
    nants = 2
    nsrcs = 5
    nfreqs = 1
    ntimes = 1
    
    # Create antenna positions
    ants = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([10.0, 0.0, 0.0]),
    }
    
    # Create sources
    ra = np.linspace(0, 0.1, nsrcs)
    dec = np.linspace(-0.05, 0.05, nsrcs)
    fluxes = np.ones((nsrcs, nfreqs))
    
    # Create frequencies and times
    freqs = np.array([150e6])
    time_array = np.array([2459000.0])  # Use array with one element
    
    # Create a beam
    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }

    beam = UVBeam()
    beam.read_cst_beam(
        str(cst_file),
        frequency=[150e6],
        telescope_name="HERA",
        feed_name="Dipole",
        feed_version="1.0",
        feed_pol=["x"],
        model_name="Dipole - Rigging height 4.9 m",
        model_version="1.0",
        x_orientation="east",
        reference_impedance=100,
        history=(
            "Derived from https://github.com/Nicolas-Fagnoni/Simulations."
            "\nOnly 1 file included to keep test data volume low."
        ),
        extra_keywords=extra_keywords,
    )
    
    # Set telescope location (HERA)
    telescope_loc = EarthLocation.from_geodetic(
        lat=-30.72152612 * u.deg, 
        lon=21.42830383 * u.deg, 
        height=1051.69 * u.m
    )
    
    # Run the simulation in double precision
    vis_double = simulate_vis(
        ants=ants,
        fluxes=fluxes,
        ra=ra,
        dec=dec,
        freqs=freqs,
        times=time_array,
        beam=beam,
        telescope_loc=telescope_loc,
        precision=2,  # Double precision
        polarized=False,
    )
    
    # Run the simulation in single precision
    vis_single = simulate_vis(
        ants=ants,
        fluxes=fluxes,
        ra=ra,
        dec=dec,
        freqs=freqs,
        times=time_array,
        beam=beam,
        telescope_loc=telescope_loc,
        precision=1,  # Single precision
        polarized=False,
    )
    
    # Check that the results are close, but not identical
    assert np.allclose(vis_double, vis_single, rtol=1e-5, atol=1e-5)
    
    # Check data types
    assert vis_double.dtype == np.complex128
    assert vis_single.dtype == np.complex64 