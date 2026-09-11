"""Regression tests for the standalone A-projection profiling prototype.

These tests intentionally live beside the experiment rather than testing the
fftvis simulator API.  They establish the discrete Fourier/sign convention
that a future integrated implementation would need to preserve.
"""

import numpy as np

from aprojection_prototype import (
    apply_kernels,
    direct_visibilities,
    fftvis_style_visibilities,
    kernel_bank,
    make_apertures,
    make_layout,
    make_sky,
    matvis_style_visibilities,
    relative_errors,
    sky_uv_grid,
    voltage_beams_at_sources,
)


def test_full_aperture_kernel_matches_direct_horizon_sky():
    """A full aperture-limited kernel is exact even for near-horizon sources."""
    ngrid, du, aperture_radius = 64, 0.4, 1.0
    sky = make_sky(100, seed=3)
    layout = make_layout(4, array_radius_cells=7, seed=4)
    apertures = make_apertures(
        4,
        ngrid,
        du,
        aperture_radius,
        variation=0.8,
        seed=5,
    )
    full_radius = 2 * int(np.ceil(aperture_radius / du))
    kernels = kernel_bank(apertures, du, full_radius)
    approx = apply_kernels(
        sky_uv_grid(sky, ngrid, du, eps=1e-10),
        kernels,
        layout,
        baseline_batch=16,
    )
    reference = direct_visibilities(
        voltage_beams_at_sources(apertures, sky, du),
        sky,
        layout,
        du,
        source_mask=None,
        baseline_batch=16,
    )

    max_error, rms_error = relative_errors(approx, reference)
    assert np.rad2deg(sky.za.max()) > 89.0
    assert max_error < 1e-8
    assert rms_error < 1e-8

    matvis_style = matvis_style_visibilities(
        voltage_beams_at_sources(apertures, sky, du),
        sky,
        layout,
        du,
        source_mask=None,
    )
    max_error, rms_error = relative_errors(matvis_style, reference)
    assert max_error < 1e-12
    assert rms_error < 1e-12


def test_uniform_aperture_airy_case_matches_direct_horizon_sky():
    """A uniform aperture keeps the full-support identity at the horizon."""
    ngrid, du, aperture_radius = 64, 0.4, 1.0
    sky = make_sky(100, seed=11)
    layout = make_layout(4, array_radius_cells=7, seed=12)
    apertures = make_apertures(
        4,
        ngrid,
        du,
        aperture_radius,
        variation=0.2,
        seed=13,
        illumination="uniform",
    )
    full_radius = 2 * int(np.ceil(aperture_radius / du))
    approx = apply_kernels(
        sky_uv_grid(sky, ngrid, du, eps=1e-10),
        kernel_bank(apertures, du, full_radius),
        layout,
        baseline_batch=16,
    )
    reference = direct_visibilities(
        voltage_beams_at_sources(apertures, sky, du),
        sky,
        layout,
        du,
        source_mask=None,
        baseline_batch=16,
    )
    max_error, rms_error = relative_errors(approx, reference)
    assert np.rad2deg(sky.za.max()) > 89.0
    assert max_error < 1e-8
    assert rms_error < 1e-8

    fftvis_style = fftvis_style_visibilities(
        voltage_beams_at_sources(apertures, sky, du),
        sky,
        layout,
        du,
        eps=1e-10,
        nthreads=1,
        method="type1",
        upsample_factor=2.0,
    )
    max_error, rms_error = relative_errors(approx, fftvis_style)
    assert max_error < 1e-8
    assert rms_error < 1e-8
