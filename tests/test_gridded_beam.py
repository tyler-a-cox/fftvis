"""Tests for sampling an analytic beam onto a grid.

``to_gridded_beam`` is what lets an analytic beam be interpolated on the GPU.
It is an approximation, so these tests check that it converges to the analytic
beam as the grid is refined, and that the interpolation-order plumbing behaves.
These run on CPU: the accuracy of the grid is a property of the grid, not of
the device that later interpolates it.
"""

import numpy as np
import pytest
from pyuvdata.analytic_beam import GaussianBeam
from pyuvdata.beam_interface import BeamInterface

from fftvis.core.beams import to_gridded_beam
from fftvis.cpu.beams import CPUBeamEvaluator
from fftvis.gpu.beams import GPUBeamEvaluator

FREQS = np.array([1.0e8, 1.5e8])


def _sample_directions(n=400, za_max=np.pi / 3, seed=0):
    """Directions well inside the grid, avoiding the horizon edge."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, n), rng.uniform(0, za_max, n)


def test_gridded_beam_is_a_uvbeam():
    """An analytic beam becomes a gridded az_za UVBeam."""
    gb = to_gridded_beam(GaussianBeam(diameter=14.0), FREQS, naz=181, nza=91)
    assert isinstance(gb, BeamInterface)
    assert gb._isuvbeam
    assert gb.beam.pixel_coordinate_system == "az_za"


def test_uvbeam_passes_through_unchanged():
    """An already-gridded beam is returned as-is, not re-sampled."""
    gb = to_gridded_beam(GaussianBeam(diameter=14.0), FREQS, naz=181, nza=91)
    again = to_gridded_beam(gb, FREQS, naz=31, nza=17)
    assert again is gb


@pytest.mark.parametrize("order", [1, 3])
def test_converges_to_analytic_beam(order):
    """Refining the grid drives the interpolation error down."""
    analytic = BeamInterface(GaussianBeam(diameter=14.0), beam_type="efield")
    az, za = _sample_directions()
    cpu = CPUBeamEvaluator()

    exact = cpu.evaluate_beam(analytic, az, za, True, FREQS[0])

    errs = []
    for naz, nza in [(91, 46), (361, 181), (721, 361)]:
        gridded = to_gridded_beam(analytic, FREQS, naz=naz, nza=nza)
        got = cpu.evaluate_beam(
            gridded, az, za, True, FREQS[0],
            spline_opts={"order": order},
            interpolation_function="az_za_map_coordinates",
        )
        errs.append(np.abs(got - exact).max() / np.abs(exact).max())

    assert errs[0] > errs[1] > errs[2], f"error not decreasing: {errs}"
    # Cubic on a 0.5-degree grid should be very close to the analytic beam.
    tol = 1e-4 if order == 3 else 1e-2
    assert errs[-1] < tol, f"order={order} converged only to {errs[-1]:.2e}"


def test_cubic_beats_linear_on_the_same_grid():
    """Order 3 is more accurate than order 1 at fixed grid resolution."""
    analytic = BeamInterface(GaussianBeam(diameter=14.0), beam_type="efield")
    az, za = _sample_directions()
    cpu = CPUBeamEvaluator()
    exact = cpu.evaluate_beam(analytic, az, za, True, FREQS[0])
    gridded = to_gridded_beam(analytic, FREQS, naz=181, nza=91)

    def err(order):
        got = cpu.evaluate_beam(
            gridded, az, za, True, FREQS[0],
            spline_opts={"order": order},
            interpolation_function="az_za_map_coordinates",
        )
        return np.abs(got - exact).max() / np.abs(exact).max()

    assert err(3) < err(1)


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5])
def test_device_path_accepts_supported_orders(order):
    """The auto gate admits every order the device path implements."""
    gridded = to_gridded_beam(GaussianBeam(diameter=14.0), FREQS, naz=91, nza=46)
    assert GPUBeamEvaluator()._can_use_gpu_interp(gridded, {"order": order})


def test_device_path_rejects_analytic_beams():
    """Analytic beams cannot be interpolated on the device."""
    analytic = BeamInterface(GaussianBeam(diameter=14.0), beam_type="efield")
    assert not GPUBeamEvaluator()._can_use_gpu_interp(analytic, {"order": 3})


def test_device_path_off_without_spline_opts():
    """With no spline_opts the host path is used, as before."""
    gridded = to_gridded_beam(GaussianBeam(diameter=14.0), FREQS, naz=91, nza=46)
    assert not GPUBeamEvaluator()._can_use_gpu_interp(gridded, None)
