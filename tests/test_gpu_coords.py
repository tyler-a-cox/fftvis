"""The GPU engine must construct GPU coordinate rotators with ``gpu=True``.

matvis's ``GPUCoordinateRotationERFA`` launches cupy kernels directly on the
arrays its base class allocates. Those arrays only live on the device when the
rotator is built with ``gpu=True``; without it the kernel launch fails with
``TypeError: You are trying to pass a numpy.ndarray ... as a kernel parameter``.

These tests use a stub rotator so they run without cupy or a device.
"""

import numpy as np
import pytest
from matvis.core.coords import CoordinateRotation

from fftvis.gpu import gpu_simulate
from fftvis.gpu.gpu_simulate import GPUSimulationEngine


@pytest.fixture(autouse=True)
def _pretend_cuda_is_available(monkeypatch):
    """Get past the import guard.

    Every test here aborts inside the coordinate-rotator constructor, which the
    engine reaches before touching cupy, so no device is needed.
    """
    monkeypatch.setattr(gpu_simulate, "HAVE_CUDA", True)


class _RecordingRotator(CoordinateRotation):
    """Records the kwargs it was constructed with, then aborts the run."""

    requires_gpu = True
    last_kwargs: dict = {}

    class Abort(Exception):
        """Raised to stop the simulation once the kwargs are captured."""

    def __init__(self, **kwargs):
        type(self).last_kwargs = dict(kwargs)
        raise _RecordingRotator.Abort()

    def rotate(self, t):  # pragma: no cover - never reached
        raise NotImplementedError


class _RecordingCPURotator(_RecordingRotator):
    """Same, but without ``requires_gpu``."""

    requires_gpu = False


def _params(polarized_sky: bool):
    """Minimal simulate() kwargs; the run aborts before any compute."""
    nsrc, nfreq = 4, 1
    fluxes = (
        np.ones((nsrc, nfreq, 4)) if polarized_sky else np.ones((nsrc, nfreq))
    )
    return dict(
        ants={0: np.array([0.0, 0.0, 0.0]), 1: np.array([10.0, 0.0, 0.0])},
        freqs=np.array([1e8]),
        fluxes=fluxes,
        beam_list=[None],  # never evaluated; the run aborts first
        ra=np.zeros(nsrc),
        dec=np.zeros(nsrc),
        times=np.linspace(2459000.0, 2459000.1, 2),
        telescope_loc=None,
        polarized=True,
    )


def test_gpu_rotator_gets_gpu_true():
    """A ``requires_gpu`` rotator is constructed with ``gpu=True``."""
    engine = GPUSimulationEngine()
    with pytest.raises(_RecordingRotator.Abort):
        engine.simulate(coord_method="_RecordingRotator", **_params(False))
    assert _RecordingRotator.last_kwargs.get("gpu") is True


def test_cpu_rotator_does_not_get_gpu_kwarg():
    """A host rotator is constructed without a ``gpu`` kwarg, as before."""
    engine = GPUSimulationEngine()
    with pytest.raises(_RecordingCPURotator.Abort):
        engine.simulate(coord_method="_RecordingCPURotator", **_params(False))
    assert "gpu" not in _RecordingCPURotator.last_kwargs


def test_gpu_rotator_with_polarized_sky_raises():
    """GPU rotator + polarized sky model is rejected with a clear message."""
    engine = GPUSimulationEngine()
    with pytest.raises(ValueError, match="polarized sky model"):
        engine.simulate(coord_method="_RecordingRotator", **_params(True))


def test_user_supplied_coord_params_survive():
    """``coord_method_params`` are not clobbered by the injected ``gpu`` kwarg."""
    engine = GPUSimulationEngine()
    with pytest.raises(_RecordingRotator.Abort):
        engine.simulate(
            coord_method="_RecordingRotator",
            coord_method_params={"update_bcrs_every": 1e9},
            **_params(False),
        )
    assert _RecordingRotator.last_kwargs.get("gpu") is True
    assert _RecordingRotator.last_kwargs.get("update_bcrs_every") == 1e9
