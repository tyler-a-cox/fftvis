"""Shared pytest fixtures and GPU gating for the fftvis test suite."""

import pytest

# Monkey patch pyuvdata.telescopes before matvis is imported anywhere.
import pyuvdata.telescopes

if not hasattr(pyuvdata.telescopes, "get_telescope"):
    from pyuvdata import Telescope

    def get_telescope(telescope_name, **kwargs):
        """Compatibility shim to build a Telescope from its name."""
        return Telescope.from_known_telescopes(telescope_name, **kwargs)

    pyuvdata.telescopes.get_telescope = get_telescope


def _gpu_available() -> bool:
    """Whether cupy, cufinufft and an actual device are all present."""
    try:
        import cupy as cp
        import cufinufft  # noqa: F401
    except ImportError:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pragma: no cover - driver present but unusable
        return False


HAVE_GPU = _gpu_available()

requires_gpu = pytest.mark.skipif(
    not HAVE_GPU, reason="requires cupy, cufinufft and a CUDA device"
)
