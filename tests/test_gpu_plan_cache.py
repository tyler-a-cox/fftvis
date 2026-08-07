"""Tests for the type-1 plan cache keying, runnable without a GPU.

The correctness of the cached transform is covered by ``test_gpu_nufft.py`` on
a real device. What can be checked anywhere is the cache *key*: it must include
everything a cufinufft plan is fixed by, and must not include the number of
nonuniform points -- that omission is the whole reason one plan can serve every
chunk of a run.
"""

import numpy as np
import pytest

from fftvis.gpu import nufft as gpu_nufft


class _FakePlan:
    """Stands in for cufinufft.Plan; records how it was built."""

    built = 0

    def __init__(self, nufft_type, n_modes, **kwargs):
        type(self).built += 1
        self.nufft_type = nufft_type
        self.n_modes = n_modes
        self.kwargs = kwargs


@pytest.fixture
def fake_cufinufft(monkeypatch):
    """Swap in a fake cufinufft.Plan and start from an empty cache."""
    _FakePlan.built = 0
    gpu_nufft.clear_plan_cache()
    monkeypatch.setattr(
        gpu_nufft, "cufinufft", type("m", (), {"Plan": _FakePlan})
    )
    yield _FakePlan
    gpu_nufft.clear_plan_cache()


def test_same_config_reuses_one_plan(fake_cufinufft):
    """Repeated calls with identical parameters build the plan once."""
    for _ in range(5):
        gpu_nufft._type1_plan(21, 4, 1e-9, np.complex64, 2.0)
    assert fake_cufinufft.built == 1
    assert len(gpu_nufft._TYPE1_PLANS) == 1


@pytest.mark.parametrize(
    "changed",
    [
        dict(n_modes=33),
        dict(n_trans=1),
        dict(eps=1e-6),
        dict(cdtype=np.complex128),
        dict(upsampfac=1.25),
    ],
)
def test_every_plan_parameter_is_in_the_key(fake_cufinufft, changed):
    """Changing any plan-defining parameter forces a new plan."""
    base = dict(n_modes=21, n_trans=4, eps=1e-9, cdtype=np.complex64, upsampfac=2.0)
    gpu_nufft._type1_plan(**base)
    gpu_nufft._type1_plan(**{**base, **changed})
    assert fake_cufinufft.built == 2, f"{changed} did not produce a new plan"


def test_plan_is_built_with_the_expected_arguments(fake_cufinufft):
    """The plan is type 1, square in n_modes, and FFT-ordered."""
    plan = gpu_nufft._type1_plan(21, 4, 1e-9, np.complex64, 2.0)
    assert plan.nufft_type == 1
    assert plan.n_modes == (21, 21)
    assert plan.kwargs["n_trans"] == 4
    assert plan.kwargs["dtype"] == "complex64"
    # modeord=1 is what makes signed integer mode indexing work downstream.
    assert plan.kwargs["modeord"] == 1
    assert plan.kwargs["upsampfac"] == 2.0


def test_clear_plan_cache_releases_plans(fake_cufinufft):
    """clear_plan_cache drops the references so cuFFT workspaces are freed."""
    gpu_nufft._type1_plan(21, 4, 1e-9, np.complex64, 2.0)
    assert gpu_nufft._TYPE1_PLANS
    gpu_nufft.clear_plan_cache()
    assert not gpu_nufft._TYPE1_PLANS
