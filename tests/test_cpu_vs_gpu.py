"""End-to-end parity between the CPU and GPU simulation engines.

The GPU engine is a port of the CPU engine, so the only acceptance criterion
that matters is that they produce the same visibilities. Everything here runs
the same simulation through both backends and compares.
"""

import numpy as np
import pytest

from conftest import requires_gpu
from fftvis.wrapper import simulate_vis

pytestmark = requires_gpu


def _params(polarized, use_analytic_beam=True, **kw):
    """Standard small simulation parameters shared by both engines."""
    from matvis._test_utils import get_standard_sim_params

    params, *_ = get_standard_sim_params(
        use_analytic_beam=use_analytic_beam, polarized=polarized, **kw
    )
    params.pop("beam_idx", None)
    # get_standard_sim_params returns `beams` (a list); simulate_vis takes `beam`.
    params["beam"] = params.pop("beams")[0]
    return params


def _run(params, backend, **kw):
    """Run a simulation through the named backend."""
    return simulate_vis(backend=backend, **params, **kw)


def _compare(ref, got, eps):
    """Assert two visibility arrays agree to the transform's own accuracy."""
    assert ref.shape == got.shape
    scale = np.abs(ref).max()
    np.testing.assert_allclose(got, ref, rtol=0, atol=max(100 * eps * scale, 1e-12))


@pytest.mark.parametrize("polarized", [False, True])
def test_cpu_vs_gpu_analytic_beam(polarized):
    """Unpolarized and polarized simulations agree between backends."""
    params = _params(polarized)
    eps = 1e-13
    ref = _run(params, "cpu", precision=2, eps=eps)
    got = _run(params, "gpu", precision=2, eps=eps)
    _compare(ref, got, eps)


def test_cpu_vs_gpu_single_precision():
    """Single precision agrees to single-precision accuracy."""
    params = _params(polarized=False)
    eps = 1e-6
    ref = _run(params, "cpu", precision=1, eps=eps)
    got = _run(params, "gpu", precision=1, eps=eps)
    _compare(ref, got, eps)


def test_cpu_vs_gpu_gridded_beam():
    """A gridded UVBeam agrees between backends on the host-fallback path."""
    params = _params(polarized=False, use_analytic_beam=False)
    eps = 1e-13
    ref = _run(params, "cpu", precision=2, eps=eps)
    got = _run(params, "gpu", precision=2, eps=eps)
    _compare(ref, got, eps)


def test_cpu_vs_gpu_gridded_array_type1():
    """A griddable array takes the type-1 path on both backends."""
    params = _params(polarized=False)
    params["ants"] = {
        i: np.array([10.0 * (i % 3), 10.0 * (i // 3), 0.0]) for i in range(9)
    }
    eps = 1e-13
    ref = _run(params, "cpu", precision=2, eps=eps)
    got = _run(params, "gpu", precision=2, eps=eps)
    _compare(ref, got, eps)


def test_cpu_vs_gpu_matvis_beam_kernel():
    """matvis's bilinear cupy kernel matches linear interpolation on the CPU."""
    params = _params(polarized=False, use_analytic_beam=False)
    eps = 1e-13
    spline = {"order": 1}
    ref = _run(params, "cpu", precision=2, eps=eps, beam_spline_opts=spline)
    got = _run(params, "gpu", precision=2, eps=eps, beam_spline_opts=spline)
    # Interpolation differences at the beam edge dominate here, not the NUFFT.
    assert ref.shape == got.shape
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-8 * np.abs(ref).max())


def test_cpu_vs_gpu_per_antenna_beams():
    """The standard multi-beam path agrees between backends."""
    params = _params(polarized=False)
    nant = len(params["ants"])
    params["beam"] = [params["beam"]] * 2
    params["beam_idx"] = np.arange(nant) % 2

    eps = 1e-13
    ref = _run(params, "cpu", precision=2, eps=eps)
    got = _run(params, "gpu", precision=2, eps=eps)
    _compare(ref, got, eps)


def test_cpu_vs_gpu_tilted_array_uses_3d():
    """A non-coplanar array exercises the 3D type-3 path on both backends."""
    params = _params(polarized=False)
    params["ants"] = {
        ant: np.array(pos) + np.array([0.0, 0.0, 5.0 * i])
        for i, (ant, pos) in enumerate(params["ants"].items())
    }

    eps = 1e-13
    ref = _run(params, "cpu", precision=2, eps=eps)
    got = _run(params, "gpu", precision=2, eps=eps)
    _compare(ref, got, eps)


def test_cpu_vs_gpu_chunked():
    """Chunking the source axis does not change the result on either backend."""
    params = _params(polarized=False)
    eps = 1e-13
    ref = _run(params, "cpu", precision=2, eps=eps, min_chunks=1)
    got = _run(params, "gpu", precision=2, eps=eps, min_chunks=3)
    _compare(ref, got, eps)


def test_cpu_vs_gpu_polarized_sky():
    """A polarized sky model agrees between backends."""
    params = _params(polarized=True, use_polarized_sky=True)
    eps = 1e-13
    ref = _run(params, "cpu", precision=2, eps=eps)
    got = _run(params, "gpu", precision=2, eps=eps)
    _compare(ref, got, eps)


def test_gpu_ignores_multiprocessing_args(caplog):
    """nprocesses > 1 is ignored with a warning rather than silently honoured."""
    params = _params(polarized=False)
    eps = 1e-13
    ref = _run(params, "cpu", precision=2, eps=eps)
    with caplog.at_level("WARNING"):
        got = _run(params, "gpu", precision=2, eps=eps, nprocesses=4)
    assert any("single device" in r.message for r in caplog.records)
    _compare(ref, got, eps)
