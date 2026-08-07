"""Tests for the GPU beam evaluator and GPU utility functions."""

import numpy as np
import pytest

from conftest import HAVE_GPU, requires_gpu
from fftvis.cpu.beams import CPUBeamEvaluator
from fftvis.gpu.beams import GPUBeamEvaluator
from fftvis.gpu.utils import inplace_rot

if HAVE_GPU:
    import cupy as cp


def test_gpu_beam_evaluator_init():
    """The evaluator constructs without a GPU present."""
    evaluator = GPUBeamEvaluator()

    assert evaluator.beam_list == []
    assert evaluator.beam_idx is None
    assert evaluator.polarized is False
    assert evaluator.nant == 0
    assert evaluator.freq == 0.0
    assert evaluator.nsrc == 0
    assert evaluator.precision == 2
    assert evaluator.use_gpu_interp is None


def test_gpu_beam_evaluator_import_error_without_cuda():
    """Evaluating a beam without cupy gives a helpful ImportError."""
    if HAVE_GPU:
        pytest.skip("cupy is installed")
    with pytest.raises(ImportError, match="fftvis\\[gpu\\]"):
        GPUBeamEvaluator().evaluate_beam(
            beam=None,
            az=np.array([0.0]),
            za=np.array([0.0]),
            polarized=False,
            freq=150e6,
        )


def test_gpu_inplace_rot_import_error_without_cuda():
    """inplace_rot without cupy gives a helpful ImportError."""
    if HAVE_GPU:
        pytest.skip("cupy is installed")
    with pytest.raises(ImportError, match="fftvis\\[gpu\\]"):
        inplace_rot(np.eye(3), np.zeros((3, 10)))


@requires_gpu
def test_gpu_inplace_rot_matches_cpu():
    """The GPU rotation matches the numba CPU rotation."""
    from fftvis.cpu.utils import inplace_rot as cpu_inplace_rot

    rng = np.random.default_rng(0)
    rot = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    b = rng.standard_normal((3, 500))

    ref = b.copy()
    cpu_inplace_rot(rot, ref)

    got = cp.asarray(b)
    inplace_rot(rot, got)

    np.testing.assert_allclose(cp.asnumpy(got), ref, rtol=1e-12, atol=1e-12)


@requires_gpu
@pytest.mark.parametrize(
    "kernel",
    [
        "get_apparent_flux_polarized_beam",
        "get_apparent_flux_polarized",
        "get_apparent_flux_polarized_beam_pair",
        "get_apparent_flux_polarized_pair",
    ],
)
def test_apparent_flux_kernels_match_cpu(kernel):
    """Each cupy apparent-flux kernel reproduces its numba counterpart."""
    rng = np.random.default_rng(3)
    nsrc = 64

    def cplx(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
            np.complex128
        )

    beam_i = cplx((2, 2, nsrc))
    beam_j = cplx((2, 2, nsrc))
    flux = rng.standard_normal(nsrc).astype(np.complex128)
    coherency = cplx((2, 2, nsrc))

    if kernel == "get_apparent_flux_polarized_beam":
        ref = beam_i.copy()
        CPUBeamEvaluator.get_apparent_flux_polarized_beam(ref, flux)
        got = cp.asarray(beam_i)
        GPUBeamEvaluator.get_apparent_flux_polarized_beam(got, cp.asarray(flux))
    elif kernel == "get_apparent_flux_polarized":
        ref = beam_i.copy()
        CPUBeamEvaluator.get_apparent_flux_polarized(ref, coherency)
        got = cp.asarray(beam_i)
        GPUBeamEvaluator.get_apparent_flux_polarized(got, cp.asarray(coherency))
    elif kernel == "get_apparent_flux_polarized_beam_pair":
        ref = np.empty_like(beam_i)
        CPUBeamEvaluator.get_apparent_flux_polarized_beam_pair(
            beam_i, beam_j, flux, ref
        )
        got = cp.empty_like(cp.asarray(beam_i))
        GPUBeamEvaluator.get_apparent_flux_polarized_beam_pair(
            cp.asarray(beam_i), cp.asarray(beam_j), cp.asarray(flux), got
        )
    else:
        ref = np.empty_like(beam_i)
        CPUBeamEvaluator.get_apparent_flux_polarized_pair(
            beam_i, beam_j, coherency, ref
        )
        got = cp.empty_like(cp.asarray(beam_i))
        GPUBeamEvaluator.get_apparent_flux_polarized_pair(
            cp.asarray(beam_i), cp.asarray(beam_j), cp.asarray(coherency), got
        )

    np.testing.assert_allclose(cp.asnumpy(got), ref, rtol=1e-12, atol=1e-12)


@requires_gpu
@pytest.mark.parametrize("polarized", [False, True])
def test_evaluate_beam_matches_cpu(polarized):
    """Host-fallback beam evaluation returns the same values as the CPU path."""
    from matvis._test_utils import get_standard_sim_params

    params, *_ = get_standard_sim_params(
        use_analytic_beam=True, polarized=polarized
    )
    beam = params["beams"][0]
    freq = float(np.atleast_1d(params["freqs"])[0])

    rng = np.random.default_rng(5)
    za = rng.uniform(0, np.pi / 3, 128)
    az = rng.uniform(0, 2 * np.pi, 128)

    cpu_eval = CPUBeamEvaluator()
    gpu_eval = GPUBeamEvaluator(use_gpu_interp=False)

    ref = cpu_eval.evaluate_beam(beam, az, za, polarized, freq)
    got = gpu_eval.evaluate_beam(beam, az, za, polarized, freq)

    assert got.shape == ref.shape
    np.testing.assert_allclose(cp.asnumpy(got), ref, rtol=1e-10, atol=1e-12)
