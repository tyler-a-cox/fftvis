"""Tests for the GPU NUFFT wrappers.

Each test checks the GPU wrapper against its CPU counterpart at the same
accuracy, since parity with :mod:`fftvis.cpu.nufft` is the whole contract.
"""

import numpy as np
import pytest

from conftest import HAVE_GPU, requires_gpu
from fftvis.cpu.nufft import cpu_nufft2d, cpu_nufft2d_type1, cpu_nufft3d
from fftvis.gpu.nufft import gpu_nufft2d, gpu_nufft2d_type1, gpu_nufft3d

if HAVE_GPU:
    import cupy as cp


EPS = 1e-12


def _problem(nsrc=2000, ntgt=64, ntrans=4, seed=0):
    """Source and target point sets shaped like an fftvis chunk."""
    rng = np.random.default_rng(seed)
    src = rng.uniform(-np.pi, np.pi, (3, nsrc))
    tgt = rng.uniform(-20, 20, (3, ntgt))
    weights = rng.standard_normal((ntrans, nsrc)) + 1j * rng.standard_normal(
        (ntrans, nsrc)
    )
    return src, tgt, weights


def test_gpu_nufft_import_error_without_cuda():
    """The wrappers raise a helpful ImportError when cupy is missing."""
    if HAVE_GPU:
        pytest.skip("cupy is installed")
    src, tgt, weights = _problem(nsrc=4, ntgt=2, ntrans=1)
    with pytest.raises(ImportError, match=r"fftvis\[gpu"):
        gpu_nufft2d(src[0], src[1], weights, tgt[0], tgt[1], eps=EPS)


@requires_gpu
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_gpu_nufft2d_matches_cpu(dtype):
    """2D type-3 agrees with the CPU wrapper."""
    src, tgt, weights = _problem()
    weights = weights.astype(dtype)
    rdtype = np.float32 if dtype == np.complex64 else np.float64
    src = src.astype(rdtype)
    tgt = tgt.astype(rdtype)
    eps = 1e-6 if dtype == np.complex64 else EPS

    ref = cpu_nufft2d(src[0], src[1], weights, tgt[0], tgt[1], eps=eps)
    got = gpu_nufft2d(
        cp.asarray(src[0]),
        cp.asarray(src[1]),
        cp.asarray(weights),
        cp.asarray(tgt[0]),
        cp.asarray(tgt[1]),
        eps=eps,
    )

    assert got.shape == ref.shape
    np.testing.assert_allclose(
        cp.asnumpy(got), ref, rtol=0, atol=10 * eps * np.abs(ref).max()
    )


@requires_gpu
def test_gpu_nufft3d_matches_cpu():
    """3D type-3 agrees with the CPU wrapper."""
    src, tgt, weights = _problem()

    ref = cpu_nufft3d(
        src[0], src[1], src[2], weights, tgt[0], tgt[1], tgt[2], eps=EPS
    )
    got = gpu_nufft3d(
        *[cp.asarray(a) for a in src],
        cp.asarray(weights),
        *[cp.asarray(a) for a in tgt],
        eps=EPS,
    )

    assert got.shape == ref.shape
    np.testing.assert_allclose(
        cp.asnumpy(got), ref, rtol=0, atol=10 * EPS * np.abs(ref).max()
    )


@requires_gpu
def test_gpu_nufft2d_type1_matches_cpu():
    """Type-1 plus signed-integer mode indexing agrees with the CPU wrapper."""
    src, _, weights = _problem()
    rng = np.random.default_rng(2)
    n_modes = 21
    index = rng.integers(-10, 11, (2, 64))

    ref = cpu_nufft2d_type1(
        src[0], src[1], weights, n_modes=n_modes, index=index, eps=EPS
    )
    got = gpu_nufft2d_type1(
        cp.asarray(src[0]),
        cp.asarray(src[1]),
        cp.asarray(weights),
        n_modes=n_modes,
        index=index,
        eps=EPS,
    )

    assert got.shape == ref.shape
    np.testing.assert_allclose(
        cp.asnumpy(got), ref, rtol=0, atol=10 * EPS * np.abs(ref).max()
    )


@requires_gpu
def test_gpu_nufft2d_single_transform():
    """A 1-D weights array (no transform axis) is handled."""
    src, tgt, weights = _problem(ntrans=1)
    w = weights[0]

    ref = cpu_nufft2d(src[0], src[1], w, tgt[0], tgt[1], eps=EPS)
    got = gpu_nufft2d(
        cp.asarray(src[0]),
        cp.asarray(src[1]),
        cp.asarray(w),
        cp.asarray(tgt[0]),
        cp.asarray(tgt[1]),
        eps=EPS,
    )
    np.testing.assert_allclose(
        cp.asnumpy(got), ref, rtol=0, atol=10 * EPS * np.abs(ref).max()
    )
