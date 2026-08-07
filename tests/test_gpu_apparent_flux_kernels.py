"""Validate the fused apparent-flux CUDA kernels without a GPU.

``test_gpu_beams.py`` checks the real kernels against the numba reference on a
device. That skips on CPU-only CI, which would leave the CUDA source -- where a
transposed index or a missing ``conj`` is easy to introduce -- entirely
unchecked.

Here the kernel bodies are transcribed into numpy, using the *same* flat
indexing the CUDA uses for a C-contiguous ``(2, 2, nsrc)`` array
(``A[a][p][s]`` at ``a*2*n + p*n + s``), and compared against the numba
implementations. A transcription error in the CUDA shows up as a mismatch here
so long as the transcription is kept in step with the source.
"""

import numpy as np
import pytest

from fftvis.cpu.beams import CPUBeamEvaluator
from fftvis.gpu import beams as gpu_beams

NSRC = 37


def _flat(a):
    """Flatten a (2, 2, nsrc) array the way the CUDA kernel indexes it."""
    assert a.shape[:2] == (2, 2) and a.flags["C_CONTIGUOUS"]
    return a.reshape(-1)


def _cplx(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex128
    )


def _kernel_ahha(A, f, n):
    """Transcription of the ``ahha`` CUDA kernel."""
    a = _flat(A)
    for s in range(n):
        a00, a01 = a[0 * 2 * n + 0 * n + s], a[0 * 2 * n + 1 * n + s]
        a10, a11 = a[1 * 2 * n + 0 * n + s], a[1 * 2 * n + 1 * n + s]
        fl = f[s]
        i00 = np.conj(a00) * a00 + np.conj(a10) * a10
        i01 = np.conj(a00) * a01 + np.conj(a10) * a11
        i11 = np.conj(a01) * a01 + np.conj(a11) * a11
        a[0 * 2 * n + 0 * n + s] = i00 * fl
        a[0 * 2 * n + 1 * n + s] = i01 * fl
        a[1 * 2 * n + 0 * n + s] = np.conj(i01) * fl
        a[1 * 2 * n + 1 * n + s] = i11 * fl


def _kernel_ahhb(Ai, Aj, f, O, n):
    """Transcription of the ``ahhb`` CUDA kernel."""
    ai, aj, o = _flat(Ai), _flat(Aj), _flat(O)
    for s in range(n):
        a00, a01 = ai[0 * 2 * n + 0 * n + s], ai[0 * 2 * n + 1 * n + s]
        a10, a11 = ai[1 * 2 * n + 0 * n + s], ai[1 * 2 * n + 1 * n + s]
        b00, b01 = aj[0 * 2 * n + 0 * n + s], aj[0 * 2 * n + 1 * n + s]
        b10, b11 = aj[1 * 2 * n + 0 * n + s], aj[1 * 2 * n + 1 * n + s]
        fl = f[s]
        o[0 * 2 * n + 0 * n + s] = (np.conj(a00) * b00 + np.conj(a10) * b10) * fl
        o[0 * 2 * n + 1 * n + s] = (np.conj(a00) * b01 + np.conj(a10) * b11) * fl
        o[1 * 2 * n + 0 * n + s] = (np.conj(a01) * b00 + np.conj(a11) * b10) * fl
        o[1 * 2 * n + 1 * n + s] = (np.conj(a01) * b01 + np.conj(a11) * b11) * fl


def _tmp_ahc(a00, a01, a10, a11, c00, c01, c10, c11):
    """The shared ``A^H C`` block of the ahca/ahcb kernels."""
    return (
        np.conj(a00) * c00 + np.conj(a10) * c10,
        np.conj(a00) * c01 + np.conj(a10) * c11,
        np.conj(a01) * c00 + np.conj(a11) * c10,
        np.conj(a01) * c01 + np.conj(a11) * c11,
    )


def _kernel_ahca(A, C, n):
    """Transcription of the ``ahca`` CUDA kernel."""
    a, c = _flat(A), _flat(C)
    for s in range(n):
        a00, a01 = a[0 * 2 * n + 0 * n + s], a[0 * 2 * n + 1 * n + s]
        a10, a11 = a[1 * 2 * n + 0 * n + s], a[1 * 2 * n + 1 * n + s]
        c00, c01 = c[0 * 2 * n + 0 * n + s], c[0 * 2 * n + 1 * n + s]
        c10, c11 = c[1 * 2 * n + 0 * n + s], c[1 * 2 * n + 1 * n + s]
        t00, t01, t10, t11 = _tmp_ahc(a00, a01, a10, a11, c00, c01, c10, c11)
        a[0 * 2 * n + 0 * n + s] = t00 * a00 + t01 * a10
        a[0 * 2 * n + 1 * n + s] = t00 * a01 + t01 * a11
        a[1 * 2 * n + 0 * n + s] = t10 * a00 + t11 * a10
        a[1 * 2 * n + 1 * n + s] = t10 * a01 + t11 * a11


def _kernel_ahcb(Ai, Aj, C, O, n):
    """Transcription of the ``ahcb`` CUDA kernel."""
    ai, aj, c, o = _flat(Ai), _flat(Aj), _flat(C), _flat(O)
    for s in range(n):
        a00, a01 = ai[0 * 2 * n + 0 * n + s], ai[0 * 2 * n + 1 * n + s]
        a10, a11 = ai[1 * 2 * n + 0 * n + s], ai[1 * 2 * n + 1 * n + s]
        b00, b01 = aj[0 * 2 * n + 0 * n + s], aj[0 * 2 * n + 1 * n + s]
        b10, b11 = aj[1 * 2 * n + 0 * n + s], aj[1 * 2 * n + 1 * n + s]
        c00, c01 = c[0 * 2 * n + 0 * n + s], c[0 * 2 * n + 1 * n + s]
        c10, c11 = c[1 * 2 * n + 0 * n + s], c[1 * 2 * n + 1 * n + s]
        t00, t01, t10, t11 = _tmp_ahc(a00, a01, a10, a11, c00, c01, c10, c11)
        o[0 * 2 * n + 0 * n + s] = t00 * b00 + t01 * b10
        o[0 * 2 * n + 1 * n + s] = t00 * b01 + t01 * b11
        o[1 * 2 * n + 0 * n + s] = t10 * b00 + t11 * b10
        o[1 * 2 * n + 1 * n + s] = t10 * b01 + t11 * b11


@pytest.fixture
def data():
    """Random beams, fluxes and coherency matrices."""
    rng = np.random.default_rng(0)
    return dict(
        beam_i=_cplx(rng, (2, 2, NSRC)),
        beam_j=_cplx(rng, (2, 2, NSRC)),
        flux=rng.standard_normal(NSRC).astype(np.complex128),
        coherency=_cplx(rng, (2, 2, NSRC)),
    )


def test_ahha_matches_numba(data):
    """``A^H A * flux`` kernel matches get_apparent_flux_polarized_beam."""
    ref = data["beam_i"].copy()
    CPUBeamEvaluator.get_apparent_flux_polarized_beam(ref, data["flux"])
    got = np.ascontiguousarray(data["beam_i"].copy())
    _kernel_ahha(got, data["flux"], NSRC)
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13)


def test_ahhb_matches_numba(data):
    """``A_i^H A_j * flux`` kernel matches the numba pair version."""
    ref = np.empty_like(data["beam_i"])
    CPUBeamEvaluator.get_apparent_flux_polarized_beam_pair(
        data["beam_i"], data["beam_j"], data["flux"], ref
    )
    got = np.ascontiguousarray(np.empty_like(data["beam_i"]))
    _kernel_ahhb(
        np.ascontiguousarray(data["beam_i"]),
        np.ascontiguousarray(data["beam_j"]),
        data["flux"],
        got,
        NSRC,
    )
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13)


def test_ahca_matches_numba(data):
    """``A^H C A`` kernel matches get_apparent_flux_polarized."""
    ref = data["beam_i"].copy()
    CPUBeamEvaluator.get_apparent_flux_polarized(ref, data["coherency"])
    got = np.ascontiguousarray(data["beam_i"].copy())
    _kernel_ahca(got, np.ascontiguousarray(data["coherency"]), NSRC)
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13)


def test_ahcb_matches_numba(data):
    """``A_i^H C A_j`` kernel matches the numba polarized pair version."""
    ref = np.empty_like(data["beam_i"])
    CPUBeamEvaluator.get_apparent_flux_polarized_pair(
        data["beam_i"], data["beam_j"], data["coherency"], ref
    )
    got = np.ascontiguousarray(np.empty_like(data["beam_i"]))
    _kernel_ahcb(
        np.ascontiguousarray(data["beam_i"]),
        np.ascontiguousarray(data["beam_j"]),
        np.ascontiguousarray(data["coherency"]),
        got,
        NSRC,
    )
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13)


def test_all_kernels_present_in_cuda_source():
    """Every kernel this file transcribes is actually exported by the module."""
    src = gpu_beams._APPARENT_FLUX_SRC
    for name in ("ahha", "ahhb", "ahca", "ahcb"):
        for suffix in ("c64", "c128"):
            assert f"{name}_{suffix}(" in src, f"{name}_{suffix} missing"
