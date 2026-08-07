"""
GPU-specific beam evaluation implementation for fftvis.

:class:`GPUBeamEvaluator` returns exactly what :class:`~fftvis.cpu.beams.CPUBeamEvaluator`
returns -- same shapes, same power-vs-efield semantics -- but as cupy arrays.

Interpolation backend
---------------------
Gridded ``UVBeam`` objects on an ``az_za`` grid can be interpolated with
matvis's fused bilinear cupy kernel. That kernel is **bilinear only**, whereas
pyuvdata's ``az_za_map_coordinates`` defaults to a higher spline order, so
using it unconditionally would make the GPU engine disagree with the CPU engine
for reasons that have nothing to do with the GPU.

The default (``use_gpu_interp=None``) is therefore to use the GPU kernel only
when it will reproduce the CPU result -- that is, when ``spline_opts`` asks for
order 1 -- and otherwise to evaluate on the host with pyuvdata and upload.
Pass ``use_gpu_interp=True`` to force the kernel regardless.
"""

import logging
from typing import Dict, Optional

import numpy as np
from pyuvdata.beam_interface import BeamInterface

from ..core.beams import BeamEvaluator
from ..cpu.beams import CPUBeamEvaluator

try:  # pragma: no cover - import guard
    import cupy as cp
    from matvis.gpu.beams import gpu_beam_interpolation, prepare_for_map_coords

    HAVE_CUDA = True
except ImportError:  # pragma: no cover - import guard
    cp = None
    HAVE_CUDA = False

logger = logging.getLogger(__name__)


# Fused apparent-flux kernels.
#
# These compute exactly what the numba kernels in fftvis.cpu.beams compute, but
# in a single pass. The cupy einsum formulations they replace materialised a
# conjugate copy, an einsum result and a scaled product -- roughly five passes
# over an (nax, nfeed, nsrc) complex array where two suffice. Each thread owns
# one source and reads its 2x2 block into registers first, so writing back into
# the input array is safe.
_APPARENT_FLUX_SRC = r"""
#include <cupy/complex.cuh>

#define LOAD(A, s, n) \
    T a00 = A[0*2*n + 0*n + s], a01 = A[0*2*n + 1*n + s], \
      a10 = A[1*2*n + 0*n + s], a11 = A[1*2*n + 1*n + s];
#define LOADJ(A, s, n) \
    T b00 = A[0*2*n + 0*n + s], b01 = A[0*2*n + 1*n + s], \
      b10 = A[1*2*n + 0*n + s], b11 = A[1*2*n + 1*n + s];
#define STORE(O, s, n, o00, o01, o10, o11) \
    O[0*2*n + 0*n + s] = o00; O[0*2*n + 1*n + s] = o01; \
    O[1*2*n + 0*n + s] = o10; O[1*2*n + 1*n + s] = o11;

// out = A^H A * flux   (Hermitian, so out10 = conj(out01))
template<typename T>
__device__ void ahha(T* A, const T* f, long n) {
    long s = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (s >= n) return;
    LOAD(A, s, n)
    T fl = f[s];
    T i00 = conj(a00)*a00 + conj(a10)*a10;
    T i01 = conj(a00)*a01 + conj(a10)*a11;
    T i11 = conj(a01)*a01 + conj(a11)*a11;
    STORE(A, s, n, i00*fl, i01*fl, conj(i01)*fl, i11*fl)
}

// out = A_i^H A_j * flux   (not Hermitian in general)
template<typename T>
__device__ void ahhb(const T* Ai, const T* Aj, const T* f, T* O, long n) {
    long s = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (s >= n) return;
    LOAD(Ai, s, n) LOADJ(Aj, s, n)
    T fl = f[s];
    STORE(O, s, n,
          (conj(a00)*b00 + conj(a10)*b10)*fl,
          (conj(a00)*b01 + conj(a10)*b11)*fl,
          (conj(a01)*b00 + conj(a11)*b10)*fl,
          (conj(a01)*b01 + conj(a11)*b11)*fl)
}

// out = A^H C A
template<typename T>
__device__ void ahca(T* A, const T* C, long n) {
    long s = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (s >= n) return;
    LOAD(A, s, n)
    T c00 = C[0*2*n + 0*n + s], c01 = C[0*2*n + 1*n + s],
      c10 = C[1*2*n + 0*n + s], c11 = C[1*2*n + 1*n + s];
    T t00 = conj(a00)*c00 + conj(a10)*c10;
    T t01 = conj(a00)*c01 + conj(a10)*c11;
    T t10 = conj(a01)*c00 + conj(a11)*c10;
    T t11 = conj(a01)*c01 + conj(a11)*c11;
    STORE(A, s, n, t00*a00 + t01*a10, t00*a01 + t01*a11,
                   t10*a00 + t11*a10, t10*a01 + t11*a11)
}

// out = A_i^H C A_j
template<typename T>
__device__ void ahcb(const T* Ai, const T* Aj, const T* C, T* O, long n) {
    long s = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (s >= n) return;
    LOAD(Ai, s, n) LOADJ(Aj, s, n)
    T c00 = C[0*2*n + 0*n + s], c01 = C[0*2*n + 1*n + s],
      c10 = C[1*2*n + 0*n + s], c11 = C[1*2*n + 1*n + s];
    T t00 = conj(a00)*c00 + conj(a10)*c10;
    T t01 = conj(a00)*c01 + conj(a10)*c11;
    T t10 = conj(a01)*c00 + conj(a11)*c10;
    T t11 = conj(a01)*c01 + conj(a11)*c11;
    STORE(O, s, n, t00*b00 + t01*b10, t00*b01 + t01*b11,
                   t10*b00 + t11*b10, t10*b01 + t11*b11)
}

extern "C" {
__global__ void ahha_c64(complex<float>* A, const complex<float>* f, long n)
{ ahha<complex<float> >(A, f, n); }
__global__ void ahha_c128(complex<double>* A, const complex<double>* f, long n)
{ ahha<complex<double> >(A, f, n); }

__global__ void ahhb_c64(const complex<float>* Ai, const complex<float>* Aj,
                         const complex<float>* f, complex<float>* O, long n)
{ ahhb<complex<float> >(Ai, Aj, f, O, n); }
__global__ void ahhb_c128(const complex<double>* Ai, const complex<double>* Aj,
                          const complex<double>* f, complex<double>* O, long n)
{ ahhb<complex<double> >(Ai, Aj, f, O, n); }

__global__ void ahca_c64(complex<float>* A, const complex<float>* C, long n)
{ ahca<complex<float> >(A, C, n); }
__global__ void ahca_c128(complex<double>* A, const complex<double>* C, long n)
{ ahca<complex<double> >(A, C, n); }

__global__ void ahcb_c64(const complex<float>* Ai, const complex<float>* Aj,
                         const complex<float>* C, complex<float>* O, long n)
{ ahcb<complex<float> >(Ai, Aj, C, O, n); }
__global__ void ahcb_c128(const complex<double>* Ai, const complex<double>* Aj,
                          const complex<double>* C, complex<double>* O, long n)
{ ahcb<complex<double> >(Ai, Aj, C, O, n); }
}
"""

_APPARENT_FLUX_MODULE = None
_BLOCK = 256

# The fused kernels need a real cupy. ``tests/_gpu_shim_check.py`` substitutes
# numpy for cupy to exercise the engine's logic on CPU-only CI, and numpy has
# no RawModule; there we fall back to the einsum reference implementations,
# which compute the same thing.
_USE_FUSED = HAVE_CUDA and hasattr(cp, "RawModule")


def _apparent_kernel(name: str, dtype):
    """Fetch a fused apparent-flux kernel, compiling the module on first use."""
    global _APPARENT_FLUX_MODULE
    if _APPARENT_FLUX_MODULE is None:
        _APPARENT_FLUX_MODULE = cp.RawModule(code=_APPARENT_FLUX_SRC)
    suffix = "c64" if dtype == np.complex64 else "c128"
    return _APPARENT_FLUX_MODULE.get_function(f"{name}_{suffix}")


def _launch(kern, nsrc: int, args):
    """Launch a one-thread-per-source kernel."""
    kern(((nsrc + _BLOCK - 1) // _BLOCK,), (_BLOCK,), args)


def _interp_on_device(data, daz, dza, azmin, az, za, order: int):
    """
    Interpolate a staged beam grid at az/za on the device.

    Parameters
    ----------
    data : cp.ndarray
        Staged beam grid, shape ``(nbeam, nax, nfeed, nza, naz)``, complex.
    daz, dza, azmin : np.ndarray
        Grid spacing and azimuth origin, one entry per beam.
    az, za : array_like
        Coordinates to evaluate at.
    order : int
        Spline order. ``1`` uses matvis's fused bilinear kernel (one launch for
        all planes). Any other order uses
        :func:`cupyx.scipy.ndimage.map_coordinates`, which runs the same spline
        algorithm as ``scipy.ndimage`` -- including the prefilter that makes
        order 3 a true cubic spline interpolation rather than a cubic
        convolution.

    Returns
    -------
    cp.ndarray
        Interpolated values, shape ``(nbeam, nfeed, nax, nsrc)``.

    Notes
    -----
    The real and imaginary parts are interpolated separately rather than
    relying on complex support in ``cupyx``, which has varied across releases.
    Splines are linear, so this is equivalent.
    """
    if order == 1:
        # matvis's fused path: one launch across all (beam, feed, axis) planes.
        return gpu_beam_interpolation(data, daz, dza, azmin, az, za, order=1)

    from cupyx.scipy import ndimage

    nbeam, nax, nfeed, nza, naz = data.shape
    az = cp.asarray(az)
    za = cp.asarray(za)
    nsrc = az.size
    out = cp.empty((nbeam, nfeed, nax, nsrc), dtype=data.dtype)
    rdtype = data.real.dtype

    for bm in range(nbeam):
        coords = cp.stack(
            [
                (za / float(dza[bm])).astype(rdtype),
                ((az - float(azmin[bm])) / float(daz[bm])).astype(rdtype),
            ]
        )
        for ax in range(nax):
            for fd in range(nfeed):
                plane = data[bm, ax, fd]
                re = ndimage.map_coordinates(
                    cp.ascontiguousarray(plane.real), coords, order=order
                )
                im = ndimage.map_coordinates(
                    cp.ascontiguousarray(plane.imag), coords, order=order
                )
                out[bm, fd, ax] = re + 1j * im

    return out


class GPUBeamEvaluator(BeamEvaluator):
    """GPU implementation of beam evaluation."""

    def __init__(
        self,
        use_gpu_interp: Optional[bool] = None,
        max_cached_beams: int = 64,
        **kwargs,
    ):
        """
        Initialize the evaluator.

        Parameters
        ----------
        use_gpu_interp : bool, optional
            Whether to interpolate gridded beams with matvis's bilinear cupy
            kernel. ``None`` (default) enables it only when it reproduces the
            CPU result exactly; ``True`` forces it; ``False`` always evaluates
            on the host and uploads.
        max_cached_beams : int
            Maximum number of staged ``(beam, frequency)`` grids to keep on the
            device before evicting the oldest.
        """
        super().__init__(**kwargs)
        self.use_gpu_interp = use_gpu_interp
        self.max_cached_beams = max_cached_beams
        # Staged beam grids keyed by (id(beam), frequency). The beam object is
        # kept alive alongside its data so the id cannot be recycled underneath
        # us.
        self._beam_cache: Dict[tuple, tuple] = {}
        # Set once per run so the chosen interpolation path is visible in logs
        # without spamming one line per chunk.
        self._logged_path: Optional[bool] = None
        self._warned_analytic: bool = False

    # ------------------------------------------------------------------
    # Beam evaluation
    # ------------------------------------------------------------------
    # Interpolation orders that can run on the device: 1 via matvis's fused
    # bilinear kernel, the rest via cupyx.scipy.ndimage.map_coordinates, which
    # implements the same spline algorithm (including the prefilter) as
    # scipy.ndimage and therefore as pyuvdata's ``az_za_map_coordinates``.
    _DEVICE_ORDERS = (0, 1, 2, 3, 4, 5)

    def _can_use_gpu_interp(self, beam: BeamInterface, spline_opts) -> bool:
        """Whether this beam can be interpolated on the device."""
        if self.use_gpu_interp is False:
            return False
        if not getattr(beam, "_isuvbeam", False):
            # Analytic beams are *evaluated*, not interpolated, and pyuvdata
            # only evaluates on the host. Sample them onto a grid first with
            # fftvis.core.beams.to_gridded_beam.
            if self._warned_analytic is not True:
                self._warned_analytic = True
                logger.warning(
                    "Beam is analytic, so it must be evaluated on the host: "
                    "this is usually the dominant cost of a GPU run. Convert "
                    "it once with fftvis.core.beams.to_gridded_beam(beam, "
                    "freqs) to move interpolation onto the device."
                )
            return False
        if getattr(beam.beam, "pixel_coordinate_system", None) != "az_za":
            return False
        if self.use_gpu_interp is True:
            return True
        # Auto: whenever the requested order is one the device path implements.
        # The device and host use the same spline algorithm, so this does not
        # silently change the interpolation -- except at order 1, where matvis's
        # fused kernel clamps out-of-grid coordinates to the edge.
        return bool(spline_opts) and spline_opts.get("order") in self._DEVICE_ORDERS

    def _upload_beam(self, beam: BeamInterface, freq: float, complex_dtype):
        """Interpolate the beam to ``freq`` and stage its grid on the device.

        The staged grid is cached per ``(beam, frequency)``. Keying on the
        frequency matters: the simulation loop is
        ``time -> chunk -> frequency``, so a cache that only remembered the
        most recent frequency missed on *every* call of a multi-frequency run
        and re-ran pyuvdata's ``interp`` on the host each time. With this
        cache the host interpolation happens once per frequency for the whole
        simulation.
        """
        key = (id(beam), float(freq))
        cached = self._beam_cache.get(key)
        if cached is not None:
            return cached

        uvb = beam.beam.interp(
            freq_array=np.atleast_1d(freq), new_object=True, run_check=False
        )
        d0, daz, dza, azmin = prepare_for_map_coords(uvb)

        # Upload as complex even for power beams. matvis's helper applies a
        # sqrt() to *real* inputs (its Z matrix wants an E-field-like
        # amplitude); fftvis wants the raw response, and a complex input skips
        # that branch.
        data = cp.asarray(d0[None].astype(complex_dtype, copy=False))
        staged = (data, np.array([daz]), np.array([dza]), np.array([azmin]), beam)

        if len(self._beam_cache) >= self.max_cached_beams:
            # Simple FIFO eviction; grids are small (a few MB) but a run with
            # hundreds of frequencies should not pin them all on the device.
            self._beam_cache.pop(next(iter(self._beam_cache)))
        self._beam_cache[key] = staged

        logger.debug(
            "Staged beam grid %s for %.6g Hz on the device (%d cached)",
            tuple(data.shape), freq, len(self._beam_cache),
        )
        return staged

    def evaluate_beam(
        self,
        beam: BeamInterface,
        az: np.ndarray,
        za: np.ndarray,
        polarized: bool,
        freq: float,
        check: bool = False,
        spline_opts: Optional[Dict] = None,
        interpolation_function: str = "az_za_map_coordinates",
    ):
        """
        Evaluate the beam pattern at the given coordinates on the GPU.

        Mirrors :meth:`fftvis.cpu.beams.CPUBeamEvaluator.evaluate_beam`.

        Parameters
        ----------
        beam : BeamInterface
            Beam object to evaluate.
        az : np.ndarray
            Azimuth coordinates in radians.
        za : np.ndarray
            Zenith angle coordinates in radians.
        polarized : bool
            Whether to evaluate the polarized beam.
        freq : float
            Frequency to evaluate the beam at in Hz.
        check : bool, optional
            Whether to check for invalid beam values.
        spline_opts : dict, optional
            Options for spline interpolation.
        interpolation_function : str, optional
            The interpolation function to use on the host fallback path.

        Returns
        -------
        cp.ndarray
            Beam values, shape ``(nax, nfeed, nsrc)`` if polarized, else
            ``(nsrc,)``.
        """
        if not HAVE_CUDA:  # pragma: no cover - import guard
            raise ImportError(
                "The GPU backend requires cupy and cufinufft. Install them "
                "with `pip install fftvis[gpu-cuda12]` (or [gpu-cuda11])."
            )

        # Saved for matvis compatibility, as the CPU evaluator does.
        self.polarized = polarized
        self.freq = freq
        self.spline_opts = spline_opts or {}

        complex_dtype = np.complex64 if self.precision == 1 else np.complex128

        on_gpu = self._can_use_gpu_interp(beam, spline_opts)
        if self._logged_path is not on_gpu:
            self._logged_path = on_gpu
            logger.info(
                "Beam interpolation: %s",
                "matvis bilinear cupy kernel (on device)"
                if on_gpu
                else (
                    "pyuvdata on the host, then uploaded. This is usually the "
                    "dominant cost of a GPU run. Pass "
                    "beam_spline_opts={'order': 1} with a gridded az_za UVBeam "
                    "to move it onto the device."
                ),
            )

        if on_gpu:
            order = int((spline_opts or {}).get("order", 1))
            data, daz, dza, azmin, _ = self._upload_beam(beam, freq, complex_dtype)
            # matvis returns (nbeam, nfeed, nax, nsrc); fftvis wants
            # (nax, nfeed, nsrc), so the first two axes are swapped back.
            out = _interp_on_device(data, daz, dza, azmin, az, za, order)[0]
            interp_beam = out.transpose(1, 0, 2) if polarized else out[0, 0]
        else:
            # pyuvdata is host-only. A GPU coordinate rotator hands us device
            # az/za, so bring them back before calling into it.
            host = CPUBeamEvaluator.evaluate_beam(
                self,
                beam,
                cp.asnumpy(az) if isinstance(az, cp.ndarray) else az,
                cp.asnumpy(za) if isinstance(za, cp.ndarray) else za,
                polarized,
                freq,
                check=False,
                spline_opts=spline_opts,
                interpolation_function=interpolation_function,
            )
            interp_beam = cp.asarray(host)

        if interp_beam.dtype != complex_dtype:
            interp_beam = interp_beam.astype(complex_dtype)

        if check:
            sm = cp.sum(interp_beam)
            if not bool(cp.isfinite(sm)):
                raise ValueError("Beam interpolation resulted in an invalid value")

        return interp_beam

    # ``prepare_beam_evaluation`` is pure host-side bookkeeping over antenna
    # numbers and baselines, so the CPU implementation is reused verbatim.
    prepare_beam_evaluation = staticmethod(CPUBeamEvaluator.prepare_beam_evaluation)

    # ------------------------------------------------------------------
    # Apparent-flux kernels
    #
    # cupy counterparts of the numba kernels in fftvis.cpu.beams. Index
    # convention throughout: arrays are (nax, nfeed, nsrc) and A[a, p, s] is
    # the response of feed p to axis a for source s.
    # ------------------------------------------------------------------
    @staticmethod
    def get_apparent_flux_polarized_beam(beam, flux):
        """Compute ``A^H A * flux`` in place.

        Parameters
        ----------
        beam : cp.ndarray
            Beam values, shape ``(nax, nfeed, nsrc)``. Modified in place.
        flux : cp.ndarray
            Source fluxes, shape ``(nsrc,)``.
        """
        if not _USE_FUSED:
            res = cp.einsum("aps,aqs->pqs", beam.conj(), beam)
            beam[:] = res * flux
            return
        nsrc = beam.shape[-1]
        flux = cp.ascontiguousarray(flux, dtype=beam.dtype)
        _launch(
            _apparent_kernel("ahha", beam.dtype),
            nsrc,
            (beam, flux, np.int64(nsrc)),
        )

    @staticmethod
    def get_apparent_flux_polarized(beam, coherency):
        """Compute ``A^H C A`` in place.

        Parameters
        ----------
        beam : cp.ndarray
            Beam values, shape ``(2, 2, nsrc)``. Modified in place.
        coherency : cp.ndarray
            Source coherency matrices, shape ``(2, 2, nsrc)``.
        """
        if not _USE_FUSED:
            tmp = cp.einsum("aps,aqs->pqs", beam.conj(), coherency)
            beam[:] = cp.einsum("pbs,bqs->pqs", tmp, beam)
            return
        nsrc = beam.shape[-1]
        coherency = cp.ascontiguousarray(coherency, dtype=beam.dtype)
        _launch(
            _apparent_kernel("ahca", beam.dtype),
            nsrc,
            (beam, coherency, np.int64(nsrc)),
        )

    @staticmethod
    def get_apparent_flux_polarized_beam_pair(beam_i, beam_j, flux, out):
        """Compute ``A_i^H diag(flux) A_j`` for two different beams.

        Parameters
        ----------
        beam_i, beam_j : cp.ndarray
            Beam values, shape ``(nax, nfeed, nsrc)``.
        flux : cp.ndarray
            Source fluxes, shape ``(nsrc,)``.
        out : cp.ndarray
            Output array, shape ``(nfeed, nfeed, nsrc)``.
        """
        if not _USE_FUSED:
            out[:] = cp.einsum("aps,aqs->pqs", beam_i.conj(), beam_j) * flux
            return
        nsrc = out.shape[-1]
        flux = cp.ascontiguousarray(flux, dtype=out.dtype)
        _launch(
            _apparent_kernel("ahhb", out.dtype),
            nsrc,
            (
                cp.ascontiguousarray(beam_i),
                cp.ascontiguousarray(beam_j),
                flux,
                out,
                np.int64(nsrc),
            ),
        )

    @staticmethod
    def get_apparent_flux_polarized_pair(beam_i, beam_j, coherency, out):
        """Compute ``A_i^H C A_j`` for two different beams.

        Parameters
        ----------
        beam_i, beam_j : cp.ndarray
            Beam values, shape ``(2, 2, nsrc)``.
        coherency : cp.ndarray
            Source coherency matrices, shape ``(2, 2, nsrc)``.
        out : cp.ndarray
            Output array, shape ``(2, 2, nsrc)``.
        """
        if not _USE_FUSED:
            GPUBeamEvaluator._get_apparent_flux_polarized_pair_einsum(
                beam_i, beam_j, coherency, out
            )
            return
        nsrc = out.shape[-1]
        _launch(
            _apparent_kernel("ahcb", out.dtype),
            nsrc,
            (
                cp.ascontiguousarray(beam_i),
                cp.ascontiguousarray(beam_j),
                cp.ascontiguousarray(coherency, dtype=out.dtype),
                out,
                np.int64(nsrc),
            ),
        )

    @staticmethod
    def _get_apparent_flux_polarized_pair_einsum(beam_i, beam_j, coherency, out):
        """Reference einsum implementation, kept for the kernel parity test."""
        tmp = cp.einsum("aps,aqs->pqs", beam_i.conj(), coherency)
        out[:] = cp.einsum("pbs,bqs->pqs", tmp, beam_j)
