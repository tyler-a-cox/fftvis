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


class GPUBeamEvaluator(BeamEvaluator):
    """GPU implementation of beam evaluation."""

    def __init__(self, use_gpu_interp: Optional[bool] = None, **kwargs):
        """
        Initialize the evaluator.

        Parameters
        ----------
        use_gpu_interp : bool, optional
            Whether to interpolate gridded beams with matvis's bilinear cupy
            kernel. ``None`` (default) enables it only when it reproduces the
            CPU result exactly; ``True`` forces it; ``False`` always evaluates
            on the host and uploads.
        """
        super().__init__(**kwargs)
        self.use_gpu_interp = use_gpu_interp
        # Uploaded beam grids, keyed by id(beam). The beam object is kept alive
        # alongside its data so the id cannot be recycled underneath us.
        self._beam_cache: Dict[int, tuple] = {}

    # ------------------------------------------------------------------
    # Beam evaluation
    # ------------------------------------------------------------------
    def _can_use_gpu_interp(self, beam: BeamInterface, spline_opts) -> bool:
        """Whether matvis's bilinear kernel applies to this beam."""
        if self.use_gpu_interp is False:
            return False
        if not getattr(beam, "_isuvbeam", False):
            return False
        if getattr(beam.beam, "pixel_coordinate_system", None) != "az_za":
            return False
        if self.use_gpu_interp is True:
            return True
        # Auto: only when the CPU side is also doing linear interpolation.
        return bool(spline_opts) and spline_opts.get("order", None) == 1

    def _upload_beam(self, beam: BeamInterface, freq: float, complex_dtype):
        """Interpolate the beam to ``freq`` and stage its grid on the device."""
        key = id(beam)
        cached = self._beam_cache.get(key)
        if cached is not None and cached[0] == freq:
            return cached[1:]

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
        self._beam_cache[key] = (freq,) + staged
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
                "with `pip install fftvis[gpu]`."
            )

        # Saved for matvis compatibility, as the CPU evaluator does.
        self.polarized = polarized
        self.freq = freq
        self.spline_opts = spline_opts or {}

        complex_dtype = np.complex64 if self.precision == 1 else np.complex128

        if self._can_use_gpu_interp(beam, spline_opts):
            data, daz, dza, azmin, _ = self._upload_beam(beam, freq, complex_dtype)
            # matvis returns (nbeam, nfeed, nax, nsrc); fftvis wants
            # (nax, nfeed, nsrc), so the first two axes are swapped back.
            out = gpu_beam_interpolation(data, daz, dza, azmin, az, za, order=1)[0]
            interp_beam = out.transpose(1, 0, 2) if polarized else out[0, 0]
        else:
            host = CPUBeamEvaluator.evaluate_beam(
                self,
                beam,
                az,
                za,
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
        res = cp.einsum("aps,aqs->pqs", beam.conj(), beam)
        beam[:] = res * flux

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
        tmp = cp.einsum("aps,aqs->pqs", beam.conj(), coherency)
        beam[:] = cp.einsum("pbs,bqs->pqs", tmp, beam)

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
        out[:] = cp.einsum("aps,aqs->pqs", beam_i.conj(), beam_j) * flux

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
        tmp = cp.einsum("aps,aqs->pqs", beam_i.conj(), coherency)
        out[:] = cp.einsum("pbs,bqs->pqs", tmp, beam_j)
