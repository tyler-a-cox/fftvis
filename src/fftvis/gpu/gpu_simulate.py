"""
GPU-specific simulation implementation for fftvis.

This is a direct port of :mod:`fftvis.cpu.cpu_simulate`: the setup, the
time/chunk/frequency loop, the beam-pair bookkeeping and the basis-visibility
path all mirror the CPU engine, with numpy replaced by cupy and finufft
replaced by cufinufft.

Deliberate differences from the CPU engine
------------------------------------------
* **No multiprocessing.** The CPU engine parallelises over (time, frequency)
  with ray. This engine targets a single GPU and ignores ``nprocesses``,
  ``nthreads`` and ``force_use_ray``.
* **Coordinate rotation follows ``coord_method``.** A CPU rotator
  (``CoordinateRotationERFA``) runs on the host and each chunk is uploaded; a
  GPU rotator (``GPUCoordinateRotationERFA``) is constructed with ``gpu=True``
  and keeps its arrays on the device. The GPU rotators cannot be combined with
  a *polarized sky model* -- matvis's coherency-rotation branch indexes host
  ``SkyCoord`` arrays with device index arrays -- and that combination raises a
  clear error rather than failing inside matvis.
* **Memory tracing** (``trace_mem``, ``enable_memory_monitor``) is accepted and
  ignored.

Neither of the first two is a performance ceiling; both are places to optimise
once parity is established.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Literal, Union

import numpy as np
from astropy import units as un
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from matvis import coordinates
from matvis.core.coords import CoordinateRotation
from pyuvdata import UVBeam
from pyuvdata.beam_interface import BeamInterface

from .. import utils
from ..core.antenna_gridding import check_antpos_griddability
from ..core.simulate import SimulationEngine, default_accuracy_dict
from ..cpu import utils as cpu_utils
from .beams import GPUBeamEvaluator
from .nufft import gpu_nufft2d, gpu_nufft2d_type1, gpu_nufft3d
from . import utils as gpu_utils

try:  # pragma: no cover - import guard
    import cupy as cp

    HAVE_CUDA = True
except ImportError:  # pragma: no cover - import guard
    cp = None
    HAVE_CUDA = False

logger = logging.getLogger(__name__)

# Module-level evaluator, mirroring the CPU engine's `_cpu_beam_evaluator`.
_gpu_beam_evaluator = GPUBeamEvaluator()

# Set ``STAGE_TIMING = True`` to accumulate per-stage wall times into
# ``LAST_RUN_STATS``. This inserts a device synchronisation around every stage,
# so it slows the run down and is off by default:
#
#     import fftvis.gpu.gpu_simulate as gs
#     gs.STAGE_TIMING = True
#     fftvis.simulate_vis(..., backend="gpu")
#     gs.LAST_RUN_STATS
#
STAGE_TIMING = False
LAST_RUN_STATS: dict = {}


@contextmanager
def _stage(stats: dict, name: str):
    """Accumulate the wall time of a pipeline stage when STAGE_TIMING is on."""
    if not STAGE_TIMING:
        yield
        return
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        cp.cuda.Device().synchronize()
        stats[name] = stats.get(name, 0.0) + (time.perf_counter() - t0)


def _require_cuda():
    """Raise a helpful error if the GPU stack is unavailable."""
    if not HAVE_CUDA:  # pragma: no cover - import guard
        raise ImportError(
            "The GPU backend requires cupy and cufinufft. Install them with "
            "`pip install fftvis[gpu-cuda12]` (or [gpu-cuda11]). GPU type-3 transforms need "
            "finufft >= 2.4."
        )


def estimate_type3_grid(topo, uvw, dim: int, upsample_factor: float = 2.0) -> tuple:
    """
    Estimate finufft's internal type-3 grid, in modes and in bytes.

    finufft sizes the internal upsampled grid from the product of the source
    and target half-widths, so a non-coplanar array with long baselines can
    demand a grid far larger than the visibility output. This is the dominant
    out-of-memory risk on a GPU.

    Parameters
    ----------
    topo : array_like
        Source coordinates, shape ``(3, nsrc)``, already scaled by ``2*pi``.
    uvw : array_like
        Baseline coordinates, shape ``(3, nbls)``.
    dim : int
        Transform dimensionality (2 or 3).
    upsample_factor : float
        finufft's ``upsampfac``.

    Returns
    -------
    modes : list of int
        Estimated number of modes per dimension.
    nbytes : int
        Estimated size of the grid, assuming complex128.
    """
    modes = []
    for d in range(dim):
        x = float(0.5 * (topo[d].max() - topo[d].min()))
        s = float(0.5 * (uvw[d].max() - uvw[d].min()))
        modes.append(max(int(upsample_factor * x * s / np.pi), 1))
    return modes, int(np.prod(modes)) * 16


def _evaluate_beam_list(
    beam_list: list,
    az: np.ndarray,
    za: np.ndarray,
    polarized: bool,
    freq: float,
    beam_spline_opts: dict,
    interpolation_function: str,
    complex_dtype: np.dtype,
) -> list:
    """Evaluate every beam in ``beam_list`` at the given az/za positions.

    GPU counterpart of :func:`fftvis.cpu.cpu_simulate._evaluate_beam_list`.

    Parameters
    ----------
    beam_list : list
        List of beam objects (UVBeam or BeamInterface).
    az, za : np.ndarray
        Azimuth and zenith angle arrays (radians), shape ``(nsrc,)``.
    polarized : bool
        Whether to evaluate polarized beam components.
    freq : float
        Frequency in Hz.
    beam_spline_opts : dict or None
        Options passed to the spline interpolator.
    interpolation_function : str
        Name of the interpolation function to use.
    complex_dtype : np.dtype
        Complex dtype to cast the result to if needed.

    Returns
    -------
    list of cp.ndarray
        One evaluated beam array per beam in ``beam_list``.
    """
    beam_evaluations = []
    for beam in beam_list:
        be = _gpu_beam_evaluator.evaluate_beam(
            beam,
            az,
            za,
            polarized,
            freq,
            spline_opts=beam_spline_opts,
            interpolation_function=interpolation_function,
        )
        beam_evaluations.append(
            be if be.dtype == complex_dtype else be.astype(complex_dtype)
        )
    return beam_evaluations


def _compute_apparent_coherency(
    beam_evaluations: list,
    bi: int,
    bj: int,
    flux_here,
    freqidx: int,
    polarized: bool,
    polarized_sky_model: bool,
    nfeeds: int,
    nsim_sources: int,
    complex_dtype: np.dtype,
    apparent_buf,
):
    """Compute beam-weighted sky coherency for a single beam pair.

    GPU counterpart of
    :func:`fftvis.cpu.cpu_simulate._compute_apparent_coherency`; the branch
    structure and the axis flips are identical.

    Parameters
    ----------
    beam_evaluations : list of cp.ndarray
        Pre-evaluated beams, one per unique beam in ``beam_list``.
    bi, bj : int
        Indices into ``beam_evaluations`` for the two antennas of this pair.
    flux_here : cp.ndarray
        Source flux array.
    freqidx : int
        Frequency index into ``flux_here``.
    polarized : bool
        Whether the simulation is polarized.
    polarized_sky_model : bool
        Whether the sky model is itself polarized.
    nfeeds : int
        Number of feed dimensions (1 or 2).
    nsim_sources : int
        Number of simulated sources in this chunk.
    complex_dtype : np.dtype
        Complex dtype to ensure the output array matches.
    apparent_buf : cp.ndarray
        Pre-allocated work buffer.

    Returns
    -------
    cp.ndarray
        Apparent coherency shaped ``(nfeeds**2, nsrc)``.
    """
    is_cross_pair = bi != bj

    if polarized and polarized_sky_model:
        coherency = cp.transpose(flux_here[:, freqidx], (1, 2, 0))
        if is_cross_pair:
            apparent_buf[:] = 0
            apparent_coherency = apparent_buf
            _gpu_beam_evaluator.get_apparent_flux_polarized_pair(
                beam_i=cp.ascontiguousarray(cp.flip(beam_evaluations[bi], axis=0)),
                beam_j=cp.ascontiguousarray(cp.flip(beam_evaluations[bj], axis=0)),
                coherency=coherency,
                out=apparent_coherency,
            )
        else:
            # The CPU engine flips a view and mutates through it, ending up
            # with f(flip(A)). Flipping into the buffer first is equivalent and
            # keeps negative-strided views out of the cupy kernels.
            apparent_buf[:] = cp.flip(beam_evaluations[bi], axis=0)
            _gpu_beam_evaluator.get_apparent_flux_polarized(apparent_buf, coherency)
            apparent_coherency = apparent_buf

    elif polarized:
        if is_cross_pair:
            apparent_buf[:] = 0
            apparent_coherency = apparent_buf
            _gpu_beam_evaluator.get_apparent_flux_polarized_beam_pair(
                beam_i=beam_evaluations[bi],
                beam_j=beam_evaluations[bj],
                flux=flux_here[:, freqidx],
                out=apparent_coherency,
            )
        else:
            apparent_buf[:] = beam_evaluations[bi]
            apparent_coherency = apparent_buf
            _gpu_beam_evaluator.get_apparent_flux_polarized_beam(
                apparent_coherency, flux_here[:, freqidx]
            )

    else:
        cp.multiply(beam_evaluations[bi], beam_evaluations[bj], out=apparent_buf)
        cp.sqrt(apparent_buf, out=apparent_buf)
        apparent_buf *= flux_here[:, freqidx]
        apparent_coherency = apparent_buf

    apparent_coherency = cp.reshape(apparent_coherency, (nfeeds**2, nsim_sources))

    if apparent_coherency.dtype != complex_dtype:
        apparent_coherency = apparent_coherency.astype(complex_dtype)

    return apparent_coherency


def _run_nufft(
    apparent_coherency,
    topo,
    uvw,
    bls,
    flipped,
    bls_idxs,
    use_type1: bool,
    is_coplanar: bool,
    tx,
    ty,
    type1_n_modes: int,
    eps: float,
    n_threads: int,
    upsample_factor: float,
    nfeeds: int,
):
    """Dispatch to the appropriate GPU NUFFT and return shaped visibilities.

    GPU counterpart of :func:`fftvis.cpu.cpu_simulate._run_nufft`.

    Parameters
    ----------
    apparent_coherency : cp.ndarray
        Beam-weighted sky coherency, shape ``(nfeeds**2, nsrc)``.
    topo : cp.ndarray
        Topocentric source coordinates, shape ``(3, nsrc)``.
    uvw : cp.ndarray
        Frequency-scaled baseline vectors, shape ``(3, nbls)``.
    bls : cp.ndarray
        Integer gridded baseline indices (type-1 path only).
    flipped : cp.ndarray
        Boolean mask of baselines whose UVW was negated.
    bls_idxs : cp.ndarray
        Indices of the baselines this beam pair contributes to.
    use_type1 : bool
        Use type-1 NUFFT (gridded array).
    is_coplanar : bool
        Use 2D instead of 3D NUFFT.
    tx, ty : cp.ndarray
        Frequency-scaled ``topo[0]``/``topo[1]`` (type-1 path only).
    type1_n_modes : int
        Grid size for type-1 transform.
    eps, n_threads, upsample_factor : float / int
        NUFFT accuracy and performance parameters. ``n_threads`` is ignored.
    nfeeds : int
        Number of feed dimensions.

    Returns
    -------
    cp.ndarray
        Visibilities shaped ``(nbls_here, nfeeds, nfeeds)``.
    """
    nbls_here = len(bls_idxs)

    if use_type1:
        bls_here = cp.where(flipped, -bls[:, bls_idxs], bls[:, bls_idxs])
        _vis_here = gpu_nufft2d_type1(
            tx,
            ty,
            apparent_coherency,
            n_modes=type1_n_modes,
            index=bls_here,
            eps=eps,
            n_threads=n_threads,
            upsample_factor=upsample_factor,
        )
    else:
        _uvw = cp.where(flipped, -uvw[:, bls_idxs], uvw[:, bls_idxs])
        if is_coplanar:
            _vis_here = gpu_nufft2d(
                topo[0],
                topo[1],
                apparent_coherency,
                _uvw[0],
                _uvw[1],
                eps=eps,
                n_threads=n_threads,
                upsample_factor=upsample_factor,
            )
        else:
            _vis_here = gpu_nufft3d(
                topo[0],
                topo[1],
                topo[2],
                apparent_coherency,
                _uvw[0],
                _uvw[1],
                _uvw[2],
                eps=eps,
                n_threads=n_threads,
                upsample_factor=upsample_factor,
            )

    _vis_here = cp.where(flipped, cp.conj(_vis_here), _vis_here)

    return cp.swapaxes(_vis_here.reshape(nfeeds, nfeeds, nbls_here), 2, 0)


def _compute_basis_visibilities(
    beam_evaluations: list,
    flux_here,
    ant1_idxs: np.ndarray,
    ant2_idxs: np.ndarray,
    beam_coefs: np.ndarray,
    freqidx: int,
    topo,
    uvw,
    bls,
    tx,
    ty,
    nbls: int,
    nfeeds: int,
    nsim_sources: int,
    complex_dtype: np.dtype,
    use_type1: bool,
    is_coplanar: bool,
    type1_n_modes: int,
    eps: float,
    n_threads: int,
    upsample_factor: float,
    polarized: bool = False,
    polarized_sky_model: bool = False,
):
    """Compute the basis visibility tensor for all basis-beam pairs.

    GPU counterpart of
    :func:`fftvis.cpu.cpu_simulate._compute_basis_visibilities`. See that
    function for the measurement-equation derivation; the loop structure,
    the ``k <= l`` conjugate-symmetry shortcut and the coefficient contraction
    are identical here.

    Parameters
    ----------
    beam_evaluations : list of cp.ndarray
        Evaluated basis beams, length ``nbasis``.
    flux_here : cp.ndarray
        Source flux.
    ant1_idxs, ant2_idxs : np.ndarray
        Antenna indices for each baseline, each shape ``(nbls,)``.
    beam_coefs : np.ndarray
        Per-antenna basis coefficients, shape ``(nant, nbasis, nfreqs)``.
    freqidx : int
        Frequency index.
    topo : cp.ndarray
        Topocentric source coordinates, shape ``(3, nsrc)``.
    uvw : cp.ndarray
        Frequency-scaled baseline vectors, shape ``(3, nbls)``.
    bls : cp.ndarray
        Integer gridded baselines (type-1 path).
    tx, ty : cp.ndarray
        Frequency-scaled topo coordinates (type-1 path).
    nbls : int
        Total number of baselines.
    nfeeds : int
        Number of feed dimensions (1 or 2).
    nsim_sources : int
        Number of simulated sources.
    complex_dtype : np.dtype
        Complex dtype for accumulation.
    use_type1, is_coplanar : bool
        NUFFT mode flags.
    type1_n_modes : int
        Grid size for type-1 transform.
    eps, n_threads, upsample_factor : float / int
        NUFFT accuracy and performance parameters.
    polarized : bool
        Whether basis beams have polarization structure.
    polarized_sky_model : bool
        Whether the sky model carries full coherency.

    Returns
    -------
    cp.ndarray
        Visibilities, shape ``(nbls, nfeeds, nfeeds)``.
    """
    nbasis = len(beam_evaluations)

    vis_out = cp.zeros((nbls, nfeeds, nfeeds), dtype=complex_dtype)

    flipped = cp.zeros(nbls, dtype=bool)
    bls_idxs = cp.arange(nbls)

    if polarized:
        _apparent_buf = cp.empty((nfeeds, nfeeds, nsim_sources), dtype=complex_dtype)
    else:
        _apparent_buf = cp.empty(nsim_sources, dtype=complex_dtype)

    # V_ij = A_i^H C A_j, so the left (ant1) coefficients are conjugated.
    ant1_c = cp.asarray(beam_coefs[ant1_idxs, :, freqidx].conj())
    ant2_c = cp.asarray(beam_coefs[ant2_idxs, :, freqidx])

    for k in range(nbasis):
        for l in range(k, nbasis):
            phi_kl = _compute_apparent_coherency(
                beam_evaluations=beam_evaluations,
                bi=k,
                bj=l,
                flux_here=flux_here,
                freqidx=freqidx,
                polarized=polarized,
                polarized_sky_model=polarized_sky_model,
                nfeeds=nfeeds,
                nsim_sources=nsim_sources,
                complex_dtype=complex_dtype,
                apparent_buf=_apparent_buf,
            )

            vis_kl = _run_nufft(
                apparent_coherency=phi_kl,
                topo=topo,
                uvw=uvw,
                bls=bls,
                flipped=flipped,
                bls_idxs=bls_idxs,
                use_type1=use_type1,
                is_coplanar=is_coplanar,
                tx=tx,
                ty=ty,
                type1_n_modes=type1_n_modes,
                eps=eps,
                n_threads=n_threads,
                upsample_factor=upsample_factor,
                nfeeds=nfeeds,
            )

            w_kl = ant1_c[:, k] * ant2_c[:, l]
            vis_out += w_kl[:, None, None] * vis_kl

            if l != k:
                w_lk = ant1_c[:, l] * ant2_c[:, k]
                vis_out += w_lk[:, None, None] * vis_kl.swapaxes(1, 2)

    return vis_out


class GPUSimulationEngine(SimulationEngine):
    """GPU implementation of the simulation engine."""

    def simulate(
        self,
        ants: dict,
        freqs: np.ndarray,
        fluxes: np.ndarray,
        beam_list: list[Union[UVBeam, BeamInterface]],
        ra: np.ndarray,
        dec: np.ndarray,
        times: Union[np.ndarray, Time],
        telescope_loc: EarthLocation,
        baselines: list[tuple] = None,
        beam_idx: np.ndarray = None,
        precision: int = 2,
        polarized: bool = False,
        eps: float = None,
        upsample_factor: Literal[1.25, 2] = 2,
        beam_spline_opts: dict = None,
        flat_array_tol: float = 1e-6,
        interpolation_function: str = "az_za_map_coordinates",
        nprocesses: int | None = 1,
        nthreads: int | None = None,
        coord_method: Literal[
            "CoordinateRotationAstropy", "CoordinateRotationERFA"
        ] = "CoordinateRotationERFA",
        coord_method_params: dict | None = None,
        force_use_ray: bool = False,
        force_use_type3: bool = False,
        trace_mem: bool = False,
        enable_memory_monitor: bool = False,
        nchunks: int = 1,
        source_buffer=1.0,
        beam_coefs: np.ndarray = None,
    ) -> np.ndarray:
        """
        Simulate visibilities using the GPU implementation.

        The signature matches
        :meth:`fftvis.cpu.cpu_simulate.CPUSimulationEngine.simulate`.
        ``nprocesses``, ``nthreads``, ``force_use_ray``, ``trace_mem`` and
        ``enable_memory_monitor`` are accepted for compatibility and ignored;
        see the module docstring.

        Parameters
        ----------
        beam_coefs : np.ndarray, optional
            Per-antenna basis coefficients, shape ``(nant, K, nfreqs)``. When
            provided, ``beam_list`` is interpreted as K basis beams.

        See base class for all other parameter descriptions.
        """
        _require_cuda()

        if nprocesses not in (None, 1) or force_use_ray:
            logger.warning(
                "The GPU engine runs on a single device; nprocesses=%s and "
                "force_use_ray=%s are ignored.",
                nprocesses,
                force_use_ray,
            )

        nfreqs = np.size(freqs)
        ntimes = len(times)
        nbeam = len(beam_list)
        nant = len(ants)

        nax = nfeeds = 2 if polarized else 1

        if precision == 1:
            real_dtype = np.float32
            complex_dtype = np.complex64
        else:
            real_dtype = np.float64
            complex_dtype = np.complex128

        _gpu_beam_evaluator.precision = precision

        if eps is None:
            eps = default_accuracy_dict[precision]

        if ra.dtype != real_dtype:
            ra = ra.astype(real_dtype)
        if dec.dtype != real_dtype:
            dec = dec.astype(real_dtype)
        if freqs.dtype != real_dtype:
            freqs = freqs.astype(real_dtype)

        beam_idx = utils.validate_beam_idx(beam_idx, beam_coefs, nbeam, nant)

        if baselines is None:
            reds = utils.get_pos_reds(ants, include_autos=True)
            baselines = [red[0] for red in reds]

        nbls = len(baselines)

        coherency, polarized_sky_model = cpu_utils.prepare_source_catalog(
            fluxes, polarized_beam=polarized
        )
        if coherency.dtype != complex_dtype:
            coherency = coherency.astype(complex_dtype)

        antnums = list(ants.keys())
        antkey_to_idx = dict(zip(ants.keys(), range(len(ants))))
        antvecs = np.array([ants[ant] for ant in ants], dtype=real_dtype)

        basis_matrix = None
        n_modes = None

        if np.abs(antvecs[:, -1]).max() > flat_array_tol or force_use_type3:
            is_gridded = False
        else:
            is_gridded, gridded_antpos, basis_matrix = check_antpos_griddability(ants)

        if not is_gridded:
            rotation_matrix = utils.get_plane_to_xy_rotation_matrix(antvecs)
            rotation_matrix = np.ascontiguousarray(rotation_matrix.T)
            rotated_antvecs = np.dot(rotation_matrix, antvecs.T)
            rotated_ants = {
                ant: rotated_antvecs[:, antkey_to_idx[ant]] for ant in ants
            }
            rotation_matrix = rotation_matrix.astype(real_dtype)

            bls = np.array(
                [rotated_ants[bl[1]] - rotated_ants[bl[0]] for bl in baselines]
            )[:, :].T

            is_coplanar = np.all(np.less_equal(np.abs(bls[2]), flat_array_tol))

            bls /= utils.speed_of_light
            bls = bls.astype(real_dtype)
        else:
            logger.info(
                "Using gridded coordinates for the array. Type 1 transform will be used."
            )
            bls = np.array(
                [gridded_antpos[bl[1]] - gridded_antpos[bl[0]] for bl in baselines]
            ).T
            bls = np.round(bls).astype(int)

            n_modes = 2 * int(np.round(np.max(np.abs(bls)))) + 1

            basis_matrix *= 1 / utils.speed_of_light
            basis_matrix = basis_matrix.astype(real_dtype)

            is_coplanar = True
            rotation_matrix = np.eye(3, dtype=real_dtype)

        if isinstance(times, np.ndarray):
            times = Time(times, format="jd")

        chunk_size = int(np.ceil(dec.size / nchunks))

        coord_method = CoordinateRotation._methods[coord_method]
        coord_method_params = coord_method_params or {}

        # matvis's GPU rotators (``requires_gpu``) launch cupy kernels directly
        # on the arrays the base class allocates, so they must be constructed
        # with gpu=True or those arrays stay on the host and the kernel launch
        # fails with "trying to pass a numpy.ndarray as a kernel parameter".
        coord_on_gpu = bool(getattr(coord_method, "requires_gpu", False))

        if coord_on_gpu and polarized_sky_model:
            raise ValueError(
                f"coord_method={coord_method.__name__!r} cannot be used with a "
                "polarized sky model: its coherency-rotation branch indexes "
                "host SkyCoord arrays with device index arrays. Use "
                "coord_method='CoordinateRotationERFA' instead -- the "
                "rotation is a small fraction of the total run time."
            )

        if coord_on_gpu:
            coord_method_params = {"gpu": True, **coord_method_params}

        coord_mgr = coord_method(
            flux=coherency,
            times=times,
            telescope_loc=telescope_loc,
            skycoords=SkyCoord(ra=ra * un.rad, dec=dec * un.rad, frame="icrs"),
            precision=precision,
            source_buffer=source_buffer,
            chunk_size=chunk_size,
            **coord_method_params,
        )

        if getattr(coord_mgr, "update_bcrs_every", 0) > (times[-1] - times[0]).to(un.s):
            coord_mgr._set_bcrs(0)  # pragma: no cover

        init_time = time.time()

        vis = self._evaluate_vis_chunk(
            time_idx=slice(None),
            freq_idx=slice(None),
            beam_list=beam_list,
            coord_mgr=coord_mgr,
            rotation_matrix=rotation_matrix,
            antnums=antnums,
            baselines=baselines,
            bls=bls,
            freqs=freqs,
            complex_dtype=complex_dtype,
            nfeeds=nfeeds,
            beam_idx=beam_idx,
            polarized=polarized,
            polarized_sky_model=polarized_sky_model,
            eps=eps,
            upsample_factor=upsample_factor,
            beam_spline_opts=beam_spline_opts,
            interpolation_function=interpolation_function,
            n_threads=1,
            is_coplanar=is_coplanar,
            use_type1=is_gridded,
            basis_matrix=basis_matrix if is_gridded else None,
            type1_n_modes=n_modes if is_gridded else None,
            trace_mem=False,
            nchunks=nchunks,
            beam_coefs=beam_coefs,
        )

        logger.info(f"Main loop evaluation time: {time.time() - init_time}")

        return (
            np.transpose(vis, (4, 0, 2, 3, 1))
            if polarized
            else np.moveaxis(vis[..., 0, 0, :], 2, 0)
        )

    def _evaluate_vis_chunk(
        self,
        time_idx: slice,
        freq_idx: slice,
        beam_list: list[Union[UVBeam, BeamInterface]],
        coord_mgr: CoordinateRotation,
        rotation_matrix: np.ndarray,
        antnums: list,
        baselines: list[tuple],
        bls: np.ndarray,
        freqs: np.ndarray,
        complex_dtype: np.dtype,
        nfeeds: int,
        beam_idx: np.ndarray = None,
        polarized: bool = False,
        polarized_sky_model: bool = False,
        eps: float = None,
        upsample_factor: Literal[1.25, 2] = 2,
        beam_spline_opts: dict = None,
        interpolation_function: str = "az_za_map_coordinates",
        n_threads: int = 1,
        is_coplanar: bool = False,
        basis_matrix: np.ndarray = None,
        type1_n_modes: int = None,
        use_type1: bool = False,
        trace_mem: bool = False,
        nchunks: int = 1,
        beam_coefs: np.ndarray = None,
    ) -> np.ndarray:
        """
        Evaluate a chunk of visibility data on the GPU.

        Mirrors
        :meth:`fftvis.cpu.cpu_simulate.CPUSimulationEngine._evaluate_vis_chunk`.
        Returns a host array so the caller sees the same type as the CPU engine.

        See base class for parameter descriptions.
        """
        _require_cuda()

        nbls = bls.shape[1]
        ntimes = len(coord_mgr.times)
        nfreqs = len(freqs)

        nt_here = len(coord_mgr.times[time_idx])
        nf_here = len(freqs[freq_idx])
        vis = cp.zeros(
            (nt_here, nbls, nfeeds, nfeeds, nf_here), dtype=complex_dtype
        )

        coord_mgr.setup()

        bls_gpu = cp.asarray(bls)

        use_basis = beam_coefs is not None

        if use_basis:
            ant1_idxs = np.array([antnums.index(bl[0]) for bl in baselines])
            ant2_idxs = np.array([antnums.index(bl[1]) for bl in baselines])
        else:
            (
                unique_beam_pairs,
                beam_pair_to_bls_idxs,
                beam_pair_to_flipped,
            ) = _gpu_beam_evaluator.prepare_beam_evaluation(
                antnums=antnums,
                baselines=baselines,
                beam_idx=beam_idx,
            )
            # Hoist the per-pair index arrays onto the device once.
            gpu_bls_idxs = {
                bp: cp.asarray(np.asarray(idxs, dtype=np.int64))
                for bp, idxs in beam_pair_to_bls_idxs.items()
            }
            gpu_flipped = {
                bp: cp.asarray(np.asarray(fl, dtype=bool))
                for bp, fl in beam_pair_to_flipped.items()
            }

        is_rotation_identity = np.allclose(rotation_matrix, np.eye(3))
        checked_grid = False
        stats: dict = {}
        t_start = time.perf_counter()

        for time_index, ti in enumerate(range(ntimes)[time_idx]):
            with _stage(stats, "rotate"):
                coord_mgr.rotate(ti)

            for chunk in range(nchunks):
                with _stage(stats, "select_chunk"):
                    topo, flux, nsim_sources = coord_mgr.select_chunk(chunk, ti)

                if nsim_sources == 0:
                    continue

                topo = topo[:, :nsim_sources]
                flux = flux[:nsim_sources]

                if not use_basis:
                    if polarized:
                        _apparent_buf = cp.empty(
                            (nfeeds, nfeeds, nsim_sources), dtype=complex_dtype
                        )
                    else:
                        _apparent_buf = cp.empty(nsim_sources, dtype=complex_dtype)

                with _stage(stats, "coords"):
                    # az/za come from the unrotated topocentric coordinates.
                    # enu_to_az_za dispatches on the array module, so these are
                    # host or device arrays depending on the coordinate rotator;
                    # GPUBeamEvaluator handles either.
                    az, za = coordinates.enu_to_az_za(
                        enu_e=topo[0], enu_n=topo[1], orientation="uvbeam"
                    )

                    # No-ops when the rotator already produced device arrays.
                    topo_gpu = cp.asarray(topo)
                    flux_gpu = cp.asarray(flux)

                    if not is_rotation_identity:
                        gpu_utils.inplace_rot(rotation_matrix, topo_gpu)

                    if basis_matrix is not None:
                        gpu_utils.inplace_rot(basis_matrix.T, topo_gpu)

                    topo_gpu *= 2 * np.pi

                for freqidx in range(nfreqs)[freq_idx]:
                    freq = freqs[freqidx]

                    uvw = None
                    if not use_type1:
                        uvw = bls_gpu * freq

                        if not checked_grid:
                            checked_grid = True
                            modes, nbytes = estimate_type3_grid(
                                topo_gpu, uvw, 2 if is_coplanar else 3,
                                upsample_factor,
                            )
                            free = cp.cuda.Device().mem_info[0]
                            logger.info(
                                "Estimated type-3 grid %s (~%.2f GB); %.2f GB free",
                                modes, nbytes / 1024**3, free / 1024**3,
                            )
                            if nbytes > free:
                                raise MemoryError(
                                    f"The type-3 internal grid is estimated at "
                                    f"{nbytes / 1024**3:.1f} GB ({modes}) but only "
                                    f"{free / 1024**3:.1f} GB of device memory is "
                                    f"free. This is usually a non-coplanar array "
                                    f"with long baselines; the 3D transform is far "
                                    f"more expensive than the 2D one."
                                )

                    with _stage(stats, "beam"):
                        beam_evaluations = _evaluate_beam_list(
                            beam_list=beam_list,
                            az=az,
                            za=za,
                            polarized=polarized,
                            freq=freq,
                            beam_spline_opts=beam_spline_opts,
                            interpolation_function=interpolation_function,
                            complex_dtype=complex_dtype,
                        )

                    tx = ty = None
                    if use_type1:
                        tx = topo_gpu[0] * freq
                        ty = topo_gpu[1] * freq

                    # ---------------------------------------------------
                    # Basis visibility path
                    # ---------------------------------------------------
                    if use_basis:
                        vis_basis = _compute_basis_visibilities(
                            beam_evaluations=beam_evaluations,
                            flux_here=flux_gpu,
                            ant1_idxs=ant1_idxs,
                            ant2_idxs=ant2_idxs,
                            beam_coefs=beam_coefs,
                            freqidx=freqidx,
                            topo=topo_gpu,
                            uvw=uvw,
                            bls=bls_gpu,
                            tx=tx,
                            ty=ty,
                            nbls=nbls,
                            nfeeds=nfeeds,
                            nsim_sources=nsim_sources,
                            complex_dtype=complex_dtype,
                            use_type1=use_type1,
                            is_coplanar=is_coplanar,
                            type1_n_modes=type1_n_modes,
                            eps=eps,
                            n_threads=n_threads,
                            upsample_factor=upsample_factor,
                            polarized=polarized,
                            polarized_sky_model=polarized_sky_model,
                        )

                        # Integers and slices only, so this is a view and the
                        # in-place add writes through to `vis`.
                        vis[time_index, :, :, :, freqidx] += vis_basis

                    # ---------------------------------------------------
                    # Standard beam-pair path
                    # ---------------------------------------------------
                    else:
                        for bi, bj in unique_beam_pairs:
                            bls_idxs = gpu_bls_idxs[(bi, bj)]
                            flipped = gpu_flipped[(bi, bj)]

                            with _stage(stats, "coherency"):
                                apparent_coherency = _compute_apparent_coherency(
                                    beam_evaluations=beam_evaluations,
                                    bi=bi,
                                    bj=bj,
                                    flux_here=flux_gpu,
                                    freqidx=freqidx,
                                    polarized=polarized,
                                    polarized_sky_model=polarized_sky_model,
                                    nfeeds=nfeeds,
                                    nsim_sources=nsim_sources,
                                    complex_dtype=complex_dtype,
                                    apparent_buf=_apparent_buf,
                                )

                            with _stage(stats, "nufft"):
                                _vis_here = _run_nufft(
                                    apparent_coherency=apparent_coherency,
                                    topo=topo_gpu,
                                    uvw=uvw,
                                    bls=bls_gpu,
                                    flipped=flipped,
                                    bls_idxs=bls_idxs,
                                    use_type1=use_type1,
                                    is_coplanar=is_coplanar,
                                    tx=tx,
                                    ty=ty,
                                    type1_n_modes=type1_n_modes,
                                    eps=eps,
                                    n_threads=n_threads,
                                    upsample_factor=upsample_factor,
                                    nfeeds=nfeeds,
                                )

                            with _stage(stats, "accumulate"):
                                # Basic-index first to get a view, then scatter
                                # the baseline subset into it. Mixing an index
                                # array with scalars in one subscript works in
                                # numpy but is patchier in cupy.
                                vis[time_index, :, :, :, freqidx][bls_idxs] += _vis_here

        out = cp.asnumpy(vis)

        if STAGE_TIMING:
            stats["total"] = time.perf_counter() - t_start
            stats["_accounted"] = sum(
                v for k, v in stats.items() if not k.startswith(("total", "_"))
            )
            LAST_RUN_STATS.clear()
            LAST_RUN_STATS.update(stats)
            logger.info("GPU stage timing (s): %s", stats)

        return out
