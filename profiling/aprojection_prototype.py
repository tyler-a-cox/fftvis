"""A small, self-contained, horizon-aware A-projection forward-model prototype.

This is deliberately *not* wired into :mod:`fftvis`.  It is a profiling and
validation harness for the algorithmic idea before an API or a GPU design is
committed to.  It models a scalar, coplanar array with arbitrary voltage
apertures per antenna and point sources over the complete visible hemisphere.

The important split is::

    V_ij(b) = sum_delta K_ij(delta) S_hat(b + delta)

where ``S_hat`` is the transform of the *hard-horizon-masked* sky, while
``K_ij`` is the cross-correlation of the two antenna apertures.  The horizon
therefore does not enlarge the A-kernel.  The aperture support, rather than
beam similarity, determines the number of UV samples per baseline.

The script generates intentionally different antenna apertures, and compares
the convolution result with a matvis-style exact ``Z Z^H`` source sum.  An
optional bright-rim correction evaluates only selected near-horizon sources
with the same dense-matrix method:

    V = V_Aproj(all) + V_direct(rim) - V_Aproj(rim).

Examples
--------
Validate the full-support identity (the reported error should be close to the
``--eps`` value)::

    python profiling/aprojection_prototype.py --verify

Deliberately crop the kernel, then correct bright sources within 10 degrees of
the horizon::

    python profiling/aprojection_prototype.py --verify --kernel-radius 4 \
        --correct-rim --rim-za 80 --rim-min-flux 0.1

Profile a moderately larger problem without the direct reference::

    python profiling/aprojection_prototype.py --nants 32 --nsrc 100000 \
        --ngrid 512 --reps 3

Use an exactly uniform circular aperture (a classical Airy voltage beam when
``--variation 0``), while retaining realistic antenna-to-antenna perturbations
when ``--variation`` is nonzero::

    python profiling/aprojection_prototype.py --illumination uniform --variation 0.15 \
        --compare-fftvis --verify

The aperture model is synthetic by design.  A production implementation would
replace :func:`make_apertures` with a conversion from beam grids or a physical
aperture model, keep the kernel bank cached by frequency/beam state, and move
``apply_kernels`` plus kernel construction to the GPU.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable

import finufft
import numpy as np


@dataclass(frozen=True)
class Sky:
    """Point-source sky coordinates and fluxes on the visible hemisphere."""

    lm: np.ndarray  # (nsrc, 2), direction cosines
    za: np.ndarray  # (nsrc,), zenith angle in radians
    flux: np.ndarray  # (nsrc,), real Stokes-I-like weights


@dataclass(frozen=True)
class ArrayLayout:
    """Antenna lattice locations and the corresponding unique baselines."""

    positions: np.ndarray  # (nant, 2), integer UV-grid cells
    pairs: np.ndarray  # (nbl, 2), antenna indices
    baselines: np.ndarray  # (nbl, 2), integer UV-grid cells


def _median_time(fn, reps: int) -> tuple[float, object]:
    """Time a callable, returning its median time and final result."""
    result = fn()  # warm-up: plan construction and imports should not dominate
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times)), result


def make_sky(nsrc: int, seed: int, bright_horizon_sources: int = 4) -> Sky:
    """Generate a source catalogue covering the full visible hemisphere.

    Sampling ``cos(za)`` uniformly, rather than ``l``/``m`` uniformly, yields
    a catalogue uniform in solid angle and naturally populates the horizon.
    A few bright sources are injected just above it so a rim correction has a
    meaningful, repeatable workload to target.
    """
    if nsrc < bright_horizon_sources:
        raise ValueError("nsrc must be >= bright_horizon_sources")

    rng = np.random.default_rng(seed)
    az = rng.uniform(0.0, 2.0 * np.pi, nsrc)
    cos_za = rng.uniform(0.0, 1.0, nsrc)
    za = np.arccos(cos_za)
    radius = np.sqrt(1.0 - cos_za**2)
    lm = np.column_stack((radius * np.cos(az), radius * np.sin(az)))

    # A shallow source-count distribution with a non-pathological finite mean.
    flux = (0.05 + rng.pareto(1.5, nsrc)).astype(np.float64)

    # Make a reproducible set of difficult, bright, nearly horizon sources.
    if bright_horizon_sources:
        horizon_za = np.deg2rad(np.linspace(89.1, 89.8, bright_horizon_sources))
        horizon_az = np.linspace(0.23, 5.71, bright_horizon_sources)
        za[:bright_horizon_sources] = horizon_za
        lm[:bright_horizon_sources] = np.column_stack(
            (
                np.sin(horizon_za) * np.cos(horizon_az),
                np.sin(horizon_za) * np.sin(horizon_az),
            )
        )
        flux[:bright_horizon_sources] = flux.max() * np.linspace(
            0.40, 1.00, bright_horizon_sources
        )

    return Sky(lm=lm, za=za, flux=flux)


def make_layout(nant: int, array_radius_cells: int, seed: int) -> ArrayLayout:
    """Place antennas on a random integer UV lattice and form all i <= j pairs."""
    if array_radius_cells < 1:
        raise ValueError("array_radius_cells must be positive")

    rng = np.random.default_rng(seed)
    radius_sq = array_radius_cells**2
    candidates = np.array(
        [
            (x, y)
            for x in range(-array_radius_cells, array_radius_cells + 1)
            for y in range(-array_radius_cells, array_radius_cells + 1)
            if x * x + y * y <= radius_sq
        ],
        dtype=np.int64,
    )
    if nant > len(candidates):
        raise ValueError(
            f"{nant} antennas do not fit in a radius-{array_radius_cells} lattice."
        )
    positions = candidates[rng.choice(len(candidates), size=nant, replace=False)]

    pairs = np.array(
        [(i, j) for i in range(nant) for j in range(i, nant)], dtype=np.int64
    )
    baselines = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    return ArrayLayout(positions=positions, pairs=pairs, baselines=baselines)


def make_apertures(
    nant: int,
    ngrid: int,
    du: float,
    aperture_radius: float,
    variation: float,
    seed: int,
    illumination: str = "cosine",
) -> np.ndarray:
    """Create compact but antenna-dependent complex aperture illuminations.

    The differing phase screen, taper, and sub-aperture ripple make the far
    field beams quite distinct without changing their physical support.  That
    is precisely the regime in which a kernel method should remain viable even
    when an image-domain SVD has no attractive low rank.
    """
    if not 0.0 <= variation <= 1.0:
        raise ValueError("variation must be in [0, 1]")
    if illumination not in {"uniform", "cosine"}:
        raise ValueError("illumination must be 'uniform' or 'cosine'")

    cell_radius = int(np.ceil(aperture_radius / du))
    center = ngrid // 2
    y, x = np.indices((ngrid, ngrid))
    xx = (x - center) * du
    yy = (y - center) * du
    rr = np.hypot(xx, yy)
    support = rr <= aperture_radius

    # A uniform circular aperture produces the familiar Airy voltage pattern
    # (2 J1(x) / x).  The cosine option is the previous lower-sidelobe model.
    # Both remain *exactly* compact in aperture space, so their full A-kernels
    # have the same finite support; only their coefficients differ.
    taper = np.zeros_like(rr)
    if illumination == "uniform":
        taper[support] = 1.0
    else:
        taper[support] = np.cos(0.5 * np.pi * rr[support] / aperture_radius) ** 1.5

    rng = np.random.default_rng(seed)
    apertures = np.empty((nant, ngrid, ngrid), dtype=np.complex128)
    theta = np.arctan2(yy, xx)
    for ant in range(nant):
        # Low-order aperture errors make distinct, visibly asymmetric beams;
        # without them, illumination="uniform" is a classical Airy beam.
        # They do not grant the algorithm any beam-similarity shortcut.
        p0, p1, p2 = rng.normal(scale=variation, size=3)
        a0, a1 = rng.normal(scale=variation, size=2)
        phase = p0 * xx / aperture_radius + p1 * yy / aperture_radius
        phase += p2 * (rr / aperture_radius) ** 2
        amp = 1.0 + 0.30 * a0 * (xx / aperture_radius)
        amp += 0.20 * a1 * np.cos(3.0 * theta)
        aperture = taper * amp * np.exp(1j * phase)
        aperture[~support] = 0.0

        # Normalize every voltage beam to one at zenith.  The du**2 is the
        # Riemann weight used when transforming aperture -> far-field voltage.
        zenith = du**2 * aperture.sum()
        apertures[ant] = aperture / zenith

    # This is intentionally retained as an invariant for kernel support below.
    assert cell_radius * 2 < ngrid // 2
    return apertures


def voltage_beams_at_sources(apertures: np.ndarray, sky: Sky, du: float) -> np.ndarray:
    """Evaluate scalar voltage beams directly from their compact apertures."""
    ngrid = apertures.shape[-1]
    center = ngrid // 2
    nonzero = np.any(apertures != 0.0, axis=0)
    iy, ix = np.nonzero(nonzero)
    aperture_uv = np.column_stack(((ix - center) * du, (iy - center) * du))
    values = apertures[:, iy, ix]

    phase = np.exp(2j * np.pi * (sky.lm @ aperture_uv.T))
    return du**2 * (values @ phase.T)


def kernel_bank(
    apertures: np.ndarray,
    du: float,
    kernel_radius_cells: int,
) -> np.ndarray:
    """Return K[dy, dx, i, j] = du^4 sum_p conj(a_i[p]) a_j[p + delta].

    This is the forward-model A-kernel.  The implementation deliberately forms
    all antenna-pair kernels at a given offset with a GEMM.  A GPU version
    should use batched/tensor-core GEMMs here and cache the result by frequency
    and beam state; it must not run one FFT/NUFFT for every beam pair.
    """
    nant, _, _ = apertures.shape
    # Correlation is identically zero outside the union of the physical
    # apertures.  Restricting each GEMM to this compact window is not an
    # approximation: it prevents a prototype-only O(N_uv^2) cost from hiding
    # the intended O(N_aperture) construction cost.
    occupied = np.any(apertures != 0.0, axis=0)
    iy, ix = np.nonzero(occupied)
    y0, y1 = iy.min(), iy.max() + 1
    x0, x1 = ix.min(), ix.max() + 1
    apertures = apertures[:, y0:y1, x0:x1]
    _, ny, nx = apertures.shape
    offsets = range(-kernel_radius_cells, kernel_radius_cells + 1)
    side = 2 * kernel_radius_cells + 1
    out = np.empty((side, side, nant, nant), dtype=np.complex128)

    for out_y, dy in enumerate(offsets):
        if abs(dy) >= ny:
            out[out_y, :, :, :] = 0.0
            continue
        if dy >= 0:
            left_y, right_y = slice(0, ny - dy), slice(dy, ny)
        else:
            left_y, right_y = slice(-dy, ny), slice(0, ny + dy)

        for out_x, dx in enumerate(offsets):
            if abs(dx) >= nx:
                out[out_y, out_x] = 0.0
                continue
            if dx >= 0:
                left_x, right_x = slice(0, nx - dx), slice(dx, nx)
            else:
                left_x, right_x = slice(-dx, nx), slice(0, nx + dx)

            left = apertures[:, left_y, left_x].reshape(nant, -1)
            right = apertures[:, right_y, right_x].reshape(nant, -1)
            out[out_y, out_x] = left.conj() @ right.T

    return out * du**4


def sky_uv_grid(sky: Sky, ngrid: int, du: float, eps: float) -> np.ndarray:
    """Compute S_hat on a regular UV grid with the hard horizon in the sky.

    ``modeord=1`` makes index zero the DC mode and signed integer UV cells
    addressable with Python's normal modulo indexing.  Since all sources are
    already restricted to the visible hemisphere, no horizon edge is placed in
    the per-baseline A-kernel.
    """
    x = np.ascontiguousarray(2.0 * np.pi * du * sky.lm[:, 0])
    y = np.ascontiguousarray(2.0 * np.pi * du * sky.lm[:, 1])
    weights = np.ascontiguousarray(sky.flux.astype(np.complex128))
    return finufft.nufft2d1(
        x,
        y,
        weights,
        ngrid,
        isign=1,
        modeord=1,
        eps=eps,
        showwarn=0,
    )


def check_sky_grid(sky: Sky, grid: np.ndarray, du: float) -> float:
    """Check the type-1 transform convention against several direct modes."""
    ngrid = grid.shape[0]
    checks: Iterable[tuple[int, int]] = ((0, 0), (1, -2), (-3, 4), (7, 5))
    errors = []
    for ku, kv in checks:
        direct = np.sum(
            sky.flux
            * np.exp(2j * np.pi * du * (ku * sky.lm[:, 0] + kv * sky.lm[:, 1]))
        )
        errors.append(abs(grid[ku % ngrid, kv % ngrid] - direct))
    return max(errors) / max(np.abs(sky.flux).sum(), np.finfo(float).tiny)


def apply_kernels(
    sky_uv: np.ndarray,
    kernels: np.ndarray,
    layout: ArrayLayout,
    baseline_batch: int,
) -> np.ndarray:
    """Convolve a single UV sky grid with selected baseline A-kernels."""
    ngrid = sky_uv.shape[0]
    kr = (kernels.shape[0] - 1) // 2
    offsets = np.arange(-kr, kr + 1, dtype=np.int64)
    dx, dy = np.meshgrid(offsets, offsets, indexing="xy")
    out = np.empty(len(layout.pairs), dtype=np.complex128)

    # This chunking is essential at realistic array sizes: the temporary UV
    # gather is nbaseline_batch x kernel_cells rather than nbl x kernel_cells.
    for start in range(0, len(layout.pairs), baseline_batch):
        stop = min(start + baseline_batch, len(layout.pairs))
        baseline = layout.baselines[start:stop]
        samples = sky_uv[
            (baseline[:, None, None, 0] + dx[None]) % ngrid,
            (baseline[:, None, None, 1] + dy[None]) % ngrid,
        ]
        pair = layout.pairs[start:stop]
        pair_kernels = kernels[:, :, pair[:, 0], pair[:, 1]].transpose(2, 0, 1)
        out[start:stop] = np.einsum("bij,bij->b", pair_kernels, samples)
    return out


def direct_visibilities(
    beams: np.ndarray,
    sky: Sky,
    layout: ArrayLayout,
    du: float,
    source_mask: np.ndarray | None,
    baseline_batch: int,
) -> np.ndarray:
    """Exact scalar measurement equation for selected sources and baselines."""
    if source_mask is None:
        source_mask = np.ones(len(sky.flux), dtype=bool)
    lm = sky.lm[source_mask]
    flux = sky.flux[source_mask]
    beam = beams[:, source_mask]
    out = np.empty(len(layout.pairs), dtype=np.complex128)

    for start in range(0, len(layout.pairs), baseline_batch):
        stop = min(start + baseline_batch, len(layout.pairs))
        pair = layout.pairs[start:stop]
        baseline = layout.baselines[start:stop]
        apparent_sky = (
            beam[pair[:, 0]].conj() * beam[pair[:, 1]] * flux[None, :]
        )
        phase = np.exp(2j * np.pi * du * (baseline @ lm.T))
        out[start:stop] = np.einsum("bs,bs->b", apparent_sky, phase)
    return out


def matvis_style_visibilities(
    beams: np.ndarray,
    sky: Sky,
    layout: ArrayLayout,
    du: float,
    source_mask: np.ndarray | None,
) -> np.ndarray:
    """Evaluate selected sources exactly using matvis's ``V = Z Z^H`` idea.

    With scalar voltage beams, define::

        Z[i, s] = J_i(s) sqrt(I_s) exp(2 pi i r_i . l_s).

    Then ``Z.conj() @ Z.T`` contains every visibility simultaneously.  This
    has the same arithmetic scaling as a baseline/source sum, but presents the
    work as one dense Hermitian rank-k product, which maps to BLAS/cuBLAS in
    exactly the way matvis does.  It also avoids a Python loop over baselines.

    The full-polarization version stacks feed rows and E-field/source columns
    before the same matrix product.  The standalone prototype remains scalar.
    """
    if source_mask is None:
        source_mask = np.ones(len(sky.flux), dtype=bool)
    if not np.any(source_mask):
        return np.zeros(len(layout.pairs), dtype=np.complex128)

    lm = sky.lm[source_mask]
    sqrt_flux = np.sqrt(sky.flux[source_mask])
    phase = np.exp(2j * np.pi * du * (layout.positions @ lm.T))
    z = beams[:, source_mask] * sqrt_flux[None, :] * phase
    visibility_matrix = z.conj() @ z.T
    return visibility_matrix[layout.pairs[:, 0], layout.pairs[:, 1]]


def fftvis_style_visibilities(
    beams: np.ndarray,
    sky: Sky,
    layout: ArrayLayout,
    du: float,
    eps: float,
    nthreads: int,
    method: str,
    upsample_factor: float,
) -> np.ndarray:
    """Mirror fftvis's all-unique-beam computation for this scalar problem.

    Every antenna owns a unique voltage beam, so every baseline belongs to a
    distinct beam-pair group.  Like fftvis's standard path, this constructs a
    new apparent sky and invokes one NUFFT for every such group.  Each group
    has just one target baseline -- the pathological regime the A-projection
    prototype is intended to remove.

    ``method='type1'`` mirrors fftvis's branch for a griddable, coplanar array.
    It therefore computes a full regular UV grid *for every beam pair* then
    selects one mode.  ``method='type3'`` mirrors the arbitrary-array branch
    and computes a single-target type-3 NUFFT for every beam pair.

    This purposefully does not call :func:`fftvis.simulate_vis`: the synthetic
    apertures here are not UVBeam instances.  It does, however, use the same
    FINUFFT calls, coordinate convention, per-pair loop structure, and
    apparent-sky operation as fftvis's scalar polarized measurement equation.
    It is consequently a useful lower-bound comparison: it omits pyuvdata beam
    interpolation, Python beam-pair bookkeeping, and result scattering.
    """
    if method not in {"type1", "type3"}:
        raise ValueError("method must be 'type1' or 'type3'")

    # fftvis multiplies the source direction cosines by 2*pi before passing
    # them to its type-3 wrapper.  For its gridded type-1 branch, the lattice
    # coordinate transform adds the equivalent du factor to the source points.
    x_type3 = np.ascontiguousarray(2.0 * np.pi * sky.lm[:, 0])
    y_type3 = np.ascontiguousarray(2.0 * np.pi * sky.lm[:, 1])
    x_type1 = np.ascontiguousarray(x_type3 * du)
    y_type1 = np.ascontiguousarray(y_type3 * du)
    n_modes = 2 * int(np.abs(layout.baselines).max()) + 1
    out = np.empty(len(layout.pairs), dtype=np.complex128)

    for ibl, (ant_i, ant_j) in enumerate(layout.pairs):
        # Scalar form of A_i^H C A_j.  This is intentionally evaluated inside
        # the pair loop, just as fftvis creates apparent_coherency per group.
        weights = np.ascontiguousarray(
            beams[ant_i].conj() * beams[ant_j] * sky.flux
        )
        bu, bv = layout.baselines[ibl]
        if method == "type3":
            vis = finufft.nufft2d3(
                x_type3,
                y_type3,
                weights,
                np.array([bu * du], dtype=float),
                np.array([bv * du], dtype=float),
                isign=1,
                modeord=0,
                eps=eps,
                nthreads=nthreads,
                showwarn=0,
                upsampfac=upsample_factor,
            )
            out[ibl] = vis[0]
        else:
            model = finufft.nufft2d1(
                x_type1,
                y_type1,
                weights,
                n_modes,
                isign=1,
                modeord=1,
                eps=eps,
                nthreads=nthreads,
                showwarn=0,
                upsampfac=upsample_factor,
            )
            out[ibl] = model[bu % n_modes, bv % n_modes]

    return out


def relative_errors(got: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Return peak-normalized max and RMS visibility errors."""
    scale = max(float(np.abs(reference).max()), np.finfo(float).tiny)
    delta = got - reference
    return float(np.abs(delta).max() / scale), float(np.sqrt(np.mean(abs(delta) ** 2)) / scale)


def parse_args() -> argparse.Namespace:
    """Parse command-line configuration for one self-contained experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nants", type=int, default=12)
    parser.add_argument("--nsrc", type=int, default=4_000)
    parser.add_argument("--ngrid", type=int, default=256)
    parser.add_argument("--du", type=float, default=0.40, help="UV grid spacing in wavelengths")
    parser.add_argument("--array-radius", type=float, default=12.0, help="antenna radius in wavelengths")
    parser.add_argument("--aperture-radius", type=float, default=3.5, help="aperture radius in wavelengths")
    parser.add_argument(
        "--illumination",
        choices=("uniform", "cosine"),
        default="cosine",
        help="uniform = Airy voltage beam at variation=0; cosine = tapered aperture",
    )
    parser.add_argument(
        "--kernel-radius",
        type=float,
        default=None,
        help="kernel radius in wavelengths; default is its exact aperture-limited support",
    )
    parser.add_argument("--variation", type=float, default=0.65, help="independent aperture variation [0, 1]")
    parser.add_argument("--eps", type=float, default=1e-10, help="type-1 NUFFT tolerance")
    parser.add_argument("--upsample-factor", type=float, default=2.0, choices=(1.25, 2.0))
    parser.add_argument("--baseline-batch", type=int, default=512)
    parser.add_argument("--reps", type=int, default=1, help="median profile repetitions after warm-up")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run the exact matvis-style Z Z^H visibility reference",
    )
    parser.add_argument(
        "--compare-fftvis",
        action="store_true",
        help="benchmark the one-NUFFT-per-unique-beam-pair fftvis-style path",
    )
    parser.add_argument(
        "--fftvis-method",
        choices=("type1", "type3"),
        default="type1",
        help="fftvis branch to mimic; type1 is the griddable-array branch",
    )
    parser.add_argument(
        "--fftvis-nthreads",
        type=int,
        default=1,
        help="threads per FFTvis-style FINUFFT call",
    )
    parser.add_argument(
        "--correct-rim",
        action="store_true",
        help="add exact-minus-A-projection correction for selected horizon sources",
    )
    parser.add_argument("--rim-za", type=float, default=80.0, help="rim begins at this zenith angle in degrees")
    parser.add_argument(
        "--rim-min-flux",
        type=float,
        default=0.10,
        help="correct rim sources above this fraction of the catalogue peak flux",
    )
    parser.add_argument(
        "--assert-tol",
        type=float,
        default=None,
        help="fail if full A-projection's peak-normalized error exceeds this value",
    )
    return parser.parse_args()


def main() -> None:
    """Run the A-projection experiment and print timings plus validation errors."""
    args = parse_args()
    if not 0.0 < args.du < 0.5:
        raise ValueError("du must lie in (0, 0.5) to keep the horizon inside one image period")
    if args.reps < 1 or args.baseline_batch < 1 or args.fftvis_nthreads < 1:
        raise ValueError("reps, baseline-batch, and fftvis-nthreads must be positive")

    aperture_radius_cells = int(np.ceil(args.aperture_radius / args.du))
    full_kernel_radius_cells = 2 * aperture_radius_cells
    kernel_radius_cells = (
        full_kernel_radius_cells
        if args.kernel_radius is None
        else int(np.ceil(args.kernel_radius / args.du))
    )
    if not 0 <= kernel_radius_cells <= full_kernel_radius_cells:
        raise ValueError(
            "kernel-radius must be non-negative and no larger than the aperture-limited support "
            f"({full_kernel_radius_cells * args.du:.3g} wavelengths)."
        )

    array_radius_cells = int(np.floor(args.array_radius / args.du))
    # Need enough UV room for b + delta; otherwise periodic array indexing
    # aliases the convolution rather than representing the requested modes.
    required_half_grid = 2 * array_radius_cells + kernel_radius_cells + 2
    if required_half_grid >= args.ngrid // 2:
        raise ValueError(
            "ngrid is too small for the requested array and kernel support: need "
            f"ngrid > {2 * required_half_grid}, got {args.ngrid}."
        )

    sky = make_sky(args.nsrc, args.seed)
    layout = make_layout(args.nants, array_radius_cells, args.seed + 1)
    apertures = make_apertures(
        args.nants,
        args.ngrid,
        args.du,
        args.aperture_radius,
        args.variation,
        args.seed + 2,
        args.illumination,
    )

    rim_mask = (sky.za >= np.deg2rad(args.rim_za)) & (
        sky.flux >= args.rim_min_flux * sky.flux.max()
    )

    # Beam evaluation is only an oracle/correction cost.  Do not accidentally
    # include it in an A-projection-only profile, where apertures are consumed
    # directly and no image-domain beam values are required.  If the rim is the
    # only exact operation, evaluate voltage beams only for its selected sources
    # before forming its matvis-style Z matrix.
    beams = None
    beam_sky = None
    t_beam = 0.0
    need_full_beams = args.verify or args.compare_fftvis
    if need_full_beams:
        beam_sky = sky
    elif args.correct_rim:
        beam_sky = Sky(
            lm=sky.lm[rim_mask], za=sky.za[rim_mask], flux=sky.flux[rim_mask]
        )

    if beam_sky is not None:
        t_beam, beams = _median_time(
            lambda: voltage_beams_at_sources(apertures, beam_sky, args.du), args.reps
        )
    t_kernel, kernels = _median_time(
        lambda: kernel_bank(apertures, args.du, kernel_radius_cells), args.reps
    )
    t_sky, sky_uv = _median_time(
        lambda: sky_uv_grid(sky, args.ngrid, args.du, args.eps), args.reps
    )
    t_apply, aproj_all = _median_time(
        lambda: apply_kernels(sky_uv, kernels, layout, args.baseline_batch), args.reps
    )

    aproj_corrected = None
    t_rim_sky = t_rim_apply = t_rim_direct = 0.0
    if args.correct_rim:
        rim_sky = Sky(lm=sky.lm, za=sky.za, flux=sky.flux * rim_mask)
        t_rim_sky, rim_uv = _median_time(
            lambda: sky_uv_grid(rim_sky, args.ngrid, args.du, args.eps), args.reps
        )
        t_rim_apply, aproj_rim = _median_time(
            lambda: apply_kernels(rim_uv, kernels, layout, args.baseline_batch), args.reps
        )
        if need_full_beams:
            exact_rim = lambda: matvis_style_visibilities(  # noqa: E731
                beams, sky, layout, args.du, rim_mask
            )
        else:
            exact_rim = lambda: matvis_style_visibilities(  # noqa: E731
                beams, beam_sky, layout, args.du, None
            )
        t_rim_direct, direct_rim = _median_time(exact_rim, args.reps)
        aproj_corrected = aproj_all + direct_rim - aproj_rim

    direct_all = None
    t_direct = 0.0
    if args.verify:
        t_direct, direct_all = _median_time(
            lambda: matvis_style_visibilities(
                beams, sky, layout, args.du, None
            ),
            args.reps,
        )

    fftvis_result = None
    t_fftvis = 0.0
    if args.compare_fftvis:
        t_fftvis, fftvis_result = _median_time(
            lambda: fftvis_style_visibilities(
                beams,
                sky,
                layout,
                args.du,
                args.eps,
                args.fftvis_nthreads,
                args.fftvis_method,
                args.upsample_factor,
            ),
            args.reps,
        )

    print("A-projection prototype (scalar, coplanar, hard horizon in sky grid)")
    print(
        f"  antennas={args.nants}  baselines={len(layout.pairs):,}  "
        f"sources={args.nsrc:,}  grid={args.ngrid}^2  du={args.du:g} λ"
    )
    print(
        f"  aperture radius={args.aperture_radius:g} λ ({aperture_radius_cells} cells)  "
        f"illumination={args.illumination}  variation={args.variation:g}  "
        f"kernel radius={kernel_radius_cells * args.du:g} λ "
        f"({2 * kernel_radius_cells + 1}^2 = {(2 * kernel_radius_cells + 1) ** 2:,} cells)"
    )
    print(
        f"  max |baseline|={np.linalg.norm(layout.baselines * args.du, axis=1).max():.3g} λ  "
        f"nearest catalogue horizon source: za={np.rad2deg(sky.za.max()):.3f} deg"
    )
    print()
    print("Stage timings (median after one warm-up)")
    if beams is None:
        print("  direct aperture -> source voltage beams:       not run")
    else:
        print(
            f"  direct aperture -> {len(beam_sky.flux):,} source voltage beams: "
            f"{t_beam * 1e3:9.2f} ms"
        )
    print(f"  construct all-pair kernel bank:          {t_kernel * 1e3:9.2f} ms")
    print(f"  one hard-horizon sky type-1 NUFFT:       {t_sky * 1e3:9.2f} ms")
    print(f"  baseline kernel gathers/convolutions:    {t_apply * 1e3:9.2f} ms")
    print(f"  A-projection total (cached kernels):     {(t_sky + t_apply) * 1e3:9.2f} ms")

    grid_error = check_sky_grid(sky, sky_uv, args.du)
    print(f"  type-1 grid convention check:            {grid_error:9.2e}")

    if args.correct_rim:
        print()
        print(
            f"Bright rim correction: {rim_mask.sum():,} sources at za >= {args.rim_za:g} deg "
            f"and flux >= {args.rim_min_flux:g} * max(flux)"
        )
        print(f"  rim sky type-1 NUFFT:                    {t_rim_sky * 1e3:9.2f} ms")
        print(f"  rim A-projection:                        {t_rim_apply * 1e3:9.2f} ms")
        print(f"  exact matvis-style rim correction:       {t_rim_direct * 1e3:9.2f} ms")

    if direct_all is not None:
        print()
        print(f"  exact matvis-style Z Z^H reference:      {t_direct * 1e3:9.2f} ms")
        max_error, rms_error = relative_errors(aproj_all, direct_all)
        print(f"  all-sky A-projection error: max={max_error:.2e}  rms={rms_error:.2e}")
        if aproj_corrected is not None:
            max_error_corrected, rms_error_corrected = relative_errors(
                aproj_corrected, direct_all
            )
            print(
                "  rim-corrected A-projection error: "
                f"max={max_error_corrected:.2e}  rms={rms_error_corrected:.2e}"
            )
        if args.assert_tol is not None and max_error > args.assert_tol:
            raise SystemExit(
                f"A-projection error {max_error:.3e} exceeds --assert-tol {args.assert_tol:.3e}."
            )

    if fftvis_result is not None:
        max_error_fftvis, rms_error_fftvis = relative_errors(aproj_all, fftvis_result)
        print()
        print(
            f"  fftvis-style {args.fftvis_method} ({len(layout.pairs):,} NUFFTs, "
            f"{args.fftvis_nthreads} thread(s)/call): {t_fftvis * 1e3:9.2f} ms"
        )
        print(
            "  A-projection vs fftvis-style: "
            f"max={max_error_fftvis:.2e}  rms={rms_error_fftvis:.2e}  "
            f"steady-state speedup={t_fftvis / (t_sky + t_apply):.1f}x"
        )


if __name__ == "__main__":
    main()
