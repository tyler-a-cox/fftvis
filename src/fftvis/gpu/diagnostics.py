"""Diagnostics for the fftvis GPU backend.

Moving beam evaluation onto the device is usually the single largest speed-up
available to a GPU run, but it is a numerical change: an analytic beam has to
be sampled onto a grid first, and the device interpolates that grid rather than
evaluating the beam exactly. :func:`beam_interpolation_error` measures the
resulting visibility error on *your* beam, sky and array, rather than leaving
it to be guessed at.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def eps_convergence(
    *,
    eps_values: Sequence[float] = (1e-13, 1e-11, 1e-9, 1e-7, 1e-6),
    backend: str = "gpu",
    reference_eps: Optional[float] = None,
    baselines=None,
    verbose: bool = True,
    **simulate_kwargs,
):
    """
    Measure what loosening the NUFFT tolerance costs, in error and in time.

    The transform's cost grows steeply as ``eps`` tightens: finufft widens the
    spreading kernel, and the work per source scales as the kernel width to the
    power of the dimensionality. At ``eps=1e-13`` the kernel is roughly twice
    as wide as at ``eps=1e-9``, which is about 4x the spreading work in 2D.
    Whether that buys anything depends on the dynamic range your science needs,
    which this function measures rather than assumes.

    Each ``eps`` is compared against the tightest one (or ``reference_eps``),
    so the reported error is the error *introduced by loosening*, not the
    absolute error of the simulation.

    Parameters
    ----------
    eps_values : sequence of float
        Tolerances to test, tightest first.
    backend : str
        ``"gpu"`` or ``"cpu"``. Timings are only meaningful on the backend you
        intend to run.
    reference_eps : float, optional
        Tolerance to treat as truth. Defaults to ``min(eps_values)``.
    baselines : list of tuple, optional
        Passed through to ``simulate_vis``.
    verbose : bool
        Print the table.
    **simulate_kwargs
        Forwarded to :func:`fftvis.simulate_vis`. Do not pass ``eps`` or
        ``backend``. Decimate the sky; this runs the simulation once per
        tolerance.

    Returns
    -------
    dict
        ``eps_values``, ``times`` (seconds), ``max_rel``, ``rms_rel``, and
        ``reference_eps``.

    Examples
    --------
    >>> from fftvis.gpu import eps_convergence  # doctest: +SKIP
    >>> eps_convergence(  # doctest: +SKIP
    ...     ants=h6c_antpos, fluxes=fluxes[::256], ra=ra[::256], dec=dec[::256],
    ...     freqs=freqs, times=times[:2], beam=gbeam,
    ...     telescope_loc=telescope_loc, baselines=h6c_baselines,
    ...     polarized=True, precision=2, beam_spline_opts={"order": 3},
    ... )
    """
    from ..wrapper import simulate_vis

    for forbidden in ("eps", "backend"):
        if forbidden in simulate_kwargs:
            raise ValueError(
                f"{forbidden!r} is set by eps_convergence; remove it from the call."
            )

    common = dict(simulate_kwargs)
    if baselines is not None:
        common["baselines"] = baselines

    eps_values = list(eps_values)
    reference_eps = float(reference_eps if reference_eps is not None else min(eps_values))

    logger.info("Running reference at eps=%.1e ...", reference_eps)
    reference = simulate_vis(backend=backend, eps=reference_eps, **common)
    scale = np.abs(reference).max()
    if scale == 0:  # pragma: no cover - degenerate sky
        raise ValueError("Reference visibilities are identically zero.")

    times, max_rel, rms_rel = [], [], []
    for eps in eps_values:
        t0 = time.perf_counter()
        vis = simulate_vis(backend=backend, eps=eps, **common)
        times.append(time.perf_counter() - t0)
        d = np.abs(vis - reference)
        max_rel.append(float(d.max() / scale))
        rms_rel.append(float(np.sqrt((d**2).mean()) / scale))

    if verbose:
        print(f"NUFFT tolerance sweep ({backend} backend, reference eps={reference_eps:.1e})")
        print(f"  {'eps':>10}{'wall (s)':>12}{'speedup':>10}"
              f"{'max |dV|/peak':>16}{'rms |dV|/peak':>16}")
        slowest = max(times)
        for eps, t, mx, rm in zip(eps_values, times, max_rel, rms_rel):
            print(f"  {eps:>10.1e}{t:>12.2f}{slowest / t:>9.2f}x{mx:>16.2e}{rm:>16.2e}")
        print(
            "\n  The error column is what loosening eps costs you, relative to\n"
            "  the reference. Pick the loosest eps whose error sits below your\n"
            "  dynamic-range requirement.\n"
            "\n  Note: upsample_factor=1.25 does NOT help at very tight eps --\n"
            "  it trades a smaller FFT for a wider kernel, and the kernel is\n"
            "  what dominates when spreading ~10^6 sources."
        )

    return {
        "eps_values": eps_values,
        "times": times,
        "max_rel": max_rel,
        "rms_rel": rms_rel,
        "reference_eps": reference_eps,
    }


def beam_interpolation_error(
    *,
    beam,
    freqs,
    order: int = 3,
    naz: int = 721,
    nza: int = 361,
    baselines=None,
    verbose: bool = True,
    **simulate_kwargs,
):
    """
    Measure what you lose by moving beam evaluation onto the device.

    Runs the same simulation twice on the GPU backend:

    * **reference** -- ``beam`` exactly as you pass it, evaluated on the host
      with pyuvdata's ``az_za_simple`` spline. For an analytic beam this is an
      exact evaluation of the analytic function.
    * **device** -- ``beam`` sampled onto an ``naz`` x ``nza`` az/za grid by
      :func:`fftvis.core.beams.to_gridded_beam`, then interpolated on the
      device at spline order ``order``.

    Both runs use the same backend, so the difference isolates the beam
    treatment. Two separate approximations contribute: the grid resolution and
    the interpolation order.

    Parameters
    ----------
    beam : AnalyticBeam, UVBeam or BeamInterface
        The beam you intend to simulate with.
    freqs : array_like
        Simulation frequencies, also used to sample the gridded beam.
    order : int
        Spline order for the device path. ``3`` matches ``scipy.ndimage``'s
        cubic spline; ``1`` is matvis's faster fused bilinear kernel.
    naz, nza : int
        Grid size for the sampled beam.
    baselines : list of tuple, optional
        Passed through to ``simulate_vis``, and used to break the error down by
        baseline length when ``ants`` is given.
    verbose : bool
        Print a summary table.
    **simulate_kwargs
        Forwarded to :func:`fftvis.simulate_vis`. Do not pass ``backend``,
        ``interpolation_function`` or ``beam_spline_opts``. Decimate the sky --
        this runs the simulation twice.

    Returns
    -------
    dict
        ``max_rel``, ``rms_rel`` (relative to the peak reference visibility),
        ``vis_reference``, ``vis_device`` and ``per_baseline_max_rel``.

    Examples
    --------
    >>> from fftvis.gpu import beam_interpolation_error  # doctest: +SKIP
    >>> stats = beam_interpolation_error(  # doctest: +SKIP
    ...     beam=beam, freqs=freqs, order=3,
    ...     ants=h6c_antpos, fluxes=fluxes[::256],
    ...     ra=ra[::256], dec=dec[::256], times=times[:2],
    ...     telescope_loc=telescope_loc, baselines=h6c_baselines,
    ...     polarized=True, precision=1, eps=1.5e-7,
    ... )
    """
    from ..core.beams import to_gridded_beam
    from ..wrapper import simulate_vis

    for forbidden in ("backend", "interpolation_function", "beam_spline_opts"):
        if forbidden in simulate_kwargs:
            raise ValueError(
                f"{forbidden!r} is set by beam_interpolation_error; remove it "
                "from the call."
            )

    common = dict(simulate_kwargs)
    common["freqs"] = freqs
    if baselines is not None:
        common["baselines"] = baselines

    logger.info("Running reference (host, exact/spline) simulation...")
    vis_reference = simulate_vis(
        backend="gpu",
        beam=beam,
        interpolation_function="az_za_simple",
        beam_spline_opts=None,
        **common,
    )

    logger.info("Sampling the beam onto a %d x %d grid...", naz, nza)
    gridded = to_gridded_beam(
        beam,
        freqs,
        naz=naz,
        nza=nza,
        polarized=simulate_kwargs.get("polarized", True),
    )

    logger.info("Running device simulation at spline order %d...", order)
    vis_device = simulate_vis(
        backend="gpu",
        beam=gridded,
        interpolation_function="az_za_map_coordinates",
        beam_spline_opts={"order": order},
        **common,
    )

    diff = np.abs(vis_device - vis_reference)
    scale = np.abs(vis_reference).max()
    if scale == 0:  # pragma: no cover - degenerate sky
        raise ValueError("Reference visibilities are identically zero.")

    # Baseline axis position differs between the polarized and unpolarized
    # output layouts; find it from the baseline count instead of hard-coding.
    per_bl = None
    if baselines is not None:
        nbls = len(baselines)
        axes = [i for i, n in enumerate(diff.shape) if n == nbls]
        if len(axes) == 1:
            other = tuple(i for i in range(diff.ndim) if i != axes[0])
            per_bl = diff.max(axis=other) / scale

    stats = {
        "max_rel": float(diff.max() / scale),
        "rms_rel": float(np.sqrt((diff**2).mean()) / scale),
        "vis_reference": vis_reference,
        "vis_device": vis_device,
        "per_baseline_max_rel": per_bl,
    }

    if verbose:
        print(f"Beam on device (grid {naz}x{nza}, order {order}) vs host reference")
        print(f"  visibility shape        : {vis_reference.shape}")
        print(f"  peak |V| (reference)    : {scale:.4e}")
        print(f"  max  |dV| / peak        : {stats['max_rel']:.3e}")
        print(f"  rms  |dV| / peak        : {stats['rms_rel']:.3e}")
        if per_bl is not None and baselines is not None:
            ants = simulate_kwargs.get("ants")
            if ants is not None:
                lengths = np.array(
                    [
                        np.linalg.norm(np.asarray(ants[b[1]]) - np.asarray(ants[b[0]]))
                        for b in baselines
                    ]
                )
                by_length = np.argsort(lengths)
                print("\n  error vs baseline length (quintiles):")
                print(f"    {'|b| (m)':>14}{'max |dV| / peak':>20}")
                for part in np.array_split(by_length, 5):
                    if len(part):
                        print(
                            f"    {lengths[part].mean():>14.1f}"
                            f"{per_bl[part].max():>20.3e}"
                        )
        print(
            "\n  Compare max |dV| / peak against your dynamic-range budget.\n"
            "  Two knobs if it is too large: raise naz/nza (grid resolution)\n"
            "  or raise order (interpolation accuracy). Grid resolution\n"
            "  usually dominates."
        )

    return stats
