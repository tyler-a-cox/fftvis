"""Diagnostics for the fftvis GPU backend.

The GPU beam path uses matvis's bilinear cupy kernel, which is not the same
interpolator as pyuvdata's spline-based ``az_za_simple``. Moving beam
evaluation onto the device is usually the single largest speed-up available,
but it is a numerical change, so :func:`beam_interpolation_error` measures that
change on *your* beam, sky and array rather than leaving it to be guessed at.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def beam_interpolation_error(
    *,
    baselines=None,
    verbose: bool = True,
    **simulate_kwargs,
):
    """
    Measure the visibility error from bilinear vs spline beam interpolation.

    Runs the same simulation twice on the GPU backend -- once evaluating the
    beam on the host with pyuvdata's spline interpolator, once with matvis's
    bilinear cupy kernel -- and compares. Both runs use the same backend, so
    the difference isolates the interpolator and nothing else.

    Parameters
    ----------
    baselines : list of tuple, optional
        Passed through to ``simulate_vis``. Also used to report the error as a
        function of baseline length when ``ants`` is available.
    verbose : bool
        Print a summary table.
    **simulate_kwargs
        Everything else is forwarded to :func:`fftvis.simulate_vis`. Do not
        pass ``backend``, ``interpolation_function`` or ``beam_spline_opts`` --
        this function sets them. Keep the source count modest; this runs the
        simulation twice.

    Returns
    -------
    dict
        ``max_rel``, ``rms_rel`` (relative to the peak spline visibility),
        ``vis_spline``, ``vis_bilinear``, and ``per_baseline_max_rel``.

    Examples
    --------
    >>> from fftvis.gpu.diagnostics import beam_interpolation_error  # doctest: +SKIP
    >>> stats = beam_interpolation_error(  # doctest: +SKIP
    ...     ants=h6c_antpos, fluxes=fluxes[::64], ra=ra[::64], dec=dec[::64],
    ...     freqs=freqs, times=times[:2], beam=beam,
    ...     telescope_loc=telescope_loc, baselines=h6c_baselines,
    ...     polarized=True, precision=1,
    ... )
    """
    from ..wrapper import simulate_vis

    for forbidden in ("backend", "interpolation_function", "beam_spline_opts"):
        if forbidden in simulate_kwargs:
            raise ValueError(
                f"{forbidden!r} is set by beam_interpolation_error; remove it "
                "from the call."
            )

    common = dict(simulate_kwargs)
    if baselines is not None:
        common["baselines"] = baselines

    logger.info("Running reference (host, spline) simulation...")
    vis_spline = simulate_vis(
        backend="gpu",
        interpolation_function="az_za_simple",
        beam_spline_opts=None,
        **common,
    )

    logger.info("Running bilinear (device) simulation...")
    vis_bilinear = simulate_vis(
        backend="gpu",
        interpolation_function="az_za_map_coordinates",
        beam_spline_opts={"order": 1},
        **common,
    )

    diff = np.abs(vis_bilinear - vis_spline)
    scale = np.abs(vis_spline).max()
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
        "vis_spline": vis_spline,
        "vis_bilinear": vis_bilinear,
        "per_baseline_max_rel": per_bl,
    }

    if verbose:
        print("Beam interpolation: bilinear (device) vs spline (host)")
        print(f"  visibility shape        : {vis_spline.shape}")
        print(f"  peak |V| (spline)       : {scale:.4e}")
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
                order = np.argsort(lengths)
                print("\n  error vs baseline length (quintiles):")
                print(f"    {'|b| (m)':>14}{'max |dV| / peak':>20}")
                for part in np.array_split(order, 5):
                    if len(part):
                        print(
                            f"    {lengths[part].mean():>14.1f}"
                            f"{per_bl[part].max():>20.3e}"
                        )
        print(
            "\n  Compare max |dV| / peak against your dynamic-range budget.\n"
            "  If it is too large, upsample the beam grid before simulating --\n"
            "  bilinear on a finer grid converges to the spline result."
        )

    return stats
