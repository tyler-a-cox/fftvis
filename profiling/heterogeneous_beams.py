"""Options for fftvis when the per-antenna beams do NOT compress under SVD.

Run: python profiling/heterogeneous_beams.py

Three independent results:

1. Transform-count economics for the basis path, and a check of
   ``compute_beam_basis``'s ``threshold`` default. The break-even against the
   standard path is ``K < Nbeam`` -- *any* compression wins -- so "the beams
   don't compress" is a much stronger claim than it sounds.

2. Compressing the RESIDUAL rather than the beam. Splitting
   ``A_i = A_0 + d_i`` makes the pair-dependent work second order in
   ``|d|/|A_0|``, so the SVD only has to work on ``d``, and only loosely.

3. A-projection: grid the sky ONCE with no beam applied, then apply each
   baseline's beam pair as a small convolution in the uv plane. Cost becomes
   independent of whether the beams resemble each other at all.
"""

from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(11)


# ---------------------------------------------------------------------------
# Shared: a deliberately hard-to-compress beam ensemble
# ---------------------------------------------------------------------------
def hard_beam_ensemble(nbeam=350, npix=96, ncoef=4, amp=0.05, noise=0.0, rng=RNG):
    """Beams with an independent smooth random perturbation field per antenna.

    This is the pessimistic case: no shared low-dimensional structure beyond
    the common envelope, so a global SVD compresses poorly.

    ``noise`` adds independent per-pixel scatter, which makes the ensemble
    genuinely FULL RANK (no singular value is ever zero). Use it to check that
    a conclusion does not depend on the smooth model's finite mode count.
    """
    x = np.linspace(-1, 1, npix)
    L, M = np.meshgrid(x, x, indexing="ij")
    horizon = np.hypot(L, M) < 1
    a0 = np.exp(-0.5 * (np.hypot(L, M) / 0.18) ** 2) * horizon

    basis = np.array(
        [
            f(np.pi * i * L) * g(np.pi * j * M)
            for i in range(ncoef)
            for j in range(ncoef)
            for f in (np.cos, np.sin)
            for g in (np.cos, np.sin)
            if (i, j) != (0, 0)
        ]
    )
    coefs = amp * rng.standard_normal((nbeam, len(basis)))
    B = a0 * (1 + np.einsum("nk,kij->nij", coefs, basis))
    if noise:
        B = B + noise * a0 * rng.standard_normal(B.shape)
    return B.reshape(nbeam, -1), a0


def tail_error(sv):
    """Relative Frobenius error of a rank-K truncation, for every K."""
    return np.sqrt(np.cumsum(sv[::-1] ** 2)[::-1]) / np.linalg.norm(sv)


# ---------------------------------------------------------------------------
# 1. Transform-count economics and the threshold default
# ---------------------------------------------------------------------------
def economics_report(nbeam=350):
    """How many NUFFTs the basis path needs vs the standard path."""
    print("=" * 76)
    print("1. TRANSFORM COUNT: WHAT 'DOESN'T COMPRESS' ACTUALLY COSTS")
    print("=" * 76)
    std = nbeam * (nbeam + 1) // 2
    print(f"  Nbeam={nbeam}: standard path needs {std:,} NUFFTs per (time, chunk, freq)")
    print("  (with one unique beam per antenna, each of those transforms has a")
    print("   SINGLE target baseline -- the NUFFT's amortisation is entirely lost)\n")

    # Two ensembles: band-limited (rank capped by construction) and one with
    # per-pixel scatter added, which is mathematically full rank.
    for label, noise in [("smooth perturbations", 0.0), ("+ per-pixel scatter (FULL RANK)", 0.01)]:
        B, _ = hard_beam_ensemble(nbeam, noise=noise)
        sv = np.linalg.svd(B, compute_uv=False)
        tail = tail_error(sv)
        print(f"  ensemble: {label}")
        print(f"    {'target vis error':>18}{'K':>6}{'NUFFTs = K(K+1)/2':>20}{'speedup':>10}")
        for tol in (1e-2, 1e-3, 1e-4):
            K = int(np.argmax(tail < tol) + 1) if (tail < tol).any() else nbeam
            n = K * (K + 1) // 2
            print(f"    {tol:>18.0e}{K:>6}{n:>20,}{std / n:>9.0f}x")

        thresh = 1e-12
        K_default = int(np.sum(sv / sv[0] >= thresh))
        nd = max(K_default * (K_default + 1) // 2, 1)
        print(f"    compute_beam_basis(threshold={thresh:g}) keeps K = {K_default} "
              f"of {nbeam}  ->  {std / nd:.1f}x\n")
    print(
        "\n  The default threshold is a singular-value floor near machine epsilon,\n"
        "  so it retains essentially every mode by construction. It also thresholds\n"
        "  INDIVIDUAL singular values, whereas the quantity that sets the error is\n"
        "  the TAIL ENERGY sqrt(sum_{k>K} s_k^2)/||s||. Choose K from an error\n"
        "  budget, not from a floor."
    )


# ---------------------------------------------------------------------------
# 2. Compress the residual, not the beam
# ---------------------------------------------------------------------------
def residual_report(nbeam=350):
    """Rank needed for the beam itself vs for the mean-subtracted residual."""
    print()
    print("=" * 76)
    print("2. COMPRESS THE RESIDUAL, NOT THE BEAM")
    print("=" * 76)
    print("  A_i A_j* = A0 A0*        <- 1 transform, fully redundant across baselines")
    print("           + A0 d_j*       <- Nant transforms (depends on j only)")
    print("           + d_i A0*       <- free: conj of the above at -b")
    print("           + d_i d_j*      <- the only Nant^2 term, and it is second order\n")
    print("  So only the d-d term needs the SVD, and it is already suppressed by")
    print("  |d|^2/|A|^2. The gain over compressing A directly scales as ~1/(2|d|/|A|),")
    print("  i.e. the LESS your beams vary, the more this buys you.\n")

    print(f"    {'|d|/|A|':>9}{'K':>5}{'rel err on A':>15}{'vis err via d-d':>18}{'gain':>8}")
    for amp in (0.02, 0.05, 0.10):
        B, _ = hard_beam_ensemble(nbeam, amp=amp)
        A0 = B.mean(axis=0)
        D = B - A0
        delta = np.linalg.norm(D) / np.linalg.norm(B)
        sv_full = tail_error(np.linalg.svd(B, compute_uv=False))
        sv_res = tail_error(np.linalg.svd(D, compute_uv=False))
        for K in (8, 16):
            # approximating d to relative error e perturbs a term of size
            # delta^2 by ~2e, so the visibility error is 2 * e * delta^2
            vis = 2 * sv_res[K - 1] * delta**2
            print(f"    {delta:>9.3f}{K:>5}{sv_full[K - 1]:>15.2e}{vis:>18.2e}"
                  f"{sv_full[K - 1] / vis:>7.1f}x")

    print(
        "\n  Demanding 1e-4 from the SVD of A may be hopeless, but the d-d term\n"
        "  only needs the SVD of d to be good to O(1). A rank the full-beam SVD\n"
        "  would call useless is ample once the mean beam is split off."
    )


# ---------------------------------------------------------------------------
# 3. A-projection
# ---------------------------------------------------------------------------
def aprojection_report(nant=10, nsrc=4000, ngrid=128, du=0.4, ap_radius=6):
    """Grid the sky once, then apply beams as a uv-plane convolution.

    Uses a self-consistent discrete Fourier pair so the reference and the
    A-projection result are comparable exactly, isolating the one real
    approximation: truncating the aperture kernel.
    """
    print()
    print("=" * 76)
    print("3. A-PROJECTION: MOVE THE BEAM INTO THE UV PLANE")
    print("=" * 76)
    rng = np.random.default_rng(3)

    # uv grid. Sampling at du periodises the image with period 1/du, so the
    # sky must fit inside |l| < 1/(2 du).
    g = (np.arange(ngrid) - ngrid // 2) * du
    U, V = np.meshgrid(g, g, indexing="ij")
    smax = 0.4 / du  # keep the sky well inside the alias-free box

    # sources
    lm = rng.uniform(-smax, smax, (2, nsrc))
    flux = rng.pareto(1.5, nsrc) + 0.1

    # per-antenna apertures: compactly supported, INDEPENDENT, no shared structure
    ap_mask = np.hypot(U / du, V / du) <= ap_radius
    aper = np.zeros((nant, ngrid, ngrid), dtype=complex)
    for a in range(nant):
        aper[a][ap_mask] = (
            rng.standard_normal(ap_mask.sum()) + 1j * rng.standard_normal(ap_mask.sum())
        )

    # the implied beams: A_i(s) = du^2 sum_u g_i(u) exp(2 pi i u.s)
    phase = np.exp(2j * np.pi * (np.outer(lm[0], U.ravel()) + np.outer(lm[1], V.ravel())))
    beams = (phase @ aper.reshape(nant, -1).T).T * du**2  # (nant, nsrc)

    # exact reference: baselines placed on grid points so no sub-cell
    # interpolation is involved (that is an engineering detail, not physics)
    pairs = [(i, j) for i in range(nant) for j in range(i + 1, nant)]
    bidx = rng.integers(-ngrid // 8, ngrid // 8, (len(pairs), 2))
    b = bidx * du
    exact = np.array(
        [
            np.sum(beams[i] * beams[j].conj() * flux
                   * np.exp(2j * np.pi * (b[p, 0] * lm[0] + b[p, 1] * lm[1])))
            for p, (i, j) in enumerate(pairs)
        ]
    )

    # --- A-projection ---
    # ONE sky transform, no beam applied
    sky_hat = (flux * np.exp(2j * np.pi * (np.outer(U.ravel(), lm[0])
                                           + np.outer(V.ravel(), lm[1])))).sum(1)
    sky_hat = sky_hat.reshape(ngrid, ngrid)

    print(f"  Nant={nant} ({len(pairs)} pairs)  Nsrc={nsrc}  grid={ngrid}x{ngrid} "
          f"du={du}  aperture radius={ap_radius} cells")
    print(f"  Apertures are independent random draws -- zero cross-antenna similarity.\n")
    print("  The kernel's exact support is 2 x the aperture radius. Rows below that")
    print("  show what TRUNCATING costs -- the point is that you should not.\n")
    print(f"    {'kernel radius (cells)':>23}{'kernel cells':>14}{'max rel err':>14}")

    for kr in (2 * ap_radius, ap_radius + 2, ap_radius, ap_radius // 2):
        got = np.empty(len(pairs), dtype=complex)
        # K_ij = cross-correlation of the two apertures, truncated at radius kr
        wi = np.arange(-kr, kr + 1)
        WX, WY = np.meshgrid(wi, wi, indexing="ij")
        keep = np.hypot(WX, WY) <= kr
        for p, (i, j) in enumerate(pairs):
            Fi = np.fft.fft2(aper[i])
            Fj = np.fft.fft2(aper[j])
            K = np.fft.ifft2(Fi * Fj.conj()) * du**2  # correlation, wrapped
            K = np.fft.fftshift(K)
            c0 = ngrid // 2
            Kw = K[c0 - kr: c0 + kr + 1, c0 - kr: c0 + kr + 1] * keep
            # V_ij = du^2 sum_w K(w) skyhat(b + w)
            r = (bidx[p, 0] + c0 + WX) % ngrid
            s = (bidx[p, 1] + c0 + WY) % ngrid
            got[p] = np.sum(Kw * sky_hat[r, s]) * du**2
        err = np.abs(got - exact).max() / np.abs(exact).max()
        print(f"    {kr:>23}{int(keep.sum()):>14}{err:>14.2e}")

    print(
        "\n  At the full support the method is EXACT to rounding -- it is an\n"
        "  identity, not an approximation, and the support is bounded by physics\n"
        "  (the dish diameter), not by any similarity between antennas.\n"
        "\n  These apertures have hard edges, so truncation is punishing. A real\n"
        "  tapered illumination falls off far faster; but since the full support\n"
        "  is already cheap, the right move is simply not to truncate.\n"
        "\n  Cost: ONE sky transform per (time, chunk, freq) regardless of Nbeam,\n"
        "  plus Nbls x (kernel cells) MACs."
    )


def cost_model():
    """Back-of-envelope for HERA-350 at one time, one frequency."""
    print()
    print("=" * 76)
    print("COST MODEL: HERA-350, 1e6 sources, one integration, one frequency")
    print("=" * 76)
    nant, nsrc = 350, 1_000_000
    nbls = nant * (nant + 1) // 2
    # measured on this machine: 2D type-3, 200k src, 61k targets, ~0.75 s
    t_transform = 0.75 * (nsrc / 200_000)

    print(f"  standard path, {nant} unique beams:")
    print(f"    {nbls:,} transforms x {t_transform:.1f} s = "
          f"{nbls * t_transform / 3600:,.0f} hours per integration")
    print(f"  basis path at K=32:")
    n = 32 * 33 // 2
    print(f"    {n:,} transforms x {t_transform:.1f} s = {n * t_transform:,.0f} s")
    # HERA: D = 14 m, lambda = 2 m at 150 MHz -> aperture radius 3.5 lambda,
    # kernel radius 7 lambda; du = 0.4 -> kernel radius 17.5 cells, 35x35.
    kcells = 35 * 35
    macs = nbls * kcells
    # the gather is random-access into a ~100 MB grid, so assume memory-bound
    t_gather = macs / 1e9
    print(f"  A-projection (HERA 14 m dish, du=0.4 -> {kcells}-cell kernel):")
    print(f"    1 transform ({t_transform:.1f} s) + {macs / 1e6:.0f}e6 MACs "
          f"(~{t_gather:.2f} s, memory-bound) = ~{t_transform + t_gather:.1f} s")
    print(
        "\n  The standard path is not slow, it is structurally wrong for this\n"
        "  regime: with one beam per antenna each transform serves a single\n"
        "  baseline, so you pay full spreading cost for one visibility."
    )


if __name__ == "__main__":
    economics_report()
    residual_report()
    aprojection_report()
    cost_model()
