"""Benchmark cufinufft against finufft at fftvis's actual transform shapes.

Run on a GPU box:  python profiling/nufft_bench.py --nsrc 1000000 --nbls 61075

Answers the two questions that gate the GPU port:

1. At which ``eps`` does cufinufft actually beat CPU finufft for our shapes?
   cuFINUFFT's own paper reports 3D type-1 only winning for eps >= 1e-10 and
   merely matching at the tightest accuracies, so fftvis's precision=2 default
   (eps=1e-13) may be exactly where the GPU has no advantage.

2. Does the type-3 internal grid fit in device memory? For type 3, finufft
   sizes an internal upsampled grid from the *product* of the source and target
   half-widths. With long baselines and a full-sky source list this can be
   enormous in 3D, which is the main OOM risk in the port.

Both the CPU-only benchmark and the memory estimate run without a GPU.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    import cupy as cp
    import cufinufft

    HAVE_GPU = True
except ImportError:  # pragma: no cover
    HAVE_GPU = False

import finufft

# fftvis's defaults, from core/simulate.py
DEFAULT_EPS = {1: 6e-8, 2: 1e-13}
EPS_GRID = (1e-4, 1e-6, 6e-8, 1e-9, 1e-12, 1e-13)


def make_problem(nsrc, nbls, ntrans, bmax_wavelengths, dim, seed=0):
    """Source/target point sets shaped like an fftvis chunk.

    Sources are direction cosines on the visible hemisphere scaled by 2*pi (as
    fftvis does with ``topo *= 2 * np.pi``); targets are baseline vectors in
    wavelengths.
    """
    rng = np.random.default_rng(seed)
    lm = rng.uniform(-1, 1, (2, 2 * nsrc))
    lm = lm[:, np.hypot(*lm) < 1][:, :nsrc]
    n = np.sqrt(1 - lm[0] ** 2 - lm[1] ** 2)
    src = np.vstack([lm, n - 1]) * 2 * np.pi

    tgt = rng.uniform(-bmax_wavelengths, bmax_wavelengths, (3, nbls))
    if dim == 2:
        tgt[2] = 0.0
    c = (rng.standard_normal((ntrans, nsrc)) + 1j * rng.standard_normal((ntrans, nsrc)))
    return src, tgt, c


def type3_grid_estimate(src, tgt, dim, upsampfac=2.0, itemsize=16):
    """Estimate finufft's internal type-3 grid size and its memory.

    finufft chooses per-dimension mode counts from the half-widths X_d (source)
    and S_d (target) roughly as N_d ~ upsampfac * X_d * S_d / pi, then works on
    an upsampled grid of that size.
    """
    dims, total = [], 1
    for d in range(dim):
        X = 0.5 * (src[d].max() - src[d].min())
        S = 0.5 * (tgt[d].max() - tgt[d].min())
        N = max(int(upsampfac * X * S / np.pi), 1)
        dims.append(N)
        total *= N
    return dims, total * itemsize / 1024**3


def bench_cpu(src, tgt, c, dim, eps, nthreads, upsampfac, reps=3):
    """Median wall time of the CPU type-3 transform."""
    # finufft wants each coordinate array contiguous in its own right; rows of a
    # (3, nsrc) block are views and trigger an internal copy + UserWarning.
    # fftvis's CPU path passes topo[0]/topo[1] directly and hits this too.
    s = [np.ascontiguousarray(src[d]) for d in range(dim)]
    t = [np.ascontiguousarray(tgt[d]) for d in range(dim)]
    args = (*s, c, *t)
    fn = finufft.nufft2d3 if dim == 2 else finufft.nufft3d3
    kw = dict(eps=eps, nthreads=nthreads, showwarn=0, upsampfac=upsampfac)
    out = fn(*args, **kw)  # warmup
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn(*args, **kw)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), out


def bench_gpu(src, tgt, c, dim, eps, upsampfac, reps=3):  # pragma: no cover
    """Median wall time of the cufinufft type-3 transform, plus H2D transfer."""
    rdtype = c.real.dtype
    g_src = [cp.asarray(s, dtype=rdtype) for s in src[:dim]]
    g_tgt = [cp.asarray(t, dtype=rdtype) for t in tgt[:dim]]
    g_c = cp.asarray(c)

    fn = cufinufft.nufft2d3 if dim == 2 else cufinufft.nufft3d3
    # NOTE: nthreads/showwarn are CPU-only and must not be forwarded.
    kw = dict(eps=eps, upsampfac=upsampfac)

    def call():
        return fn(*g_src, g_c, *g_tgt, **kw)

    out = call()
    cp.cuda.Device().synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        call()
        cp.cuda.Device().synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), cp.asnumpy(out)


def main():
    """Run the eps sweep and print the comparison table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsrc", type=int, default=200_000)
    ap.add_argument("--nbls", type=int, default=61_075, help="HERA-350 all pairs")
    ap.add_argument("--ntrans", type=int, default=4, help="nfeeds**2")
    ap.add_argument("--bmax", type=float, default=150.0, help="max |b|/lambda")
    ap.add_argument("--dim", type=int, default=2, choices=(2, 3))
    ap.add_argument("--upsampfac", type=float, default=2.0)
    ap.add_argument("--nthreads", type=int, default=0, help="0 = finufft default")
    ap.add_argument("--precision", type=int, default=2, choices=(1, 2))
    args = ap.parse_args()

    cdtype = np.complex64 if args.precision == 1 else np.complex128
    src, tgt, c = make_problem(
        args.nsrc, args.nbls, args.ntrans, args.bmax, args.dim
    )
    src = src.astype(np.float32 if args.precision == 1 else np.float64)
    tgt = tgt.astype(src.dtype)
    c = c.astype(cdtype)

    dims, gb = type3_grid_estimate(
        src, tgt, args.dim, args.upsampfac, np.dtype(cdtype).itemsize
    )
    print(f"nsrc={args.nsrc:,}  nbls={args.nbls:,}  n_trans={args.ntrans}  "
          f"dim={args.dim}D  bmax={args.bmax}λ  {np.dtype(cdtype).name}")
    print(f"estimated type-3 internal grid: {dims}  ->  {gb:.2f} GB "
          f"(x{args.ntrans} transforms if not batched internally)")
    if gb > 4:
        print("  !! this is the OOM risk described in the module docstring")
    print()

    nthreads = args.nthreads
    print(f"{'eps':>10}{'CPU (ms)':>12}{'GPU (ms)':>12}{'speedup':>10}{'max rel diff':>15}")
    for eps in EPS_GRID:
        if args.precision == 1 and eps < 1e-6:
            continue  # unreachable in single precision
        t_cpu, ref = bench_cpu(src, tgt, c, args.dim, eps, nthreads, args.upsampfac)
        if HAVE_GPU:  # pragma: no cover
            t_gpu, got = bench_gpu(src, tgt, c, args.dim, eps, args.upsampfac)
            d = np.abs(got - ref).max() / np.abs(ref).max()
            print(f"{eps:>10.0e}{t_cpu * 1e3:>12.1f}{t_gpu * 1e3:>12.1f}"
                  f"{t_cpu / t_gpu:>9.2f}x{d:>15.2e}")
        else:
            print(f"{eps:>10.0e}{t_cpu * 1e3:>12.1f}{'n/a':>12}{'n/a':>10}{'n/a':>15}")

    if not HAVE_GPU:
        print("\ncupy/cufinufft not importable: CPU column only.")
        print("Install with:  pip install cufinufft  (needs finufft >= 2.4 for GPU type 3)")


if __name__ == "__main__":
    main()
