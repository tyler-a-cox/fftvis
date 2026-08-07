"""Execute the fftvis GPU engine with cupy/cufinufft shimmed onto numpy/finufft.

This does **not** test GPU behaviour -- it cannot catch a cupy advanced-indexing
gap or a cufinufft kwarg rejection. What it does catch is everything else: the
einsums, the axis flips, the reshapes, the accumulation indexing and the
beam-pair bookkeeping, all checked against the CPU engine on a machine with no
GPU. That keeps the port's logic covered in CPU-only CI, where
``test_cpu_vs_gpu.py`` can only skip.

It replaces entries in ``sys.modules``, so it is run in a subprocess by
``test_gpu_shim.py`` rather than collected directly. Run standalone with:

    python tests/_gpu_shim_check.py
"""

import sys
import types

import numpy as np
import finufft

# ---------------------------------------------------------------- fake cupy
fake_cp = types.ModuleType("cupy")
for name in dir(np):
    if not name.startswith("_"):
        setattr(fake_cp, name, getattr(np, name))
fake_cp.ndarray = np.ndarray
fake_cp.asnumpy = lambda a: np.asarray(a)


class _Dev:
    mem_info = (1 << 62, 1 << 62)


fake_cp.cuda = types.SimpleNamespace(Device=lambda *a, **k: _Dev())
fake_cp.get_array_module = lambda *a, **k: np
sys.modules["cupy"] = fake_cp

# matvis probes for these submodules too
sys.modules["cupyx"] = types.ModuleType("cupyx")
sys.modules["cupyx.scipy"] = types.ModuleType("cupyx.scipy")
_ndi = types.ModuleType("cupyx.scipy.ndimage")
sys.modules["cupyx.scipy.ndimage"] = _ndi

# ------------------------------------------------------------ fake cufinufft
fake_cufi = types.ModuleType("cufinufft")


def _n2d3(x, y, c, s, t, eps=1e-6, isign=1, **kw):
    return finufft.nufft2d3(
        x, y, c, np.ascontiguousarray(s), np.ascontiguousarray(t),
        eps=eps, isign=isign, showwarn=0, upsampfac=kw.get("upsampfac", 2),
    )


def _n3d3(x, y, z, c, s, t, u, eps=1e-6, isign=1, **kw):
    return finufft.nufft3d3(
        x, y, z, c,
        np.ascontiguousarray(s), np.ascontiguousarray(t), np.ascontiguousarray(u),
        eps=eps, isign=isign, showwarn=0, upsampfac=kw.get("upsampfac", 2),
    )


def _n2d1(x, y, c, n_modes, eps=1e-6, isign=1, modeord=0, **kw):
    return finufft.nufft2d1(
        x, y, c, tuple(n_modes), eps=eps, isign=isign, modeord=modeord,
        showwarn=0, upsampfac=kw.get("upsampfac", 2),
    )


fake_cufi.nufft2d3 = _n2d3
fake_cufi.nufft3d3 = _n3d3
fake_cufi.nufft2d1 = _n2d1
sys.modules["cufinufft"] = fake_cufi

# ------------------------------------------- stub matvis.gpu.beams (unused here)
stub = types.ModuleType("matvis.gpu.beams")
stub.gpu_beam_interpolation = None
stub.prepare_for_map_coords = None
sys.modules["matvis.gpu.beams"] = stub

# ---------------------------------------------------------------- run the sims
import pyuvdata.telescopes
from pyuvdata import Telescope

if not hasattr(pyuvdata.telescopes, "get_telescope"):
    pyuvdata.telescopes.get_telescope = (
        lambda n, **k: Telescope.from_known_telescopes(n, **k)
    )

from matvis._test_utils import get_standard_sim_params  # noqa: E402
from fftvis.wrapper import simulate_vis  # noqa: E402

EPS = 1e-12


def compare(label, params, **kw):
    """Run both backends and report the max relative difference."""
    ref = simulate_vis(backend="cpu", **params, **kw)
    got = simulate_vis(backend="gpu", **params, **kw)
    assert ref.shape == got.shape, f"{label}: shape {ref.shape} vs {got.shape}"
    scale = np.abs(ref).max()
    err = np.abs(got - ref).max() / scale
    # Both engines request the same eps from finufft, so agreement should be at
    # that level up to a small constant.
    tol = max(100 * kw.get("eps", EPS), 1e-13)
    status = "OK " if err < tol else "FAIL"
    print(f"  [{status}] {label:<38} shape={str(ref.shape):<22} "
          f"max rel diff={err:.2e}  (tol {tol:.0e})")
    return err < tol


def base(polarized, **kw):
    """Standard small sim params, with beam_idx dropped."""
    params, *_ = get_standard_sim_params(
        use_analytic_beam=True, polarized=polarized, **kw
    )
    params.pop("beam_idx", None)
    # get_standard_sim_params returns `beams` (a list); simulate_vis takes `beam`.
    params["beam"] = params.pop("beams")[0]
    return params


ok = True
print("CPU vs GPU engine (cupy/cufinufft shimmed onto numpy/finufft)\n")

ok &= compare("unpolarized, analytic beam", base(False), precision=2, eps=EPS)
ok &= compare("polarized, analytic beam", base(True), precision=2, eps=EPS)
ok &= compare("single precision", base(False), precision=1, eps=1e-6)
ok &= compare("min_chunks=3", base(False), precision=2, eps=EPS, min_chunks=3)

p = base(False)
nant = len(p["ants"])
p["beam"] = [p["beam"]] * 2
p["beam_idx"] = np.arange(nant) % 2
ok &= compare("per-antenna beams (2 beams)", p, precision=2, eps=EPS)

p = base(True)
p["beam"] = [p["beam"]] * 2
p["beam_idx"] = np.arange(len(p["ants"])) % 2
ok &= compare("polarized + per-antenna beams", p, precision=2, eps=EPS)

p = base(False)
p["ants"] = {
    a: np.array(v) + np.array([0.0, 0.0, 5.0 * i])
    for i, (a, v) in enumerate(p["ants"].items())
}
ok &= compare("tilted array (3D type-3)", p, precision=2, eps=EPS)

p = base(False)
p["ants"] = {i: np.array([10.0 * (i % 3), 10.0 * (i // 3), 0.0]) for i in range(9)}
ok &= compare("gridded array (type-1)", p, precision=2, eps=EPS)

p = base(True, use_polarized_sky=True)
ok &= compare("polarized sky model", p, precision=2, eps=EPS)

print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
sys.exit(0 if ok else 1)
