# Plan: a GPU backend for fftvis

Audit of `fftvis` @ `99d6ae4` (branch `gpu_new`) and a staged plan to make the
GPU backend real.

Decisions taken as given: **correctness-first vertical slice**, **reuse
matvis's GPU beam kernel**, **single-GPU workstation**, **precision default
settled by benchmark** (harness written, see §6).

---

## 1. Where things actually stand

The GPU backend is scaffolded but empty. Every entry point raises:

| file | state |
|---|---|
| `gpu/gpu_simulate.py` | `GPUSimulationEngine.simulate` + `_evaluate_vis_chunk` → `NotImplementedError` |
| `gpu/nufft.py` | `gpu_nufft2d`, `gpu_nufft3d` → `NotImplementedError`; no type-1 stub at all |
| `gpu/beams.py` | `GPUBeamEvaluator.evaluate_beam`, `get_apparent_flux_polarized` → `NotImplementedError` |
| `gpu/utils.py` | `inplace_rot` → `NotImplementedError` (with a commented sketch) |
| `wrapper.py` | `create_simulation_engine(backend="gpu")` already dispatches; `create_beam_evaluator` still raises |
| `tests/test_gpu_*.py` | assert that the stubs raise — these get deleted, not extended |

The good news is that the abstraction boundaries are already in the right
places, and a surprising amount of the hard work is done or free.

### Already GPU-ready, no work needed

- **Coordinate rotation.** fftvis takes `coord_method` by name out of
  `matvis.core.coords.CoordinateRotation._methods`, and matvis ships
  `GPUCoordinateRotationERFA` (`requires_gpu = True`). `CoordinateRotation`
  switches `self.xp` to cupy when constructed with `gpu=True`, so
  `select_chunk` returns device arrays. The GPU engine just has to pass
  `gpu=True` — the current CPU call site doesn't.
- **Beam basis / eigenbeams.** `core/beam_basis.py` is setup-time numpy +
  pyuvdata. Runs once, off the hot path. No change.
- **Antenna gridding.** `core/antenna_gridding.py` likewise — setup-time only.
- **Source truncation.** `_evaluate_vis_chunk` already does
  `topo = topo[:, :nsim_sources]`, so fftvis does *not* have the padded-buffer
  waste that matvis does. Nothing to fix here.

### Must be written

1. `gpu/nufft.py` — cufinufft wrappers (2D/3D type 3, 2D type 1). ~120 lines.
2. `gpu/beams.py` — beam evaluation via matvis's kernel + the apparent-flux
   kernels. ~200 lines.
3. `gpu/utils.py` — `inplace_rot` as a cupy matmul. ~20 lines.
4. `gpu/gpu_simulate.py` — the engine. ~400 lines, mostly mirroring
   `cpu_simulate.py`.
5. Four numba kernels ported to cupy (§4).

---

## 2. The dependency gate, and it's open

fftvis's general path is type 3 (`nufft2d3` / `nufft3d3`). Historically
cufinufft was types 1 and 2 only — that would have killed the port.

**Type 3 on the GPU landed in finufft 2.4.0**, in 1D, 2D and 3D. The functional
interface (`cufinufft.nufft2d3`, `cufinufft.nufft3d3`) exists and
`Plan.execute` documents type-3 input semantics. Note the `Plan` constructor
docstring still says "type of NUFFT (1 or 2)" — that's stale, not a limitation.

**Pin `finufft >= 2.4` and add `cufinufft` as a `[gpu]` extra.**

Note on cupy: the `cupy` name on PyPI is a *source* distribution, so listing it
as a dependency makes pip compile the whole library (slow, and routinely
OOM-killed during Cythonize). The prebuilt wheels are published under
CUDA-specific names, hence the `[gpu-cuda12]` / `[gpu-cuda11]` extras.

### API mapping, verified against finufft 2.5.1

I checked the semantics fftvis actually relies on:

| behaviour | result |
|---|---|
| batched type 3: `c` shape `(n_tr, M)` → out `(n_tr, N)` | ✅ confirmed |
| default `isign=+1` for type 3, matches fftvis's implicit convention | ✅ confirmed |
| `modeord` is a no-op for type 3 | ✅ confirmed (fftvis passes `modeord=0` harmlessly) |
| type-1 `modeord=1` + signed integer indexing (`model[..., i0, i1]`) | ✅ confirmed |

Kwarg differences to get right — this is where a silent port bug would live:

- **CPU-only, must be stripped:** `nthreads`, `showwarn`.
- **Shared:** `eps`, `isign`, `upsampfac`, `modeord`.
- **GPU-only, worth exposing:** `gpu_method` (1 = points-driven, 2 = shared
  memory), `gpu_sort`, `gpu_kerevalmeth`, `gpu_stream`.

Minor CPU finding while testing: finufft emits a copy warning for the
coordinate arrays fftvis passes as `topo[0]` / `topo[1]`. `u`/`v` are already
wrapped in `np.ascontiguousarray` but `x`/`y` aren't. Cheap to fix on both
backends.

---

## 3. The one architectural decision that matters

**The per-beam-pair NUFFT loop is the wrong granularity for a GPU, and the
eigenbeam path is the way out.**

`_evaluate_vis_chunk`'s standard path loops over unique beam pairs and calls
`_run_nufft` once per pair per (time, chunk, freq). With per-antenna beams
that's `Nbeam(Nbeam+1)/2` = **61,425 transforms per integration** at 350 beams.

On CPU each call is large enough that finufft's per-call planning is amortized.
On GPU it is not: cufinufft's functional interface builds and tears down a plan
on every call — for type 3 that means sizing and allocating the internal
upsampled grid and precomputing phase corrections each time. Sixty-one thousand
of those per integration will dominate everything else.

Two properties rescue this, and they only both hold on the eigenbeam path:

- **Source points are shared across all pairs** at a given (time, chunk) — the
  sky doesn't know which beams you're pairing.
- **Target points are shared across all pairs** — but only in the basis path,
  where `bls_idxs = np.arange(nbls)`. In the standard path each beam pair gets
  its own baseline subset, so targets differ per pair.

So on the basis path you can build **one plan per (time, chunk, freq)**,
`setpts` once, and batch every `(k, l, feed, feed)` combination into a single
`execute` via `n_trans`. For `K=8` basis beams that's
`K(K+1)/2 × nfeeds² = 144` transforms in one call instead of 36 separate
planned calls.

Rough device memory for that batch at 200k sources / 61,075 baselines,
complex128: ~460 MB input strengths + ~140 MB output + ~6 MB internal grid per
transform. Comfortable on a 24 GB card; halve it in single precision.

**Recommendation:** make the basis/eigenbeam path the primary GPU path. Support
the standard per-antenna-beam path for parity and small runs, but document that
it does not batch and will be slow on GPU. This is the same conclusion the
matvis analysis reached from the other direction — the low-rank beam basis is
what makes the fast algorithm usable with heterogeneous beams.

---

## 4. Numba kernels to port

Four `@nb.jit` functions in `cpu/beams.py`, all elementwise 2×2 matrix algebra
over the source axis — ideal for one fused `cp.RawKernel`:

| function | what it computes |
|---|---|
| `get_apparent_flux_polarized_beam` | `A^H A · flux`, in place |
| `get_apparent_flux_polarized` | `A^H C A`, in place |
| `get_apparent_flux_polarized_beam_pair` | `A_i^H A_j · flux` → `out` |
| `get_apparent_flux_polarized_pair` | `A_i^H C A_j` → `out` |

For the first milestone, `cp.einsum` is a correct and much shorter stand-in;
fuse later once parity tests pass. `cpu/utils.py::inplace_rot` is a 3×3 rotation
that becomes `cp.matmul` — or disappears entirely if the rotation is folded into
the coordinate rotator.

---

## 5. Milestones

### M0 — plumbing (½ day)
Add the `[gpu]` extra (`cupy`, `cufinufft`, `finufft>=2.4`). Delete the
`test_gpu_*` stub-assertion tests. Add a `pytest.importorskip("cupy")` fixture
and a `--gpu` marker mirroring matvis's layout.

### M1 — vertical slice (the correctness milestone)
Unpolarized, single shared beam, type-3, coplanar 2D, one GPU, `nprocesses=1`,
no ray, no streams, no plan reuse. Straight-line port of `_evaluate_vis_chunk`
with `np` → `cp` and `finufft` → `cufinufft`.

**Exit criterion:** a CPU-vs-GPU parity test agreeing to the NUFFT's own `eps`
across a small array, several times and frequencies. This is the artifact
everything later is checked against — get it before optimizing anything.

### M2 — feature parity
Polarized beams and polarized sky (the four ported kernels). Per-antenna beams
via the standard path. 3D type-3 for non-coplanar arrays. Type-1 gridded path.
Extend parity tests to cover each.

### M3 — the eigenbeam fast path
Port `_compute_basis_visibilities`. Hoist `cufinufft.Plan` out of the `(k, l)`
loop, `setpts` once per (time, chunk, freq), batch all pairs × feeds into
`n_trans`. This is where the GPU port actually earns its speedup for the HERA
per-antenna-beam case.

### M4 — pipelining
Overlap beam evaluation, coherency assembly and the transform on CUDA streams.
Measure with `nsys`; matvis's NVTX-annotated loop is a good model. Only worth
doing once M3's numbers show host stalls.

---

## 6. Risks, with numbers

**Type-3 internal grid memory — the main OOM risk.** finufft sizes its internal
upsampled grid from the *product* of source and target half-widths. Measured
with `profiling/nufft_bench.py`:

| configuration | internal grid | memory |
|---|---|---|
| 2D, coplanar, `b_max = 150λ` (HERA core) | 599 × 599 | **0.01 GB** |
| 3D, non-coplanar, `b_max = 500λ` (outriggers) | 1983 × 1953 × 986 | **56.9 GB** |

The 2D path is free; the 3D path does not fit on any single GPU. fftvis rotates
the array to the XY plane and takes the 2D path whenever `is_coplanar`, so HERA
is fine — but the GPU engine should *estimate this before allocating* and raise
a clear error rather than dying inside cufinufft. `type3_grid_estimate()` in the
harness does the arithmetic.

**Precision default.** cuFINUFFT's own paper reports 3D type-1 beating CPU
finufft only for `eps >= 1e-10`, merely matching at the tightest accuracies.
fftvis's `precision=2` default is `eps=1e-13` — plausibly right where the GPU
has no advantage. `profiling/nufft_bench.py` sweeps eps at fftvis's real shapes
and prints CPU/GPU/speedup/error; run it before setting the default. (On CPU
alone at 200k sources the runtime is nearly flat across eps, because the
target-side spreading over 61k baselines dominates — so the answer genuinely
depends on the shape, not just the tolerance.)

**Ray.** The CPU engine's ray layer parallelizes over (time, freq). For a single
GPU it should be bypassed entirely (`nprocesses=1`, `use_ray=False`), not
adapted. Don't let it into the GPU engine; it's the kind of thing that's much
harder to remove later.

**pyuvdata on the critical path.** matvis's `gpu_beam_interpolation` needs the
raw beam grids uploaded once at setup via `prepare_for_map_coords`, and requires
`pixel_coordinate_system == "az_za"` and all beams on a common grid. The
eigenbeams from `compute_beam_basis` already satisfy this by construction (they
are copies of a common interpolated reference). Analytic beams do not go through
that kernel — matvis falls back to CPU evaluation and uploads, which is fine at
`K` basis beams but would be a bottleneck at 350.

---

## 7. Testing

Mirror matvis's structure: a `tests/test_cpu_vs_gpu.py` that runs the same
small simulation through both engines and compares, parameterized over
`polarized`, `precision`, per-antenna vs basis beams, and 2D vs 3D. Gate on
`pytest.importorskip("cupy")` so CPU-only CI stays green.

The existing `test_gpu_beams.py` / `test_gpu_nufft.py` assert that stubs raise
`NotImplementedError`; they'll fail as soon as anything is implemented and
should be replaced in M0 rather than patched.

CI needs a self-hosted GPU runner — matvis already has one, and its workflow is
the obvious template.

---

## 8. Suggested first commit

M0 + the M1 skeleton: `gpu/nufft.py` with the three cufinufft wrappers and the
CPU-only-kwarg stripping, `gpu/utils.py::inplace_rot`, and a single parity test
for the unpolarized shared-beam 2D case. That's a small, reviewable diff that
turns the branch from scaffolding into something that runs.

---

## References

- [finufft v2.4.0 release notes](https://github.com/flatironinstitute/finufft/releases/tag/v2.4.0) — GPU type 3 in 1D/2D/3D
- [cufinufft Python interface](https://finufft.readthedocs.io/en/latest/python_gpu.html) — `nufft2d3`/`nufft3d3`, `Plan`, `n_trans`, GPU kwargs
- [cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs](https://arxiv.org/pdf/2102.08463) — the 4–11x exec speedups and the eps-dependence caveat
- [fftvis (arXiv:2506.02130)](https://arxiv.org/abs/2506.02130)
