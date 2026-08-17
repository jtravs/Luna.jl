# GPU / fixed-quadrature modal transform — deferred work and next steps

Branch: `modal-fixed` (off `gpu`, which is off `perf` → `slurm` → `master`).
State at the time of writing (commit `06be376`): `TransModalFixed` is the default modal
transform (`modal_integral=:fixed`, `nr=64`, `nθ=16`), the smooth PPT rate
(`sum_integral=true`) is the default ionisation model, plasma/Kerr/Raman-envelope run
batched on host threads and on CUDA, and the JLArray + hardware-gated CUDA tests pass on
the host side. Nothing below has yet been **measured on a real GPU** — the A40 numbers are
the first thing to get.

Legend — **Effort**: S ≲ ½ day / < 50 lines, M ≈ 1–2 days / 50–300 lines, L ≳ 3 days.
**Payoff** is for the production use cases (RDW/DUV/VUV multimode HCF with ionisation;
sweeps of ~200 runs; occasional 2-D vector sets and large time grids).
Priority: P0 = blocks intended use, P1 = do next, P2 = worthwhile, P3 = when needed.

§§0–6 concern the **modal** (`TransModalFixed`) path. **§7 covers the separate 3-D
free-space (`TransFree`) path** — the ModelPNPS TG-FROG campaigns — which *has* now been
measured on an A40 and whose bottlenecks are different. The downstream half of that work
lives in `ModelPNPS/GPU-TODO.md`.

---

## 0. Bugs / blockers found while planning (fix before the first GPU scan)

- [x] **P0 — lazy `arraytype=:cuda` fails inside a single top-level call (world age).**
  (Fixed.) `Luna.resolve_arraytype(:cuda)` loads CUDA.jl with `Base.require` *inside*
  `prop_capillary_args`; every subsequent CUDA method call in the same frame
  (`CuArray{T}(undef, …)`, `Adapt.adapt(CuArray, …)`, cuFFT plans) then fails with
  `MethodError … too new`. This is exactly the scan-script use case the `Symbol` form was
  designed for (login node without a GPU submits; compute node runs the closure).
  Fix: `prop_capillary` / `prop_capillary_args` resolve a `Luna.lazy_arraytype` symbol and
  re-enter through `Base.invokelatest` with the concrete type. Tests: the mechanism with a
  stand-in in `test_device.jl`; the real thing in a fresh subprocess without `import
  CUDA` in `test_cuda.jl` (hardware-gated — **still to be run on the A40**). Users of
  `prop_capillary_args` + `Luna.run` inside one function must `invokelatest(Luna.run, …)`
  themselves (documented in `resolve_arraytype`).

- [x] **P1 — `mode_error` statistic round-trips the state through the host on a device.**
  (Fixed.) `Stats.collect_stats` now returns a `StatsCollector` which accepts the device
  array itself (`Luna.device_stats`), copies it into its own host buffer for the host
  statistics, and hands the device array to statistics flagged `Stats.wants_state`
  (`StateStat`); `mode_reconstruction_error(::TransModalFixed)` evaluates the transform on
  the device state directly and only copies the small results back. `HostOutput` no
  longer copies `y` per step for such collectors (`needs_host_y`). An `HDF5Output` with
  `cache=true` still needs the host copy, and the statistic uploads it in that case.

---

## 1. Measure first (user actions on the DMOG A40 — no code)

- [ ] **Submit `test/manual/hpc_gpu_suite.sbatch`** (one-time env setup in its header):
  precompile on the node → CPU tests → `test_cuda.jl` (cuFFT semantics, PPT/ADK/plasma/
  Raman kernels, native scan, RK45 norm host-vs-device with cancellation — ported from
  `ModelPNPS/examples/check_device_norm.jl` —, transform/end-to-end agreement, lazy
  `:cuda` in a fresh subprocess) → `test/manual/hpc_gpu_bench.jl` (cases vuv/ctc/vector/
  bignt on cpu and cuda: isolated RHS, ms/step, transform share, agreement, device memory,
  primitives; Markdown summary to paste here). ~1–1.5 h with the defaults.
- [ ] `julia --project -t 8 test/manual/modal_fixed_rdw.jl vuv quick 1.5 cuda` and the same
  with `cpu` on a full CPU node (`-t 32`). The script prints ms/step, the isolated RHS time
  and 7×RHS for every run, so the transform share vs "everything else" (linear op, RK45,
  stats, output) is read off directly. Also record `Luna.device_memory_status()` peak.
- [ ] The same for a 2-D vector case (e.g. `modes=(:HE11,:HE21,:TE01,:TM01)`, `nr=64`,
  `nθ=16`) and a large-nt case (`trange`/`λlims` giving nt ≥ 2^16): these are the cases
  where the GPU is expected to pay (≥10⁷ points per RHS).
- Expected (unmeasured): scalar 4-mode nt=8192 → GPU ≈ 8-thread laptop, slower than a
  32-core node (launch-latency bound); 2-D vector / large nt → 5–10× over the laptop,
  2–3× over a node; sweeps → throughput via several processes per GPU.
  Decisions in §2 hinge on these numbers.

---

## 2. Device performance

- [x] **Native CUDA `cumsum!` for the plasma cumulative integrals** (done, `06be376`):
  `Maths.scan!`/`scan_scratch` with a `CuArray` method in `ext/LunaCUDAExt.jl`; ~80 →
  ~6 launches per plasma evaluation at nt=8192; two field-sized device buffers fewer.

- [x] **P1 — `stats_period=k`.** (Done.) `Output.PeriodicStats(statsfun, k)` evaluates
  the statistics on the first and then every k-th accepted step and returns `nothing` in
  between; both output handlers skip those steps. `prop_capillary(...; stats_period=k)`.

- [ ] **P2 — Modal statistics on device.** Today `HostOutput` copies `Eω` (nω×nmodes:
  0.26 MB at nt=8192, ~6 MB at 2^17) to the host every step and *all* stats run there:
  host IFFT (`plan_analytic`), on-axis columnwise ionisation for `electrondensity`,
  `peakintensity`, `fwhm_r`, plus the `mode_error` transform call (§0). Effort **M
  (~300–400 lines, almost all `Stats.jl`)**: a device method per field-touching stat
  written as broadcasts/reductions returning scalars via a small copy — `energy`, `ω0`,
  `energy_λ` are one-line column reductions; `peakpower`, `peakintensity`, `fwhm_t`,
  `fwhm_r`, `electrondensity` need a device FFT plan held by the closure, the on-axis
  column via a GEMV against `mode_matrix(…, [(0,0)])`, then `Ionisation.ionrate_device!` +
  `Maths.scan!` (both exist); keep host methods untouched (bit-identical host results), so
  it is a second method per stat; `collect_stats`/`HostOutput` hand the device array to
  the stats; JLArray + CUDA tests. Payoff: ≤ 15 % of a step at nt=8192 (half of which is
  the mode-error transform call, already on device), 25–50 % at nt=2^17. **Decide after
  the §1 measurement** (the round trip and `stats_period` are done).

- [ ] **P2 — CUDA graph capture of the RHS.** The fixed rule has static control flow and
  no allocation after warm-up, so one RHS (IFFT, GEMM, responses, GEMM, FFT, scaling) can
  be captured once and replayed: launch overhead → a few µs. Effort M–L: `CUDA.@captured`
  around `transform(nl, Eω, z)` with fixed buffers; `z`-dependent scalars (density,
  taper scale) must become device scalars/1-element arrays updated by `copyto!` rather
  than captured constants; the PPT rate table and mode matrices are fine; the RK45 loop
  stays host code with one sync per step. Payoff: only for the launch-bound (small nt,
  few columns) regime — 2–3× there; ~nothing for the large cases. Measure first.

- [ ] **P2 — Column-chunk fusion for very large `nt` (also memory; see also the fused
  CPU plasma column in §2b, which removes most of the CPU-side buffers).** The transform
  materialises `Et`, `Pt` (nto×npol×Np) and the plasma response five more arrays of that
  size. At nt=2^17, 65 nodes, scalar: ~7×135 MB ≈ 1 GB — fine. At 2-D vector nr=64×nθ=16
  ≈ 1000 columns and 2^17: ~7 GB — too much for one A40 alongside the RK45 registers, and
  on the CPU the block leaves L2/L3. Effort M: loop the synthesis GEMM → responses →
  projection GEMM over column blocks (the FFTs are on the modal side and are unaffected;
  the plasma buffers become block-sized). Payoff: enables the largest 2-D/large-nt cases
  on one GPU; 1.5–2× on CPU at nt ≥ 2^16 from cache residency. Threading granularity
  comes free.

- [ ] **P3 — Several simulations per GPU for sweeps.** No code: `SlurmExec(gres="gpu:1",
  instances=k)` already pins one process per visible device; for small problems run k
  processes on one device (CUDA MPS, or simply share) and compare *jobs per node-hour*
  against a CPU node. Document the recipe (see §5).

- [ ] **P3 — FP32 projection GEMMs / rate evaluation** (FP64 elsewhere). Only if the GEMMs
  or the rate kernel ever dominate on a card with weak FP64 (B300-class); test against the
  RK45 error controller, not only the spectrum. Not for A40/H200.

- [ ] **P3 — Move `linop` z-dependent (gradient) evaluation to device.** Today the
  z-dependent modal linear operator (`LinearOps` closure, e.g. the RDW gradient example) is
  evaluated on the host into a buffer and uploaded per stage (`RK45.make_prop!`). It is
  nω×nmodes×16 B per RHS (0.26 MB at nt=8192) — negligible until nt ≳ 2^17 with many
  modes. Effort M (device-native `neff(ω; z)` broadcast); payoff small.

---

## 2b. Next performance steps for `TransModalFixed` (measured, laptop 8 threads, 2026-08-17)

Stage breakdown of one RHS (`scratchpad/stagebreak.jl`):

| | VUV (nto 8192, M 4, 65 nodes) | CtC-type (nto 65536, M 6, 65 nodes) |
|---|---|---|
| **RHS total** | **1.26 ms** | **18.6 ms** |
| plasma (rate + 3 cumulative integrals + elementwise) | 0.85 (68 %) | 8.1 (44 %) |
| Raman (batched doubled-grid FFT pairs) | – | 7.0 (38 %) |
| Kerr | 0.14 | 0.4 |
| synthesis + projection GEMM | 0.10 | 1.2 |
| modal IFFT + FFT + windows | 0.10 | 1.0 |

Rate evaluation per sample: cached PPT spline ≈ 1–2 ns/thread, ADK ≈ 2.5 ns, **direct PPT
≈ 120 ns** → recomputing the rate instead of the spline is ruled out (the plasma alone would
be ~10× the whole RHS). FFTW *does* thread the batched Raman plan over the columns (31 → 7.2 ms
from 1 to 8 threads); Julia-threading per-column plans is only ~20 % better, so the Raman
cost is real work.

- [ ] **P1 (no code) — tune `nr`.** With the smooth rate the plasma is converged at nr≈32
  for 4 modes; the default 64 is a 2× safety margin. `nr=32–40` (check
  `transverse_integral_error_rel` stays ~1e-9) is ≈1.6–2× on everything, CPU and GPU:
  VUV RHS 1.26 → ~0.7 ms, CtC 18.6 → ~11 ms. Consider lowering the default once the
  mode-count study (§4) is done.
- [x] **`stats_period` / `mode_error=false`** for sweeps: −14 % (one of 7 transform calls
  per step). Done.
- [ ] **P1 — CPU: fuse the plasma column into two passes** (`_plasma_scalar!`/`_plasma_vector!`).
  Today ~7 passes over the column through 5 intermediate arrays (rate, fraction, phase,
  J, P). Instead: one SIMD pass for the rate, one serial pass carrying the three running
  trapezoid sums in registers, reading E once and writing P once, with the *same*
  arithmetic order as `Maths.cumtrapz!` so results stay bit-identical to the columnwise
  reference (testable exactly). Expect the plasma to halve (VUV RHS −35 %, CtC −25 %),
  and the five nto×Np buffers disappear on the CPU (the CtC column working set is 3 MB
  today, past L2 — this also fixes that). **CPU only**: one thread walking a column is
  exactly what a GPU must not do; the GPU counterparts are the fusion / time-tiled scan
  items below. Worth doing regardless of the A40 results. Effort S–M (~60 lines).
- [ ] **P2 — Raman by mode pairs (CtC-type runs).** The convolution is linear, so
  h∗E² = Σ_mn (h∗A_mA_n) e_m e_n: M(M+1)/2 pair convolutions (21 at M=6) instead of one
  per node (65), then a small GEMM and the pointwise product with E. Gain on the Raman
  term = Np / [M(M+1)/2]: 65/21 ≈ 3× at nr=64, M=6 (CtC RHS −25 %), 65/10 ≈ 6.5× at
  M=4 — but only 33/21 ≈ 1.6× at nr=32, so **tune `nr` first** and revisit this only
  if a case with large Np or small M keeps Raman dominant. Needs a transform-aware
  Raman path (the transform owns Emt and the pair products) — the one place the
  analysis doc's tensor idea pays, and only because Raman carries two long FFTs per
  node. Do **not** do the same for Kerr: it is already exact on the fixed grid (5e-16
  at nr=24), its grid evaluation is ~5 FLOP per sample on a field the plasma needs
  anyway (0.14 of 1.26 ms), and through the pair machinery it would cost a 21×21 GEMM
  per sample (~3× more FLOPs) and break response encapsulation. Effort M.
- [ ] **P2 — GPU: fuse the plasma's elementwise passes** between the three scans (rate;
  1−exp + phase; loss term + accumulate) — traffic −2.5×, intensity 0.5 → ~1.5 FLOP/B;
  matters on H100-class (bandwidth-bound), little on the A40 (then FP64-compute-bound).
  Then, if the scans dominate, the **time-tiled scan** (parallel over columns × time
  blocks with carried block prefixes; analysis doc §9.3) — the GPU form of the fused CPU
  column. **Decide after the A40 numbers.**
- [ ] **P2 — GPU: FP32 rate evaluation** (spline + exp only; everything else FP64). The
  rate is accurate to ~1e-4 by construction; FP64 exp/log are software sequences on
  NVIDIA, so this lifts the compute co-limit on weak-FP64 cards (A40/L40S) ~30×; no
  effect on H100. Test against the RK45 controller (step count), not only the spectrum.

Arithmetic-intensity estimate (from the code; see the conversation of 2026-08-17): plasma
device path ≈ 0.5 FLOP/B (bandwidth-bound; A40 ridge 0.85, H100 ~10), Kerr 0.3, GEMMs
M/4, FFTs ~2; CPU columns are L2-resident at nto ≤ 8k (compute-bound on exp/log, ~4
FLOP/B vs DRAM), DRAM-bound at nto ≥ 2^15. Best hardware for this code: HBM bandwidth
(H100/H200/B200) ≫ A100 > A40 ≈ 64-core EPYC for the small scalar case; 5–10× in favour
of any HBM GPU for the vector / large-nt cases. To measure instead of estimate: Nsight
Compute `--set roofline` on the A40 job; LIKWID `MEM_DP` on a CPU node.

---

## 3. Coverage: things that cannot yet run on a device (or not batched/threaded)

- [ ] **P1 — Plasma in `TransFree` (3-D free space) on GPU.** `PlasmaCumtrapzBatched` now
  exists, but `TransFree` on a device requires the fast path (EnvGrid, no oversampling)
  and plasma needs a real field (RealGrid), whose general path uses host `copy_scale!`
  loops and columnwise responses and is refused on device. The device `copy_scale!` /
  `copy_scale_both!` view-broadcast methods were added for the modal path, so this is now
  mostly wiring: a device method of `_trans_free_general!` using them and
  `Et_to_Pt_ordered!` with `batched_responses`, plus the (nto,ny,nx) → (nto,1,ny·nx)
  reshape for the batched plasma. Effort M (~80–120 lines + JLArray/CUDA tests). Payoff:
  high if 3-D + ionisation runs are on the roadmap (filamentation, free-space RDW) — the
  3-D case is where the GPU already showed the largest speedups (Kerr only).

- [x] **P1 — `RamanPolarField` batched (real-field Raman on device / threaded).** (Done:
  `Nonlinear.RamanPolarFieldBatched`, thg true/false, host threaded + device broadcasts,
  picked up automatically by `batched_response`.) This was more urgent than first
  thought: with the fixed transform now the default, Raman gases (H₂, N₂, SF₆, air —
  the CtC-type runs) were falling to the *serial* columnwise fallback over 65 nodes,
  slower than the old adaptive path, and refused on device. Measured on the CtC-type case
  (H₂, 515 nm, 6 modes, nt=32768/nto=65536): 19 ms per RHS fixed nr=64 vs 72 ms adaptive
  on 8 laptop threads. **Caveat found on the way (P2, S):** FFTW `PATIENT` planning of a
  fresh doubled-grid batched shape (2·nto × npts, e.g. 131072 × 65) takes tens of minutes
  on first use (once per machine/thread count — wisdom is cached); the benchmark script
  uses `MEASURE`. Consider `MEASURE` for batched plans above some size, or document
  `Luna.set_fftw_mode(:measure)` for large-nt Raman runs.

- [ ] **P2 — `Kerr_field_nothg` and `Kerr_env_thg` as batched responses.** Both are
  closures over a Hilbert plan / time vector and are neither pointwise nor batched, so
  `thg=false` on a RealGrid and `thg=true` on an EnvGrid are columnwise-serial and refused
  on device. Effort S each (~30–40 lines: struct + `batched` trait + FFT-along-dim-1
  Hilbert on the whole array / broadcast with a `t` column vector; keep the old
  constructors returning the new structs). Payoff: enables those two switches on device
  and threads them on CPU.

- [ ] **P2 — Scalar gas mixtures batched/threaded.** With `npol == 1` and a mixture
  (`density::Vector`, tuple of response tuples) `Et_to_Pt_ordered!` falls back to the
  serial columnwise `Et_to_Pt!` (`src/NonlinearRHS.jl:206`) — works, but serial and
  host-only; the vector path (`_apply_vector!`) already handles mixtures correctly.
  Effort S (~10 lines: fill once, then the ordered loop per `(responses_i, ρ_i)` without
  refilling). Payoff: mixtures on device and threaded on CPU.

- [ ] **P3 — `TransRadial` (radially symmetric free space) on device.** Depends on a
  device Hankel transform (`Hankel.QDHT` is a dense matrix product — trivially a
  `CuArray` GEMM — plus its scaling vectors); the rest is the same responses. Effort M.
  Payoff: moderate — radial runs are cheap already.

- [ ] **P3 — `TransModeAvg` on device.** Refused today (`_check_modeavg_arraytype`). The
  problem is a single column; a GPU only pays for enormous nt. Effort S–M; payoff low.

- [ ] **P3 — Non-scale-invariant z-dependent modes on device.** `update_matrices!`
  re-evaluates the mode fields on the host and uploads `S`, `Wp`, `Wc` per RHS when the
  modes are z-dependent but not scale-invariant (e.g. `RectMode` with callable
  dimensions, antiresonant models with z-dependent wall). Marcatili tapers use the exact
  rescaling and cost two scalars. Effort M (device evaluation of `field`); payoff low
  unless such runs are common.

- [ ] **P3 — Step-index fibres: composite quadrature with a breakpoint at `r = a`.**
  `StepIndexMode` has `dimlimits = 10a` and a kink at the core boundary; the single
  Gauss–Legendre panel converges slowly there. Effort S–M (`transverse_quadrature` with
  breakpoints, weights concatenated); payoff only for `StepIndexFibre` users.

- [ ] **P3 — Adaptive `TransModal` on device.** Never: it is the reference oracle and stays
  host-only by design (documented).

---

## 4. Validation and physics still outstanding (from the plan's Phase 6)

- [ ] **P1 — Kerr energy / photon-number conservation over long propagation.** The fixed
  symmetric rule makes the Kerr term derive from an exactly symmetric overlap tensor, so
  conservation should hold to rounding independent of `nr`; the adaptive rule breaks it at
  tolerance level. Effort S (script under `test/manual/`, loss-free Kerr-only run, plot
  Σ energy and photon number vs z for both transforms). Payoff: a clean demonstration of
  the accuracy argument for the paper/docs, and a regression test candidate.

- [ ] **P1 — Mode-convergence rerun (4 vs 6 vs 8 modes) with and without plasma.** With
  the quadrature error now fixed and tiny, mode truncation can be measured on its own;
  expect worse convergence in the ionising regime (plasma index is sharply peaked on
  axis). Effort S (script; the RDW examples at 1.5 m ×3 mode counts ×2). Payoff: tells
  whether the production `modes=4` is adequate — a physics result, not a code one.

- [ ] **P2 — 2-D vector-set convergence and the `nθ` default.** `modal_fixed_convergence.jl`
  covers HE1m sets and the x+y HE11 pair; the fully vectorial TE01/TM01/HE21/HE11 set with
  plasma has been tested for correctness (`test_modal_fixed.jl`) but not swept in `nθ`
  (default 16, warning below the Kerr bound `4·h_max+1`). Effort S. Payoff: a justified
  `nθ` default (probably 16 for Kerr-exactness, ≥ 32 with plasma at high intensity).

- [ ] **P2 — Third-gas check of smooth vs channel-sum PPT (Kr or Xe at 800 nm, or Ar at
  1.8 µm; Keldysh γ ≈ 1–2).** He (γ≈0.4) and 80 mbar Ar (γ≈1) are within the channel-sum
  model's own quadrature scatter. Effort S (`modal_fixed_rdw.jl` gains a case). If a
  case ever disagrees, the right fix is a **soft channel-closing regularisation** in
  `IonRatePPT` (smooth step over the last channel; effort S–M) rather than resolving the
  hard step with `nr ≥ 256`.

- [ ] **P2 — PPT spline accuracy audit.** Compare the cached `IonRatePPTAccel` spline
  against direct `IonRatePPT` at a few thousand random fields over the production
  intensity range, for the default knot count; the spline (not the quadrature) now sets
  the plasma error floor. Effort S. Also consider storing `ln W` on uniform `ln E` knots
  with cubic coefficients (device-friendly, no search) if the audit shows the current
  table limits accuracy.

- [ ] **P3 — Thread-scaling study on a 32-core node** (1/8/16/32 threads; FFTW threads vs
  Julia threads for the batched plans; `Utils.tforeach` `minlen` gates for the plasma
  columns). Effort S. Payoff: know the CPU baseline the GPU has to beat, and tune the
  gates.

---

## 5. Documentation, infrastructure, merge

- [ ] **P1 — A user-facing GPU page** (`docs/src/gpu.md` or a section in `interface.md`):
  `arraytype=:cuda`, what runs on device (modal fixed: yes; `TransFree` fast path: yes;
  plasma in 3-D: not yet; adaptive/mode-averaged: no), the statistics caveat and
  `allow_device_stats`, `stats_period`, `step_on` vs dense output, memory
  footprint per field, `SlurmExec(gres, instances)` recipe with a scan-script skeleton
  (after the §0 fix), and how to run the hardware-gated tests. Effort S–M.

- [ ] **P2 — Precompile workload.** Luna has no `PrecompileTools` workload; the fixed
  transform + batched plasma add noticeable first-call latency (kernel/plan compilation).
  Effort S–M; payoff: `instances`-style scans pay TTFX per process.

- [ ] **P2 — Merge plan.** Commits on `modal-fixed` are phase-ordered for cherry-picking:
  (1) `3ed6007` bug fix; (2) `7c11178` quadrature/mode matrices; (3) `699474f`
  `TransModalFixed` host path; (4) batched responses/scan/ionisation; (5) Interface/Stats/
  docs; (6) device plumbing; (7) validation + defaults; (8) `06be376` native scan. Steps
  1–5, 7 are CPU-only and could go to `master` ahead of the GPU work if the `gpu` branch
  itself is not merged first; the `gpu` branch's array-generic refactors (`Utils.backend`,
  `tchunks`, `plan_*_backend`, `device_zeros`, `GridVectors`) are prerequisites of 6.
  Decide the order once the A40 numbers are in.

- [ ] **P3 — README / CHANGELOG entries** for the new default modal integral, the PPT
  default change (`sum_integral=true` — a physics-visible change: peak electron density
  moves by up to ~10 % in the multiphoton regime, final fields ≪ 1 %), and `arraytype`.

- [ ] **P3 — Retire the old `TransModal` default keyword paths in examples/tests** once the
  fixed transform has been in production for a while (keep the type as the oracle).

---

## 6. Ideas considered and rejected (so they are not re-litigated)

- Exact overlap tensor Γ_mnpq for Kerr/Raman (the "hybrid" of the analysis doc): the fixed
  Gauss–Legendre grid is already exact for the quartic at nr≈24–32; at M ≲ 10 the Γ
  contraction costs the same FLOPs as the grid GEMMs and both are dwarfed by the 2M
  FFTs; it breaks response encapsulation. Revisit only if M ≳ 50 becomes a use case
  (then flip Kerr to Γ, keep plasma on the grid).
- Amortised adaptivity (re-mesh every k steps): a fixed operator is better for long
  propagation and for the GPU; the sizing turned out small.
- Wiring the embedded error estimate into automatic refinement: alarm, not controller
  (documented); over-resolve deliberately instead.
- Threading `pointcalc!` of the adaptive transform: shared `ToSpace.Ems`, still no device
  path.
- Re-tuning `nt` / RK45 tolerance as part of this work: do it after the spatial error is
  fixed — now possible (§4).

---

## 7. Three-dimensional free space (`TransFree`) — the ModelPNPS TG-FROG path

Unlike §§0–6 this path **has** been measured on an A40, against the 04 Kerr production
campaign (`Nω=256`, `N=768`, `rtol=1e-7`, 16 z-saves, field = 2.25 GiB):

| | GPU (1×A40) | CPU (8 threads) |
|---|---|---|
| wall / delay point | 265–281 s | 2873–3150 s |
| of which propagation | 227–230 s | 2779–3048 s |
| **non-propagation overhead** | **38–50 s (14–18 %)** | 94–102 s (3.2 %) |
| RK45 steps | 73–77 | 86–88 |
| device resident | 22.5 GiB (**exactly 10.0 fields**) | — |
| host peak RSS | 14.2 GiB | 32.8–33.3 GiB |

Speedup 11.4×; accuracy ≤7.4e-4 of the trace maximum against a bit-identical CPU
reference; the RK45 error norm was verified host-vs-device on hardware (ratio
1.000000000, reldiff ~1e-13, flat up to a cancellation factor of 2.3e8), so the lower GPU
step count is trajectory divergence seeded by cuFFT-vs-FFTW, not an accuracy difference.

The cuFFT plan workspace — the one quantity the original plan could not predict — is
**negligible**: 10.0 fields is exactly 9 RK45 registers + 1 transform buffer.

The bottlenecks here are therefore **not** the modal ones. Two thirds of the remaining
headroom is host-side and I/O, not kernel time. Items below are ordered by expected value
for that campaign; see `ModelPNPS/GPU-TODO.md` for the downstream half of each.

- [ ] **P0 — Measure the device RHS breakdown** (Effort S, script under `test/manual/`).
  Everything below is a traffic-model estimate. Time, in isolation on the device: the two
  3-D FFTs, `_prop_factored!` (8×/step), the response broadcast, `_apply_towin!`,
  `_scale_nl!`, the RK45 stage combines, `weaknorm_fused`. The single decision this
  informs is **whether the step is FFT/FP64-bound or bandwidth-bound**, because that
  determines whether renting an A100 is worth 2× or 4×: the A40 (GA102) has 1/64-rate
  FP64 (~0.58 TFLOPS) against the A100's 1/2 rate (~9.7 TFLOPS, **16.7×**), while
  bandwidth differs by only 2.2× (0.70 vs 1.55 TB/s). If the FFTs dominate, an A100 is
  worth far more than its bandwidth ratio suggests — and that is a procurement decision,
  not a code one, so it is worth knowing before any of the kernel work below.

- [ ] **P1 — Skip the host FFT prototype and host plan on the device path** (Effort S).
  `setup(::EnvGrid, ::FreeGrid)` (`src/Luna.jl:377-407`) unconditionally allocates
  `xr = Array{ComplexF64}(undef, nt, ny, nx)` purely as an FFTW planning prototype, plans
  `FTh` against it, allocates `Eωk` as a host array, and only then adapts to the device.
  At production that is **two full host fields (4.5 GiB) and a 3-D FFTW plan that the
  device path never uses** — ModelPNPS passes `inputs = ()`, so `doinputs_fs!` is a no-op
  and `FTh` is discarded. Fix: when `ondevice && isempty(inputs)`, allocate `Eωk` with
  `device_zeros(arraytype, ComplexF64, …)` and build neither `xr` nor `FTh`. Keep the host
  build when there are inputs (the field builders are host code). Payoff: 4.5 GiB off the
  host peak and the host planning time, for a few lines. This is the largest single item
  in the 14.2 GiB host figure after the beamlets.

- [x] ~~**P1 — Let an output handler consume the *device* state.**~~ **Not needed** — this
  was listed as the enabler for ModelPNPS's save-time extraction, and that landed with
  **no Luna change at all** (`ModelPNPS.TraceExtractOutput`). Three existing pieces of
  design made it work: `needs_host_y` already defaults to `false`, so `HostOutput` passes
  `y` through untouched; `nostats_only` is a plain generic function, so a handler can opt
  out of the device-statistics refusal with a method on *its own* type; and
  `RK45.interpolate` snaps to the step endpoint and returns `s.yn` **itself**
  (`src/RK45.jl:321`), so a handler whose saves are pinned with `step_on` receives the
  device array rather than a copy. Recorded because the general lesson is worth keeping:
  the output interface was already device-transparent enough for a downstream package to
  reduce on device without touching Luna.

- [ ] **P2 — `needs_host_save(o)`, the save-side twin of `needs_host_y`** (Effort S).
  What the item above *should* have said. `HostOutput` unconditionally allocates `ibuf`
  (a full host field — 2.25 GiB at production) and unconditionally wraps `yfun` to copy
  into it, with no way to decline. A handler that reduces on device needs neither. The
  fix is the same shape as `needs_host_y`: a trait defaulting to `true`, checked when
  allocating `ibuf` and when building the closure, so behaviour for `MemoryOutput` and
  `HDF5Output` is unchanged by construction.
  Payoff, measured against the shipped ModelPNPS implementation: the 2.25 GiB host
  buffer, one device-to-host copy per delay point, and the whole host-fallback branch in
  `_reduce_slice!`. That branch exists for exactly one slice — **`z = 0` is the step
  *start*, not an endpoint**, so `Luna.run` reaches it through the interpolant. Note
  `interpolate` already returns `s.y` (the device array) for `ti == s.t`, so it is
  `HostOutput`, not the interpolant, that forces the copy: this trait alone would make
  every slice device-resident and let the downstream reducer be a single code path.

- [ ] **P3 — Factor out the save-condition loop** (Effort S). `MemoryOutput`,
  `HDF5Output` and now `ModelPNPS.TraceExtractOutput` each carry the same
  `while save; …; saved += 1; save, ts = save_cond(…); end` bookkeeping. An
  `Output.foreach_save(f, o, y, t, dt, yfun)` helper would let a handler supply only the
  per-slice action. Also worth documenting the handler contract while doing it: writing a
  new one currently means grepping `Luna.run` to discover that it calls the 4-arg save,
  `willsave`, `check_cache`, and two metadata forms — the last of which a handler must
  swallow with a catch-all `(o)(d; kwargs...) = nothing` that could silently absorb a call
  it ought to service.

- [ ] **P2 — Drop the `ScaledPlan` normalisation pass in the fast-path RHS** (Effort S for
  the pointwise case, M in general). On device `IFT = inv(FT)` (`src/NonlinearRHS.jl:1162`)
  is an `AbstractFFTs.ScaledPlan`, so `mul!(t.Eto, IFT, Eωk)` runs cuFFT and then a
  *separate* full-array `Eto .*= 1/N`. At production that is a 4.5 GiB read+write per RHS,
  7× per step — an estimated **~7 % of step traffic**, entirely avoidable:
  - *(S, exact, no assumptions)* for the pointwise path (`t.Pto === nothing`, which is the
    Kerr-only campaign), pass the `1/N` into `pointwise_Pt!` as a prescale applied on read.
    It fuses into a pass that already exists.
  - *(M)* for the general/batched path (Kerr+Raman), give responses a homogeneity-degree
    trait: both Kerr and the Raman envelope polarisation are degree 3 in `Et`, so an
    unnormalised inverse plus a `(1/N)^d` factor folded into `_scale_nl!` is exact. Guard
    on all responses sharing a degree and fall back otherwise.

- [ ] **P2 — Fuse `_apply_towin!` into the response evaluation** (Effort S–M). A second
  standalone full-array pass per RHS (`Pto .*= towin`, `src/NonlinearRHS.jl:1273`), same
  4.5 GiB × 7 per step, another estimated **~7 %**. The pointwise path can apply `towin` in
  the same broadcast that writes `Pto`; the batched path needs it after the accumulation,
  so it is only foldable into the last response's write.

- [ ] **P2 — Plasma in `TransFree` on device** — already listed at §3 P1; noted here
  because the TG-FROG campaigns are the use case that would need it (they are Kerr+Raman
  today). Nothing to add beyond what §3 says.

**Not worth doing** (analysed, rejected): reducing the resident field count to fit two
delay points on one A40 — two points need 45 GiB against the A40's 44.4, and on a
bandwidth- or FFT-bound kernel concurrency buys nothing anyway; FP32 anywhere in the
propagation — it would invalidate the `rtol=1e-7` accuracy campaign.
