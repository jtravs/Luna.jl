# Brief for the agent on the H100/H200 pod

You are on a rented Runpod GPU pod, **paid by the minute**. Two GPU suites run this trip,
in this order:

| | suite | transform | what it stresses |
|---|---|---|---|
| **A** | `Luna.jl/test/manual/h200_gpu_suite.sh` | `TransModalFixed` (multimode HCF) | many small kernels; launch/sync-bound |
| **B** | `ModelPNPS.jl/examples/h200_modelpnps_suite.sh` | `TransFree` (3-D free space, TG-FROG) | one huge array; bandwidth-bound |

They share Luna's device machinery (backend trait, `HostOutput`, the CUDA extension,
RK45) but sit at opposite ends of the GPU's operating range — see §6, which is the point
of running both.

**Ground rules.** Optimise for *fewest GPU-minutes to a decisive answer*. Do not run CPU
benchmarks or the CPU test suite — those belong on our HPC. Do not start experiments
beyond what is listed here without a clear reason. Never leave the pod idle waiting on a
run you will not use; if you are blocked on a decision, say so immediately rather than
exploring.

---

## 1. Layout (from `runpodcoldstart.sh`)

- Volume `/workspace` (persistent). Checkouts `/workspace/code/Luna.jl` (branch
  `modal-fixed`) and `/workspace/code/ModelPNPS.jl` (branch `gpu`). Julia project
  `/workspace/code/dev` — both are `Pkg.develop`ed there, so edits are live.
- `source /workspace/env.sh` sets `JULIA_NUM_THREADS` (cgroup-aware, cap 8),
  `JULIA_CPU_TARGET=generic`, depot paths. Results in `/workspace/runs/<stamp>/`
  (`latest` → suite A, `latest-pnps` → suite B).
- **Design docs — read the relevant one before deciding anything looks wrong:**
  - A: `Luna.jl/docs/src/developer/modal_fixed_design.md` (§5.2–5.5 costs, §8 roadmap)
  - B: `Luna.jl/docs/src/developer/transfree_design.md` (§7 performance, §10 outstanding)
- Device code: `Luna/src/{Device.jl,NonlinearRHS.jl,Nonlinear.jl,Ionisation.jl,Maths.jl,RK45.jl,Capillary.jl,Stats.jl}`,
  `Luna/ext/LunaCUDAExt.jl`; `ModelPNPS/src/ModelPNPS.jl`.

## 2. Run

```bash
source /workspace/env.sh
bash /workspace/code/Luna.jl/test/manual/h200_gpu_suite.sh            # A, 20-30 min
bash /workspace/code/ModelPNPS.jl/examples/h200_modelpnps_suite.sh    # B, 15-40 min
```

Steps rerun individually — **never rerun a whole suite to test one fix**:
A: `STEPS=tests|bench|production|scan`. B: `STEPS=tests|accuracy|bench|scan` (+ opt-in
`share`), with `CASES=`, `POINTS=`, `ACC_CASE=`.

**Do not run A and B at once.** Both are expected to be bandwidth-bound, so overlapping
them contaminates every timing. If you must overlap, only overlap `tests`/`accuracy`.

Start B narrow: `STEPS=bench CASES=04 POINTS=1`. `04` is the only shape with an A40
reference, so it is the only one where "slow" is distinguishable from "as expected".

If a step is clearly hung (no output for >10 min outside a propagation), kill it, note
where, continue with the next.

---

## 3. Suite A — modal (`TransModalFixed`)

1. **Header**: `Luna modal-fixed @ <sha>`, GPU name, driver/runtime, threads.
2. **tests** — all pass. Acceptance `@info` lines: `device norm` ratio 1 ± ~1e-12 at every
   `delta`; `CUDA modal transform vs host rel` ≲ 1e-11 (ADK, PPT, Raman thg on/off);
   `CUDA modal propagation vs host` < 1e-8; `LAZY_CUDA_OK=true` (fresh subprocess).
3. **bench phase A** — `RHS x.xxx ms` for `vuv`, `ctc`, then the primitives line. Laptop
   CPU (8 threads) reference: 0.69 ms (vuv) / 14.5 ms (ctc). *Interpretation*: primitives
   summing to ≪ RHS means launch/sync-bound (expected for `vuv`, ~30 launches + syncs);
   comparable means bandwidth-bound. Record, do not optimise.
4. **bench phase B** — `ms/step` and `7×RHS/step`. The remainder is host work: RK45
   error-norm sync, statistics, windows, output.
5. **production** — wall and ms/step for RDW VUV 1.5 m (nr 32/64), lean vs default stats
   (the difference *is* the host statistics); CtC 0.3 m. Laptop CPU (8 threads): VUV 1.5 m
   32 s (nr=32) / 52 s (nr=64), default stats.
6. **scan** — s/point (setup vs propagation) → the 400-point estimate. The 4-process run
   should show **3–5× throughput**: these kernels are small enough that one process leaves
   the card idle.

## 4. Suite B — 3-D free space (`TransFree`)

State is one `(Nω, N, N)` `ComplexF64`; the solver holds 9 plus 1 transform buffer.

| case | grid | field GiB | expect device | expect wall/point |
|---|---|---|---|---|
| `dd05` | 256×640² | 1.56 | ~15.6 GiB | ~7 s |
| `04` | 256×768² | 2.25 | ~22.5 GiB | ~10 s |
| `dd20` | 256×1024² | 4.00 | ~40 GiB | ~18 s |
| `100um` | 512×640² | 3.13 | ~31 GiB | ~28 s |

**The wall times are estimates from a traffic model never validated against a card — a 2×
miss either way is not a bug.** The device figures are *measured*: an A40 gave exactly
10.00 fields at `04`, so a large deviation there is a real finding (cuFFT workspace) worth
reporting, not working around. A40 reference for `04`: **276 s/point, 22.5 GiB**.

1. **tests** — all pass (includes the hardware-gated CUDA testset).
2. **accuracy** — two things, both GPU-only:
   - the two extraction routes must agree to **< 1e-10**. They reduce the same slices by
     different routes, so anything larger is a bug in the extraction, not the propagation.
   - `rtol=1e-7` vs `1e-8`: **a physics measurement, not a pass/fail.** These runs use
     `weaknorm`, which measures error against the pump-dominated whole field, so the weak
     FWM signal's own relative error is `~rtol × ‖pump‖/‖signal‖` and a wing point is the
     hard case *by design*. Record it — it has never been measured at these shapes. **Do
     not "fix" it.**
3. **bench** — wall/point, device GiB (÷ field size should be ≈10), and the `s/GiB`
   column. Flat across cases (spread < 1.3×) ⇒ the card is saturated at every shape ⇒
   concurrency cannot help. Falling with field size ⇒ headroom ⇒ `STEPS=share` is worth
   the minutes (otherwise skip it, see §6).
4. **scan** — s/point *including* setup and `scansave`. That, not the benchmark number, is
   what to multiply by the delay count when sizing a campaign.

---

## 5. Triage (shared)

**CUDA environment.** Not functional, or "precompiled for CUDA 13.x but driver is 13.y" →
`julia --project=/workspace/code/dev -e 'using CUDA; CUDA.versioninfo()'`. A driver/runtime
mismatch means a stale pin in the dev project's `LocalPreferences.toml`; **remove the pin**
(`CUDA.reset_runtime_version!()`), never add one.

**`Package X not found`.** The scripts run as scripts, so everything they `import` must be
a *direct* dependency of `/workspace/code/dev`, not merely a dependency of Luna. Fix with
that suite's `STEPS=pkgs`.

**Scalar indexing error.** A device array reached a scalar loop — **always a real bug, and
almost always a host/device residency mix rather than a kernel error.** Check
`Luna.Utils.isdevice(x)` on each operand first. This class has bitten three times.
- A: a missing `Utils.isdevice` dispatch on that path.
- B: `_reduce_slice!` (`ModelPNPS/src/ModelPNPS.jl:1422`) routing a host array into a
  device kernel, or `_extract_slice_device!` (`:1331`).

**Kernel compilation error** ("dynamic function invocation", boxed closure, non-isbits
capture) — the broadcast named in the trace, usually a `Val`/scalar captured wrongly.
A: `Capillary._marcatili_element`, `Ionisation.ionrate_device!` / `Maths.spline_eval`,
`Nonlinear._plasma_batched!(::DeviceBackend)`, `_raman_field_batched!(::DeviceBackend)`.

**A comparison off by a huge factor** (1e+6, not 1e-6) — suspect a *normalisation*, not
physics. `NonlinearRHS._ift_unscaled` (`:1309`) splits the device inverse FFT plan into raw
transform + `1/N`; losing the scale is off by `N` per transform. Isolated by the
`"unscaled inverse plan and the fused pointwise RHS"` testset in `test/test_device.jl`.

**World age `MethodError … too new`.** Suite B imports CUDA at top level and passes
`CUDA.CuArray`, so it should not appear; if it does, something passed the *symbol*
`:cuda`. Suite A uses the lazy form deliberately (`Luna.lazy_arraytype` → `invokelatest`,
tested by `LAZY_CUDA_OK`); a failure there is world-age re-entry in
`Interface.prop_capillary`.

**`Maths.scan!` / native scan failure** (A only): `ext/LunaCUDAExt.jl` uses
`Base.cumsum!(out, y; dims=1)`; the fallback is the doubling scan (`scan_scratch`).

**Slow rather than wrong** — check cheapest first:
1. Is the other suite running? `nvidia-smi`. If so the numbers are contended.
2. Power/clock cap: `nvidia-smi --query-gpu=power.limit,power.max_limit,clocks.sm,clocks.max.sm --format=csv`.
   Rented cards are sometimes capped; compare the cold start's copy-bandwidth number
   against ~4800 GB/s for an H200.
3. Step count: the Luna log `Propagation finished in …, N steps`. B/`04` at 40 µm took
   73–77 steps on the A40. Many more means the *solver* is struggling, a different problem.
4. **Host-bound** (ms/step ≫ 7×RHS, or setup dominating): `JULIA_CPU_TARGET=generic` (try
   `native` for one run and compare); default statistics (A: `stats_period`,
   `mode_error=false`); FFTW threads — `Luna.Utils.FFTWthreads()` is capped at
   `Sys.CPU_THREADS`, which **in a container reports the whole host**; if it exceeds
   `JULIA_NUM_THREADS`, call `Luna.set_fftw_threads(Threads.nthreads())` and rerun.
5. Only then the code. `CUDA.@profile` on one RHS — see §6.

**Out of memory.** B/`dd20` needs ~40 GiB; `share` needs `NPROC × 10 × field`. If device
memory climbs across delay points despite `Luna.device_reclaim()`, that is a leak — report
it. Host RAM peak in B is set by `build_setup` (four beamlet fields + window), ~18 GiB at
`N=1024`.

---

## 6. The question this trip exists to answer

Both design docs flag the same unmeasured thing: **is this code FP64-bound or
bandwidth-bound?** It decides whether a card's FP64 rate or its HBM is what you are paying
for. The A40 evidence says the 3-D free-space path is **FP64-bound** (it ran ~6× slower
than its bandwidth allows), and the H200 has **58×** the A40's FP64 against only **6.9×**
its bandwidth — so the expected outcome is that suite B becomes bandwidth-bound here.

The two suites bracket the range, which is why running both is worth more than either:

- **A is launch/sync-bound** (~0.5M elements per kernel) — the card idles between
  launches, which is why sharing it across 4 processes gains 3–5×.
- **B saturates the card** (~151M elements per kernel, 300× larger) — one delay point
  should already fill the memory system, so sharing should gain ~nothing. B's `s/GiB`
  column tests exactly this; only run `STEPS=share` if it is *not* flat.

**The single most valuable thing to bring home is `CUDA.@profile` on one RHS from each
suite**, showing the FFT-vs-GEMM-vs-elementwise split. It is minutes of GPU time and it
closes the "measure first" item in both design docs.

---

## 7. Record and commit

Copy from `/workspace/runs/<stamp>/`: `suite.log`, `bench.csv`, and B's scan
`*_collected.h5` (the accuracy artefact — it uses production parameters so a CPU run on
the HPC can be compared against it with `verify_against_collected`; read the **scan-peak**
normalised column, not the own-peak one — see `ModelPNPS/examples/scan_peaks.jl`).

Append results, with GPU name, driver/runtime, threads and both commits:
- A → `docs/src/developer/modal_fixed_design.md` §5.4
- B → `docs/src/developer/transfree_design.md` §7 (and strike the relevant §10 item)

If you changed code to make something pass: keep it minimal, add or adjust a test in
`test/test_cuda.jl` / `test/test_device.jl` (or `ModelPNPS/test/device_test.jl`), rerun
`julia --project=/workspace/code/dev /workspace/code/Luna.jl/test/runtests.jl test_cuda test_device`,
and commit with a message saying what failed *on this hardware* and why the fix is right.
**Work on a branch** — `modal-fixed` and `gpu` are being merged elsewhere.

## 8. Do not

- Run CPU delay points or CPU benchmarks (B's `--cpu=1` exists; do not use it — ~50 min
  per point at production shapes).
- Rerun a whole suite to test one fix; use `STEPS=`.
- Tune `rtol`, `max_dz` or `twin_period` — they are what the campaigns are validated at.
- "Fix" the rtol comparison in B, or a large own-peak difference in a delay wing.
- Add a CUDA runtime pin.
- Run both suites concurrently when timing matters.
