# Brief for the agent on the H100/H200 pod

You are on a rented Runpod GPU pod (paid by the minute). Your job: run Luna's GPU test and
benchmark suite, read the results, fix what is broken quickly, and record the numbers.
Do not run CPU benchmarks or the CPU test suite (they run on our HPC). Do not start long
experiments beyond what is listed here without a clear reason.

## Layout (from `runpodcoldstart.sh`)

- Volume `/workspace` (persistent). Luna checkout `/workspace/code/Luna.jl`, branch
  `modal-fixed`; Julia project `/workspace/code/dev` (Luna is `Pkg.develop`ed there, so
  edits are live). `source /workspace/env.sh` sets `JULIA_NUM_THREADS` (cgroup-aware,
  cap 8), `JULIA_CPU_TARGET=generic`, depot paths.
- Design/measurements/roadmap: `docs/src/developer/modal_fixed_design.md` (read §5.2–5.5
  and §8 first if anything looks off). GPU code paths: `src/NonlinearRHS.jl`
  (`TransModalFixed`), `src/Nonlinear.jl` (`PlasmaCumtrapzBatched`, Raman batched),
  `src/Ionisation.jl` (device rate), `src/Maths.jl` (`scan!`), `src/Device.jl`,
  `ext/LunaCUDAExt.jl`, `src/Capillary.jl` (`MarcatiliLinop`), `src/Stats.jl`
  (`StatsCollector`), `src/RK45.jl` (`make_prop!`, error norms).

## Run

```bash
source /workspace/env.sh
bash /workspace/code/Luna.jl/test/manual/h200_gpu_suite.sh
```
Results: `/workspace/runs/latest/` (`suite.log`, `bench.csv`, `scan_*`). Steps can be
rerun individually: `STEPS=tests`, `STEPS=bench`, `STEPS=production`, `STEPS=scan`.
Expected total 20–30 min. If a step is clearly hung (no output for >10 min outside a
propagation), kill it, note where, and continue with the next step.

## Check, in this order

1. **Header**: `Luna modal-fixed @ <sha>` (should match origin), GPU name, driver/runtime
   versions, `threads`. If CUDA is not functional → `julia --project=/workspace/code/dev -e
   'using CUDA; CUDA.versioninfo()'`; a driver/runtime mismatch means a stale
   `LocalPreferences.toml` in the dev project (see cold-start comments) — remove the pin.

2. **tests** — all testsets must pass. Acceptance numbers printed as `@info`:
   `device norm` ratio 1 ± ~1e-12 at every `delta`; `CUDA modal transform vs host rel`
   ≲ 1e-11 (ADK, PPT, Raman thg on/off); `CUDA modal propagation vs host` < 1e-8;
   `LAZY_CUDA_OK=true` (fresh subprocess). Failure modes and where to look:
   - kernel compilation errors ("dynamic function invocation", boxed closures, non-isbits
     captures): the broadcast in question — usually a `Val`/scalar captured wrongly.
     `Capillary._marcatili_element` (linear operator), `Ionisation.ionrate_device!` /
     `Maths.spline_eval`, `Nonlinear._plasma_batched!(::DeviceBackend)`,
     `_raman_field_batched!(::DeviceBackend)`.
   - scalar indexing error: a device array reached a scalar loop — `Utils.isdevice`
     dispatch is missing somewhere on that path.
   - `native scan` failure: `Maths.scan!(::CuArray)` in `ext/LunaCUDAExt.jl` (uses
     `Base.cumsum!(out, y; dims=1)`); fall back is the doubling scan (`scan_scratch`).
   - lazy `:cuda` subprocess failure: world-age re-entry in `Interface.prop_capillary`
     (`Luna.lazy_arraytype` → `invokelatest`); read the captured subprocess text.

3. **bench, phase A** (`RHS x.xxx ms` for `vuv` and `ctc` on cuda; then the primitives
   line): record them. Sanity: A40 reference not yet available (job pending); laptop CPU
   (8 threads) is 0.69 ms (vuv) / 14.5 ms (ctc). Interpretation: if `sum` of primitives ≪
   RHS, the RHS is launch/sync-bound (many small kernels) — that is expected for `vuv`
   (~30 launches + syncs); if it is not much smaller, it is bandwidth-bound. Do not
   optimise here; just record.

4. **bench, phase B**: `ms/step` and `7×RHS/step`. What remains of a step is host work:
   RK45 error-norm sync, statistics, windows, output. `rel L2 vs cuda` is vs itself (0)
   in this GPU-only run.

5. **production**: wall, ms/step for RDW VUV 1.5 m (nr 32/64) lean stats vs default
   stats — the difference is the host statistics; CtC 0.3 m. Laptop CPU reference (8
   threads): VUV 1.5 m 32 s (nr=32) / 52 s (nr=64) with default stats.

6. **scan rehearsal**: seconds per point (setup vs propagation) → the printed 400-point
   estimate; the 4-process run's per-point propagation time vs the 1-process one tells
   how much sharing the GPU helps (expected 3–5× throughput for this small case).
   `Processing.scanproc` must return the 2×2 band-energy and electron-density matrices.

## If something is slow rather than wrong

- Host-bound step (ms/step ≫ 7×RHS): first suspects are `JULIA_CPU_TARGET=generic`
  (try `JULIA_CPU_TARGET=native` for one production run and compare), the default
  statistics (`stats_period`, `mode_error=false` — the "lean" settings), and the FFTW
  thread count (`Luna.Utils.FFTWthreads()`; on Linux it is capped at `Sys.CPU_THREADS`,
  which in a container may report the whole host — if it is > `JULIA_NUM_THREADS`,
  set `Luna.set_fftw_threads(Threads.nthreads())` at the top of the script and rerun).
- Device-bound but slower than expected: check `nvidia-smi` power cap and clocks during
  the run; the cold start prints the measured copy bandwidth vs spec.
- Never leave the pod idle waiting on a long run you are not going to use.

## Record

Append the phase-A table, the production rows and the scan per-point numbers (with GPU
name, driver/runtime, threads, commit) to `docs/src/developer/modal_fixed_design.md`
§5.4 (device results) and commit on `modal-fixed`. If you had to change code to make
something pass, keep the change minimal, add/adjust a test in `test/test_cuda.jl` or
`test/test_device.jl`, run `julia --project=/workspace/code/dev
/workspace/code/Luna.jl/test/runtests.jl test_cuda test_device` again, and commit with a
message that says what failed on this hardware and why the fix is right.
