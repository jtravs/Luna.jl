# Three-dimensional free-space propagation: threading, memory and the device path

!!! note "Living document"
    This page records the design of the work done on the 3-D free-space transform
    (`NonlinearRHS.TransFree`) and the machinery it shares with the rest of Luna — the
    RK45 register and pass reduction, the threaded elementwise kernels, the factored
    (lazy) operators, the batched envelope Raman response, and the device (GPU) execution
    path — across the `perf` → `gpu` → `modal-fixed` branch line. It is the companion to
    [The fixed-quadrature modal transform and the device (GPU) path](@ref), which covers
    the *modal* transform; the two share the backend trait, the device execution model and
    the extension mechanism described in §5 here and §3.7 there. Numbers are quoted with
    the machine and configuration they were measured on, and estimates are marked as such.

    Driving use case: the TG-FROG campaigns built on Luna (three pump beamlets in boxcars
    geometry generating a fourth, much weaker four-wave-mixing signal), which are
    `Grid.EnvGrid` × `Grid.FreeGrid` propagations at
    ``(N_\omega, N_{k_y}, N_{k_x}) = (256, 768, 768)``. The downstream half of that work
    is not described here.

## 1. Summary

3-D free-space propagation is a different optimisation problem from the modal transform.
There is no quadrature and no synthesis/projection GEMM: the state *is* the field, one
`ComplexF64` array of ``(N_\omega, N_{k_y}, N_{k_x})``, and every operation is either a
3-D FFT or an elementwise pass over that array. At production size one array is **2.25
GiB**, so the binding constraints are (a) how many such arrays are live at once, and (b)
how many times per step each is read and written. Neither is affected by the physics; both
are set by the solver and transform plumbing.

The work therefore had three phases, in this order:

1. **Fewer arrays and fewer passes** (the `perf` branch). Lazy interpolant and error
   registers, fused RK45 stage combines, an in-place `fbar!`, `preserve_input=false`, a
   `TransFree` fast path that skips the oversampled buffers when they would be identical,
   and *factored* linear operator and normalisation objects that compute their elements on
   demand instead of storing two more field-sized arrays. Production memory **46 GB →
   30.6 GiB**.
2. **Threading** of the elementwise kernels and of FFTW, which the free-space path had
   never had. **4.18×** on 8 threads.
3. **A device (GPU) path** (the `gpu` branch) that reuses all of the above unchanged. The
   discipline that made this cheap is that every hot kernel had already been rewritten as
   a whole-array broadcast or reduction; the device work was then mostly *dispatch*, not
   new algorithms.

Headline results, all detailed below:

- **Bit-identity on the host.** Every `perf`-branch change is bit-identical to the code it
  replaced, enforced with `isequal` (not `isapprox`) in `test/test_perf_bitident.jl`, and
  confirmed at production scale: a delay point of a running campaign recomputed by the
  merged stack reproduced the stored reference with `max rel diff = 0.0` on every dataset,
  at both a central and a far-wing point.
- **Speed (CPU).** 4.18× on 8 threads for a production-like configuration; a smaller
  benchmark (``N=256``) went 4.17 GiB / 62.9 s → 3.11 GiB / 37.7 s at 4 threads with an
  identical checksum.
- **Speed (GPU, measured on an A40).** 11.4× per delay point over 8 CPU threads
  (4.6 min vs 52.5 min) at the production shape.
- **Device memory.** 22.5 GiB resident = **exactly 10.00 field copies** — 9 RK45 registers
  + 1 transform buffer. The cuFFT plan workspace, the one quantity the plan could not
  predict and had flagged as a risk to the campaign budget, is negligible.
- **Accuracy on device.** ``\text{rel }L_2 = 2.4\times10^{-11}`` (Kerr) and
  ``2.6\times10^{-12}`` (Kerr+Raman) against the host on small grids; at production
  scale, agreement with a bit-identical CPU reference to ``\le 7.4\times10^{-4}`` of the
  trace maximum, an order of magnitude *inside* the solver's own discretisation error at
  the campaign's `rtol`.
- **No CUDA dependency in Luna.** The device code is ordinary broadcasts and `mapreduce`s
  dispatched on a backend trait; the genuinely vendor-specific glue lives in a package
  extension.

## 2. How the old code worked, and what it cost

### 2.1 The propagation

`Grid.EnvGrid` × `Grid.FreeGrid` propagates ``E(\omega, k_y, k_x)`` in the interaction
picture with an RK45 Dormand–Prince 5(4) stepper (`RK45.solve_precon`,
`RK45.PreconStepper`). Each RHS evaluation is:

1. ``(\omega, k_y, k_x) \to (t, y, x)`` — one 3-D inverse FFT;
2. the nonlinear responses, elementwise in ``(t, y, x)``;
3. the temporal apodisation window;
4. ``(t, y, x) \to (\omega, k_y, k_x)`` — one 3-D forward FFT;
5. a scaling by the normalisation factor.

Seven RHS evaluations per step (FSAL), plus eight applications of the linear propagator
``\exp(\hat L\,\Delta z)``, plus the stage combines and the error norm.

### 2.2 Where the memory went

At ``(256, 768, 768)`` — 2.25 GiB per `ComplexF64` array — the original code held:

| | arrays | GiB |
|---|---|---|
| RK45 `y`, `yn`, `ks[1..7]` | 9 | 20.25 |
| RK45 `yi` (interpolant), `yerr` | 2 | 4.5 |
| `TransFree` `Eto`, `Pto`, oversampled twins | 2–4 | 4.5–9 |
| materialised `linop`, materialised norm | 2 | 4.5 |
| input copy (`preserve_input=true`) | 1 | 2.25 |

Around 46 GB in practice, against an HPC allowance that made that the limiting factor on
how many delay points could run concurrently.

Two of those entries are pure accounting. `yi` and `yerr` are only needed if something
asks for them — dense output between steps, or an error norm that wants a materialised
error estimate — and the production configuration asks for neither. The materialised
`linop` is a constant, elementwise function of ``(\omega, k_y, k_x)`` that is *separable*:
storing it is trading 2.25 GiB for an arithmetic expression that is cheap next to the
memory traffic of reading it.

### 2.3 What was not threaded

`Utils.tforeach`/`tchunks` did not exist. The elementwise kernels — stage combines, the
propagator, the transform's column loops — were serial, and FFTW ran on one thread. For a
workload that is essentially "FFT, then walk a 2.25 GiB array", that left most of a node
idle.

## 3. Design: fewer arrays, fewer passes

### 3.1 The bit-identity discipline

The single most important design rule on the `perf` branch, and the reason the production
campaign could adopt it without re-validating the physics: **every change must be
bit-identical to the code it replaces**, tested with `isequal`.

This is stricter than it sounds, and it constrained the implementation everywhere. Fusing

```julia
y .= y0
y .+= a1 .* k1
y .+= a2 .* k2
```

into one broadcast is only bit-identical if the fused expression is *left-associated in
the same order*: `@. y = y0 + a1*k1 + a2*k2` accumulates in the same sequence the
successive `.+=` did. Reordering for symmetry, or letting a compiler contract a multiply
and add into an FMA, changes the last bits and breaks the guarantee. The kernels are
therefore written with explicit left-associated sums, and `test/test_perf_bitident.jl`
compares against serialised reference outputs from the pre-change code.

The payoff was concrete: when the merged stack was checked against a running production
campaign at ``N=768``, all three trace datasets came back at `max rel diff = 0.0` — not
"1e-12", exactly zero — which retired the question of whether any of this work perturbed
the physics.

The discipline explicitly does **not** extend to the device path, where a parallel
reduction sums in a different order by construction. That boundary is drawn in §5.6.

### 3.2 Lazy registers and fused stages

`RK45.Stepper`/`PreconStepper` make `yi` and `yerr` `Union{Nothing, T}`, allocated on first
use. With `step_on` aligned to the save positions (so no interpolation is ever requested)
and a *fused* error norm (§3.3), neither is ever allocated: the stepper holds **9**
field-sized registers (`y`, `yn`, `ks[1..7]`) instead of 11.

Because this is conditional, it is also a documented constraint on how the solver is
driven: on a device, `step_on` **must** match the save positions, or the interpolant costs
two more fields.

The stage combines were fused into single left-associated broadcasts, `fbar!` was made
in-place, and `Luna.run` gained `preserve_input=false` so the solver adopts the caller's
input array as a working buffer rather than copying it.

### 3.3 The fused error norm

The default `RK45.weaknorm` needs ``\|y\|``, ``\|y_n\|`` and ``\|y_\text{err}\|``, where
``y_\text{err}`` is a weighted combination of six stage derivatives. Materialising it costs
a tenth field-sized array and an extra pass.

`weaknorm_fused` computes the error estimate *element by element inside the reduction*, so
it is never materialised. A norm opts in by defining
`fused_errnorm(::typeof(mynorm)) = my_fused_version`; norms without one fall back to the
materialising path on the host and **error loudly on a device**, rather than silently
allocating a 2.25 GiB buffer per step.

On the host the fused version performs the same floating-point operations in the same
order as the materialised one, so it is bit-identical.

### 3.4 The `TransFree` fast path

`TransFree` supports oversampled grids (`grid.to` longer than `grid.t`), real fields and
noise fields; each needs its own buffers and a `copy_scale!` between them. For an
`EnvGrid` with no oversampling, no noise field and `scale == 1`, those buffers are
*identical to the ones already held* and the copies are no-ops.

`trans_free_fast!` takes that path: it transforms the solver's state array directly (using
the property, asserted in the hardware tests, that an out-of-place c2c transform preserves
its input), applies the responses in place where they allow it, and windows and scales in
two broadcasts. The gate is explicit:

```julia
fast = (fastpath && TT <: Complex && length(grid.ωo) == length(grid.ω)
        && scale == 1 && isnothing(noise_field))
```

The general path is retained and is the reference the fast path is tested against; the
docstring states the two are bit-identical.

### 3.5 Pointwise responses

A response that is a pure function of the local field — Kerr is ``\propto |E|^2E`` for an
envelope, ``\propto E^3`` for a real field — needs no buffer at all: it can overwrite the
time-domain array in place. `Nonlinear.KerrEnv`/`KerrField` became structs carrying
``\gamma_3`` with traits

```julia
pointwise(::KerrEnv) = true
pointwise_P(K::KerrEnv, E, ρ) = 3/4*(ρ*ε_0*K.γ3)*abs2(E)*E
```

so `TransFree` can skip allocating `Pto` entirely when *all* responses are pointwise
(`t.Pto === nothing`), saving a further field. The old closure-based constructors still
work and return the new structs.

This also turns the Kerr response into something a GPU can execute: a scalar function of
scalars, inlined into a broadcast, with no captured buffers.

### 3.6 Factored operators

`LinearOps.FactoredFreeLinop <: AbstractArray{ComplexF64,3}` stores only the separable
factors — ``k^2(\omega)`` (a vector) and ``k_\perp^2(k_y,k_x)`` (a small matrix) — and
computes elements on demand. `NonlinearRHS.FreeNorm` does the same for the normalisation.
Both are opt-in via `factored=true` and replace two field-sized arrays with two small ones.

The element formula is shared between the materialising fill loop and the lazy
`getindex`, so they cannot drift:

```julia
@inline function _linop_xy_element(ω, β1, k2ω, kperp2i)
    βsq = k2ω - kperp2i
    if βsq < 0
        return -im*(-β1*ω) - min(sqrt(abs(βsq)), 200)  # evanescent → attenuation
    else
        return -im*(sqrt(βsq) - β1*ω)
    end
end
```

`RK45.make_prop!` has a specialised kernel for `FactoredFreeLinop`, because applying the
propagator is the single largest elementwise cost of a step — it runs eight times per
step, and each call reads and writes the whole state.

A subtlety that cost a debugging cycle: parametrising the array fields for device support
(§5.3) silently removed the implicit conversion of the *scalar* fields, and `β0ref`
arrives as a `Complex` with zero imaginary part whenever the refractive-index function is
complex-valued. An explicit converting outer constructor restores the pre-existing
behaviour.

### 3.7 Batched envelope Raman

`Nonlinear.RamanPolarEnvBatched` replaces the columnwise envelope Raman response with one
that operates on the whole ``(t, y, x)`` array: four broadcasts around two in-place
batched FFTs, with the small ``h(\omega)`` response vector staged to the working array's
type. It is picked up automatically by `batched_response`, and it is what makes Raman
runs threadable on the CPU and possible at all on a device.

### 3.8 `twin_period`

The spectral/temporal apodisation windows feed back into the propagation, so applying them
on every accepted step is not the same calculation as applying them only at the saves.
`twin_period=k` applies them every ``k``-th accepted step **and always immediately before
a save** (`Output.willsave` exists for exactly this). This is a physics-affecting knob, not
a pure optimisation, and is documented as such: the production campaign records a ~7 %
change in shape at 40 µm between per-step and per-save windowing.

### 3.9 Grid sizing

Transverse grid sizes are chosen with `nextprod([2,3,5], N)` rather than `nextpow(2, N)`:
FFTW and cuFFT are efficient for 2,3,5-smooth sizes, and rounding up to a power of two can
almost double the work. The production grid moved 1024 → 640 this way (and later to 768,
which is ``2^8\times 3``).

When comparing spectra across grid sizes, note that the k-integrated datasets are in
FFT-bin units that scale as ``N^4`` at fixed physical extent (Parseval over the transverse
FFT gives ``\sum_k|E_k|^2 = N^2\sum_x|E_x|^2``, and ``\sum_x|E_x|^2 \propto N^2`` at fixed
energy). A comparison at ``N=640`` against an ``N=1024`` reference initially showed a
"0.847 relative difference" that was exactly ``1-(640/1024)^4``. Verification harnesses
must rescale.

## 4. Threading

`Utils.tforeach`/`tchunks` thread an elementwise-independent kernel over chunks. Two design
points:

- **A length threshold.** `THREADING_MINLEN = 1<<20` elements; below it the kernel runs
  serially, because thread startup dominates and — more importantly — because the small
  grids used throughout the test suite must not take a different code path from production.
- **Threading is a property of the *call*, not of the data.** `tchunks` returns index
  ranges, so the caller writes one loop body that is correct either way.

FFTW threading needed a workaround. FFTW.jl registers a Julia-task-based threading callback
with libfftw3, which **segfaults on Julia ≥ 1.12** under the partr scheduler. `set_fftw_threads`
therefore deregisters that callback and lets libfftw3 use its own pthreads:

```
[ Info: Julia ≥ 1.12 with 8 threads: using libfftw3's native pthreads for FFTs
        (FFTW.jl's Julia-task callback deregistered).
```

FFTW wisdom is keyed by *(shape, thread count)*, so a run with a thread count that has
never been planned pays a full `MEASURE` (or, worse, `PATIENT`) planning cost on first use.
`examples/pregenerate_wisdom.jl` exists to pay that once per machine.

## 5. The device (GPU) path

### 5.1 Principles

1. **The CPU path does not change.** Every device addition is a new method dispatched on a
   backend trait, or a new type parameter defaulting to the current host type. CPU method
   bodies were *moved verbatim*, never rewritten, and the bit-identity tests must keep
   passing with `isequal`.
2. **Luna core gains no CUDA dependency.** The device code is broadcasts and `mapreduce`s,
   never hand-written kernels, so generic costs the same as vendor-specific. Core uses
   `Adapt`, `AbstractFFTs` and `GPUArraysCore`; the genuinely CUDA-specific glue (device
   selection, pool reclaim, memory query) lives in `ext/LunaCUDAExt.jl`.
3. **Never put a lazy operator struct in a device broadcast.** Broadcast the separable
   *factors* instead (§5.3).
4. **Unsupported combinations error loudly** rather than silently crawling.

### 5.2 The backend trait

```julia
struct CPUBackend <: Backend end
struct DeviceBackend <: Backend end
backend(::Type) = CPUBackend()
backend(::Type{<:GPUArraysCore.AbstractGPUArray}) = DeviceBackend()
backend(::Type{<:SubArray{T,N,P}}) where {T,N,P} = backend(P)
backend(::Type{<:Base.ReshapedArray{T,N,P}}) where {T,N,P} = backend(P)
backend(::Type{<:PermutedDimsArray{T,N,A,B,P}}) where {T,N,A,B,P} = backend(P)
```

The wrapper-unwrapping matters: a plain `::Array` vs `::AbstractArray` split is not robust
because `tchunks` and the transform pass views.

The lever that made most of the solver work with **zero call-site edits** was making
`tchunks` itself backend-dispatched: the device method calls the supplied closure *once*,
with the whole arrays, which turns each `@.` inside it into a single device kernel. That
one change made `RK45.step!`, `stage!`, `interpolant!` and the generic
`make_prop!(::AbstractArray, y0)` device-correct as they stood.

### 5.3 Operators on a device

The factored operators carry an `Adapt.adapt_structure` rule, so their small factors move
to the device. Their `getindex` then becomes a host-side scalar read of device memory and
correctly fails — on a device the operator is only ever consumed by the broadcast kernel:

```julia
function _prop_factored!(::Utils.DeviceBackend, y, linop, dt)
    ω = reshape(linop.ω, :, 1, 1)
    k2 = reshape(linop.k2, :, 1, 1)
    kperp2 = reshape(linop.kperp2, 1, size(linop.kperp2)...)
    @. y *= exp(_linop_xy_dt(ω, β1, k2, kperp2, βref, subref, dt))
end
```

Broadcasting the reshaped factors rather than the struct keeps this independent of whether
a given backend adapts non-native `AbstractArray` operands inside a broadcast, and hoists
the reference-phase test out of the per-element work.

!!! note "A claim that was wrong, and how it was caught"
    This design was originally justified with the assertion that CUDA *rejects* a lazy
    operator struct in a broadcast. Hardware measurement showed no such exception — CUDA
    adapts non-native operands, and the `Adapt` rule makes the struct route work too. The
    factor form is still the right choice (portability, and hoisting the branch), but the
    stated reason was wrong and the source comments were corrected. `test_cuda.jl` now
    asserts the two routes *agree*, rather than that one fails.

### 5.4 Device error norms

`weaknorm_fused` splits by backend. The device method is a single `mapreduce` over eight
arrays with a three-tuple accumulator computing ``(\sum|y|^2, \sum|y_n|^2,
\sum|y_\text{err}|^2)`` in one pass, still without materialising the error estimate.

!!! warning "`mapreduce` over several arrays does not broadcast shapes"
    This cost a full debugging cycle and an incorrect intermediate design. Base throws
    `DimensionMismatch`; a GPU backend may quietly compute something else. Any mask or
    per-column weight must therefore be applied *after* the reduction, to the small
    partials — not folded into it — unless it has exactly the same shape as the data.

### 5.5 Transform and FFT plans

`TransFree` gained an explicit `IFT` field and a `GridVectors` mirror of the grid's
``\omega``/window vectors adapted to the transform's array type — a host `Vector` cannot be
broadcast against a device array, and making `Grid` itself device-aware would leak device
arrays into the metadata written to output files. On the host `GridVectors` aliases the
grid's own vectors: no copy, no extra memory, CPU path unchanged.

The three elementwise steps split by backend (`_to_time!`, `_apply_towin!`, `_scale_nl!`),
with the host bodies moved verbatim. The inverse transform is stored explicitly rather than
using `ldiv!`, because on a device `ldiv!` would construct an `AbstractFFTs.ScaledPlan` per
call.

Plans are built through backend-generic helpers (`Utils.plan_fft_backend` and friends)
because FFTW takes planning flags and cuFFT does not, and `loadFFTwisdom` is gated on the
backend. On a device the plan's prototype is the state array itself, so no host prototype
is allocated for the plan — though `Luna.setup` still builds one for other reasons, which
is recorded as outstanding work in §8.

### 5.6 The device-to-host boundary

`Device.HostOutput` wraps an output handler when the propagating state is a device array,
so **`Output.jl` required no edits at all**. It copies the interpolated (saved) field into
one reusable host buffer, and copies the per-step `y` only when the handler actually needs
it (`needs_host_y`) — copying `y` every step rather than every save would dominate the
runtime.

`Luna.run` refuses a device propagation with a statistics function unless
`allow_device_stats=true`, for the same reason. `Output.nostats` (the default) is free.

Because a parallel reduction and a different FFT library round differently, **CPU and GPU
results cannot be bit-identical**, and the adaptive stepper amplifies that: the two take
genuinely different step sequences. Tolerance-level agreement is the standard, and §6.2
records what was measured.

### 5.7 Deferred loading, and world age

The scan scripts are parsed and executed on a **login node with no GPU**, whose CPU faults
while precompiling CUDA.jl, and then again on a compute node. Nothing may `using CUDA` at
top level. `Luna.resolve_arraytype(:cuda)` therefore loads CUDA by UUID with `Base.require`
— no lexical reference, and it still triggers the package extension.

!!! danger "Deferred package loading has bitten three times"
    Each failure had a different surface and the same root cause. Anything executed *after*
    a runtime package load, in the same call, is running in an older world age and cannot
    see the newly-defined methods.

    1. **Extension precompilation.** Defining `Luna.device_reclaim()` in the extension
       *overwrites* the stub, which Julia rejects during precompilation. The extension
       still loaded — just recompiled in every process, once per delay point under
       `instances`, which is easy to miss. Fixed with function-valued `Ref` hooks that the
       extension *installs* from its `__init__` rather than methods it defines.
    2. **Calling the hook.** The stored function's methods are newer than the frame calling
       `device_reclaim()`, so the call itself failed. Fixed by invoking the hooks through
       `Base.invokelatest` — they run once per simulation, never per step, so the dynamic
       dispatch costs nothing.
    3. **Everything downstream.** The same applies to every device kernel reached from the
       call that triggered the load. Callers that resolve an array type and then propagate
       within one function must re-enter through `Base.invokelatest`.

    **Tests cannot catch this**: they `import CUDA` at the top of the file, so their world
    age is already current. Only the real entry point exercises it, which is why
    `test_cuda.jl` runs the lazy-`:cuda` case in a fresh subprocess.

### 5.8 Slurm

`Scans.SlurmExec` gained a `gres` field (emitted as `#SBATCH --gres=...`, validated
against shell injection) and per-instance `CUDA_VISIBLE_DEVICES` pinning inside the
instances subshell, so each respawning one-shot process owns exactly one GPU. `procs>0`
with a GPU is rejected: workers inherit the environment and would all target device 0.

## 6. Accuracy

### 6.1 Host

`test/test_perf_bitident.jl` compares every fast path against the code it replaced with
`isequal`. At production scale, the merged stack recomputed delay points of a running
campaign and reproduced the stored reference with `max rel diff = 0.0` on all datasets, at
a central point and at a far-wing point.

### 6.2 Device

| test | measured |
|---|---|
| Kerr, end-to-end vs host (small grid) | ``\text{rel }L_2 = 2.4\times10^{-11}`` |
| Kerr+Raman, end-to-end vs host | ``2.6\times10^{-12}`` |
| production delay point vs bit-identical CPU reference | ``\le 7.4\times10^{-4}`` of the trace maximum |

The production figure needs its context. It is *larger* than the small-grid figures because
the adaptive stepper takes different steps once anything differs in the last bits, and it
is nonetheless an order of magnitude **inside** the discretisation error both runs already
carry: these runs use `weaknorm`, which measures the error relative to the whole field,
which the pump beamlets dominate. The weak four-wave-mixing signal's own relative error is
therefore ``\sim\text{rtol}\times\|\text{pump}\|/\|\text{signal}\|``.

!!! note "Normalise to the trace peak, not the point's own peak"
    Raw per-point relative differences were badly misleading. A far-wing delay point of a
    scan carries a signal ``1.6\times10^{-5}`` of the scan maximum, so a difference that is
    ``10^{-5}`` of the assembled data appears as **0.63** against that point's own peak.
    The ranking inverts under the correct normalisation: the alarming point was
    ``9.8\times10^{-6}`` of the maximum while an innocuous-looking central point was
    ``6.1\times10^{-4}`` — the worst of the set.

### 6.3 The error norm, verified on hardware

The GPU consistently took **fewer RK45 steps** than the CPU for the same delay point (73 vs
88, 77 vs 86). Since the step controller sets ``\Delta z`` from the error estimate, a
systematically smaller estimate would mean the GPU was running at an effectively looser
`rtol` than requested — and the norm was validated host-vs-device only under JLArrays,
which is CPU-backed. The end-to-end hardware test could not catch a norm *scale* error
either, because both runs would remain accurate and only the step count would move.

Measured directly on an A40, with the stages built as a common base field plus a swept
relative perturbation so the reduction genuinely exercises the cancellation that
Dormand–Prince's error estimate involves (its `errest` coefficients sum to zero, so
identical stages give ``y_\text{err} = 0`` exactly and independent random stages would
prove nothing):

| cancellation factor | device/host ratio | rel. difference |
|---|---|---|
| ``2.3\times10^2`` | 1.000000000 | ``4.0\times10^{-13}`` |
| ``2.3\times10^4`` | 1.000000000 | ``3.4\times10^{-14}`` |
| ``2.3\times10^6`` | 1.000000000 | ``1.0\times10^{-13}`` |
| ``2.3\times10^8`` | 1.000000000 | ``2.6\times10^{-13}`` |

A ratio near 0.45 would have been needed to explain the step deficit; the norms agree to
``10^{-13}``, and — the informative part — the residual is **flat in cancellation**, so
neither FMA contraction nor the sequential-host-sum versus device-tree-reduction difference
is being amplified. The step-count difference is trajectory divergence seeded by
cuFFT-vs-FFTW at the ``10^{-15}`` level in the RHS, which is expected and unavoidable.

## 7. Performance

### 7.1 CPU

Production-like configuration: memory **46 GB → 30.6 GiB**, thread scaling **4.18×** on 8
threads. Smaller benchmark (``N=256``): 4.17 GiB / 62.9 s → 3.11 GiB / 37.7 s on 4 threads,
identical `Iω_win` checksum.

### 7.2 GPU (A40, production shape ``N_\omega=256``, ``N=768``, `rtol=1e-7`, 16 z-saves)

| | GPU (1×A40) | CPU (8 threads) |
|---|---|---|
| wall / delay point | 265–281 s | 2873–3150 s |
| of which propagation | 227–230 s | 2779–3048 s |
| RK45 steps | 73–77 | 86–88 |
| device resident | 22.5 GiB | — |
| host peak RSS | 14.2 GiB | 32.8–33.3 GiB |

**11.4×** per delay point. Two A40s per node run two points concurrently.

### 7.3 Device memory: exactly the budget

22.5 GiB / 2.25 GiB per field = **10.00 fields**: 9 RK45 registers + 1 transform buffer,
with the factored operators contributing only a vector and a small matrix. The cuFFT plan
workspace is therefore negligible at this shape — the one quantity the plan could not
predict, and which had been carried as a risk to the campaign memory budget with a
provisional "≤ 1 field" allowance.

### 7.4 H200 (measured, 2026-08-18)

Runpod H200 (143.8 GB, 700 W, driver CUDA 13.3), Julia 1.12.7, 8 host threads. Warm
propagations only — the first case in a process carries ~11 s of CUDA kernel compilation.

| case | grid | field GiB | steps | s/step | **s/step per GiB** | device/field | wall/point |
|---|---|---|---|---|---|---|---|
| `dd05` | 256×640² | 1.56 | 74 | 0.123 | 0.079 | 10.56 | ~9.9 s |
| `04` | 256×768² | 2.25 | 57 | 0.180 | 0.080 | 10.50 | 10.3 s |
| `dd20` | 256×1024² | 4.00 | 70 | 0.284 | 0.071 | 10.50 | 19.9 s |
| `100um` | 512×640² | 3.12 | 85 | 0.241 | 0.077 | 10.52 | 20.5 s |

**The bound question of §7.5 is settled: this path is bandwidth-bound on an H200.**
Cost per GiB of state is flat to **1.13×** across a 2.6× range of field size, which is
what saturation looks like; against the traffic model that is **≈2340 GB/s, 49 % of the
card's 4.8 TB/s**, a normal achieved fraction for FP64 3-D FFTs plus elementwise passes,
while using **5.8 % of its 34 TFLOPS of FP64**. The A40 was therefore FP64-bound, as §7.4
inferred from its running ~6× slower than its own bandwidth allowed.

At matched step count the H200 is **20× the A40** at the `04` shape (0.180 vs 3.68 s/step).

Two consequences:

- **Running several delay points on one card cannot help.** One point already saturates
  the memory system; concurrency would only split the same bandwidth. (Contrast the modal
  path, whose ~0.5M-element kernels leave the card idle between launches — see
  [The fixed-quadrature modal transform and the device (GPU) path](@ref) §5.)
- **The cuFFT plan workspace is ≈0.5 of a field** (10.5 resident fields, against exactly
  10.0 on an A40) — larger than on the A40 but small, and constant across all four shapes,
  so the memory budget stays predictable. `dd20` at 42 GiB would be marginal on a 48 GB
  card and is comfortable here.

Per-point overhead outside the propagation is **~1 s** (garbage collection, pool reclaim,
the delayed input, the on-device extraction). What is *not* small is **`build_setup`:
41–72 s**, host-side, once per process — negligible over a 200-point scan but the
dominant cost of a short one, and the thing to attack if campaigns ever run many short
processes. That is `ModelPNPS/GPU-TODO.md` §2.

Campaign estimates from these numbers, one card, one process: `dd05` and `04` ~45–90 min
for 200 points, `dd20` ~80 min, `100um` ~95 min for 241 points. The `04` campaign
currently takes ~9 h wall on 20 concurrent HPC points.

### 7.5 What is *not* known

The split of GPU step time between the two 3-D FFTs, the eight propagator applications, the
response and window broadcasts, the stage combines and the norm has **not** been measured.
This matters more than the individual kernel optimisations it would rank, because it also
decides a hardware question: the A40 (GA102) runs FP64 at 1/64 rate (~0.58 TFLOPS) against
the A100's 1/2 rate (~9.7 TFLOPS, **16.7×**), while memory bandwidth differs by only 2.2×
(0.70 vs 1.55 TB/s). If the FFTs dominate, an A100 is worth far more than its bandwidth
ratio suggests; if the path is bandwidth-bound, ~2.2×. This is the first item in §8.

## 8. Testing

| file | what it covers | where it runs |
|---|---|---|
| `test/test_perf_bitident.jl` | every `perf` fast path vs the code it replaced, with `isequal` | everywhere |
| `test/test_full_freespace.jl` | the 3-D free-space propagation itself | everywhere |
| `test/test_device.jl` | the device *logic* under JLArrays, with `AbstractFFTs` plan shims | everywhere |
| `test/test_cuda.jl` | cuFFT semantics, codegen, agreement, resident memory | `LUNA_TEST_CUDA=1` + a GPU |

`runtests.jl` takes substring arguments so a subset can be run during development
(`julia --project test/runtests.jl rk45 freespace`); the patterns are consumed with
`empty!(ARGS)` because `Luna.Scans` parses `ARGS` itself and would reject leftovers.

### 8.1 What JLArrays can and cannot catch

`JLArrays.JLArray` is a CPU-backed `AbstractGPUArray` enforcing the same no-scalar-indexing
contract as `CuArray`, which makes the whole device path testable on a laptop. The plan
shims in `test_device.jl` deliberately do **not** define `ldiv!`, so the generic
`inv` → `ScaledPlan` route is exercised — the route cuFFT will take.

Its blind spots are structural, and two of them have bitten:

- **A broadcast mixing host and device operands just works** under JLArrays, because it is
  CPU-backed. CUDA rejects it outright. This is how a host-resident delay-phase vector
  reached a device broadcast unnoticed.
- **A non-native `AbstractArray` operand in a device broadcast** also just works.
- **CUDA code generation** is not exercised at all: boxed closures, dynamic dispatch and
  non-isbits captures pass under JLArrays and fail on hardware.

The rule adopted in response: **assert residency structurally, on types, not by running.**
For example

```julia
wrapper(x) = Base.typename(typeof(x)).wrapper
@test wrapper(s.ωd) === wrapper(s.Eωk_g12)
```

which fails at test time on any backend, rather than only on hardware.

### 8.2 A cautionary case

An early device-vs-host discrepancy was diagnosed as a strided-view bug on CUDA and the
code was reworked around it — twice, the first rework being itself invalid (it relied on
`mapreduce` broadcasting shapes, §5.4). The original failure turned out to be a **bad test
configuration** (`rtol=1e-8` with the default `floor_rel=1e-6`), which fails identically on
the host. Two lessons went into the working practice: reproduce a suspected device bug on
the host before designing around it, and do not leave an unverified mechanism in a
docstring or commit message.

## 9. Known limitations

- The general (non-fast-path) `TransFree` is host-only: `RealGrid`, oversampled grids,
  noise fields and columnwise responses are all refused on a device.
- Plasma in `TransFree` on a device is not implemented (`PlasmaCumtrapzBatched` exists, but
  the general path it needs does not) — see the roadmap in the modal design document, §3.
- `factored=false` with a device array is refused: a materialised operator is a whole extra
  device field.
- Statistics on a device require `allow_device_stats=true` and cost a full transfer per
  step.
- CPU and GPU results are not bit-identical and cannot be.

## 10. Outstanding work

The single list for this path is §7 of `GPU-TODO.md` on this branch.

### 10.1 Done

- [x] **Skip the host FFT prototype on the device path.** `setup(::EnvGrid, ::FreeGrid)`
  built a full host array purely as an FFTW planning prototype, planned a 3-D host
  transform against it and allocated a second host array, before adapting to the device.
  When the caller supplies the state itself (`inputs === ()`, which is how the free-space
  scan drivers work) none of that is used: the state is now allocated directly on the
  device with `device_zeros`, and no host plan is built or inverted. **4.5 GiB off the
  host peak at production shapes**, plus the planning pass. Guarded by `_no_inputs`, which
  dispatches on the type rather than calling `isempty` — `inputs` may be a single
  `Fields.SpatioTemporalField`, which is not iterable.
- [x] **`needs_host_save(o)`**, the save-side twin of `needs_host_y`. `HostOutput` now
  allocates its saved-field buffer only for handlers that want one, and passes the
  interpolant through unwrapped otherwise. The default is `true`, so `MemoryOutput` and
  `HDF5Output` are unchanged by construction. A device-reducing handler gets the solver's
  own array for **every** save, including `z = 0` (the step start, reached through the
  interpolant) — which removed a whole branch from the downstream ModelPNPS extractor.
- [x] **`Output.foreach_save(f, o, y, t, dt, yfun)`**, and the output-handler contract
  documented on it. One accepted step can produce several saves or none, and every handler
  needs the same loop; `MemoryOutput` and `HDF5Output` are both refactored onto it, so the
  helper is proven rather than merely offered.
- [x] **The `ScaledPlan` pass and `_apply_towin!`, fused — for the pointwise device path.**
  A device inverse plan is an `AbstractFFTs.ScaledPlan`, so applying it ran the transform
  and *then* a separate full-array multiply by `1/N`. When every response is pointwise the
  next operation is itself a broadcast, so the transform is now applied unnormalised
  (`_ift_unscaled`) and the `1/N` folded in as a prescale on read, together with the
  temporal window. **Two whole-array passes per RHS**, an estimated ~14 % of step traffic.
  Nothing is assumed about the responses: `f(s·E)` is exactly what the normalised field
  would have produced. The host path is untouched — `ldiv!` through the forward plan is
  FFTW's fused normalisation and never had the extra pass — so bit-identity is unaffected.

### 10.2 Remaining

1. ~~**Measure the device RHS breakdown** (P0).~~ **Partly answered** by the H200 run
   (§7.4): the path is bandwidth-bound there, at ~49 % of peak, so the hardware question
   is settled — bandwidth, not FP64 rate, is what this code buys. A per-kernel breakdown
   (FFT vs propagator vs stage combines) would still be needed to rank items 2–3 below,
   and is one `CUDA.@profile` on a single RHS.
2. **The same normalisation fold for the *general* (batched) path** (P2, effort M). With a
   non-pointwise response the polarisation is accumulated across responses into a separate
   buffer, so there is no single broadcast to fold into and the `ScaledPlan` is still used.
   The route is a homogeneity-degree trait — both Kerr and the Raman envelope polarisation
   are degree 3 in `Et`, so an unnormalised inverse plus a `(1/N)^d` factor folded into
   `_scale_nl!` is exact — guarded on all responses declaring the same degree. Deliberately
   **not** done with the rest: it adds a trait to `Nonlinear`'s public surface, which the
   modal transform shares, and the benefit is one arm of one campaign. A mis-declared
   degree would be caught immediately rather than silently (the error is a factor of
   ``N^{\Delta d}``), so it is safe to add later with a device-vs-host A/B.
3. **`_apply_towin!` for the general path** (P3). Same obstruction: the window must be
   applied after all responses have accumulated, so it can only fold into the last
   response's write.

## 11. Where things live

| | |
|---|---|
| backend trait, threading, FFT planning | `src/Utils.jl` |
| device scaffolding, `HostOutput`, `resolve_arraytype` | `src/Device.jl` |
| CUDA glue (hooks installed from `__init__`) | `ext/LunaCUDAExt.jl` |
| solver registers, fused norms, propagator kernels | `src/RK45.jl` |
| `TransFree`, `FreeNorm`, the fast path | `src/NonlinearRHS.jl` |
| `FactoredFreeLinop`, element formula | `src/LinearOps.jl` |
| pointwise/batched response traits, batched Raman | `src/Nonlinear.jl` |
| `setup`, `run`, the device output boundary | `src/Luna.jl` |
| `SlurmExec` `gres` and device pinning | `src/Scans.jl` |
