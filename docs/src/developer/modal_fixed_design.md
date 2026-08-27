# The fixed-quadrature modal transform and the device (GPU) path

!!! note "Living document"
    This page records the design of the fixed-quadrature multimode nonlinear transform
    (`NonlinearRHS.TransModalFixed`), the batched nonlinear responses and the device (GPU)
    execution path introduced on the `modal-fixed` branch (built on `gpu`, which builds on
    `perf` → `slurm` → `master`), together with the measurements that motivated each
    decision, the evidence for its accuracy, and the outstanding work. It is meant to be
    kept up to date as the branch evolves; the [Roadmap and open items](@ref "8. Roadmap and open items") section at
    the end is the single list of deferred work (formerly `GPU-TODO.md`). Numbers are
    quoted with the machine and configuration they were measured on; anything not yet
    measured on GPU hardware is marked as an estimate.

    Its companion is
    [Three-dimensional free-space propagation: threading, memory and the device path](@ref),
    which covers the `TransFree` path on the same branch line. The two transforms have
    different bottlenecks — quadrature and per-node work there, array count and passes over
    a multi-GB state here — but share the backend trait, the device execution model, the
    package-extension mechanism and the deferred-loading/world-age constraints; those are
    described in §3.7 below and §5 there, and the free-space document carries the A40
    measurements that exist today.

## 1. Summary

The multimode nonlinear polarisation ``P_j(\omega)`` (see
[Multi-mode guided](@ref) for the model) was evaluated by
[`NonlinearRHS.TransModal`](@ref) with **adaptive cubature** over the fibre cross-section:
for every quadrature point the integrator asked for, the field was synthesised from the
modes, transformed to the time domain, passed through the nonlinear responses, transformed
back and projected onto the modes. That is inherently serial, does one FFT pair per
quadrature point, cannot run on a GPU, and controls its error with a single L2 number over
a pump-dominated vector.

[`NonlinearRHS.TransModalFixed`](@ref) evaluates the same integral on a **fixed** Gauss
rule ([`Modes.TransverseQuadrature`](@ref)): the whole modal field is transformed to the
time domain with one batched FFT, synthesised on all nodes with one matrix product, the
responses are applied to the whole ``(t, \text{polarisation}, \text{node})`` array
(array-level "batched" response implementations), the result is projected back with a
second matrix product and one batched FFT. Everything is a GEMM, a batched FFT plan or a
broadcast, so it is threaded on the CPU and runs unchanged on device arrays. It is the
default (`modal_integral=:fixed`); the adaptive transform is kept as the reference oracle
(`modal_integral=:adaptive`).

The headline findings, all detailed below:

- **Accuracy.** For the Kerr (and Raman) polarisation, which are polynomial in the mode
  fields, the fixed Gauss rule is *exact* to rounding at a few nodes per radial mode order
  (``5\times10^{-16}`` at ``n_r=24`` for four ``\mathrm{HE}_{1m}`` modes). For the plasma
  polarisation the convergence is spectral **provided the ionisation rate is a smooth
  function of the field** — which the default PPT rate was not: its channel-closing kinks
  limited *every* quadrature, adaptive or fixed, to slow algebraic convergence
  (``\sim 10^{-3}`` at 256–512 nodes). The smooth integral form of the PPT sum is now the
  default; with it the plasma term converges to ``\sim10^{-9}`` at ``n_r\approx 24``–32,
  the fixed rule reproduces the adaptive one to ``10^{-8}`` in the final field of a 1.5 m
  RDW propagation, and the embedded Gauss–Kronrod error estimate becomes trustworthy.
- **Speed (CPU, laptop, 8 threads).** RDW VUV example, 1.5 m: 18.6 min (adaptive, old
  rate) → 138 s (adaptive, smooth rate) → 67 s (fixed, default ``n_r=64``) → 48 s (fixed,
  ``n_r=32``), with the fused two-pass plasma kernel; 52 s / 32 s with the broadcast (device-native)
  ``z``-dependent linear operator as well. CtC-type H₂ supercontinuum (nt = 32768, 6
  modes, Raman + plasma): 72 ms → 14.5 ms per RHS. The transform scales with cores; the
  adaptive one did not.
- **GPU.** The same code runs on CUDA arrays (validated with JLArrays on the host and by
  hardware-gated tests; the first A40 measurements are pending — see the roadmap). The
  plasma path is bandwidth-bound (≈0.5 FLOP/B), so HBM bandwidth, not FP64 rate, is what a
  GPU needs for this code.

## 2. How the old code worked

### 2.1 `TransModal`

`TransModal` (`src/NonlinearRHS.jl`) holds host `Array` buffers for one spatial point at a
time: `Erω (nω × npol)`, `Er`/`Pr (nto × npol)`, `Prω`, `Prmω (nω × nmodes)`. Its call
`(t::TransModal)(nl, Eω, z)` hands `Cubature.pcubature_v` (radial-only, `full=false`) or
`Cubature.hcubature_v` (2-D, `full=true`) an integrand `pointcalc!(fval, xs, t)` over a
vector of `length(Eω)*2` real outputs (the re/im parts of the modal polarisation). For each
point of the batch `xs` the integrand did, serially (`pointcalc!` carried a
`TODO: parallelize`):

1. `Modes.to_space!`: evaluate every mode's transverse field at the point (`Exy`) into a
   shared `(nmodes × npol)` matrix `ts.Ems`, then `Erω = Emω * Ems` — a `(nω × nmodes)`
   GEMV per point;
2. `to_time!`: zero-pad to the oversampled spectral grid and inverse-FFT (one c2r plan);
3. `Et_to_Pt!`: apply the responses to the `(nto × npol)` field (Kerr, `PlasmaCumtrapz`,
   `RamanPolarField`, … — each a columnwise callable with its own internal buffers);
4. time window, forward FFT, spectral window, `norm!` (the ``-i\omega/4`` factor);
5. project: `Prmω = Prω * transpose(Ems)`, times the Jacobian (`2πr` or `r`).

Cubature then decided, from its embedded error estimate with `error_norm=Cubature.L2` and
`reltol=1e-3` (`radial_integral_rtol`), whether to refine.

Three properties of this scheme matter for what follows:

- **Cost per point** was dominated by the FFT pair and the plasma response, not by the
  synthesis: measured on the RDW VUV example (nt=4096/nto=8192, laptop) ≈130 µs per point
  (Kerr 2 µs, PPT-spline plasma 59 µs, FFT pair 60–190 µs depending on thread contention).
  Two FFTs of length nto per *point* rather than per *mode*.
- **The point count** at production tolerance was small most of the time: 31 points along
  most of the fibre and 63–511 around self-compression (``z\approx0.3``–0.4 m). We show
  in §4 that the increase near the peak was caused by the non-smoothness of the PPT rate,
  not by the physics.
- **Error control** bounded ``\|\varepsilon\|_2 \le \texttt{reltol}\,\|P\|_2`` over the
  whole output vector, which is dominated by the pump; weak spectral components far below
  the pump were formally unconstrained. In practice this fear turned out to be unfounded
  for Kerr (the nested Clenshaw–Curtis rule was exact for it, ``5\times10^{-16}`` at
  `rtol=1e-3`), and real for the plasma with the kinked rate (§4.3), whose own estimate
  was 10–1000× optimistic when the tolerance was tightened.

### 2.2 Why `pcubature` was not used for the 2-D integral

`pcubature` (p-adaptive tensor Clenshaw–Curtis) was found, in earlier work, to fail for
polar 2-D integrands: the estimate "always cancelled". The likely mechanism (consistent
with that observation; not re-verified here): its low-degree levels sample ``\theta`` at
``0, \pi/2, \pi, 3\pi/2``, where products of vector components such as ``E_xE_y`` for
``\mathrm{HE}_{11}`` x+y, or ``\mathrm{HE}_{11}\cdot\mathrm{HE}_{21}`` terms
(``\propto\sin2\theta``, ``\sin^22\theta``), vanish identically; two successive levels
then both return zero, the level-to-level error estimate is zero, and the integrator
declares convergence on a wrong value — a pure aliasing artefact. Hence `hcubature` for
`full=true`. The fixed rule is immune by
construction (§3.3): the periodic trapezoid rule is exact for trigonometric polynomials up
to degree ``n_\theta-1``, cubic products of modes with azimuthal order ``h`` have degree
``\le 4h``, and there is no convergence decision to fool.

### 2.3 The `gpu` branch before this work

The `gpu` branch had made the 3-D free-space transform (`TransFree`) array-generic:
`Utils.backend` (a `CPUBackend`/`DeviceBackend` trait on the array type),
`Utils.tchunks`/`tforeach` (host chunked/threaded loops which become one kernel on a
device), `Utils.plan_*_backend` (FFTW with Luna's flags on the host, `AbstractFFTs` on a
device), `Luna.device_zeros`, `Luna.GridVectors` (grid vectors mirrored on the device),
`Luna.HostOutput` (device-to-host copies at saves), fused device error norms in `RK45`, and
a package extension `ext/LunaCUDAExt.jl` for the few CUDA-specific hooks (device
selection, memory reclaim/query). Nothing on that branch touched the modal path, and
**plasma could not run on a device at all**: `PlasmaCumtrapz` was a columnwise response
with a sequential cumulative integral.

## 3. Design of the new code

### 3.1 Measurements that shaped the design

Before writing the transform, the following was measured on the RDW VUV example
(`examples/simple_interface/RDWemission_VUV_gradient.jl`: He pressure gradient
0.8→0 bar, 125 µm core, 4 ``\mathrm{HE}_{1m}`` modes, 275 µJ, 7.5 fs at 800 nm, PPT
plasma, nt=4096/nto=8192; Apple-silicon laptop, 8 Julia threads). Frozen fields were
taken from a short propagation at several ``z``, and the projection ``P_m(\omega)`` was
computed with each method against a reference (adaptive at `rtol=1e-10`, or a 2048-node
Gauss rule).

| Quantity | Measured |
|---|---|
| Cubature points per RHS, production `rtol=1e-3` | 31 along most of the fibre; 63–511 around self-compression |
| Time per adaptive RHS | 4 ms (31 pts) … 18 ms (127 pts); ≈130 µs per point |
| Kerr projection error of the adaptive scheme at `rtol=1e-3` | ``5\times10^{-16}`` (nested Clenshaw–Curtis is exact for the quartic) |
| Kerr projection on a fixed Gauss–Legendre radial rule | machine precision at ``n_r=24`` (4 modes); overlap tensor ``\Gamma_{mnpq}`` to ``4\times10^{-16}`` at ``n_r=32`` |
| Plasma, ADK rate, fixed rule | ``10^{-9}`` at ``n_r=24``; ``10^{-11}`` restricted to the 100–140 nm band |
| Plasma, cached PPT rate (then the default), fixed rule | algebraic ``\sim n_r^{-1.5}``, non-monotone: ``10^{-2}`` @64, ``3\times10^{-3}`` @128, ``10^{-3}`` @256–512, ``5\times10^{-5}`` @1024 |
| Plasma, PPT, adaptive | ``5\times10^{-4}`` with 511 pts at `rtol=1e-3`; 8k–65k pts for ``\sim10^{-5}``; self-estimate 10–1000× optimistic when tightened |
| Plasma, PPT with `sum_integral=true` (smooth channel sum), fixed rule | ``3\times10^{-10}`` at ``n_r=24`` |
| Prototype fixed-rule RHS (host, plasma columns threaded), vs adaptive 17.7 ms at the same field | ``n_r=32``: 2.6 ms serial / 0.87 ms 8 threads; 128: 10.6 / 2.1 ms; 512: 36.6 / 6.7 ms |

Two conclusions drove everything else. First, **one fixed collocation grid handles Kerr and
plasma**: the grid is already exact for Kerr, and at ``M\lesssim10`` modes an exact overlap
tensor ``\Gamma_{mnpq}`` for Kerr (the "hybrid" route considered in the pre-implementation
analysis) would cost the same FLOPs as the grid GEMMs, both dwarfed by the FFTs, while
breaking response encapsulation. Second, **the plasma accuracy limit was the PPT rate's
non-smoothness, for the old and the new scheme alike** (§4.3); it was verified not to be
the spline (tables with ``2^{14}``, ``2^{16}``, ``2^{18}`` knots gave identical curves) but
the lower limit ``n_0=\lceil v\rceil`` of the multiphoton sum in
[`Ionisation.IonRatePPT`](@ref), which jumps whenever the ponderomotive shift closes a
channel.

### 3.2 Alternatives considered and rejected

- **Exact overlap tensor for Kerr/Raman + grid for plasma** ("hybrid"): no gain at
  ``M\lesssim10``, per-response tensor code, breaks the response abstraction. Revisit only
  if ``M\gtrsim50`` ever matters. (Raman by mode pairs is the one place the idea pays, for a
  different reason: see the roadmap.)
- **Amortised adaptivity** (re-mesh every ``k`` steps, freeze, reuse): a fixed operator is
  better for long propagation and for a GPU, and the sizing turned out small enough that
  deliberate over-resolution is cheaper and simpler.
- **Threading `pointcalc!`**: `Cubature`'s vectorised interface does hand over batches, but
  the buffers (`ToSpace.Ems`, the transform's per-point arrays, the responses' internal
  state) are shared, the p-adaptive levels add 2, 4, 8, 16 points per dependent round, and
  none of it runs on a device. Perhaps 3–4× on 8 threads for the 1-D case; not pursued.
- **Wiring the embedded error estimate into automatic refinement**: switching rules
  step-to-step injects broadband, non-cancelling perturbations; a fixed rule that is
  slightly wrong behaves better over ``10^4`` steps than one that changes. The estimate is
  an *alarm, not a controller*.
- **Re-tuning `nt`/RK45 tolerance** as part of this work: deliberately not, until the
  spatial error was fixed and measurable — which it now is.

### 3.3 The quadrature rule

[`Modes.TransverseQuadrature`](@ref) / [`Modes.transverse_quadrature`](@ref):

- **Polar, radial only (`full=false`)** — for mode sets whose intensity and component
  products are azimuthally symmetric, i.e. any set of ``\mathrm{HE}_{1m}`` modes in any
  polarisation (their transverse fields are ``f(r)\hat{x}``, ``f(r)\hat{y}``, so
  ``|E|^2``, ``E_xE_y``, … depend on ``r`` only). ``n_r`` Gauss–Legendre nodes
  ``\xi_i\in(0,1)`` on ``r=a\xi``, weights ``w_i = a^2\,\xi_i\,w^{GL}_i\cdot 2\pi`` (the
  Jacobian ``r`` and the azimuthal ``2\pi`` folded in). With `kronrod=true`, ``n_r`` is
  rounded up to ``2n+1`` and the rule is the ``(2n+1)``-point Kronrod extension of the
  ``n``-point Gauss rule (`QuadGK.kronrod`), whose Gauss subset provides the *embedded
  coarse weights* used for the error estimate. Gauss nodes are strictly interior, so the
  boundary zeroing in `to_space!` never triggers and ``r=0`` (where HE fields are maximal)
  is never sampled exactly — irrelevant for a Gauss rule.
- **Polar, full (`full=true`)** — needed as soon as a mode with azimuthal dependence is
  present (``\mathrm{HE}_{nm}``, ``n\ge2``, TE, TM); `Interface.needfull` decides.
  ``n_\theta`` equispaced nodes ``\theta_j = (j-\tfrac12)\,2\pi/n_\theta`` with weight
  ``2\pi/n_\theta``: the periodic trapezoid rule, which is a Gauss rule on the circle,
  exact for trigonometric polynomials of degree ``\le n_\theta-1``. Cubic products of modes
  with azimuthal order up to ``h_{\max}`` (`Modes.azimuthal_order`: ``n-1`` for
  ``\mathrm{HE}_{nm}``, 1 for TE/TM) have degree ``\le 4h_{\max}``, so the constructor
  warns if ``n_\theta < 4h_{\max}+1``. The default ``n_\theta=16`` is exact for
  ``h_{\max}\le3``. The coarse rule takes every other ``\theta`` node.
- **Cartesian (`RectMode`)** — Gauss–Legendre in both directions.

The nodes and weights are computed once; the fine and coarse projection matrices are
built from them (§3.4).

### 3.4 Mode matrices, synthesis and projection

`Modes.mode_matrix(ms, indices, quad; z)` evaluates the normalised transverse fields
`Exy` of all modes at all nodes: an array `Ems (nmodes × npol × npts)`. From it the
transform builds (`_mode_matrices`)

```math
S = \mathrm{reshape}(E,\; M\times n_{\rm pol}n_p),\qquad
W_p = \big(\mathrm{reshape}(E\,w,\; M\times n_{\rm pol}n_p)\big)^{\!T},\qquad
W_c = \text{same with the coarse weights},
```

with column ``p + (i-1)n_{\rm pol}`` of ``S`` (row of ``W_p``) being polarisation
component ``p`` at node ``i``, so that a `(nto, npol, npts)` reshape of the real-space
arrays is free and `view(Et, :, :, i)` is a contiguous `(nto × npol)` matrix — exactly what
every existing columnwise response takes. ``S`` and ``W_p`` are stored in the field's
element type (`Float64`/`ComplexF64`) so `mul!` stays dgemm/zgemm on BLAS and cuBLAS.

Three mode traits control ``z`` dependence (`Modes.zconstant`, `Modes.scale_invariant`,
`Modes.azimuthal_order`, delegated through `Modes.@delegated`/`@arbitrary` and the
antiresonant wrappers): for `MarcatiliMode` with a numeric radius the profile is
``z``-independent; for a tapered Marcatili mode ``e(r;a) = f(r/a)``, ``N\propto a^2``, so
`update_matrices!` rescales ``S(z) = S(0)\,a(0)/a(z)``, ``W_p(z) = W_p(0)\,a(z)/a(0)``
exactly (two scalars per RHS instead of ~``8000`` Bessel evaluations); for other
``z``-dependent modes the fields are re-evaluated on the host and uploaded.

### 3.5 One evaluation of `TransModalFixed`

For `(t::TransModalFixed)(nl, Eω, z)` with `Eω (nω × M)`:

1. `update_matrices!(t, z)`; `t.density = densityfun(z)`.
2. `to_time!(Emt, Eω, Emωo, IFT)`: zero-pad to the oversampled spectral grid
   (`copy_scale!`, backend-dispatched: host loops as before, view broadcasts on a device)
   and one batched inverse FFT along dim 1 → `Emt (nto × M)`. With a modified-shot-noise
   field, `Emt_nl = Emt + Emt_noise` (the noise is transformed to the time domain **once**
   at construction — exact, since everything is linear — and the propagating field is
   never contaminated).
3. `mul!(Et, Emt, S)`: synthesis on all nodes, `Et (nto × npol·npts)`.
4. `apply_responses!(Pt, Et, resp_eval, density, npol)`: for scalar fields via
   `Et_to_Pt_ordered!` on the `(nto, 1, npts)` array (pointwise responses as one broadcast,
   batched ones as one array-level call, legacy ones column by column on the host); for
   vector fields the array-level `(nto, npol, npts)` methods where `Nonlinear.batched(resp,
   npol)` holds, otherwise the serial per-node call. Gas mixtures (a tuple of response
   tuples and a density vector) are handled by both paths.
5. `mul!(Pmt, Pt, Wp)`: projection with the weights folded in, `Pmt (nto × M)`.
6. `Pmt .*= towin`; one batched forward FFT; `to_freq!`; `nl .*= ωwin`; `norm!(nl)`. All
   ω-diagonal factors act on the small `(nω × M)` array (they commute with the spatial
   integral; the old code applied them per point).

Buffers: `Emωo, Pmωo (nωo × M)`, `Emt, Pmt (nto × M)`, `Et, Pt (nto × npol·npts)`, `err
(nω × M)`, `Et_scratch (nt × M)` (handed to `Luna.run` as its window scratch through
`NonlinearRHS.scratch`, so `run` never does the destructive c2r `FT \ Eω` on a device), plus
whatever the batched responses allocate lazily on the field's array type.

`integral_error!(t)` fills `t.err` with ``P_{\rm coarse}-P_{\rm fine}`` of the *last*
evaluation from the real-space polarisation still held in `Pt` (one extra GEMM against
``W_c - W_p`` and ``M`` FFTs), on demand — the statistics call it, not the RHS.

### 3.6 Batched responses

The transform hands the responses whole arrays. Two traits describe what a response can do
([`Nonlinear.pointwise`](@ref), [`Nonlinear.batched`](@ref)); `Nonlinear.batched_response`
converts a columnwise response into its array-level equivalent where one exists
(the transform keeps the originals in `resp` for `show`/`Stats` and evaluates `resp_eval`):

| Response | Scalar field | Vector field | Device |
|---|---|---|---|
| `Kerr_field`, `Kerr_env` (`KerrField`/`KerrEnv`) | pointwise broadcast | array-level (`batched(K, 2)`), same arithmetic as `KerrVector!` | yes |
| `PlasmaCumtrapz` → [`Nonlinear.PlasmaCumtrapzBatched`](@ref) | array-level | array-level | yes |
| `RamanPolarField` → [`Nonlinear.RamanPolarFieldBatched`](@ref) | array-level | – (vector Raman not implemented anywhere) | yes |
| `RamanPolarEnv` → [`Nonlinear.RamanPolarEnvBatched`](@ref) | array-level | – | yes |
| `Kerr_field_nothg`, `Kerr_env_thg`, anything else | columnwise, serial | columnwise, serial | no (clear error) |

**Plasma.** `PlasmaCumtrapzBatched` computes, for all columns, the same chain as the
columnwise `PlasmaCumtrapz`: rate ``W(|E|)`` → cumulative integral → ionised fraction
``\rho/\rho_{nt} = 1-e^{-\int W}`` (plus `preionfrac`) → phase ``\propto \rho E`` →
cumulative integral (current) → ionisation-loss term → cumulative integral →
``\rho_{\rm gas}P``. On the host each column runs the existing 1-D kernels on views into
full-size buffers, threaded over columns with `Utils.tforeach` (the element-count
threading gate would otherwise disable threading at ``8192\times65``, although each column
is ~60 µs of work) — bit-identical to the columnwise response. On a device everything is
whole-array broadcasts and the cumulative integrals are prefix sums:
[`Luna.Maths.cumtrapz_scan!`](@ref) → `Maths.scan!`, a portable Hillis–Steele doubling scan on
views (any array type; ``2\lceil\log_2 n_{to}\rceil`` broadcasts) with a native
`CUDA.cumsum!` method added by the CUDA extension (one launch, no scratch buffer). Results
agree with the sequential rule to rounding.

**Ionisation rate on a device.** `IonRateADK` is isbits; the cached PPT rate
[`Ionisation.IonRatePPTAccel`](@ref) is a cubic spline in ``\log|E|`` on uniform knots.
`Maths.spline_eval` is the unchecked evaluation (the interpolated `throw` must be
*compiled out*, not merely disabled — a runtime `Bool` branch still fails GPU
compilation); `Adapt.@adapt_structure` moves the coefficient tables to the device;
`Ionisation.device_ionrate` refuses `FastFinder`-backed (non-uniform) splines and direct
`IonRatePPT`. The rate short-circuits to zero below the table's lower field, which most
samples in the pulse wings hit.

**Raman.** `RamanPolarFieldBatched` (real field, `thg` true/false) and
`RamanPolarEnvBatched` do the doubled-grid convolution of the columnwise versions with one
batched FFT pair over all columns and the density-dependent response updated once per call;
the `thg=false` analytic signal is a batched complex FFT pair. Without the real-field
version, Raman gases (H₂, N₂, SF₆, air) would fall to the *serial* columnwise fallback over
all nodes on the new default — slower than the old adaptive path — and could not run on a
device.

### 3.7 Device execution

The rules of the `gpu` branch are kept: **no CUDA dependency in Luna core**. Device code
is ordinary broadcasts, `mapreduce`s, `mul!` and `AbstractFFTs` plans, dispatched on
`Utils.backend`; the extension (`ext/LunaCUDAExt.jl`) provides only what cannot be
expressed generically — device selection, memory reclaim/query, `synchronize` — as
function-valued hooks installed from `__init__` (a method with an identical signature
cannot be *overwritten* from an extension without breaking its precompilation), plus one
genuinely CUDA-specific hot-loop method: `Maths.scan!` on `CuArray` → `Base.cumsum!`.

`Luna.setup(...; arraytype)` allocates the state and all transform buffers with
`Luna.device_zeros`, plans the state FFT on the device (`Utils.plan_rfft_backend`), builds
`norm_modal(grid; arraytype)` with the frequency axis mirrored, and adapts the input field.
`Luna.run` uploads a constant host linear operator once. A ``z``-dependent one is, for
`Capillary.MarcatiliMode` collections built by the standard constructors (tapers, pressure
gradients, both), a `Capillary.MarcatiliLinop`: ``z`` enters the Marcatili effective index
only through the scalars ``a(z)`` and ``ρ(z)`` (``ε_{\rm core} = 1+γ(ω)ρ(z)``, recognisable
because `Capillary.gradient` returns a `GradientCoreIndex` and the constructors use
`ConstIndex` for constant fills and claddings), so ``γ(ω)``, the cladding term ``v_n(ω)``
and ``u_{nm}`` are precomputed once and each stage is one broadcast over ``n_ω\times M``
(the same scalar kernel `_neff_scalar` as `Modes.neff`, hence bit-identical), evaluated
directly on the device (`RK45.device_capable`; the arrays move to the device on first
use) or on the host (~40 µs instead of ~800 µs per stage). Anything else — wrapped modes,
user-supplied ``z``-dependent index closures, other mode types — keeps the generic host
closure, which `RK45.make_prop!` evaluates into a host buffer and uploads per stage. Saved
fields come back through `Luna.HostOutput`.

**Statistics on a device run.** `Stats.collect_stats` returns a `StatsCollector` which
accepts the *device* array itself (`Luna.device_stats`). The analytic field is computed on
the device (`plan_analytic` device methods), and every default statistic is
**device-capable** (`Stats.device_capable`): `ω0`, `energy`, `energy_λ` (weighted
reductions; the `RealGrid` energy functional's Simpson weights are reconstructed and
verified against `Fields.energyfuncs` at construction), `peakpower`, `peakintensity` (the
on-axis field is one small GEMM against the cached on-axis mode matrix, `OnAxisModes`),
`fwhm_t` (|E|² formed on the device and copied down for the host crossing search — the only
remaining per-step copy, nt×M reals), `fwhm_r` (now, on both backends, from the ``M×M`` mode
coherency matrix ``G = EᴴE`` — one GEMM — with the half-maximum root found on the host by
evaluating only the mode fields, instead of re-synthesising the spectrum at every trial
radius), `electrondensity` (on-axis field GEMM, the device ionisation rate of the
transform's batched plasma response, one reduction for the final integral) and
`mode_reconstruction_error` (`wants_state`: the transform on the device state, the on-axis
reference with its own batched responses on the device, all norms as device reductions).
Field-free statistics (density, pressure, ZDW, …) are wrapped in `FieldFree`. Only if a
user-supplied (host-only) statistic is present does the collector also keep a host copy
of the state. `Output.PeriodicStats(statsfun, k)` (`prop_capillary(...; stats_period=k)`)
evaluates the statistics only every ``k``-th accepted step. `prop_capillary` passes
`allow_device_stats=true` for the modal case.

**Lazy loading and world age.** `arraytype=:cuda` (`Luna.resolve_arraytype`) loads
CUDA.jl with `Base.require` so that a scan script can request a GPU without importing
CUDA at top level (the DMOG login node faults while precompiling CUDA.jl). Methods defined
by a package loaded *during* a call are invisible to that frame (Julia's world-age rule):
`prop_capillary`/`prop_capillary_args` therefore resolve a lazy symbol
(`Luna.lazy_arraytype`) and re-enter through `Base.invokelatest` with the concrete type.
This was a real bug found while planning: the intended scan use case (login node submits,
compute node runs the closure) failed with `MethodError … too new`; it is now tested by a
fresh subprocess without `import CUDA` in `test/test_cuda.jl`. Users calling
`prop_capillary_args` + `Luna.run` inside one function must `invokelatest(Luna.run, …)`
themselves.

### 3.8 Interface, defaults and physics change

`prop_capillary` keywords: `modal_integral=:fixed|:adaptive` (default `:fixed`), `nr=64`,
`nθ=16`, `arraytype=Array|:cpu|:cuda|<type>`, `stats_period=1`, and
`stats_kwargs=Dict(:mode_error=>true, :error_window=>(λmin, λmax))`. The Kronrod rule is
used whenever the mode-error statistics are on (`kronrod = mode_error`).

**The PPT rate default changed** (commit `aa07fdf`): [`Ionisation.IonRatePPT`](@ref) now
replaces the sum over multiphoton orders by its integral form (`sum_integral=true`)
by default. This is a physics-visible change and was decided on the evidence in §4.3–4.4:
the literal channel sum has a hard step at every channel closing, a crude caricature of a
real effect that no quadrature can resolve at an affordable cost, and in the regimes
where the two rates differ appreciably (multiphoton, ``\gamma\gtrsim1``) the absolute rate
is negligible for the ionisation of the runs studied. `PPT_options=Dict(:sum_integral=>false)`
restores the literal sum, and then ``n_r\ge256`` is needed. The PPT cache is keyed on the
effective settings, so a changed default can never load a table computed with the old one.

### 3.9 The general (`RealGrid`) `TransFree` path

`TransFree` has two routes. The **fast path** transforms the solver's state array directly,
skipping the oversampled buffers; it is what every 3-D envelope run takes and what the
device support of §2.3 was built for. The **general path** goes through `Eωo`/`Pto`, and
until this work it was a serial scalar loop per stage, host only.

A **field-resolved (`RealGrid`) free-space propagation can only ever take the general
path**, for a reason that is not about performance: the inverse of a real-to-complex plan
*destroys its input*, and on the fast path that input is the solver's own state. (Measured:
FFTW's `p \ X` copies and so survives it, but `ldiv!(y, p, X)` and `mul!(y, inv(p), X)` do
not, and cuFFT's C2R is documented as destructive. `TransModalFixed` avoids the same trap
by always inverting into its own scratch — §3.5.) So the general path had to become as
capable as the fast one before a 3-D field-resolved run could be threaded, let alone run on
a device.

It is now written in the same backend-dispatched kernels as the fast path — `copy_scale!`,
`pointwise_Pt!`/`Et_to_Pt_ordered!`, `_apply_towin!`, `_scale_nl!` — reading the grid
vectors from the transform's `gv` mirror rather than the host grid. Three consequences:

- **Host**: threaded, including `copy_scale!`, which moves two field-sizes per RHS.
- **Device**: allowed whenever every response is pointwise or batched and there is no noise
  field, and refused at construction otherwise rather than deep inside the first RHS. On a
  device (only) the responses go through `Nonlinear.batched_responses` as
  `TransModalFixed` does, so a columnwise response with a batched equivalent is substituted
  silently; doing that on the host too would move existing host results at the ~1e-15 level
  for no gain.
- **Memory**: `Pωo` always aliases `Eωo` (the inverse transform consumes it and nothing
  reads it again), and `Pto` aliases `Eto` when every response is pointwise, exactly as the
  fast path does — two field-sized buffers instead of four. At 3-D field-resolved production
  shapes (``n_\omega=513``, ``768^2``) that is 18 GiB not spent.

A new `Et_win` buffer holds the coarse-grid time-domain field when the grid is oversampled,
so `scratch` never returns `nothing`: `Luna.run` uses it for the apodisation windows instead
of allocating `FT \ Eω`, and `Luna.setup`'s real free-space method plans the state rfft
against it — a real-to-complex plan needs a real prototype, which the (complex) state is
not. `Grid.RealGrid` also gained an opt-in `ffac` keyword (default 6, unchanged): the fine
grid is sampled at `ffac`×``f_{\max}``, and a response that generates no third harmonic
needs only 4, which at typical shapes removes the oversampling entirely. It changes ``δω``
and the realised time window as well, so it is for convergence-checked use only.

The regression gate for all of this is the existing `test/test_perf_bitident.jl` "TransFree
fast path" testset, which asserts general == fast **bit for bit** on an `EnvGrid`.

**Measured on an H200 (139.8 GiB, CUDA 13.3), 2026-08-26.** Host-versus-device agreement on
a `RealGrid` free-space propagation is ``7\times10^{-15}`` for the batched no-THG response
and ``9\times10^{-15}`` for the pointwise `KerrField` (`test_cuda.jl`, "RealGrid free-space
end to end"); through ModelPNPS's 3-D TG-FROG setup and extraction it is
``2\times10^{-13}``. At that campaign's production shape (``n_\omega = 513``, ``768^2``,
fine grid 2048) a delay point costs **6.0× the envelope per step** at identical step counts
— 61.1 s against 10.3 s for 57 steps each. The extra is this path's, and specifically the
no-THG response's: an 18 GiB complex analytic-signal buffer and two 1-D FFT passes over it
per RHS, on top of the oversampled `Eωo`/`Eto` the general path needs anyway. A laptop
gives ~3× for the same comparison, because there the fixed per-point costs dilute it; the
per-step figure is the one to plan with.

The resident total is 96.9 GiB at that shape, against a per-buffer model that predicts 96.9
(`ModelPNPS.memory_budget`). Note that the response's analytic buffer is allocated lazily,
on the **first RHS** rather than at setup — so a card with room after setup can still fail
on the first step.

## 4. Accuracy

### 4.1 Kerr and Raman: exact by construction

The Kerr polarisation is cubic in the field, so its projection ``\int e_m\,(e\cdot e)\,e\,dA``
is a quartic in the mode fields; the transverse profiles of Marcatili modes are Bessel
functions of ``r/a`` (times ``\cos h\theta``/``\sin h\theta``), analytic in ``r``. An
``n``-node Gauss–Legendre rule integrates the quartic to rounding once ``n`` exceeds
roughly ``\pi\times`` the highest radial order plus a few nodes; measured: ``5\times10^{-16}``
at ``n_r=24`` for four ``\mathrm{HE}_{1m}`` modes, and the Kerr transform at ``n_r=17``
matches the adaptive cubature to ``10^{-15}`` in the tests. Raman is cubic too (``E\,(h*E^2)``),
convolution being linear, and the same holds: the fixed rule at ``n_r=17`` matches the
adaptive cubature to rounding even at a loose adaptive tolerance. In ``\theta`` the
periodic trapezoid rule is exact below the aliasing bound (§3.3), so mixed
``\mathrm{HE}_{11}/\mathrm{HE}_{21}/\mathrm{TE}_{01}/\mathrm{TM}_{01}`` sets are exact
at ``n_\theta=16`` (tested: orthonormality ``\sum_i w_i e_{mi}e_{ni} = \delta_{mn}`` to
``10^{-12}``, full 2-D Kerr transform vs a high-``n_r`` reference and `hcubature` at
``10^{-10}``).

A symmetric fixed rule yields an effective overlap tensor
``\Gamma^{\rm eff}_{mnpq}=\sum_i w_i e_{mi}e_{ni}e_{pi}e_{qi}`` that is symmetric under
index permutation, so the spatial part of the Kerr term's energy conservation is exact
independent of ``n_r``. **Measured** (`test/manual/modal_fixed_conservation.jl`, RDW-type
Kerr-only loss-free case): the frozen-field energy derivative of the Kerr term
``|D|/E_{\rm tot}`` is ``10^{-7}``–``10^{-9}`` per metre and *identical* to three digits
for the fixed rule at ``n_r=17/33/65`` **and** for the adaptive cubature at
`rtol=1e-3`/`1e-6`, and the energy drift over 20 cm is the same ``9.6\times10^{-8}`` for
both. So for Kerr the spatial quadrature is not the limiting factor for either method
(the adaptive nested Clenshaw–Curtis rule is exact for the quartic too, as noted above);
the residual is the time-domain part — apodisation windows and the aliasing of the
``\partial_tE\cdot E^3`` product on the oversampled grid — common to both. The
"adaptive mesh breaks conservation" concern of the pre-implementation analysis therefore
does not apply to Kerr; it would apply to the plasma term, where the adaptive rule is not
exact.

### 4.2 Plasma: spectral convergence for a smooth rate

The plasma polarisation is not polynomial in the field, so the fixed rule is not exact;
its convergence is spectral for an analytic integrand and algebraic otherwise. The frozen
field used for the convergence tables was the RDW VUV field at ``z=0.4`` m — at the
self-compression point, i.e. where the old code needed most points. With the ADK rate
the plasma term converged to ``10^{-9}`` at ``n_r=24`` (``10^{-11}`` in the 100–140 nm
band); with the smooth PPT rate to ``3\times10^{-10}`` at ``n_r=24``. The number of nodes
needed is set by the transverse smoothness of the integrand — essentially by the highest
radial mode order — not by the intensity: at the peak the field is still a superposition
of the same four modes and the plasma a stronger but still smooth function of it. Given a
smooth rate, the *adaptive* code also converged with 31 points everywhere, peak included
(that is the 138 s run in §5.1).

**Azimuthal convergence with plasma** (`test/manual/modal_fixed_ntheta.jl`; frozen field
after 5 mm of a strongly θ-dependent case — HE₁₁ 90 µJ + HE₂₁ 50 µJ + TM₀₁ 40 µJ, 10 fs
at 800 nm, 1 bar Ar, peak field ``2.8\times10^{10}`` V/m, peak electron density
``3.9\times10^{21}\,{\rm m^{-3}}``; errors relative to ``n_\theta=128``, on the
θ-dependent mode columns): Kerr ``1.6\times10^{-1}`` at ``n_\theta=4`` (aliased: degree 4
= ``n_\theta``) and ``10^{-15}`` from ``n_\theta=6`` on (bound ``4h_{\max}+1 = 5``);
plasma spectral — ``8\times10^{-1}`` @4, ``1.3\times10^{-1}`` @6, ``3\times10^{-2}``
@8, ``2.4\times10^{-4}`` @12, ``\mathbf{8.4\times10^{-7}}`` @16 (the default),
``4\times10^{-10}`` @24, ``10^{-13}`` @32, rounding at 48. Radially (``n_\theta=32``,
vs ``n_r=256``): plasma ``1.3\times10^{-8}`` @16, ``5\times10^{-13}`` @24,
``10^{-14}`` @32. So the default ``n_\theta=16`` gives ~``10^{-6}`` on the plasma term
of a strongly vectorial field; ``n_\theta=24`` gives ``10^{-9}``. (A circularly polarised
HE₁₁ input alone has a θ-independent intensity and is exact at any ``n_\theta`` — a
useless test, tried first.)

### 4.3 The PPT channel-closing kinks

With the literal channel sum, the frozen-field convergence of the plasma term was
algebraic (``\sim n_r^{-1.5}``) and non-monotone for the fixed rule (``10^{-2}`` @64,
``3\times10^{-3}`` @128, ``10^{-3}`` @256–512, ``5\times10^{-5}`` @1024) *and* for the
adaptive one (``5\times10^{-4}`` with 511 points at `rtol=1e-3`; 8k–65k points for
``\sim10^{-5}``; its own error estimate 10–1000× optimistic when tightened). Increasing the
spline table from ``2^{14}`` to ``2^{18}`` knots changed nothing; replacing the sum by its
integral (`sum_integral=true`) restored spectral convergence.

*The physics of the sum.* The PPT rate has the structure
``W = A(E,\gamma)\sum_{n\ge n_0} e^{-\alpha(n-v)}\,\varphi_m\!\big(\sqrt{\beta(n-v)}\big)``
with ``v=(I_p+U_p)/\hbar\omega`` the ponderomotively shifted threshold in photon units,
``n_0=\lceil v\rceil``, and ``\alpha(\gamma)``, ``\beta(\gamma)`` functions of the
Keldysh parameter: each term is the partial rate of the ``n``-photon (ATI) channel, and
``\varphi_m(\sqrt{\beta x})\propto\sqrt{x}`` near threshold is the Wigner threshold law
of a channel that has just opened. As the intensity rises ``U_p`` grows, ``v`` crosses an
integer and the lowest channel closes: the rate stays continuous (``\varphi_m(0)=0``) but
has a **square-root cusp** there. Gauss quadrature of an interior ``|x-c|^{1/2}``
singularity converges as ``N^{-1.5}`` — exactly the exponent measured above, which
confirms the diagnosis. `sum_integral=true` replaces the sum by ``\int_0^\infty f(x)\,dx``
(closed form ``\sqrt{\pi}\,m!\,\beta^m/[2(\alpha+\beta)^{m+1}]\sqrt{\beta/\alpha}``),
which by Euler–Maclaurin is the sum *averaged over the channel phase* (the fractional part
of ``v``): not a biased approximation but the phase-averaged rate. In the tunnelling
regime (``\gamma\lesssim0.5``, ``\alpha\approx\tfrac43\gamma^3\ll1``) ten or more of the
``\sim70`` open channels contribute and the two forms agree to a fraction of a per cent;
near ``\gamma\approx1``–2 one or two channels dominate and the literal sum oscillates
about the integral with the channel phase. Measured (He, 800 nm): 0.1 % at
``10^{11}`` V/m (``\gamma\approx0.4``), 7 % at ``5\times10^{10}`` (``\gamma\approx0.8``),
20 % at ``2\times10^{10}`` (``\gamma\approx2``) — where the absolute rate is
``10^{4}``–``10^{11}\,{\rm s^{-1}}`` against ``10^{14}`` at the peak, i.e. negligible
ionisation in these runs.

*Physical reality.* Channel closings exist, but PPT models them crudely (monochromatic
field, purely ponderomotive shift, no resonances or Coulomb threshold effects), and for the
pulses Luna is used for the threshold is smeared by the pulse bandwidth: ``\delta v\approx
(I_p/\hbar\omega)(\Delta\omega/\omega)\approx 2`` photons for a 7 fs pulse in He at
800 nm, ``\approx1`` for 10 fs in Ar — the discrete structure is washed out over more than
a channel spacing before any intensity averaging over the beam. For few-cycle pulses the
phase-averaged form is therefore the more physical one; only for long, narrowband pulses at
``\gamma\gtrsim1`` could the channel structure be observable, and there PPT's own accuracy
(deviations of order unity from TDSE/experiment in the multiphoton–tunnelling transition)
dwarfs the ±10–20 % channel-phase oscillation. What is lost by the new default is
bit-for-bit agreement with the published channel-sum formula and with codes that implement
it (rates ~10–20 % apart of alternating sign near ``\gamma\approx1``–2; peak electron
densities in weakly ionising cases shifted by ~10 %, final fields much less, §4.4); what is
gained is a rate the quadrature can integrate spectrally, and the removal of a numerical
scatter (1–2 % between rules) larger than the physical difference between the two forms.
The physically best middle ground would be a *soft* threshold (the sum convolved with the
pulse spectrum), kept in the roadmap in case a case ever needs it.

### 4.4 End-to-end validation

Scripts: `test/manual/modal_fixed_convergence.jl` (frozen-field tables) and
`test/manual/modal_fixed_rdw.jl` (end-to-end, RDW VUV and DUV examples).

**RDW VUV example, 1.5 m, ≈4400 steps, laptop, 8 threads.**

| Run | Wall | Final field vs "fixed 512, channel-sum" (rel. L2 of ``|E|^2``) |
|---|---|---|
| adaptive, channel-sum PPT (the original code and default) | 18.6 min | 2.4 % |
| fixed ``n_r``=64 / 128 / 256 / 512, channel-sum PPT | 1.6 / 2.3 / 3.8 / 6.4 min | 2.6 % / 2.6 % / 1.2 % / — |
| fixed ``n_r``=32, smooth PPT | 56 s | 0.6 % |
| fixed ``n_r``=128, smooth PPT | – | agrees with ``n_r``=32 to ``10^{-8}`` |
| adaptive, smooth PPT | 138 s | agrees with fixed smooth to ``10^{-8}`` |

Read: with the channel-sum rate the answer itself is defined only to 1–2 % (the scatter
between quadrature rules, the adaptive one included, is that model's own quadrature
noise); with the smooth rate every rule agrees to ``10^{-8}``, and the smooth result is
*closer* to the channel-sum-512 reference than the channel-sum adaptive run is.

**Ar DUV example, 80 mbar (Keldysh ``\gamma\approx1``, weak ionisation), 3 m.** Smooth vs
channel-sum PPT: peak electron density 12 % lower; final fields within 0.2 % (0.16 % in
the 220–270 nm dispersive-wave band); channel-sum 256 vs 512 nodes agree to
``2\times10^{-6}`` there.

**Mode convergence** (`test/manual/modal_fixed_modeconv.jl`, RDW VUV example, 1.5 m,
default rule, smooth PPT; quadrature error in the 100–140 nm band ``10^{-10}``–``10^{-11}``,
so the differences below are mode truncation): without plasma 4 vs 8 modes differ by
``5.7\times10^{-4}`` in band energy and ``3\times10^{-3}`` in the band spectrum, 6 vs 8
by ``2.7\times10^{-5}`` / ``2.4\times10^{-4}`` — converged at 6; with plasma 4 vs 8
differ by ``2\times10^{-2}`` in the band spectrum and ``0.9\,\%`` in total energy
(loss), 6 vs 8 by ``5\times10^{-3}`` / ``0.14\,\%``; peak electron density
``1.89/1.85/1.86\times10^{22}\,{\rm m^{-3}}``; steps 4391/5038/5208; wall 62/92/114 s.
The ~10× worse mode convergence with ionisation (the plasma index is sharply peaked on
axis and couples to higher radial orders) was predicted in the analysis and is now
measurable because the quadrature noise that used to mask it is gone. Practical
consequence: for per-cent-level VUV band spectra in ionising RDW runs use 6 modes; 4 modes
is a 2 % answer.

### 4.5 Error monitoring

`Stats.mode_reconstruction_error(::TransModalFixed; window)` records every step:

- `transverse_integral_error_abs` / `_rel`: RMS of the embedded estimate
  ``P_{\rm coarse}-P_{\rm fine}`` (Gauss subset of the Kronrod rule in ``r``, every other
  node in ``\theta``), absolute and relative to the RMS polarisation;
- `transverse_integral_error_rel_window`: the same restricted to a wavelength window
  (`stats_kwargs=Dict(:error_window=>(λmin, λmax))`) — the relevant number for weak
  spectral features such as dispersive waves, and the direct answer to the old scheme's
  pump-dominated L2 criterion;
- `mode_reconstruction_error`: the on-axis polarisation reconstructed from its modal
  expansion against the polarisation evaluated directly on axis with the original
  columnwise responses;
- `transverse_points`.

For the smooth rate the embedded estimate tracks the true error (it did not for the kinked
rate); ``|K-G|`` bounds the error of the *coarse* Gauss rule, so it is conservative for the
Kronrod result actually used. The estimate reuses the samples (one extra GEMM and ``M``
FFTs), so it is cheap enough to monitor every step; it is an alarm, not a controller. What
still deserves care: more modes (higher radial orders need proportionally more nodes),
regimes with strong on-axis depletion (``1-e^{-\int W}`` saturating sharpens the transverse
profile), and anyone reverting to `sum_integral=false`. The cheap discipline for a new
regime is one run at ``2\times`` the nodes — a 2× cost check, not a 10× one.

## 5. Performance

All CPU numbers: Apple-silicon laptop, 8 Julia threads, FFTW planning `MEASURE` unless
stated. GPU numbers are **not yet measured** (§5.4).

### 5.1 End to end

RDW VUV example, 1.5 m: 18.6 min (adaptive, channel-sum PPT) → 138 s (adaptive, smooth
PPT) → 96 s (fixed ``n_r=64``, the default, first version) → 56 s (fixed ``n_r=32``);
with the fused two-pass plasma kernel (below) 67 s and 48 s. The fair like-for-like
comparison is therefore 138 s vs 67/48 s: once the rate is smooth the adaptive scheme needs
only ~31 points, so at ``n_r=32`` the fixed rule does the same amount of per-node work and
gains only from batching and threading, and at ``n_r=64`` it does twice the work and still
wins because it is parallel. The gain grows with core count (the adaptive integrand is
serial), with per-node work (Raman: CtC-type H₂ case, nt=32768/nto=65536, 6 modes,
Kerr+ADK+Raman: **14.5 ms per RHS fixed ``n_r=64`` vs 72 ms adaptive**), and for 2-D vector
sets.

### 5.2 Where the time goes

One RHS, before and after fusing the CPU plasma column (see below):

| | VUV (nto 8192, M 4, 65 nodes) | CtC-type (nto 65536, M 6, 65 nodes) |
|---|---|---|
| **RHS total** | **1.26 → 0.69 ms** | **18.6 → 14.5 ms** |
| plasma | 0.85 → 0.28 | 8.1 → 3.8 |
| Raman (batched doubled-grid FFT pairs) | – | 7.0–8.4 |
| Kerr | 0.14 | 0.4 |
| synthesis + projection GEMM | 0.10 | 1.2 |
| modal IFFT + FFT + windows | 0.10 | 1.0 |

**The fused plasma column.** The scalar/vector plasma kernels shared by `PlasmaCumtrapz`
and the host path of `PlasmaCumtrapzBatched` used to make ~7 passes over the column through
five intermediate arrays; they are now two passes — the rate (the rate function's array
method), then one serial sweep carrying the three running trapezoid sums in registers,
reading E once and writing P once — with exactly the arithmetic and operation order of the
multi-pass form (kept as `_plasma_*_multipass!` and tested `==`), plus two exact
shortcuts: ``e^{-\int W}`` is only re-evaluated when the integral changed (it does not over
the pulse wings, where the rate is zero — 93 % of the samples in the RDW example, and the
`exp` was the dominant cost of the sweep at ~3.5 ns per sample), and the loss term's
division is skipped where the rate vanishes (it adds exactly ±0.0). Sweep cost 6.1 → 2.3 ns
per sample per thread. The host path allocates only the rate and P.

**One RK45 step** (RDW VUV gradient case, ``n_r=32``, 11.0 ms measured before, 7.4 ms
after): 6 × (linear operator 0.78 ms + RHS 0.46 ms) + statistics 2.9 ms + windows 0.07 ms.
So after the transform work the step was dominated by (a) the **``z``-dependent linear
operator** for the pressure gradient — the generic `LinearOps.make_linop` closure evaluates
`Modes.neff(mode, ω; z)` for every (mode, ω) at every stage, ~48 ns each, re-evaluating the
gas index (a density spline × Sellmeier) and the cladding index (a `CmplxBSpline`) per mode
(~45 % of the step) — now replaced by `Capillary.MarcatiliLinop` (§3.7): one broadcast per
stage, 0.036 ms vs 0.385 ms at nω=1025 on the host, bit-identical, device-native; the
1.5 m RDW run went 67 → 52 s at ``n_r=64`` and 48 → 32 s at ``n_r=32``; and (b) the
**statistics** — of which `fwhm_t` alone was 1.3 ms because `Maths.level_xings` returned
"no crossing" (empty higher modes) by throwing and catching an exception, ~100 µs each;
fixed to return the same NaNs without throwing (statistics 2.4 → 1.4 ms per step, of which
the mode-error statistic's transform call is 0.65 ms). What remains of the 7.4 ms step is
6 × RHS 2.8 ms, statistics ~1.6 ms, and ~3 ms of RK45 stage updates/error norm/windows/
output. On a device the operator no longer caps the step: the host work per stage is the
three scalars ``a(z)``, ``ρ(z)``, ``β_1(z)``.

While doing this a latent bug was found and fixed: the older fixed-core acceleration
`Capillary.neff_wg` → `neff(m, εco, nwg)` (used by the mode-averaged ``z``-dependent
operator and by tuple collections of fixed-core modes) had the imaginary (loss) sign of
the `:reduced` model reversed, so with `model=:reduced, loss=true` the loss was clamped to
zero there (1.3 % operator difference vs `Modes.neff`); `:full`, the default, was
unaffected. Both paths now agree with `Modes.neff` bit-for-bit (tested).

Rate evaluation per sample: cached PPT spline ≈ 1–2 ns per thread, ADK ≈ 2.5 ns, direct
`IonRatePPT` ≈ 120 ns — recomputing the rate instead of the spline is ruled out on any
hardware (the plasma alone would be ~10× the whole RHS; on a GPU the branchy special-function
sum is the worst possible kernel shape whatever the peak FLOP rate). FFTW threads the
batched Raman plan over the columns (131072×65 rfft pair: 31.2 / 19.3 / 9.5 / 7.2 ms at
1/2/4/8 threads); Julia-threading single-column plans is only ~20 % faster (5.7 ms), so
the Raman cost is genuine work.

FFTW's thread count defaults to one per Julia thread with native pthreads (capped by
`Sys.CPU_THREADS`, which inside a container reports the host — set `LUNA_FFTW_THREADS`
explicitly there, as the Runpod `env.sh` does).

Every new `(2n_{to}\times n_p)` batched shape costs a one-time FFTW `PATIENT` planning
that can take **tens of minutes** (once per machine and thread count; wisdom is cached
under `~/.julia/scratchspaces/.../lunacache`). The benchmark script uses `MEASURE`; the
test suite uses `ESTIMATE`.

### 5.3 Arithmetic intensity and the choice of hardware (estimate)

From the code, per RHS (nto·Np elements; exp/log ≈ 15 FLOP-equivalents): modal FFT pair
≈ 2–3 FLOP/B (cache-resident, tiny); synthesis/projection GEMM ``M/4`` (``K=M`` is skinny:
write/read-bound); Kerr 0.3; **plasma device path ≈ 0.5 FLOP/B** (~120–150 FLOP per
element over ~15 whole-array passes and 3 scans, ~250–300 B); plasma host path: the
per-column working set (6 arrays × nto × 8 B = 400 kB at nto=8192) is L2-resident, so
~4 FLOP/B versus DRAM (compute-bound on exp/log) at small nto and DRAM-bound at
nto ≳ 2^15 (3 MB per column). Machine balance (peak FP64 / bandwidth): A40 ≈ 0.85 FLOP/B,
L40S 1.6, A100 5–6, H100/H200 ≈ 10, B200 ≈ 5, MI300X ≈ 15, 64-core EPYC 7763 ≈ 12 vs
DRAM (~1–2 vs L2). Consequences: on the A40 the plasma sits at the ridge — bandwidth- and
FP64-compute-bound at once (FP64 exp/log are software sequences on NVIDIA); on
H100/H200-class parts the code is purely bandwidth-bound and 5–7× faster per RHS than on
an A40 with no code change; a big CPU node is competitive for the small scalar case
because its columns stay in cache; the GPU pays clearly for 2-D vector sets and large
time grids (``\gtrsim10^7`` elements per RHS). To measure instead of estimate: Nsight
Compute `--set roofline` on the GPU job; LIKWID `MEM_DP` on a CPU node.

### 5.4 The HPC job

`test/manual/hpc_gpu_suite.sbatch` runs on one A40 node: precompilation on the node, the
CPU tests for the touched parts, the hardware-gated CUDA tests (`LUNA_TEST_CUDA=1`), and
`test/manual/hpc_gpu_bench.jl`, which benchmarks the transform on CPU and CUDA for
production-like cases (`vuv` RDW VUV; `ctc` H₂ supercontinuum; `vector` HE11(x,y)+TE01+
TM01+HE21 with plasma on the 2-D rule; `bignt` the VUV case on a 4× longer window):
isolated RHS time, ms/step and the transform's share of a step (6 RK45 stages + 1 call for
the mode-error statistic), GPU-vs-CPU agreement, resident device memory and a primitives
micro-benchmark at the same shapes, with a Markdown summary. Its results belong in this
document once available.

**First submission (job 442792, 2026-08-18, node gpu17, A40 48 GB, 16 CPU threads):** the
hardware tests all passed — 69/69: cuFFT semantics, native scan, RK45 error norm
host-vs-device ratio ``1\pm10^{-14}`` at every cancellation level and both shapes, modal
transform vs host ``1.2\times10^{-14}`` (ADK), ``9\times10^{-15}`` (PPT spline),
``8.6\times10^{-15}`` (Raman, `thg` on/off), end-to-end gradient propagation vs host
``1.3\times10^{-10}``, resident memory 12 fields (TransFree test), lazy `:cuda` subprocess
OK; CPU tests 243/244 (the error was a missing direct `QuadGK` dependency in the job
environment). The benchmark's per-RHS numbers were **lost**: Julia buffers stdout when it
is redirected to a file and the job hit the 4 h limit inside a 0.3 m CtC propagation
(~14 000 steps; that case takes ~50 steps/mm), so only the `@info` lines on stderr
survived. Those showed the `vuv` case at 76 ms/step on the CPU and 50 ms/step on the GPU
against 15 ms/step on the laptop — i.e. the *host* side of the node was ~5× slower than
expected, and it dominates the GPU run too (linear operator, statistics, windows). Prime
suspect, now fixed: Luna's FFTW default of 4× the Julia threads (64 pthreads on a 16-core
cgroup allocation, a 2020 default from the Julia-task-callback era) — with native pthreads
on Linux the count is now capped at the available parallelism (`Sys.CPU_THREADS`, which
respects the affinity mask; macOS keeps 4× because there `Sys.CPU_THREADS` counts only
performance cores and the extra threads measurably help). The job now runs the benchmark
before the slow CPU tests, flushes every printed line, uses per-case propagation lengths
(vuv 0.3 m, ctc 0.02 m, vector 0.1 m, bignt 0.3 m), and its environment needs `QuadGK`.
Whether the FFTW default explains all of the slowdown will be visible in the next run's
phase A (isolated RHS timings, printed first).

### 5.5 Fast parameter scans on a GPU (analysis; not yet run on hardware)

The target: a 20 × 20 energy × pressure scan of the RDW VUV example (400 propagations of
~4400 steps) on one GPU, with the lowest latency and only reduced results ("collected"
quantities) coming off the device. What each point costs today, and what to do about it:

- **Per-point host setup** (`prop_capillary_args`: gradient/density splines, mode
  matrices, PPT table load from the cache, host FFT plans for the statistics): ~0.5 s per
  point after the first (measured on the laptop; the first point pays compilation, ~4 s).
  With ~30 s of propagation per point on a GPU this is negligible; if the propagation
  ever gets to ~1 s per point it becomes the floor. Cheap wins if needed: build the mode
  collection and gradient once outside the loop and use `Luna.setup`/`Luna.run` directly
  instead of `prop_capillary` (the mode matrices and PPT table are then reused).
- **Per-step host work** dominates the device run: the default statistics are ~1.6 ms per
  step on the host (a 4400-step point → 7 s, comparable to or larger than the whole GPU
  propagation) — use `stats_kwargs=Dict(:mode_error=>false)` (drops the extra transform
  call and the on-axis reference evaluation) and `stats_period=10`; the ``z``-dependent
  linear operator is now a device broadcast (§3.7); the remaining host work per step is
  the RK45 error-norm sync and a few scalars.
- **Output**: `saveN=2` (start and end field) is enough for a scan whose reduced
  quantities are computed afterwards; each save is a small device→host copy. The
  `ScanHDF5Output` per point (~0.5 MB at saveN=2) is what `Processing.scanproc` reads to
  extract band energies, peak densities, spectra — the "collected" results. Nothing else
  needs to leave the device.
- **JIT and process model**: one Julia process per point (`SlurmExec(instances=…)` style)
  would pay compilation and CUDA initialisation (~1 min) per point — never for a GPU
  scan. Run the scan in-process (`LocalExec`, the default `runscan`), so compilation is
  paid once. `runscan` now calls `Luna.device_reclaim()` after every point (returns
  pooled device memory; a no-op on the host).
- **GPU occupancy**: a single 4-mode nt=8192 propagation is launch-latency-bound and uses a
  fraction of an H200. Throughput comes from **several processes sharing the GPU**, each
  running a slice of the scan in-process: `julia script.jl --batch k,i` for `i=1:k`
  (`Scans.BatchExec`; every process has its own CUDA context, ~300–500 MB of device
  memory each; CUDA MPS improves the overlap but plain time-slicing already helps for
  launch-bound kernels). Expect 3–5× over one process for the small case; measure with the
  rehearsal script's 4-process step. `--queue n` (`QueueExec`, `addprocs` workers pulling
  from a queue file) load-balances the same way.
- **The structural option** (roadmap): propagate all 400 points as **one batched state**
  `(nω, M, 400)` — same grid and modes, per-point density and linear operator, one RK45
  with a shared step size (set by the worst point). The transform already handles a
  column dimension; the plasma/Kerr `ρ` would become a per-column vector and the linear
  operator a per-point array; the whole scan then fills the GPU with ~4×10⁵ columns per
  RHS and finishes in one propagation. Effort M–L; only worth it if the multi-process
  approach leaves the GPU idle.

Rehearsal tooling: `test/manual/gpu_scan_rehearsal.jl` runs a small scan through the real
`Scans` machinery with the lean settings, times setup/propagation/output per point,
extrapolates to 400 points, runs with `--batch k,i` for the multi-process case, and
extracts the reduced results with `Processing.scanproc`; `test/manual/h200_gpu_suite.sh`
runs it (with the CUDA tests, the CUDA-only benchmark and full-length production runs) on
a rented Runpod H100/H200 pod prepared by `test/manual/runpodcoldstart.sh` (volume at
`/workspace`; results under `/workspace/runs/<timestamp>/`). Laptop CPU rehearsal (for the mechanics only): 0.5 s setup + propagation
per point, first point +3.5 s compilation.

### 5.6 H200 results (Runpod pod, 2026-08-18)

NVIDIA H200 (140 GiB), CUDA 13.3 driver/runtime, Julia 1.12.7, 8 host threads on a 24-vCPU
slice with `JULIA_CPU_TARGET=generic` (since changed — the host side of these numbers is
pessimistic). Log: `test/manual/runs/h200-suite-20260818-214254/`.

- **Hardware tests: 69/69 pass** in 6.4 min (host-bound; 3.6 min on the A40 node):
  device norm ratio 1 ± 10⁻¹⁴, transform vs host 10⁻¹⁵ (ADK, PPT, Raman thg on/off),
  end-to-end propagation vs host 4×10⁻¹¹.
- **Isolated RHS (phase A)**: `vuv` (nto 8192, 4 modes, 65 nodes) **0.53 ms**, kernels sum
  0.31 ms (FFT pair 0.04, GEMMs 0.08, rate 0.02, three native scans 0.17) → ~0.2 ms of
  launch/sync overhead; `ctc` (nto 65536, 6 modes, Raman) **6.2 ms** with kernels summing
  to only 1.0 ms — the gap is host work inside the RHS: the batched Raman recomputed the
  response kernel on the host, transformed it (2nto-point FFT) and uploaded it *every
  call* (0.5 ms on the laptop, several ms on this host); it is now cached on the density
  (`_update_raman_kernel!`), which removes it entirely for constant-pressure runs.
- **Production, lean statistics** (`mode_error=false`, `stats_period=10`): RDW VUV 1.5 m
  **22.5 s at nr=32 (5.1 ms/step), 18.8 s at nr=64** — i.e. the H200 ≈ the 8-thread laptop
  for this small, launch-bound case, as predicted (§5.3); with the default statistics
  42.8 s (9.8 ms/step: 4.6 ms/step of host statistics on this host). CtC-type 0.3 m:
  14 361 steps in 6.4 min (26 ms/step, both nr) — 4× the laptop; the full 1.5 m would be
  ~32 min. First propagation of a process: +95 s of compilation on this host.
- **Phase B of the benchmark** was compilation-dominated (157 "ms/step" for a 137-step
  vuv propagation): fixed with a warm-up propagation before phase B.
- **Scan rehearsal (2×2, lean, one process)**: 3.6–4.2 ms/step, 1.4–1.5 s setup per point,
  12.7 s per point on average → **400 points ≈ 85 min in one process**. **Four processes
  sharing the GPU gave no gain**: each ran 3–5× slower per step (host-bound: 4×8 threads on
  24 vCPUs, generic host code, no MPS), so total throughput was worse than sequential.
  The end of that step then hung — each process ran `Processing.scanproc` over the shared
  directory while others still wrote to it, and HDF5's file lock blocks on the network
  volume; collection is now done afterwards by one process, batch processes never read
  each other's files, and the suite bounds the wait.

**Second run** (`4753af4`, same pod, multi-target `JULIA_CPU_TARGET`, Raman kernel cached,
warmed phase B; log `test/manual/runs/h200-suite-20260818-230326/`):

| | run 1 | run 2 | laptop CPU, 8 threads |
|---|---|---|---|
| `vuv` RHS (nto 8192, 4 modes, 65 nodes) | 0.53 ms | 0.55 ms | 0.46 ms (nr=32: 0.69 at 65 nodes) |
| `ctc` RHS (nto 65536, 6 modes, Raman + plasma) | 6.2 ms | **1.83 ms** (kernels 0.95) | 13.6 ms |
| RDW VUV 1.5 m, lean stats, nr=32 / 64 | 5.1 / 4.3 ms/step | 6.7 / 6.4 ms/step | ~5–7 ms/step |
| RDW VUV 1.5 m, default stats, nr=32 | 9.8 ms/step | 15.7 ms/step | 7.4 ms/step |
| CtC 0.3 m (14 361 steps), lean, nr=64 / 32 | 27.3 / 26.2 ms/step | **13.9 / 9.3 ms/step** | ~110 ms/step |
| bench phase B `ctc` 0.02 m, default stats | 75 ms/step | 65 ms/step (7×RHS = 20 %) | – |

Reading it: (i) the Raman kernel cache took the `ctc` RHS from 6.2 to 1.8 ms and the CtC
propagation to 9.3 ms/step at nr=32 — **~12× the laptop; the full 1.5 m CtC case is
~11 min on the H200 versus ~2 h on the laptop**. (ii) The small `vuv` case is unchanged
and host-bound: 0.55 ms of RHS (~35 launches + syncs) and 6–7 ms/step, i.e. the H200 ≈ a
laptop, as §5.3 predicted; run 2's host share is even ~1.5 ms/step higher than run 1's,
which the multi-target images should not cause — treat as pod noise until an
`unset JULIA_CPU_TARGET` (native) comparison is made. (iii) **The default statistics
are the dominant host cost of a device run**: +9 ms/step on `vuv` and ~50 ms/step on
`ctc` (six modes on a 65 536-point grid: host FFTs, on-axis reference with the columnwise
Raman, fwhm's), several times the GPU work — lean settings (`mode_error=false`,
`stats_period`) are mandatory for GPU production, and on-device statistics (roadmap)
moves from P2 to P1 for large grids.

Conclusions: correctness on H200 is established; the GPU pays for the large case (~12×
a laptop for CtC, more with lean statistics or on-device statistics), while for the
small modal case the GPU is not faster than a good CPU (the RHS is launch/sync-bound and
the step host-bound) and sharing the card between processes does not help while the
host is the bottleneck — the levers there are the batched-state propagation of all scan
points at once (§5.5, roadmap), on-device statistics, and a faster host; the
gradient/taper cases no longer pay the host operator (§3.7). Recorded in the roadmap.

## 6. Testing

- `test/test_modal_quadrature.jl`: rules, nodes/weights, orthonormality of the mode sets on
  the rule (``10^{-12}``), Kerr overlap tensor vs a 2048-node reference (``10^{-13}``),
  taper scaling identity.
- `test/test_modal_fixed.jl`: frozen-field projections against `TransModal` at
  `rtol=1e-10, mfcn=10^7`: Kerr scalar radial, HE11 x+y vector (radial), TM01+HE21+HE11
  full 2-D, envelope grid, Kerr+ADK plasma scalar and vector, noise field and gas
  mixtures; batched plasma vs columnwise (bit-identical on the host, ``10^{-12}`` for the
  device path on host arrays, both rate types, scalar and vector); batched real-field
  Raman vs columnwise (``10^{-12}``, both `thg`) and the transform vs the adaptive one to
  rounding; vector Kerr array-level vs columnwise (bit-identical).
- The pre-existing `test_multimode.jl`, `test_vectorplasma.jl`, `test_tapers.jl`,
  `test_polarisation_field/env.jl`, `test_noise.jl` are parametrised over
  `modal_integral ∈ (:adaptive, :fixed)` at their existing tolerances.
- `test/test_device.jl`: the device logic without hardware — `Utils.backend` dispatch,
  `HostOutput`, array-type resolution and the world-age mechanism (with a stand-in),
  device error norms, and, under `Pkg.test` (JLArrays is a test-only dependency): FFT plan
  shims for JLArray, `TransModalFixed` on JLArrays vs the host (``10^{-10}``, Kerr+plasma,
  ADK and PPT), the mode-error statistic with a host and a device state, a whole
  propagation with a ``z``-dependent operator and default statistics vs the host,
  `stats_period`, and the error paths.
- `test/test_cuda.jl` (hardware-gated, `LUNA_TEST_CUDA=1`): cuFFT semantics, the native
  scan, the RK45 error norm host-vs-device under cancellation (ported from
  `ModelPNPS/examples/check_device_norm.jl`), the modal transform vs the host (``10^{-10}``,
  incl. PPT spline and ADK kernels and H₂ Raman), an end-to-end gradient propagation
  (``10^{-8}``), the resident memory footprint, and the lazy `:cuda` path in a fresh
  subprocess.
- `test/test_output.jl`: `PeriodicStats` with both output handlers.
- Manual (not in the suite): `test/manual/modal_fixed_convergence.jl`,
  `test/manual/modal_fixed_rdw.jl` (both backends, per-step timing),
  `test/manual/hpc_gpu_bench.jl` + `hpc_gpu_suite.sbatch`.

`TransModal` (adaptive) is the reference oracle for all of this and must never be deleted.

## 7. Known limitations

- Vector (two-component) Raman is not implemented in any transform (never was).
- `Kerr_env_thg` and user-defined columnwise responses run serially on the host and are
  refused on a device (clear error). `Kerr_field_nothg` now has a batched form,
  `KerrFieldNoTHG` — `Kerr_field_nothg(γ3)` with one argument returns it; the two-argument
  closure is the old columnwise one and is kept for reference.
- Scalar gas mixtures take the serial columnwise fallback in `Et_to_Pt_ordered!`.
- Plasma in `TransFree` (3-D free space) on a device is not wired up: the transform is
  ready (§3.9), but `PlasmaCumtrapzBatched` needs the `(nto,ny,nx) → (nto,1,ny·nx)`
  reshape to be reachable from there.
- `TransModeAvg` (mode-averaged) and `TransRadial` have no device path.
- Non-scale-invariant ``z``-dependent modes re-evaluate the mode fields on the host per RHS.
- `StepIndexMode` (`dimlimits = 10a`, kink at ``r=a``) would need a composite rule with a
  breakpoint; not validated on the single Gauss panel.
- On a device, `HDF5Output(cache=true)` needs the host copy of the state every step (the
  mode-error statistic then uploads it again).
- The device path is CUDA-only in practice (the generic code should run under AMDGPU.jl
  with a small extension; untested).

## 8. Roadmap and open items

The single list of deferred work; effort S ≲ ½ day, M ≈ 1–2 days, L ≳ 3 days; P0 blocks
intended use, P1 do next, P2 worthwhile, P3 when needed. Ticked items are done and kept for
the record.

### 8.1 Bugs / blockers found while planning

- [x] **P0 — lazy `arraytype=:cuda` inside one top-level call (world age).** Fixed, §3.7;
  hardware test still to be run.
- [x] **P1 — `mode_error` statistic round-tripping the state on a device.** Fixed, §3.7.

### 8.2 Measure first (user actions on the DMOG A40)

- [ ] Re-submit `test/manual/hpc_gpu_suite.sbatch` (§5.4): the first run validated the
  device path (69/69 hardware tests, agreement to ``10^{-14}``/``10^{-10}``) but lost the
  benchmark output; the job is reordered, flushed, shorter, and the FFTW thread default
  fixed. It now also asks for 32 CPUs and runs a **CPU thread-scaling sweep**
  (`test/manual/cpu_thread_scaling.jl`: 1–32 threads, one process each, FFTW/BLAS threads
  = Julia threads: isolated RHS of the vuv and ctc cases, one linear-operator and one
  statistics call, ms/step of a short vuv propagation) — the fair CPU baseline for the GPU
  numbers on a 128-core node, and the direct test of whether serial host work (linear
  operator, statistics) caps the step. Record the phase-A RHS table, phase-B ms/step and
  the scaling table here.
- [ ] `test/manual/modal_fixed_rdw.jl vuv quick 1.5 {cpu,cuda}` and a full CPU node
  (`-t 32`) for the per-step share of "everything else" (linear op, RK45, stats, output).
- Expected (unmeasured): scalar 4-mode nt=8192 → A40 ≈ 8-thread laptop, slower than a
  32-core node (launch-latency- and FP64-bound); 2-D vector / large nt → 5–10× over the
  laptop; sweeps → throughput via several processes per GPU. Decisions below hinge on these.

### 8.3 Performance — next steps for `TransModalFixed` (measured breakdown in §5.2)

- [ ] **P1 (no code) — tune `nr`.** With the smooth rate the plasma is converged at
  ``n_r\approx32`` for 4 modes; the default 64 is a 2× safety margin. ``n_r=32``–40 (check
  `transverse_integral_error_rel` stays ~``10^{-9}``) is ≈1.6–2× on everything, CPU and
  GPU. Consider lowering the default once the mode-count study (§8.5) is done.
- [x] **`stats_period` / `mode_error=false`** for sweeps: −14 % (one of 7 transform calls
  per step).
- [x] **P1 — CPU: fuse the plasma column into two passes.** Done (§5.2): plasma 3×
  faster on the RDW case (0.85 → 0.28 ms), 2× on CtC; RHS −45 % / −22 %; bit-identical.
- [x] **P1 — the ``z``-dependent modal linear operator, device-native** — done
  (`Capillary.MarcatiliLinop`, §3.7/§5.2): tapers, gradients and both; bit-identical to
  the generic closure; 10× faster on the host, one broadcast on a device; RDW VUV 1.5 m
  48 → 32 s at ``n_r=32``. The description below is kept for the record. Was: 0.78 ms per
  call, 6 calls per step = ~45 % of a step at ``n_r=32``.
  `LinearOps.neff_grid` for a vector of `MarcatiliMode`s should evaluate the gas index once
  per ``z`` for all modes that share it (today `Interface.makemode_s` builds a separate
  `Capillary.gradient` — and density spline — per mode, so even identity-based sharing
  fails; build it once) and the cladding index once per run where it is
  ``z``-independent (the standard constructors' `cladn` closures are; give them a shared,
  registered closure so `neff_grid` can know it exactly), then the closed-form `neff` per
  mode. Same numbers (same functions on the same inputs), ~3× faster operator, ~−30 % per
  step. Effort M (Interface + Capillary + LinearOps). **Better still, and the same
  amount of work: make it device-native.** For a Marcatili mode ``z`` enters `neff` only
  through two scalars per stage — the core radius ``a(z)`` (tapers) and the density
  ``\rho(z)`` in ``\varepsilon_{\rm core} = 1 + \gamma(\omega)\rho(z)`` (gradients) —
  so with ``\gamma(\omega)``, the cladding term and ``u_{nm}`` precomputed on the device the
  operator is one broadcast over ``n_\omega\times M`` per stage (µs), no host loop and no
  upload; tapers, gradients and both together use the same kernel, and on the CPU the same
  broadcast is ~20 µs instead of ~800. Needs `Capillary.gradient` to return a recognisable
  callable (`GradientCoreIndex(γ, dens)`) rather than an anonymous closure, and a fallback
  to the host loop for other mode types or user-supplied ``z``-dependent `coren`/`cladn`
  (`RK45.make_prop!` already has the `device_capable` hook). This matters most on
  H100-class GPUs, where the device RHS shrinks to a fraction of the host operator's
  ~0.8 ms per stage and Amdahl caps the gradient cases (§5.3).
- [x] **`fwhm_t` statistic exception path** — fixed (§5.2).
- [ ] **P2 — Raman by mode pairs.** ``h*E^2 = \sum_{mn}(h*A_mA_n)\,e_me_n``: ``M(M+1)/2``
  pair convolutions instead of one per node, then a small GEMM. Gain on the Raman term
  ``= n_p/[M(M+1)/2]``: 65/21 ≈ 3× at ``n_r=64``, ``M=6`` (CtC RHS −25 %), 65/10 ≈ 6.5× at
  ``M=4`` — but only 33/21 ≈ 1.6× at ``n_r=32``, so tune `nr` first. Needs a
  transform-aware Raman path; the one place the tensor idea pays, and only because Raman
  carries two long FFTs per node. **Not** for Kerr: already exact on the grid, ~5 FLOP per
  sample on a field the plasma needs anyway, and the pair route would cost a 21×21 GEMM
  per sample (~3× more FLOPs) and break response encapsulation. Effort M.
- [ ] **P2 — GPU: fuse the plasma's elementwise passes** between the three scans (traffic
  −2.5×, intensity 0.5 → ~1.5 FLOP/B; matters on H100-class, little on the A40). Then, if
  the scans dominate, the **time-tiled scan** (parallel over columns × time blocks with
  carried block prefixes) — the GPU form of the fused CPU column. Decide after the A40
  numbers.
- [ ] **P2 — GPU: FP32 rate evaluation** (spline + exp only). The rate is accurate to
  ~``10^{-4}`` by construction; lifts the FP64-compute co-limit on A40/L40S-class cards
  ~30×; irrelevant on H100/MI300X. Test against the RK45 controller (step count).
- [ ] **P2 — CUDA graph capture of the RHS** (static control flow, no allocation after
  warm-up): launch overhead → µs; ``z``-dependent scalars must become device scalars.
  Only pays in the launch-bound (small nt) regime; 2–3× there. Effort M–L.
- [ ] **P2 — Column-chunk fusion for very large nt (also memory).** `Et`, `Pt` and the
  plasma buffers are nto×npol×Np: ~1 GB at nt=2^17 with 65 nodes, ~7 GB for a 2-D vector
  set with ~1000 columns — too much for one A40 next to the RK45 registers, and past L3 on
  a CPU. Loop synthesis GEMM → responses → projection GEMM over column blocks (the FFTs are
  on the modal side and unaffected). Effort M. (The fused CPU column above removes most of
  the CPU-side buffers.)
- [x] **P1 — Modal statistics on device.** Done (§3.7): all default statistics run on the
  device state (weighted reductions, on-axis GEMMs, the device rate, device FFT for the
  analytic field; `fwhm_t` copies |E|² down for the host crossing search; `fwhm_r` via the
  mode coherency matrix on both backends), no host copy of the state unless a user
  statistic needs one. Was: +9 ms/step on `vuv`, ~50 ms/step on `ctc` on the H200. To be
  measured on the next GPU run (the "default stats" rows of the production timing).
- [ ] **P1 — GPU parameter scans** (§5.5, §5.6): the H200 rehearsal showed one process at
  12.7 s/point (400 points ≈ 85 min) and *no* gain from four processes sharing the GPU
  (host-bound). Next: the batched-state propagation of all points at once (per-column
  density and per-point linear operator; effort M–L) — the only lever left for the small
  case; and rerun the rehearsal with the corrected host environment.
- [ ] **P3** — FP32 GEMMs (never for A40/H200).
- [ ] **P2 (policy) — FFTW planning of large batched shapes**: `PATIENT` for a fresh
  ``2n_{to}\times n_p`` Raman shape takes tens of minutes on first use; consider `MEASURE`
  above some size, or document `Luna.set_fftw_mode(:measure)` for large-nt Raman runs.

### 8.4 Coverage — not yet on a device / not batched

- [x] **P1 — the general (RealGrid) `TransFree` path on a device.** Done (§3.9). It is now
  threaded on the host and runs on a device for any pointwise or batched response, which is
  what a 3-D field-resolved (`RealGrid`) free-space propagation needs — it can never take
  the fast path, because the inverse of a real-to-complex plan destroys its input and on
  the fast path that input is the solver's own state. Plasma in `TransFree` on a device now
  needs only `PlasmaCumtrapzBatched` to be reachable from there, i.e. nothing in the
  transform.
- [x] **P1 — `RamanPolarField` batched.** Done (§3.6).
- [x] **P2 — `Kerr_field_nothg` as a batched response.** Done: `KerrFieldNoTHG` reuses
  `_analytic_batched!` (the batched `plan_hilbert` shared with `RamanPolarFieldBatched`).
- [ ] **P2 — `Kerr_env_thg` as a batched response** (broadcast with a ``t`` vector).
  Effort S.
- [ ] **P2 — Scalar gas mixtures batched/threaded** in `Et_to_Pt_ordered!` (fill once,
  ordered loop per `(responses_i, ρ_i)`). Effort S.
- [ ] **P3** — `TransRadial` on device (device Hankel matmul); `TransModeAvg` on device;
  non-scale-invariant ``z``-dependent modes on device; StepIndex breakpoint quadrature.
  Adaptive `TransModal` on device: never (reference oracle).

### 8.5 Validation and physics still outstanding

- [x] **Kerr energy conservation, fixed vs adaptive** — done (§4.1): identical for
  both, limited by the time-domain discretisation, not the spatial quadrature.
- [x] **Mode convergence 4/6/8 with and without plasma** — done (§4.4): converged at 6
  without plasma; 4 modes is a 2 % answer with plasma, 6 modes 0.5 %.
- [x] **``n_\theta`` convergence of the 2-D rule with plasma** — done (§4.2): the
  default 16 gives ``10^{-6}``, 24 gives ``10^{-9}`` on a strongly vectorial field; Kerr
  exact from 6. Consider ``n_\theta=24`` as the default for plasma-heavy vector runs
  (1.5× cost) — or leave 16 and rely on the (very conservative) embedded θ estimate.
- [ ] **P2 — Third-gas check of smooth vs channel-sum PPT** (Kr/Xe at 800 nm or Ar at
  1.8 µm, ``\gamma\approx1``–2). If a case ever disagrees, the right fix is a *soft*
  channel-closing regularisation in `IonRatePPT`, not resolving the hard step. Effort S.
- [ ] **P2 — PPT spline accuracy audit** against direct `IonRatePPT` over the production
  field range; consider `ln W` on uniform `ln E` knots. Effort S.
- [ ] **P3 — Thread-scaling study on a 32/64-core node**; `tforeach` gates. Effort S.

### 8.6 Documentation, infrastructure, merge

- [ ] **P1 — User-facing GPU page** (or a section in the interface docs): `arraytype`, what
  runs where, statistics/`stats_period`, `step_on`, memory footprint,
  `SlurmExec(gres, instances)` recipe, how to run the hardware tests.
- [ ] **P2 — Precompile workload** (`PrecompileTools`) for the fixed transform and batched
  responses (`instances`-style scans pay TTFX per process).
- [ ] **P2 — Merge plan.** Commits are phase-ordered for cherry-picking: bug fix in the
  cartesian 2-D integrand (`3ed6007`), quadrature/mode matrices, `TransModalFixed` host
  path, batched responses/scan/ionisation, Interface/Stats/docs, device plumbing,
  validation + defaults, native scan, world-age fix, batched Raman + HPC suite,
  `stats_period` + device statistics. The CPU-only steps could go to `master` ahead of the
  GPU work; the `gpu` branch's array-generic refactors are prerequisites of the device
  plumbing.
- [ ] **P3** — README / CHANGELOG entries (new default modal integral; the PPT default
  change is physics-visible; `arraytype`).
- [ ] **P3 — PyPlot as a weak dependency.** `using Luna` loads `Plotting` → PyPlot, whose
  init needs matplotlib in PyCall's Python; on a bare GPU pod `using Luna` fails until
  matplotlib is installed (the cold start now does it). Making `Plotting` a package
  extension (or lazy) would remove a heavy, headless-hostile dependency from every
  compute job. Effort M.

### 8.7 Rejected (do not re-litigate)

Γ-tensor hybrid for Kerr (§3.2); amortised adaptivity; automatic refinement from the error
estimate; threading `pointcalc!`; re-tuning ``n_t``/RK45 tolerance before the spatial error
was under control (now possible, §8.5); recomputing the ionisation rate instead of the
spline (§5.2).

## 9. Where things live

| | |
|---|---|
| Quadrature rules, nodes/weights, mode matrices, mode traits | `src/Modes.jl` (`TransverseQuadrature`, `transverse_quadrature`, `quadrature_nodes/weights`, `mode_matrix`, `zconstant`, `scale_invariant`, `azimuthal_order`) |
| The transform | `src/NonlinearRHS.jl` (`TransModalFixed`, `update_matrices!`, `apply_responses!`, `integral_error!`, `norm_modal`, backend-dispatched `copy_scale!`; `TransModal` unchanged as the oracle) |
| Batched responses, traits | `src/Nonlinear.jl` (`pointwise`, `batched`, `batched_response(s)`, `PlasmaCumtrapzBatched`, `RamanPolarFieldBatched`, `RamanPolarEnvBatched`, vector `KerrField`/`KerrEnv`) |
| Device-safe ionisation rate | `src/Ionisation.jl` (`device_ionrate`, `ionrate_device!`, `device_capable`, Adapt rules), `src/Maths.jl` (`spline_eval`, `CSpline` Adapt) |
| Prefix sums | `src/Maths.jl` (`cumtrapz_scan!`, `scan!`, `scan_scratch`), `ext/LunaCUDAExt.jl` (native `cumsum!`) |
| Backend trait, threading helpers, backend FFT planners | `src/Utils.jl` |
| Device scaffolding: hooks, `resolve_arraytype`, `lazy_arraytype`, `device_zeros`, `GridVectors`, `HostOutput`, `needs_host_y`, `device_stats` | `src/Device.jl` |
| Setup and run with `arraytype`, linop upload | `src/Luna.jl`, `src/RK45.jl` (`make_prop!`, `device_capable`) |
| Statistics | `src/Stats.jl` (`mode_reconstruction_error`, `StatsCollector`, `StateStat`, `wants_state`), `src/Output.jl` (`PeriodicStats`) |
| Interface keywords, PPT default | `src/Interface.jl`, `src/Ionisation.jl` (`IonRatePPT`) |
| Tests and manual scripts | `test/test_modal_quadrature.jl`, `test/test_modal_fixed.jl`, `test/test_device.jl`, `test/test_cuda.jl`, `test/manual/*` |
