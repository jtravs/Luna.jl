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
  rate) → 138 s (adaptive, smooth rate) → 96 s (fixed, default ``n_r=64``) → 56 s (fixed,
  ``n_r=32``). CtC-type H₂ supercontinuum (nt = 32768, 6 modes, Raman + plasma): 72 ms →
  19 ms per RHS. The transform scales with cores; the adaptive one did not.
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
`Luna.run` uploads a constant host linear operator once; a ``z``-dependent one (the RDW
*gradient* example) is evaluated on the host into a buffer and uploaded per stage
(`RK45.make_prop!`, `device_capable(linop!)` opt-out for future device-native operators).
Saved fields come back through `Luna.HostOutput`.

**Statistics on a device run.** `Stats.collect_stats` returns a `StatsCollector` which
accepts the *device* array itself (`Luna.device_stats`): it copies it into its own host
buffer for the ordinary host statistics (the modal state is small: 0.26 MB at nt=8192)
and hands the device array to statistics flagged `Stats.wants_state` (a `StateStat`);
`Stats.mode_reconstruction_error(::TransModalFixed)` is one and evaluates the transform on
the device state directly, so nothing round-trips. `Output.PeriodicStats(statsfun, k)`
(`prop_capillary(...; stats_period=k)`) evaluates the statistics only every ``k``-th
accepted step. `prop_capillary` passes `allow_device_stats=true` for the modal case.

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

A consequence beyond accuracy: a symmetric fixed rule yields an effective overlap tensor
``\Gamma^{\rm eff}_{mnpq}=\sum_i w_i e_{mi}e_{ni}e_{pi}e_{qi}`` that is symmetric under
index permutation, so the Kerr term derives from a quartic Hamiltonian and conserves
energy/photon number exactly (independent of ``n_r``); an adaptive mesh that changes
between evaluations breaks that at the tolerance level. (A demonstration script is on the
roadmap.)

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

### 4.3 The PPT channel-closing kinks

With the literal channel sum, the frozen-field convergence of the plasma term was
algebraic (``\sim n_r^{-1.5}``) and non-monotone for the fixed rule (``10^{-2}`` @64,
``3\times10^{-3}`` @128, ``10^{-3}`` @256–512, ``5\times10^{-5}`` @1024) *and* for the
adaptive one (``5\times10^{-4}`` with 511 points at `rtol=1e-3`; 8k–65k points for
``\sim10^{-5}``; its own error estimate 10–1000× optimistic when tightened). Increasing the
spline table from ``2^{14}`` to ``2^{18}`` knots changed nothing; replacing the sum by its
integral (`sum_integral=true`) restored spectral convergence. The rate itself: the two
forms differ by 0.1 % at ``10^{11}`` V/m, 7 % at ``5\times10^{10}`` and 20 % at
``2\times10^{10}`` (He, 800 nm) — but the absolute rate at the latter fields is
``10^{4}``–``10^{11}\,{\rm s^{-1}}`` against ``10^{14}`` at the peak, i.e. negligible
ionisation. In the tunnelling regime of the RDW/VUV runs (Keldysh ``\gamma\approx0.4``) the
integral form is essentially exact.

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
PPT) → 96 s (fixed ``n_r=64``, the default) → 56 s (fixed ``n_r=32``). The fair like-for-like
comparison is therefore 138 s vs 96/56 s: once the rate is smooth the adaptive scheme needs
only ~31 points, so at ``n_r=32`` the fixed rule does the same amount of per-node work and
gains only from batching and threading, and at ``n_r=64`` it does twice the work and still
wins because it is parallel. The gain grows with core count (the adaptive integrand is
serial), with per-node work (Raman: CtC-type H₂ case, nt=32768/nto=65536, 6 modes,
Kerr+ADK+Raman: **19 ms per RHS fixed ``n_r=64`` vs 72 ms adaptive**), and for 2-D vector
sets.

### 5.2 Where the time goes (one RHS)

| | VUV (nto 8192, M 4, 65 nodes) | CtC-type (nto 65536, M 6, 65 nodes) |
|---|---|---|
| **RHS total** | **1.26 ms** | **18.6 ms** |
| plasma (rate + 3 cumulative integrals + elementwise) | 0.85 (68 %) | 8.1 (44 %) |
| Raman (batched doubled-grid FFT pairs) | – | 7.0 (38 %) |
| Kerr | 0.14 | 0.4 |
| synthesis + projection GEMM | 0.10 | 1.2 |
| modal IFFT + FFT + windows | 0.10 | 1.0 |

Rate evaluation per sample: cached PPT spline ≈ 1–2 ns per thread, ADK ≈ 2.5 ns, direct
`IonRatePPT` ≈ 120 ns — recomputing the rate instead of the spline is ruled out on any
hardware (the plasma alone would be ~10× the whole RHS; on a GPU the branchy special-function
sum is the worst possible kernel shape whatever the peak FLOP rate). FFTW threads the
batched Raman plan over the columns (131072×65 rfft pair: 31.2 / 19.3 / 9.5 / 7.2 ms at
1/2/4/8 threads); Julia-threading single-column plans is only ~20 % faster (5.7 ms), so
the Raman cost is genuine work.

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
- `Kerr_field_nothg`, `Kerr_env_thg` and user-defined columnwise responses run serially
  on the host and are refused on a device (clear error).
- Scalar gas mixtures take the serial columnwise fallback in `Et_to_Pt_ordered!`.
- Plasma in `TransFree` (3-D free space) on a device is not wired up (the pieces exist).
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

- [ ] Submit `test/manual/hpc_gpu_suite.sbatch` (§5.4). Record the summary tables here.
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
- [ ] **P1 — CPU: fuse the plasma column into two passes** (`_plasma_scalar!`/
  `_plasma_vector!`): today ~7 passes over the column through 5 intermediate arrays;
  instead one SIMD pass for the rate and one serial pass carrying the three running
  trapezoid sums in registers, reading E once and writing P once, with the *same*
  arithmetic order as `Maths.cumtrapz!` (bit-identical, testable exactly). Expect the
  plasma to halve (VUV RHS −35 %, CtC −25 %) and the five nto×Np buffers to disappear on
  the CPU (also fixes the CtC column working set falling out of L2). **CPU only** — one
  thread walking a column is what a GPU must not do; its GPU counterparts are the fusion /
  time-tiled scan below. Worth doing regardless of the A40 results. Effort S–M.
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
- [ ] **P2 — Modal statistics on device.** Effort M (~300–400 lines in `Stats.jl`, a
  device method per field-touching statistic returning scalars); payoff ≤15 % of a step
  at nt=8192, 25–50 % at nt=2^17. Decide after measurement.
- [ ] **P3** — several simulations per GPU for sweeps (`SlurmExec(gres, instances)`, no
  code; document); FP32 GEMMs (never for A40/H200); device-native ``z``-dependent linear
  operator (small).
- [ ] **P2 (policy) — FFTW planning of large batched shapes**: `PATIENT` for a fresh
  ``2n_{to}\times n_p`` Raman shape takes tens of minutes on first use; consider `MEASURE`
  above some size, or document `Luna.set_fftw_mode(:measure)` for large-nt Raman runs.

### 8.4 Coverage — not yet on a device / not batched

- [ ] **P1 — Plasma in `TransFree` (3-D free space) on GPU.** `PlasmaCumtrapzBatched` and
  the device `copy_scale!` exist; the general (RealGrid) `TransFree` path needs a device
  method using them and `Et_to_Pt_ordered!` with `batched_responses`, plus the
  `(nto,ny,nx) → (nto,1,ny·nx)` reshape. Effort M. Payoff high if 3-D + ionisation runs
  are on the roadmap.
- [x] **P1 — `RamanPolarField` batched.** Done (§3.6).
- [ ] **P2 — `Kerr_field_nothg`, `Kerr_env_thg` as batched responses** (Hilbert along dim
  1 / broadcast with a ``t`` vector). Effort S each.
- [ ] **P2 — Scalar gas mixtures batched/threaded** in `Et_to_Pt_ordered!` (fill once,
  ordered loop per `(responses_i, ρ_i)`). Effort S.
- [ ] **P3** — `TransRadial` on device (device Hankel matmul); `TransModeAvg` on device;
  non-scale-invariant ``z``-dependent modes on device; StepIndex breakpoint quadrature.
  Adaptive `TransModal` on device: never (reference oracle).

### 8.5 Validation and physics still outstanding

- [ ] **P1 — Kerr energy / photon-number conservation** over long propagation, fixed vs
  adaptive (should be exact to rounding for the fixed rule, §4.1). Effort S.
- [ ] **P1 — Mode-convergence rerun (4 vs 6 vs 8 modes) with and without plasma**, now
  that quadrature noise is gone; expect worse convergence in the ionising regime. Effort S.
- [ ] **P2 — 2-D vector-set convergence in ``n_\theta`` with plasma** (the plasma is not
  polynomial, so ``\theta`` exactness does not apply to it); a justified ``n_\theta``
  default. The A40 `vector` case gives first data. Effort S.
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
