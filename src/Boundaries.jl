"""
    Boundaries

Absorbing boundaries with **rate semantics**.

Luna's spectral and temporal absorbing boundaries used to be applied by multiplying the
solution by a fixed profile `W` after every accepted step. That makes the cumulative
absorption `W^N`, where `N` is the number of steps the adaptive controller happens to
take — so the answer depended on `rtol` and never converged. (With the step counts of a
typical run, `W^N` is not a taper at all: it is a brick wall at the edge of the flat
region, which is why the old scheme eroded genuine spectral wings.) An absorbing boundary
must instead have an absorption *coefficient per unit propagation distance* `α`, so that
traversing a distance `L` attenuates by a fixed factor however that path is subdivided.

`α` here is a **power** absorption coefficient, as everywhere else in Luna (`Modes.α`,
`LinearOps`): power falls as `exp(-αL)`, the field as `exp(-αL/2)`, and a linear operator
carries `-α/2`. This module turns the historical taper profiles into such coefficients,

    α(x) = -2 log(W(x)) / ℓ

where `ℓ` is a reference length: propagating through `ℓ` attenuates the *field* by exactly
the historical profile once, since the profiles were applied to the field. `ℓ` defaults to
`zmax/N`, i.e. "the historical window is applied `N` times over the whole propagation".

The two rates are then applied by different mechanisms, chosen by what the existing
machinery can do *exactly*:

- `α_ω` is diagonal in ω, so it is folded into the linear operator ([`addloss`](@ref)) and
  the interaction-picture propagator applies `exp(-α_ω Δz/2)` exactly, for whatever
  sub-interval the stepper uses, at no extra cost, and consistently with dense output.
- `α_t` is diagonal in `t`, so it cannot ride the propagator. It is applied as an
  exponential split-step factor `exp(-α_t Δz/2)` in `Luna.run`'s `stepfun`. Those factors
  telescope exactly to `exp(-α_t L/2)` regardless of the step layout, so the *total*
  absorption is still a rate. (This is a first-order Lie splitting, but the commutator is
  supported only inside the collar, where the field is being destroyed anyway.)

The *hard* band limit — the 0/1 part of the spectral window — is not applied here at all.
`NonlinearRHS` zeroes the nonlinear polarisation outside `grid.sidx` and the linear operator
is zero there, so once the field has been band-limited at the start of `Luna.run` nothing
can put anything back.

See also `Luna.run`'s `boundary` keyword argument.
"""
module Boundaries

import ..Maths
import ..Grid
import LinearAlgebra: mul!, ldiv!
import Logging
import Printf: @sprintf

"Default number of applications of the historical window profile over `zmax`."
const DEFAULT_N = 20

"Default minimum temporal collar width, as a fraction of the full time window."
const DEFAULT_TCOLLAR = 0.05

#= Cap on α*ℓ, i.e. on the depth of the absorber over one reference length. α is a power
   coefficient, so the field is attenuated by at most exp(-MAX_αℓ/2) over ℓ.

   RK45's `fbar!` back-propagates the RHS with exp(-linop*Δz), i.e. it *amplifies* by
   exp(+α Δz) for a decaying linop. Two things follow.

   First, α must be finite: the tapers reach exactly 0 at the band edge, and Inf*0 = NaN.

   Second — and this is the constraint that actually binds — the amplified elements must
   not pollute the step-size controller. `RK45.weaknorm` sums abs2(yerr) over *all*
   elements and divides by the global ‖y‖, and it is evaluated in the interaction-picture
   frame, before `prop!_maybe`. So an amplified band-edge element is not independent of the
   rest: it can dominate the error estimate and drive the controller into repeated
   rejections. The saving grace is that the nonlinear drive is itself ∝ ωwin = exp(-α ℓ),
   so the amplified drive scales as exp(α(Δz - ℓ)) — bounded by 1 as long as Δz ≤ ℓ, for
   any α. `Luna.run` therefore caps `max_dz` (and `init_dz`) at ℓ, and this constant then
   only has to keep exp(α Δz/2) ≤ exp(MAX_αℓ/2) finite. exp(30) ≈ 1e13, and the
   corresponding floor on the profile, exp(-30) ≈ 1e-13, is far below anything physically
   meaningful. =#
const MAX_αℓ = 60.0

"""
    reflength(grid, N, ℓ)

The absorber reference length in metres. `ℓ` wins if given; otherwise `zmax/N`.

Both are validated here so that every call site is protected. A non-positive length is not
merely useless, it is silently destructive: `ℓ == 0` makes [`rate`](@ref) evaluate `0/0` in
the interior and fills the grid with NaN, and `ℓ < 0` inverts the clamp bounds so that
`clamp` returns the (negative) upper bound everywhere — an absorber which amplifies, over
the whole grid rather than just the collar. To switch the absorbers off, use
`boundary=:none`.
"""
function reflength(grid, N, ℓ)
    if isnothing(ℓ)
        (N > 0 && isfinite(N)) || error(
            "boundary_N must be finite and positive, got $N")
        grid.zmax/N
    else
        (ℓ > 0 && isfinite(ℓ)) || error(
            "boundary_length must be finite and positive, got $ℓ")
        float(ℓ)
    end
end

"""
    rate(W, ℓ)

Convert an apodisation profile `W ∈ [0, 1]` (1 in the interior, falling to 0 at the
boundary) into a power absorption coefficient in 1/m, such that propagating through `ℓ`
attenuates the *field* by `W` — that is, `exp(-αℓ/2) == W`, since the historical profiles
multiplied the field. Exactly zero where `W == 1`, and clamped at `MAX_αℓ/ℓ` so that
`W == 0` gives a large but finite coefficient.
"""
rate(W, ℓ) = clamp.(.-2 .* log.(W) ./ ℓ, 0.0, MAX_αℓ/ℓ)

"""
    spectral_rate(grid; N=DEFAULT_N, ℓ=nothing)

Power absorption coefficient in 1/m across the frequency grid, derived from
`grid.ωwin`. Exactly zero outside `grid.sidx`, where the linear operator is zero by
construction and there is nothing left to absorb.
"""
function spectral_rate(grid; N=DEFAULT_N, ℓ=nothing)
    α = rate(grid.ωwin, reflength(grid, N, ℓ))
    α[.!grid.sidx] .= 0
    α
end

"""
    tcollarwidth(grid, collar)

Width (in seconds) of the temporal absorber collar at each end of the time window.

`grid.twin`'s own collar is only the *slack* between the requested `trange` and the
realised power-of-two window, so it can be exactly zero — for a `trange` that already lands
on a power of two, `grid.twin` is identically 1 except for a single sample, i.e. those
grids have no temporal absorber at all. The collar used here is therefore the wider of that
slack and `collar` times the full time window.
"""
function tcollarwidth(grid, collar)
    t = grid.t
    tmin, tmax = extrema(t)
    i1 = findfirst(isequal(1.0), grid.twin)
    i2 = findlast(isequal(1.0), grid.twin)
    slack = (isnothing(i1) || isnothing(i2)) ? 0.0 : min(t[i1] - tmin, tmax - t[i2])
    max(slack, collar*(tmax - tmin))
end

"""
    tprofile(grid; collar=DEFAULT_TCOLLAR)

The temporal absorber profile: the pointwise minimum of `grid.twin` and a Planck taper with
a guaranteed non-degenerate collar (see [`tcollarwidth`](@ref)). Taking the minimum means
the absorber is never weaker than the historical profile, and equals it exactly whenever
the natural collar is already wide enough.

Deliberately not stored on the grid: `Grid.to_dict`/`from_dict` serialise every grid field
and `from_dict` requires them all, so adding one would make existing output files
unreadable.
"""
function tprofile(grid; collar=DEFAULT_TCOLLAR)
    t = grid.t
    tmin, tmax = extrema(t)
    w = tcollarwidth(grid, collar)
    min.(grid.twin, Maths.planck_taper(t, tmin, tmin + w, tmax - w, tmax))
end

"""
    temporal_rate(grid; N=DEFAULT_N, ℓ=nothing, collar=DEFAULT_TCOLLAR)

Power absorption coefficient in 1/m across the time grid.
"""
temporal_rate(grid; N=DEFAULT_N, ℓ=nothing, collar=DEFAULT_TCOLLAR) =
    rate(tprofile(grid; collar), reflength(grid, N, ℓ))

"""
    addloss(linop, α)

Add the power absorption coefficient `α` (a vector over ω) to a linear operator as
`-α/2`, which is how loss enters a linop everywhere in Luna (see `LinearOps`). Returns a
linop of the same kind, so that `RK45.make_prop!` dispatch and `Luna.linoptype` are
unchanged. Handles both forms Luna uses: a materialised `AbstractArray` and a closure
`linop!(out, z)`.

ω is axis 1 for every linop shape — `(Nω,)` mode-averaged, `(Nω, nmodes)` multimode,
`(Nω, q.N)` radial, `(Nω, Nky, Nkx)` free-space — so one broadcast covers all of them.

The array method **allocates a copy** rather than subtracting in place:
`Interface.prop_capillary_args` is documented as being for repeated simulations in an
identical fibre, i.e. the same linop is deliberately reused across `Luna.run` calls, and an
in-place subtraction would compound the absorber on every reuse.
"""
addloss(linop::AbstractArray, α) = linop .- α./2

function addloss(linop!, α)
    function linop_absorbing!(out, z)
        linop!(out, z)
        out .-= α./2
        out
    end
end

"""
    walkoff_check(grid, L, αt, collarwidth; quantile=0.95, threshold=1e-2)

Warn if the temporal absorber is too weak for the group-delay walk-off the simulation can
produce.

The absorber's job in time is to kill energy before it wraps around the periodic time
window. A component walking off at `τ` (s/m) crosses the collar in `collarwidth/τ` metres,
so the relevant figure of merit is the attenuation over that distance. `τ` is read off the
linear operator: `imag(linop) = -(β - β₁ω)`, so `-d imag(L)/dω` is the group delay per unit
length relative to the reference frame. A high quantile over `grid.sidx` is used rather
than the maximum, because β₁ diverges at the band edges.

The default `threshold` is a *field* retention of 1e-2, i.e. 1e-4 in intensity — roughly
where wraparound starts to be visible on a log-scale spectral plot. A tighter threshold
warns about wraparound far below any level that could matter.

This is a diagnostic only — a bad estimate costs a spurious log line, not a wrong answer.
"""
function walkoff_check(grid, L, αt, collarwidth; quantile=0.95, threshold=1e-2)
    ω = grid.ω
    length(ω) > 2 || return nothing
    #= Group delay per unit length, on the ω midpoints; keep only pairs fully inside the
       simulation band, and only finite values. Only the first trailing slice is used — for
       a multimode or free-space linop that is the fundamental mode or zero transverse
       wavevector, which is representative, and scanning the whole array would mean tens of
       millions of points for a large 3-D grid. =#
    τ = Float64[]
    dωs = diff(ω)
    lead = ntuple(_ -> 1, max(ndims(L) - 1, 0))
    for i = 1:length(ω)-1
        (grid.sidx[i] && grid.sidx[i+1]) || continue
        v = abs((imag(L[i+1, lead...]) - imag(L[i, lead...]))/dωs[i])
        isfinite(v) && push!(τ, v)
    end
    isempty(τ) && return nothing
    sort!(τ)
    τq = τ[clamp(ceil(Int, quantile*length(τ)), 1, length(τ))]
    τq > 0 || return nothing
    zcross = collarwidth/τq # distance to walk across the collar
    #= Mean coefficient through the collar, i.e. what such a component actually
       experiences. αt is a power coefficient and `threshold` is a field retention, hence
       the factor of 2. =#
    collar = filter(>(0), αt)
    isempty(collar) && return nothing
    att = exp(-sum(collar)/length(collar)*zcross/2)
    if att > threshold
        Logging.@warn(@sprintf(
            "Temporal absorbing boundary may be too weak: a component walking off at \
             %.3g fs/mm crosses the %.3g fs collar in %.3g mm and is only attenuated by \
             %.1e. Increase boundary_N, widen trange, or increase tcollar.",
            τq*1e15/1e3, collarwidth*1e15, zcross*1e3, att))
    end
    nothing
end

# --------------------------------------------------------------------------- application

#= The three ways a boundary can be applied per accepted step. Each is a functor rather than
   a closure so that `setup` has one return type per mode, the state each needs is named,
   and a new kind of boundary (the spatial windows a free-space simulation wants) is another
   struct rather than another branch inside `Luna.run`. Each ends by calling `output`,
   because that is what `RK45.solve` expects of a `stepfun`. =#

"""
    RateAbsorber(αt, Et, FT, output, z0)

Applies the temporal absorber as `exp(-αt Δz/2)` over the distance actually travelled.
Successive factors telescope to `exp(-αt L/2)` however the solver subdivides the
propagation, which is the whole point of rate semantics. The spectral absorber is not here:
it rides the propagator, having been folded into the linear operator by [`addloss`](@ref).
"""
struct RateAbsorber{tT, fT, oT}
    αt::Vector{Float64}
    tidcs::Vector{Int} # only the collar is ever ≠ 1, so only those need updating each step
    tfac::Vector{Float64}
    Et::tT
    FT::fT
    output::oT
    zprev::Base.RefValue{Float64}
end

RateAbsorber(αt, Et, FT, output, z0) = RateAbsorber(
    αt, findall(>(0), αt), ones(Float64, length(αt)), Et, FT, output, Ref(float(z0)))

function (b::RateAbsorber)(Eω, z, dz, interpolant)
    Δz = z - b.zprev[]
    b.zprev[] = z
    if Δz > 0
        @inbounds for i in b.tidcs
            b.tfac[i] = exp(-b.αt[i]*Δz/2) # αt is a power coefficient, tfac hits the field
        end
        #= An inverse real FFT overwrites its input, so between these two lines Eω holds
           whatever FFTW left there and must not be read; the forward transform refills it. =#
        ldiv!(b.Et, b.FT, Eω)
        b.Et .*= b.tfac
        mul!(Eω, b.FT, b.Et)
    end
    b.output(Eω, z, dz, interpolant)
end

"""
    LegacyAbsorber(grid, Et, FT, output)

The historical scheme: multiply the solution by the fixed profiles once per accepted step.
Kept only so that results from before rate semantics can be reproduced exactly.
"""
struct LegacyAbsorber{gT, tT, fT, oT}
    grid::gT
    Et::tT
    FT::fT
    output::oT
end

function (b::LegacyAbsorber)(Eω, z, dz, interpolant)
    Eω .*= b.grid.ωwin
    ldiv!(b.Et, b.FT, Eω) # destroys Eω, see RateAbsorber
    b.Et .*= b.grid.twin
    mul!(Eω, b.FT, b.Et)
    b.output(Eω, z, dz, interpolant)
end

"""
    NoAbsorber(output)

No boundary at all. The field is still band-limited once at the start of `Luna.run` and the
nonlinear polarisation is still band-limited in `NonlinearRHS`, but nothing stops energy
wrapping around the time window or piling up at the edge of the frequency window.
"""
struct NoAbsorber{oT}
    output::oT
end

(b::NoAbsorber)(Eω, z, dz, interpolant) = b.output(Eω, z, dz, interpolant)

"""
    linop_array(linop, Eω, z)

The linear operator as an array, materialising the z-dependent (closure) form at `z`. The
linop has the same shape as the field, which is how `RK45.make_prop!` allocates its own
buffer.
"""
linop_array(linop::AbstractArray, Eω, z) = linop
function linop_array(linop!, Eω, z)
    out = similar(Eω)
    linop!(out, z)
    out
end

"Report the absorbing-boundary configuration."
function log_setup(grid, ℓ, collar)
    w = tcollarwidth(grid, collar)
    trange = maximum(grid.t) - minimum(grid.t)
    #= The clamp in `rate` is not worth reporting: it bites only where the profile is
       already below exp(-30), and even there the attenuation over the whole propagation is
       exp(-30 zmax/ℓ). What is worth reporting is the strength the user actually chose, so
       quote the attenuation over the propagation at the half-way point of a taper. =#
    Logging.@info(@sprintf(
        "Absorbing boundaries: rate-based, reference length %.3g m (%.3g applications of \
         the window profile over %.3g m; a 50%% point of the taper attenuates by %.1e over \
         the propagation). Temporal collar %.3g fs, %.1f%% of the time window.",
        ℓ, grid.zmax/ℓ, grid.zmax, 0.5^(grid.zmax/ℓ), w*1e15, 100*w/trange))
end

"""
    setup(boundary, grid, linop, Eω, Et, FT, output, z0, max_dz, init_dz;
          N=DEFAULT_N, ℓ=nothing, collar=DEFAULT_TCOLLAR)

Everything `Luna.run` needs in order to apply absorbing boundaries, as a named tuple
`(; stepfun, linop, max_dz, init_dz, ℓ)`. `ℓ` is the reference length actually used, or
`nothing` for the modes which do not have one.

Three of those are returned because setting up an absorber genuinely changes them, and it is
clearer to hand them back than to mutate them from inside a branch:

- `linop` gains the spectral absorber, which the interaction-picture propagator then applies
  exactly (see [`addloss`](@ref)).
- `max_dz` is capped at the reference length `ℓ`, and `init_dz` with it. RK45's `fbar!`
  amplifies the RHS by `exp(+α Δz/2)` while the nonlinear drive it acts on is ∝ `exp(-α ℓ/2)`,
  so the amplified band-edge elements stay bounded — and stay small in `weaknorm`'s global
  sum — only while `Δz ≤ ℓ`. This costs nothing in practice: it requires at least `N` steps
  and real runs take far more.
"""
function setup(boundary, grid, linop, Eω, Et, FT, output, z0, max_dz, init_dz;
               N=DEFAULT_N, ℓ=nothing, collar=DEFAULT_TCOLLAR)
    ℓabs = nothing
    if boundary === :rate
        ℓabs = reflength(grid, N, ℓ)
        if max_dz > ℓabs
            Logging.@info(@sprintf(
                "Reducing max_dz from %.3g m to the absorber reference length %.3g m.",
                max_dz, ℓabs))
            max_dz = ℓabs
        end
        init_dz = min(init_dz, max_dz)
        αt = temporal_rate(grid; N, ℓ, collar)
        walkoff_check(grid, linop_array(linop, Eω, z0), αt, tcollarwidth(grid, collar))
        linop = addloss(linop, spectral_rate(grid; N, ℓ))
        log_setup(grid, ℓabs, collar)
        stepfun = RateAbsorber(αt, Et, FT, output, z0)
    elseif boundary === :legacy
        Logging.@warn(
            "boundary=:legacy applies the absorbing boundaries once per accepted step, " *
            "so the absorption depends on the step count and the result depends on rtol.")
        stepfun = LegacyAbsorber(grid, Et, FT, output)
    elseif boundary === :none
        stepfun = NoAbsorber(output)
    else
        error("boundary must be :rate, :legacy or :none, not $boundary")
    end
    (; stepfun, linop, max_dz, init_dz, ℓ=ℓabs)
end

end
