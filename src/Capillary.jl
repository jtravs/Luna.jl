module Capillary
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import StaticArrays: SVector
import Cubature: hquadrature
using Reexport
@reexport using Luna.Modes
import Luna: Maths, Grid, LinearOps, RK45, Utils
import Luna.PhysData: c, ε_0, μ_0, ref_index_fun, roomtemp, densityspline, sellmeier_gas, wlfreq
import Luna.Modes: AbstractMode, dimlimits, neff, field, Aeff, N, modeinfo,
                   zconstant, scale_invariant, azimuthal_order
import Luna.LinearOps: make_linop, conj_clamp, neff_grid, neff_β_grid
import Luna.PhysData: wlfreq, roomtemp
import Luna.Utils: subscript
import Base: show

export MarcatiliMode, dimlimits, neff, field, N, Aeff

"""
    MarcatiliMode

Type representing a mode of a hollow capillary as presented in:

Marcatili, E. & Schmeltzer, R.
"Hollow metallic and dielectric waveguides for long distance optical transmission and lasers
(Long distance optical transmission in hollow dielectric and metal circular waveguides,
examining normal mode propagation)."
Bell System Technical Journal 43, 1783–1809 (1964).
"""
struct MarcatiliMode{Ta, Tcore, Tclad, LT} <: AbstractMode
    a::Ta # core radius callable as function of z only, or fixed core radius if a Number
    n::Int # azimuthal mode index
    m::Int # radial mode index
    kind::Symbol # kind of mode (transverse magnetic/electric or hybrid)
    unm::Float64 # mth zero of the nth Bessel function of the first kind
    ϕ::Float64 # overall rotation angle of the mode
    coren::Tcore # callable, returns (possibly complex) core ref index as function of ω
    cladn::Tclad # callable, returns (possibly complex) cladding ref index as function of ω
    model::Symbol # if :full, includes complete influence of complex cladding ref index
    loss::LT # Val{true}() or Val{false}() - whether to include the loss
    aeff_intg::Float64 # Pre-calculated integral fraction for effective area
end

function show(io::IO, m::MarcatiliMode)
    a = radius_string(m)
    loss = "loss=" * (m.loss == Val(true) ? "true" : "false")
    model = "model="*string(m.model)
    angle = "ϕ=$(m.ϕ/π)π"
    out = "MarcatiliMode{"*join([mode_string(m), a, loss, model, angle], ", ")*"}"
    print(io, out)
end

mode_string(m::MarcatiliMode) = string(m.kind)*subscript(m.n)*subscript(m.m)
radius_string(m::MarcatiliMode{<:Number, Tco, Tcl, LT}) where {Tco, Tcl, LT} = "a=$(m.a)"
radius_string(m::MarcatiliMode) = "a(z=0)=$(radius(m, 0))"

modeinfo(m::MarcatiliMode) = Dict(:kind => m.kind, :n => m.n, :m => m.m,
                                  :radius => radius(m, 0), :ϕ => m.ϕ, :model => m.model,
                                  :loss => m.loss == Val(true))

"""
    MarcatiliMode(a, n, m, kind, ϕ, coren, cladn; model=:full, loss=true)

Create a MarcatiliMode.

# Arguments
- `a` : Either a `Number` for constant core radius, or a function `a(z)` for variable radius.
- `n::Int` : Azimuthal mode index (number of nodes in the field along azimuthal angle).
- `m::Int` : Radial mode index (number of nodes in the field along radial coordinate).
- `kind::Symbol` : `:TE` for transverse electric, `:TM` for transverse magnetic,
                   `:HE` for hybrid mode.
- `ϕ::Float` : Azimuthal offset angle (for linearly polarised modes, this is the angle
                between the mode polarisation and the `:y` axis)
- `coren` : Callable `coren(ω; z)` which returns the refractive index of the core
- `cladn` : Callable `cladn(ω; z)` which returns the refractive index of the cladding
- `model::Symbol=:full` : If `:full`, use the complete Marcatili model which takes into
                          account the dispersive influence of the cladding refractive index.
                          If `:reduced`, use the simplified model common in the literature
- `loss::Bool=true` : Whether to include loss.

"""
function MarcatiliMode(a, n, m, kind, ϕ, coren, cladn; model=:full, loss=true)
    # chkzkwarg makes sure that coren and cladn take z as a keyword argument
    aeff_intg = Aeff_Jintg(n, get_unm(n, m, kind), kind)
    MarcatiliMode(a, n, m, kind, get_unm(n, m, kind), ϕ,
                   chkzkwarg(coren), chkzkwarg(cladn),
                   model, Val(loss), aeff_intg)
end

"""
    MarcatiliMode(a, gas, P; kwargs...)

Create a MarcatiliMode for a capillary with radius `a` which is filled with `gas` to
pressure `P`.
"""
function MarcatiliMode(a, gas, P;
                        n=1, m=1, kind=:HE, ϕ=0.0, T=roomtemp, model=:full,
                        clad=:SiO2, loss=true)
    rfg = ref_index_fun(gas, P, T)
    rfs = ref_index_fun(clad)
    coren = ConstIndex(rfg)
    cladn = ConstIndex(rfs)
    MarcatiliMode(a, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end

"""
    MarcatiliMode(a, gas, P, cladn; kwargs...)

Create a MarcatiliMode for a capillary made of a cladding material defined by the refractive
index `cladn(ω; z)` with a core radius `a` which is filled with `gas` to pressure `P`.
"""
function MarcatiliMode(a, gas, P, cladn;
                        n=1, m=1, kind=:HE, ϕ=0.0, T=roomtemp, model=:full, loss=true)
    rfg = ref_index_fun(gas, P, T)
    coren = ConstIndex(rfg)
    MarcatiliMode(a, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end

"""
    MarcatiliMode(a, coren; kwargs...)

Create a MarcatiliMode for a capillary with radius `a` with `z`-dependent gas fill determined
by `coren(ω; z)`.
"""
function MarcatiliMode(a, coren;
                        n=1, m=1, kind=:HE, ϕ=0.0, model=:full, clad=:SiO2, loss=true)
    rfs = ref_index_fun(clad)
    cladn = ConstIndex(rfs)
    MarcatiliMode(a, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end


"""
    MarcatiliMode(a; kwargs...)

Create a `MarcatiliMode` for a capillary with radius `a` and no gas fill.
"""
MarcatiliMode(a; kwargs...) = MarcatiliMode(a, ConstIndex(λ -> 1); kwargs...)

"""
    neff(m::MarcatiliMode, ω; z=0)

Calculate the complex effective index of Marcatili mode with dielectric core and arbitrary
(metal or dielectric) cladding.

Adapted from:

Marcatili, E. & Schmeltzer, R.
"Hollow metallic and dielectric waveguides for long distance optical transmission and lasers
(Long distance optical transmission in hollow dielectric and metal circular waveguides,
examining normal mode propagation)."
Bell System Technical Journal 43, 1783–1809 (1964).
"""
function neff(m::MarcatiliMode, ω; z=0)
    εcl = m.cladn(ω, z=z)^2
    εco = m.coren(ω, z=z)^2
    vn = get_vn(εcl, m.kind)
    neff(m, ω, εco, vn, radius(m, z))
end

# The Marcatili effective index of one mode at one frequency from the core permittivity
# `εco`, the cladding term `vn` (see get_vn), the Bessel zero `unm` and the radius `a`, for
# the four combinations of loss (Val) and model (Val). These scalar kernels are the single
# source of the arithmetic: the mode-level `neff` methods below call them, and so does the
# broadcast of `LinearOps.MarcatiliLinop`, which evaluates all modes and frequencies at once
# (on the host or a device) and is therefore bit-identical to the per-point evaluation.
@inline function _neff_scalar(::Val{true}, ::Val{:full}, unm, ω, εco, vn, a)
    k = ω/c
    n = sqrt(complex(εco - (unm/(k*a))^2*(1 - im*vn/(k*a))^2))
    return (real(n) < 1e-3) ? (1e-3 + im*clamp(imag(n), 0, Inf)) : n
end
@inline function _neff_scalar(::Val{true}, ::Val{:reduced}, unm, ω, εco, vn, a)
    return ((1 + (εco - 1)/2 - c^2*unm^2/(2*ω^2*a^2))
                + im*(c^3*unm^2)/(a^3*ω^3)*vn)
end
@inline function _neff_scalar(::Val{false}, ::Val{:full}, unm, ω, εco, vn, a)
    k = ω/c
    n = real(sqrt(εco - (unm/(k*a))^2*(1 - im*vn/(k*a))^2))
    return (n < 1e-3) ? 1e-3 : n
end
@inline function _neff_scalar(::Val{false}, ::Val{:reduced}, unm, ω, εco, vn, a)
    return real(1 + (εco - 1)/2 - c^2*unm^2/(2*ω^2*a^2))
end

# Dispatch on loss to make neff type stable
# m.loss = Val{true}() (returns ComplexF64)
function neff(m::MarcatiliMode{Ta, Tco, Tcl, Val{true}}, ω, εco, vn, a) where {Ta, Tcl, Tco}
    if m.model == :full
        return _neff_scalar(Val(true), Val(:full), m.unm, ω, εco, vn, a)
    elseif m.model == :reduced
        return _neff_scalar(Val(true), Val(:reduced), m.unm, ω, εco, vn, a)
    else
        error("model must be :full or :reduced")
    end
end

# m.loss = Val{false}() (returns Float64)
function neff(m::MarcatiliMode{Ta, Tco, Tcl, Val{false}}, ω, εco, vn, a) where {Ta, Tcl, Tco}
    if m.model == :full
        return _neff_scalar(Val(false), Val(:full), m.unm, ω, εco, vn, a)
    elseif m.model == :reduced
        return _neff_scalar(Val(false), Val(:reduced), m.unm, ω, εco, vn, a)
    else
        error("model must be :full or :reduced")
    end
end

function neff_wg(m::MarcatiliMode{Ta, Tco, Tcl, Val{true}}, ω; z=0) where {Ta, Tcl, Tco}
    εcl = m.cladn(ω, z=z)^2
    vn = get_vn(εcl, m.kind)
    a = radius(m, z)
    if m.model == :full
        k = ω/c
        return (m.unm/(k*a))^2*(1 - im*vn/(k*a))^2
    elseif m.model == :reduced
        # `neff(m, εco, nwg)` forms `1 + (εco - 1)/2 - nwg`, so the imaginary (loss) part
        # must enter with the opposite sign to the one in `_neff_scalar`, which adds it:
        # (1 + (εco - 1)/2 - re) + im*x  ==  1 + (εco - 1)/2 - (re - im*x)
        return c^2*m.unm^2/(2*ω^2*a^2) - im*(c^3*m.unm^2)/(a^3*ω^3)*vn
    else
        error("model must be :full or :reduced")
    end
end

function neff_wg(m::MarcatiliMode{Ta, Tco, Tcl, Val{false}}, ω; z=0) where {Ta, Tcl, Tco}
    εcl = m.cladn(ω, z=z)^2
    vn = get_vn(εcl, m.kind)
    a = radius(m, z)
    if m.model == :full
        k = ω/c
        return (m.unm/(k*a))^2*(1 - im*vn/(k*a))^2
    elseif m.model == :reduced
        return c^2*m.unm^2/(2*ω^2*a^2)
    else
        error("model must be :full or :reduced")
    end
end

function neff(m::MarcatiliMode{Ta, Tco, Tcl, Val{true}}, εco, nwg) where {Ta, Tcl, Tco}
    if m.model == :full
        return sqrt(complex(εco - nwg))
    elseif m.model == :reduced
        return complex((1 + (εco - 1)/2 - nwg))
    else
        error("model must be :full or :reduced")
    end
end

function neff(m::MarcatiliMode{Ta, Tco, Tcl, Val{false}}, εco, nwg) where {Ta, Tcl, Tco}
    if m.model == :full
        return real(sqrt(complex(εco - nwg)))
    elseif m.model == :reduced
        return real((1 + (εco - 1)/2 - nwg))
    else
        error("model must be :full or :reduced")
    end
end

function get_vn(εcl, kind)
    if kind == :HE
        (εcl + 1)/(2*sqrt(complex(εcl - 1)))
    elseif kind == :TE
        1/sqrt(complex(εcl - 1))
    elseif kind == :TM
        εcl/sqrt(complex(εcl - 1))
    else
        error("kind must be :TE, :TM or :HE")
    end
end

function get_unm(n, m, kind)
    if (kind == :TE) || (kind == :TM)
        if (n != 0)
            error("n=0 for TE or TM modes")
        end
        besselj_zero(1, m)
    elseif kind == :HE
        if n == 0
            error("n ≠ 0 for HE modes")
        end
        besselj_zero(n-1, m)
    else
        error("kind must be :TE, :TM or :HE")
    end
end

radius(m::MarcatiliMode{<:Number, Tco, Tcl, LT}, z) where {Tcl, Tco, LT} = m.a
radius(m::MarcatiliMode, z) = m.a(z)

dimlimits(m::MarcatiliMode; z=0) = (:polar, (0.0, 0.0), (radius(m, z), 2π))

# geometry traits used by the fixed-quadrature modal transform: a numeric radius means the
# transverse profile never changes; a tapered radius only rescales it (see Modes.scale_invariant)
zconstant(m::MarcatiliMode{<:Number, Tco, Tcl, LT}) where {Tcl, Tco, LT} = true
scale_invariant(m::MarcatiliMode) = true
azimuthal_order(m::MarcatiliMode) = m.kind == :HE ? m.n - 1 : 1

# we use polar coords, so xs = (r, θ)
function field(m::MarcatiliMode, xs; z=0)
    r, θ = xs
    if m.kind == :HE
        return (besselj(m.n-1, r*m.unm/radius(m, z)) .* SVector(
            cos(θ)*sin(m.n*(θ + m.ϕ)) - sin(θ)*cos(m.n*(θ + m.ϕ)),
            sin(θ)*sin(m.n*(θ + m.ϕ)) + cos(θ)*cos(m.n*(θ + m.ϕ))
            ))
    elseif m.kind == :TE
        return besselj(1, r*m.unm/radius(m, z)) .* SVector(-sin(θ), cos(θ))
    elseif m.kind == :TM
        return besselj(1, r*m.unm/radius(m, z)) .* SVector(cos(θ), sin(θ))
    end
end

function N(m::MarcatiliMode; z=0)
    np1 = (m.kind == :HE) ? m.n : 2
    π/2 * radius(m, z)^2 * besselj(np1, m.unm)^2 * sqrt(ε_0/μ_0)
end

function Aeff_Jintg(n, unm, kind)
    den, err = hquadrature(r -> r*besselj(n-1, unm*r)^4, 0, 1)
    np1 = (kind == :HE) ? n : 2
    num = 1/4 * besselj(np1, unm)^4
    return 2π*num/den
end

Aeff(m::MarcatiliMode; z=0) = radius(m, z)^2 * m.aeff_intg


"""
    ConstIndex(rf)

A ``z``-independent refractive index `rf(λ)`, callable as `(ω; z)` like every core/cladding
index function of a [`MarcatiliMode`](@ref). The standard constructors use it for a
constant gas fill and for the cladding, so that the linear operator can *recognise* the
``z``-independence and precompute the index once (see `LinearOps.MarcatiliLinop`); a
user-supplied closure is treated as ``z``-dependent.
"""
struct ConstIndex{F}
    rf::F
end
(c::ConstIndex)(ω; z=0.0) = c.rf(wlfreq(ω))

"""
    GradientCoreIndex(γ, dens)

The core refractive index ``\\sqrt{1 + γ(λ)ρ(z)}`` of a gas fill with polarisability
`γ(λ[µm])` (per unit density, `PhysData.sellmeier_gas`) and density profile
`dens(z)`, callable as `(ω; z)`. Returned by [`gradient`](@ref); the linear operator uses
its structure — ``z`` enters only through the scalar `dens(z)` — to evaluate the modes'
effective indices as one broadcast per step (see `LinearOps.MarcatiliLinop`).
"""
struct GradientCoreIndex{G, D}
    γ::G
    dens::D
end
(c::GradientCoreIndex)(ω; z=0.0) = sqrt(1 + c.γ(wlfreq(ω)*1e6)*c.dens(z))

"""
    gradient(gas, L, p0, p1; T=roomtemp)

Convenience function to create density and core index profiles for
simple two-point gradient fills defined by the waveguide length `L` and the pressures at
`z=0` and `z=L`. Returns `(coren, dens)` with `coren` a [`GradientCoreIndex`](@ref).
"""
function gradient(gas, L, p0, p1; T=roomtemp)
    γ = sellmeier_gas(gas)
    dspl = densityspline(gas, Pmin=p0==p1 ? 0 : min(p0, p1), Pmax=max(p0, p1); T)
    p(z) =  z > L ? p1 :
            z <= 0 ? p0 :
            sqrt(p0^2 + z/L*(p1^2 - p0^2))
    dens(z) = dspl(p(z))
    return GradientCoreIndex(γ, dens), dens
end

"""
    gradient(gas, Z, P; T=roomtemp)

Convenience function to create density and core index profiles for
multi-point gradient fills defined by positions `Z` and pressures `P`.
"""
function gradient(gas, Z, P; T=roomtemp)
    γ = sellmeier_gas(gas)
    ex = extrema(P)
    dspl = densityspline(gas, Pmin=ex[1]==ex[2] ? 0 : ex[1], Pmax=ex[2]; T)
    function p(z)
        if z <= Z[1]
            return P[1]
        elseif z >= Z[end]
            return P[end]
        else
            i = findlast(x -> x < z, Z)
            return sqrt(P[i]^2 + (z - Z[i])/(Z[i+1] - Z[i])*(P[i+1]^2 - P[i]^2))
        end
    end
    dens(z) = dspl(p(z))
    return GradientCoreIndex(γ, dens), dens
end

#= Avoid repeated calculation of the waveguide part of the effective index for modes with
    constant core radius.
    This is used by LinearOps.make_linop =#
function neff_β_grid(grid,
                   mode::MarcatiliMode{<:Number, Tco, Tcl, LT} where {Tco, Tcl, LT},
                   λ0)
    nwg = complex(zero(grid.ω))
    sidcs = (1:length(grid.ω))[grid.sidx]
    for iω in sidcs
        nwg[iω] = neff_wg(mode, grid.ω[iω]; z=0)
    end
    _neff = let nwg=nwg, ω=grid.ω, mode=mode
        _neff(iω; z) = neff(mode, mode.coren(ω[iω], z=z)^2, nwg[iω])
    end
    _β = let nwg=nwg, ω=grid.ω, _neff=_neff
        _β(iω; z) = ω[iω]/c*real(_neff(iω; z=z))
    end
    _neff, _β
end

# Collection of modes with fixed core radius
FixedCoreCollection = Union{
    Tuple{Vararg{MarcatiliMode{<:Number, Tco, Tcl, LT}} where {Tco, Tcl, LT}},
    AbstractArray{MarcatiliMode{<:Number, Tco, Tcl, LT} where {Tco, Tcl, LT}}
    }

function neff_grid(grid, modes::FixedCoreCollection, λ0; ref_mode=1)
    nwg = Array{ComplexF64, 2}(undef, (length(grid.ω), length(modes)))
    sidcs = (1:length(grid.ω))[grid.sidx]
    for (i, mi) in enumerate(modes)
        for iω in sidcs
            nwg[iω, i] = neff_wg(mi, grid.ω[iω]; z=0)
        end
    end
    _neff = let nwg=nwg, ω=grid.ω, modes=modes
        _neff(iω, iim; z) = neff(modes[iim], modes[iim].coren(ω[iω], z=z)^2, nwg[iω, iim])
    end
    _neff
end

"""
    transmission(a, λ, L; kind=:HE, n=1, m=1)

Calculate the transmission through a capillary with core radius `a` and length `L` at the
wavelength `λ` when propagating the `MarcatiliMode` defined by `kind`, `n` and `m`.
"""
function transmission(a, λ, L; kind=:HE, n=1, m=1)
    # TODO hardcoded fill needs to be updated if using absorbing materials
    mode = MarcatiliMode(a, :He, 0; n=n, m=m, kind=kind)
    Modes.transmission(mode, wlfreq(λ), L)
end


#=================================================#
#======  z-DEPENDENT LINEAR OPERATOR (fast)  =====#
#=================================================#

const MarcatiliCollection = Union{AbstractVector{<:MarcatiliMode}, Tuple{Vararg{MarcatiliMode}}}

"""
    marcatili_linop_ok(modes) -> Bool

Whether [`MarcatiliLinop`](@ref) can represent the ``z``-dependent linear operator of
`modes`: every mode is a `MarcatiliMode` (not a wrapper), with a [`ConstIndex`](@ref)
cladding, a [`ConstIndex`](@ref) or [`GradientCoreIndex`](@ref) core (all of one kind), and
all modes share the same `model` and `loss` setting.
"""
function marcatili_linop_ok(modes)
    ms = collect(modes)
    isempty(ms) && return false
    all(m -> m isa MarcatiliMode, ms) || return false
    all(m -> m.cladn isa ConstIndex, ms) || return false
    all(m -> m.coren isa ConstIndex, ms) || all(m -> m.coren isa GradientCoreIndex, ms) ||
        return false
    allequal(m.model for m in ms) || return false
    allequal(typeof(m.loss) for m in ms) || return false
    return true
end

# the fast operator where it applies, the generic host closure otherwise
function LinearOps.make_linop(grid::Grid.RealGrid, modes::MarcatiliCollection, λ0; ref_mode=1)
    marcatili_linop_ok(modes) || return LinearOps._make_linop_generic(grid, modes, λ0; ref_mode)
    MarcatiliLinop(grid, modes, λ0; ref_mode)
end

function LinearOps.make_linop(grid::Grid.EnvGrid, modes::MarcatiliCollection, λ0;
                              ref_mode=1, thg=false)
    marcatili_linop_ok(modes) || return LinearOps._make_linop_generic(grid, modes, λ0; ref_mode, thg)
    MarcatiliLinop(grid, modes, λ0; ref_mode, thg)
end

"""
    MarcatiliLinop

``z``-dependent modal linear operator for a collection of [`MarcatiliMode`](@ref)s,
callable as `linop!(out, z)` like the closure returned by the generic
`LinearOps.make_linop` and giving bit-identical results, but evaluated as **one broadcast**
over all frequencies and modes. ``z`` enters the Marcatili effective index only through two
scalars per mode — the core radius ``a(z)`` (tapers) and, for a [`GradientCoreIndex`](@ref),
the density ``ρ(z)`` in ``ε_{\\rm core} = 1 + γ(ω)ρ(z)`` (pressure gradients) — so the
``ω``-dependent parts are precomputed once (the cladding term ``v_n(ω)`` per mode, ``γ(ω)``
or ``ε_{\\rm core}(ω)`` per mode, ``u_{nm}``) and each call costs the host-side scalars
(``a(z)``, ``ρ(z)``, the reference mode's ``β_1(z)``) plus a single elementwise kernel
(`_neff_scalar`, the same arithmetic as `neff`). Because the kernel is a broadcast, the
operator writes directly into a **device** array (`RK45.device_capable`), with no host loop
and no per-stage upload: the precomputed arrays are moved to the array type of `out` on
the first call with a device array (and the operator can then no longer be evaluated
into a host array).

The generic operator costs ~0.8 ms per stage for 4 modes on a 4097-point grid (a serial
`Modes.neff` loop re-evaluating the gas and cladding indices per mode) — ~45 % of an RK45
step of the RDW gradient example, and the cap on the step time of device runs on fast
GPUs; this one costs tens of µs on the host and a few µs on a device.
"""
mutable struct MarcatiliLinop{mT, lossT, modelT}
    modes::mT
    ref_mode::Int
    ω0::Float64 # wlfreq(λ0), for β1 (and βref)
    ωc::Float64 # grid.ω0 for an EnvGrid (the carrier), 0 for a RealGrid
    envelope::Bool
    thg::Bool
    gradient::Bool # core holds γ(ω) (εco = sqrt(1 + γρ)^2) rather than εco(ω)
    loss::lossT # Val
    model::modelT # Val
    ondevice::Bool # whether the arrays below live on a device
    ω::Any # (nω) frequency grid
    mask::Any # (nω) Bool: grid.sidx
    unm::Any # (1, M)
    vn::Any # (nω, M) cladding term per mode
    core::Any # (nω, M) εco(ω) or γ(ω) per mode
    arow::Any # (1, M) core radius per mode at the current z
    ρrow::Any # (1, M) density per mode at the current z (gradients; 1 otherwise)
    arow_h::Vector{Float64} # host staging for arow/ρrow
    ρrow_h::Vector{Float64}
end

RK45.device_capable(::MarcatiliLinop) = true

function MarcatiliLinop(grid::Grid.AbstractGrid, modes, λ0; ref_mode=1, thg=false)
    ms = collect(modes)
    M = length(ms)
    ω = grid.ω
    nω = length(ω)
    mask = collect(grid.sidx)
    unm = reshape([Float64(m.unm) for m in ms], 1, M)
    # cladding term per mode: εcl(ω) -> vn (kind-dependent), z-independent by construction
    vn = Matrix{ComplexF64}(undef, nω, M)
    for (i, m) in enumerate(ms), iω in 1:nω
        vn[iω, i] = mask[iω] ? get_vn(m.cladn(ω[iω], z=0.0)^2, m.kind) : 0.0
    end
    gradient = ms[1].coren isa GradientCoreIndex
    # core: γ(ω) per mode for gradients (εco(ω, z) = sqrt(1 + γ ρ(z))^2 as in
    # GradientCoreIndex), or εco(ω) = coren(ω)^2 for constant fills
    iref = findfirst(mask)
    coreT = gradient ? Float64 : promote_type((typeof(m.coren(ω[iref], z=0.0)^2) for m in ms)...)
    core = zeros(coreT, nω, M)
    for (i, m) in enumerate(ms), iω in 1:nω
        mask[iω] || continue
        core[iω, i] = gradient ? m.coren.γ(wlfreq(ω[iω])*1e6) : m.coren(ω[iω], z=0.0)^2
    end
    envelope = grid isa Grid.EnvGrid
    ωc = envelope ? grid.ω0 : 0.0
    MarcatiliLinop(ms, ref_mode, wlfreq(λ0), ωc, envelope, thg, gradient,
                   ms[1].loss, Val(ms[1].model), false,
                   copy(ω), mask, unm, vn, core, ones(1, M), ones(1, M),
                   ones(M), ones(M))
end

# move the precomputed arrays to the array type of `out` (once)
function _adapt_to!(l::MarcatiliLinop, out)
    dev(x) = copyto!(similar(out, eltype(x), size(x)), x)
    l.ω = dev(l.ω); l.mask = dev(l.mask); l.unm = dev(l.unm)
    l.vn = dev(l.vn); l.core = dev(l.core); l.arow = dev(l.arow); l.ρrow = dev(l.ρrow)
    l.ondevice = true
    nothing
end

@inline _εco(core, ρ, ::Val{true}) = sqrt(1 + core*ρ)^2 # cf. GradientCoreIndex
@inline _εco(core, ρ, ::Val{false}) = core

# one element of the operator (see LinearOps._make_linop_generic for the reference form)
@inline function _marcatili_element(mask, ω, unm, core, vn, a, ρ, β1, βref, ωc, thg,
                                    loss, model, grad)
    mask || return zero(ComplexF64)
    n = _neff_scalar(loss, model, unm, ω, _εco(core, ρ, grad), vn, a)
    nc = LinearOps.conj_clamp(n, ω)
    v = -im*(ω/c*nc - (ω - ωc)*β1)
    thg ? v : v - (-im*βref)
end

function (l::MarcatiliLinop)(out, z)
    if Utils.isdevice(out) && !l.ondevice
        _adapt_to!(l, out)
    elseif !Utils.isdevice(out) && l.ondevice
        error("MarcatiliLinop: this operator has been used with a device array and cannot "*
              "be evaluated into a host array afterwards")
    end
    ms = l.modes
    β1 = dispersion(ms[l.ref_mode], 1, l.ω0, z=z)::Float64
    # "thg" here means "no carrier subtraction": a RealGrid never subtracts, an EnvGrid
    # only when thg=false
    thg = !l.envelope || l.thg
    βref = thg ? 0.0 : β(ms[l.ref_mode], l.ω0, z=z)
    for (i, m) in enumerate(ms)
        l.arow_h[i] = radius(m, z)
        l.ρrow_h[i] = l.gradient ? m.coren.dens(z) : 1.0
    end
    if l.ondevice
        copyto!(l.arow, l.arow_h); copyto!(l.ρrow, l.ρrow_h)
    else
        l.arow .= reshape(l.arow_h, 1, :); l.ρrow .= reshape(l.ρrow_h, 1, :)
    end
    _marcatili_broadcast!(out, l.mask, l.ω, l.unm, l.core, l.vn, l.arow, l.ρrow, β1, βref,
                          l.ωc, thg, l.loss, l.model, Val(l.gradient))
    out
end

# function barrier: the array fields of the struct are loosely typed
function _marcatili_broadcast!(out, mask, ω, unm, core, vn, arow, ρrow, β1, βref, ωc, thg,
                               loss, model, grad)
    @. out = _marcatili_element(mask, ω, unm, core, vn, arow, ρrow, β1, βref, ωc, thg,
                                loss, model, grad)
    out
end

end
