module Capillary
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import StaticArrays: SVector
using Reexport
@reexport using Luna.Modes
import Luna: Maths, Grid
import Luna.PhysData: c, ref_index_fun, roomtemp
import Luna.Modes: AbstractMode, dimlimits, neff, field

export MarcatilliMode, dimlimits, neff, field

#= dimlimits() and field() are the same for on-grid and off-grid modes =#
dimlimits(m::Union{MarcatilliMode, OnGridMarcatilliMode}) = (:polar, (0.0, 0.0), (m.a, 2π))

# we use polar coords, so xs = (r, θ)
function field(m::Union{MarcatilliMode, OnGridMarcatilliMode})
    if m.kind == :HE
        return (xs) -> besselj(m.n-1, xs[1]*m.unm/m.a) .* SVector(
            cos(xs[2])*sin(m.n*(xs[2] + m.ϕ)) - sin(xs[2])*cos(m.n*(xs[2] + m.ϕ)),
            sin(xs[2])*sin(m.n*(xs[2] + m.ϕ)) + cos(xs[2])*cos(m.n*(xs[2] + m.ϕ))
            )
    elseif m.kind == :TE
        return (xs) -> besselj(1, xs[1]*m.unm/m.a) .* SVector(-sin(xs[2]), cos(xs[2]))
    elseif m.kind == :TM
        return (xs) -> besselj(1, xs[1]*m.unm/m.a) .* SVector(cos(xs[2]), sin(xs[2]))
    end
end

"Marcatili mode without a grid"
struct MarcatilliMode{Tcore, Tclad}
    a::Float64
    n::Int
    m::Int
    kind::Symbol
    unm::Float64
    ϕ::Float64
    coren::Tcore # callable, returns (possibly complex) core ref index as function of ω
    cladn::Tclad # callable, returns (possibly complex) cladding ref index as function of ω
    model::Symbol
end

function MarcatilliMode(a, n, m, kind, ϕ, coren, cladn; model=:full)
    MarcatilliMode(a, n, m, kind, get_unm(n, m, kind), ϕ, coren, cladn, model)
end

"convenience constructor assunming single gas filling"
function MarcatilliMode(a, gas, P; n=1, m=1, kind=:HE, ϕ=0.0, T=roomtemp, model=:full, clad=:SiO2)
    rfg = ref_index_fun(gas, P, T)
    rfs = ref_index_fun(clad)
    coren = ω -> rfg(2π*c./ω)
    cladn = ω -> rfs(2π*c./ω)
    MarcatilliMode(a, n, m, kind, ϕ, coren, cladn, model=model)
end

"complex effective index of Marcatilli mode with dielectric core and arbitrary
 (metal or dielectric) cladding.

Adapted from
Marcatili, E. & Schmeltzer, R.
Hollow metallic and dielectric waveguides for long distance optical transmission and lasers
(Long distance optical transmission in hollow dielectric and metal circular waveguides,
examining normal mode propagation).
Bell System Technical Journal 43, 1783–1809 (1964).
"
function neff(m::MarcatilliMode, ω)
    εcl = m.cladn(ω)^2
    εco = m.coren(ω)^2
    vn = get_vn(εcl, m.kind)
    k = ω/c
    if m.model == :full
        sqrt(Complex(εco - (m.unm/(k*m.a))^2*(1 - im*vn/(k*m.a))^2))
    elseif m.model == :reduced
        (1 + (εco - 1)/2 - c^2*m.unm^2/(2*ω^2*m.a^2)) + im*(c^3*m.unm^2)/(m.a^3*ω^3)*vn
    else
        error("model must be :full or :reduced")
    end 
end

"Marcatili mode with a grid pre-specified for speed"
struct OnGridMarcatilliMode{gT, dT} <: MarcatilliMode
    grid::gT
    a::Float64
    n::Int
    m::Int
    kind::Symbol
    unm::Float64
    ϕ::Float64
    model::Symbol
    densityfun::dT # callable, returns density as function of z (propagation direction)
    γco::Array{ComplexF64, 1} # Polarisability (χ1 of a single particle) of core
    neff_wg::Array{ComplexF64, 1} # Pre-calculated waveguide contribution to neff
end

function OnGridMarcatilliMode(grid::Grid.AbstractGrid, a, n, m, kind, ϕ, coren, cladn;
                        model=:full)
    unm = get_unm(n, m, kind)
    εcl = @. cladn(grid.ω)^2
    γco = @. coren(grid.ω)^2 - 1
    dens(z) = 1
    vn = get_vn.(εcl, kind)
    k = grid.ω./c
    if model == :full
        neff_wg = @. -(unm/(k*a))^2*(1 - im*vn/(k*a))^2
    elseif model == :reduced
        neff_wg = @. 1 - c^2*unm^2/(2*grid.ω^2*a^2) + im*(c^3*unm^2)/(m.a^3*grid.ω^3)*vn
    else
        error("model must be :full or :reduced")
    end 
    OnGridMarcatilliMode(grid, a, n, m, kind, unm, ϕ, model, dens,
                         complex(γco), complex(neff_wg))
end

"convenience constructor assunming single gas filling"
function OnGridMarcatilliMode(grid::Grid.AbstractGrid, a, gas, P;
                        n=1, m=1, kind=:HE, ϕ=0.0, T=roomtemp, model=:full, clad=:SiO2)
    rfg = ref_index_fun(gas, P, T)
    rfs = ref_index_fun(clad)
    coren = ω -> rfg(2π*c./ω)
    cladn = ω -> rfs(2π*c./ω)
    OnGridMarcatilliMode(grid, a, n, m, kind, ϕ, coren, cladn, model=model)
end

function neff(m::OnGridMarcatilliMode, z=0)
    if m.model == :full
        @. sqrt(Complex(m.densityfun(z)*(m.γco + 1) + m.neff_wg))
    elseif m.model == :reduced
        @. m.densityfun(z)*m.γco/2 + m.neff_wg
    else
        error("model must be :full or :reduced")
    end
end

function get_vn(εcl, kind)
    if kind == :HE
        (εcl + 1)/(2*sqrt(Complex(εcl - 1)))
    elseif kind == :TE
        1/sqrt(Complex(εcl - 1))
    elseif kind == :TM
        εcl/sqrt(Complex(εcl - 1))
    else
        error("kind must be :TE, :TM or :HE")
    end
end

function get_unm(n, m, kind)
    if (kind == :TE) || (kind == :TM)
        if (n != 0) || (m != 1)
            error("n=0, m=1 for TE or TM modes")
        end
        besselj_zero(1, 1)
    elseif kind == :HE
        besselj_zero(n-1, m)
    else
        error("kind must be :TE, :TM or :HE")
    end
end

end
