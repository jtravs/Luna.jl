module LinearOps
import FFTW
import Hankel
import Adapt
import Luna: Modes, Grid, PhysData, Maths, RK45, Utils
import Luna.PhysData: wlfreq

#=================================================#
#===============    FREE SPACE     ===============#
#=================================================#

# Single-element free-space linear operator: shared between the materialising
# _fill_linop_xy! fill loops and the lazy FactoredFreeLinop, so the two can never drift
# apart (they are bit-identical by construction).
@inline function _linop_xy_element(ω, β1, k2ω, kperp2i)
    βsq = k2ω - kperp2i
    if βsq < 0
        # negative βsq -> evanescent fields -> attenuation
        return -im*(-β1*ω) - min(sqrt(abs(βsq)), 200)
    else
        return -im*(sqrt(βsq) - β1*ω)
    end
end

"""
    FactoredFreeLinop

Lazy constant linear operator for free-space 3D propagation. Stores only the separable
factors — `k²(ω)` (a vector) and `k⊥²(ky, kx)` (a small matrix) — instead of the full
`(nω, nky, nkx)` `ComplexF64` array, and computes elements on demand. `RK45.make_prop!`
has a specialised threaded kernel for it, so the array is never materialised during
propagation. Construct via `make_const_linop(...; factored=true)`.
"""
struct FactoredFreeLinop{Vt, Mt} <: AbstractArray{ComplexF64, 3}
    ω::Vt
    k2::Vt # (n(ω)·ω/c)² on the sidx region, 0 elsewhere
    kperp2::Mt # (nky, nkx)
    β1::Float64 # 1/velocity of the reference frame
    βref::Float64 # reference β0, subtracted for EnvGrid without THG
    subref::Bool # whether βref is subtracted (EnvGrid with thg=false)
end

# Parametrising the array fields removed the automatic conversion of the scalar ones, so
# restore it: `β0ref` arrives as a `Complex` with zero imaginary part whenever the
# refractive index function is complex-valued, and converting (rather than rejecting) it
# is the pre-existing behaviour.
FactoredFreeLinop(ω::Vt, k2::Vt, kperp2::Mt, β1, βref, subref) where {Vt, Mt} =
    FactoredFreeLinop{Vt, Mt}(ω, k2, kperp2, β1, βref, subref)

# The factors are small (a vector and a matrix), so a device copy is negligible next to
# the field-sized array it replaces. NOTE that `getindex` below is then a host-side
# scalar read of device memory and will (correctly) fail: on a device the operator is
# only ever consumed by the broadcast kernel in `make_prop!`, which uses the factors
# directly.
Adapt.adapt_structure(to, l::FactoredFreeLinop) =
    FactoredFreeLinop(Adapt.adapt(to, l.ω), Adapt.adapt(to, l.k2),
                      Adapt.adapt(to, l.kperp2), l.β1, l.βref, l.subref)

Base.size(l::FactoredFreeLinop) = (length(l.ω), size(l.kperp2, 1), size(l.kperp2, 2))
Base.@propagate_inbounds function Base.getindex(l::FactoredFreeLinop,
                                                iω::Int, iy::Int, ix::Int)
    val = _linop_xy_element(l.ω[iω], l.β1, l.k2[iω], l.kperp2[iy, ix])
    l.subref ? val - (-im*l.βref) : val
end

"""
RK45 propagator kernel for the factored linear operator: the operator elements and their
exponentials are computed on the fly — no array-sized reads or storage. This is the
single largest elementwise cost of a step (it runs 8 times per step), which is why it
gets a backend-specialised kernel rather than going through the generic
`make_prop!(::AbstractArray, y0)`.
"""
function RK45.make_prop!(linop::FactoredFreeLinop, y0)
    prop! = let linop=linop
        function prop!(y, t1, t2, bwd=false)
            dt = bwd ? (t1-t2) : (t2-t1)
            _prop_factored!(Utils.backend(y), y, linop, dt)
        end
    end
end

# Host: threaded over transverse points, with the ω loop innermost so each thread walks
# contiguous memory.
function _prop_factored!(::Utils.CPUBackend, y, linop, dt)
    ω = linop.ω; k2 = linop.k2; kperp2 = linop.kperp2
    β1 = linop.β1; βref = linop.βref; subref = linop.subref
    nω = length(ω)
    Utils.tforeach(length(kperp2); ntotal=length(y)) do ii
        kp = kperp2[ii]
        base = (ii-1)*nω
        @inbounds for iω in 1:nω
            l = _linop_xy_element(ω[iω], β1, k2[iω], kp)
            subref && (l -= -im*βref)
            y[base+iω] *= exp(l*dt)
        end
    end
end

# Device: one broadcast over the SEPARABLE FACTORS, reshaped to (nω,1,1) and (1,nky,nkx)
# so they expand against the state array without materialising anything.
#
# Broadcasting the factors rather than the `FactoredFreeLinop` itself keeps this
# independent of whether a given backend adapts non-native `AbstractArray` operands
# inside a broadcast. (CUDA does, via the `Adapt` rule on the struct — measured on an
# A40 — so that route also works there; it just is not something to rely on.) The
# factors additionally hoist the `subref` test out of the per-element work.
function _prop_factored!(::Utils.DeviceBackend, y, linop, dt)
    ω = reshape(linop.ω, :, 1, 1)
    k2 = reshape(linop.k2, :, 1, 1)
    kperp2 = reshape(linop.kperp2, 1, size(linop.kperp2)...)
    β1 = linop.β1; βref = linop.βref; subref = linop.subref
    @. y *= exp(_linop_xy_dt(ω, β1, k2, kperp2, βref, subref, dt))
    return nothing
end

# The exponent for one element, including the reference-phase subtraction. Kept as a
# scalar function of scalars so the device broadcast and the host loop above evaluate
# exactly the same expression.
@inline function _linop_xy_dt(ω, β1, k2ω, kperp2i, βref, subref, dt)
    l = _linop_xy_element(ω, β1, k2ω, kperp2i)
    subref && (l -= -im*βref)
    return l*dt
end

"""
    make_const_linop(grid, xygrid, n, frame_vel)

Make constant linear operator for full 3D propagation. `n` is the refractive index (array)
and β1 is 1/velocity of the reference frame. With `factored=true`, return a
[`FactoredFreeLinop`](@ref) instead of the materialised array (bit-identical results,
one field-sized array less).

With `frozen_transverse=true`, replace `k_z(ω, k⊥)` by `k_z(ω, 0)`: every k⊥ component
receives the same ω-dependent phase, so the transverse field pattern is frozen exactly
at its input form while temporal dispersion runs unchanged — a diffraction-ablation
diagnostic, not physics. Implemented as `kperp2 → zero(kperp2)` at construction; the
element math, the factored/materialised equivalence and everything downstream are
untouched.
"""
function make_const_linop(grid::Grid.RealGrid, xygrid::Grid.FreeGrid,
                          n::AbstractArray, β1::Number; factored::Bool=false,
                          arraytype=Array, frozen_transverse::Bool=false)
    kperp2 = @. (xygrid.kx^2)' + xygrid.ky^2
    frozen_transverse && (kperp2 = zero(kperp2))
    idcs = CartesianIndices((length(xygrid.ky), length(xygrid.kx)))
    k2 = zero(grid.ω)
    k2[grid.sidx] .= (n[grid.sidx] .* grid.ω[grid.sidx] ./ PhysData.c).^2
    if factored
        # A real grid never subtracts a reference phase: it propagates the field itself,
        # not an envelope about a carrier.
        l = FactoredFreeLinop(copy(grid.ω), k2, kperp2, float(β1), 0.0, false)
        return arraytype === Array ? l : Adapt.adapt(arraytype, l)
    end
    _check_materialised_linop(arraytype)
    out = zeros(ComplexF64, (length(grid.ω), length(xygrid.ky), length(xygrid.kx)))
    _fill_linop_xy!(out, grid, β1, k2, kperp2, idcs)
    return out
end

function make_const_linop(grid::Grid.RealGrid, xygrid::Grid.FreeGrid, nfun;
                          factored::Bool=false, arraytype=Array,
                          frozen_transverse::Bool=false)
    n = zero(grid.ω)
    n[grid.sidx] = nfun.(2π*PhysData.c./grid.ω[grid.sidx])
    β1 = PhysData.dispersion_func(1, nfun)(grid.referenceλ)
    make_const_linop(grid, xygrid, n, β1; factored, arraytype, frozen_transverse)
end

function make_const_linop(grid::Grid.EnvGrid, xygrid::Grid.FreeGrid,
                          n::AbstractArray, β1::Number, β0ref::Number;
                          thg=false, factored::Bool=false, arraytype=Array,
                          frozen_transverse::Bool=false)
    kperp2 = @. (xygrid.kx^2)' + xygrid.ky^2
    frozen_transverse && (kperp2 = zero(kperp2))
    idcs = CartesianIndices((length(xygrid.ky), length(xygrid.kx)))
    k2 = zero(grid.ω)
    k2[grid.sidx] .= (n[grid.sidx].*grid.ω[grid.sidx]./PhysData.c).^2
    if factored
        l = FactoredFreeLinop(copy(grid.ω), k2, kperp2, float(β1), float(β0ref), !thg)
        return arraytype === Array ? l : Adapt.adapt(arraytype, l)
    end
    _check_materialised_linop(arraytype)
    out = zeros(ComplexF64, (length(grid.ω), length(xygrid.ky), length(xygrid.kx)))
    _fill_linop_xy!(out, grid, β1, k2, kperp2, idcs, β0ref; thg=thg)
    return out
end

function make_const_linop(grid::Grid.EnvGrid, xygrid::Grid.FreeGrid, nfun,
                          thg=false; factored::Bool=false, arraytype=Array,
                          frozen_transverse::Bool=false)
    n = zero(grid.ω)
    n[grid.sidx] = nfun.(wlfreq.(grid.ω[grid.sidx]))
    β1 = PhysData.dispersion_func(1, nfun)(grid.referenceλ)
    if thg
        β0const = 0.0
    else
        β0const = grid.ω0/PhysData.c * nfun(wlfreq(grid.ω0))
    end
    make_const_linop(grid, xygrid, n, β1, β0const; thg=thg, factored, arraytype,
                     frozen_transverse)
end

# A materialised linear operator is a whole extra field-sized array on the device, and
# building it needs a scalar fill loop. Refuse rather than silently doubling the device
# memory or crawling.
function _check_materialised_linop(arraytype)
    arraytype === Array || error(
        "device propagation requires `factored=true`: a materialised linear operator "*
        "would occupy an entire extra field-sized device array (and is built by a "*
        "host-side scalar loop). Pass `factored=true` to make_const_linop.")
end

"""
    make_linop(grid, xygrid, nfun)

Make z-dependent linear operator for free-space propagation. `nfun(ω; z)` should return the
refractive index as a function of frequency `ω` and (kwarg) propagation distance `z`.
"""
function make_linop(grid::Grid.RealGrid, xygrid::Grid.FreeGrid, nfun)
    kperp2 = @. (xygrid.kx^2)' + xygrid.ky^2
    idcs = CartesianIndices((length(xygrid.ky), length(xygrid.kx)))
    k2 = zero(grid.ω)
    nfunλ(z) = λ -> nfun(wlfreq(λ), z=z)
    function linop!(out, z)
        β1 = PhysData.dispersion_func(1, nfunλ(z))(grid.referenceλ)
        k2[grid.sidx] .= (nfun.(grid.ω[grid.sidx]; z=z) .* grid.ω[grid.sidx] ./ PhysData.c).^2
        _fill_linop_xy!(out, grid, β1, k2, kperp2, idcs)
    end
end

# Internal routine -- function barrier aids with JIT compilation
function _fill_linop_xy!(out, grid::Grid.RealGrid, β1::Float64, k2, kperp2, idcs)
    for ii in idcs
        for iω in eachindex(grid.ω)
            out[iω, ii] = _linop_xy_element(grid.ω[iω], β1, k2[iω], kperp2[ii])
        end
    end
end

function make_linop(grid::Grid.EnvGrid, xygrid::Grid.FreeGrid, nfun; thg=false)
    kperp2 = @. (xygrid.kx^2)' + xygrid.ky^2
    idcs = CartesianIndices((length(xygrid.ky), length(xygrid.kx)))
    k2 = zero(grid.ω)
    nfunλ(z) = λ -> nfun(wlfreq(λ), z=z)
    function linop!(out, z)
        β1 = PhysData.dispersion_func(1, nfunλ(z))(grid.referenceλ)
        k2[grid.sidx] .= (nfun.(grid.ω[grid.sidx]; z=z).*grid.ω[grid.sidx]./PhysData.c).^2
        βref = thg ? 0.0 : grid.ω0/PhysData.c * nfun(grid.ω0; z=z)
        _fill_linop_xy!(out, grid, β1, k2, kperp2, idcs, βref; thg=thg)
    end
end

function _fill_linop_xy!(out, grid::Grid.EnvGrid, β1::Float64, k2, kperp2, idcs, βref; thg)
    for ii in idcs
        for iω in eachindex(grid.ω)
            out[iω, ii] = _linop_xy_element(grid.ω[iω], β1, k2[iω], kperp2[ii])
            if !thg
                out[iω, ii] -= -im*βref
            end
        end
    end
end

#=================================================#
#==============   RADIAL SYMMETRY   ==============#
#=================================================#
"""
    make_const_linop(grid, q::QDHT, n, frame_vel)

Make constant linear operator for radial free-space. `n` is the refractive index (array)
and β1 is 1/velocity of the reference frame.
"""
function make_const_linop(grid::Grid.RealGrid, q::Hankel.QDHT,
                          n::AbstractArray, β1::Number)
    out = Array{ComplexF64}(undef, (length(grid.ω), q.N))
    k2 = @. (n*grid.ω/PhysData.c)^2
    kr2 = q.k.^2
    _fill_linop_r!(out, grid, β1, k2, kr2, q.N)
    return out
end

function make_const_linop(grid::Grid.RealGrid, q::Hankel.QDHT, nfun)
    n = zero(grid.ω)
    n[grid.sidx] = nfun.(2π*PhysData.c./grid.ω[grid.sidx])
    β1 = PhysData.dispersion_func(1, nfun)(grid.referenceλ)
    make_const_linop(grid, q, n, β1)
end

function make_const_linop(grid::Grid.EnvGrid, q::Hankel.QDHT, nfun; thg=false)
    n = zero(grid.ω)
    n[grid.sidx] = nfun.(2π*PhysData.c./grid.ω[grid.sidx])
    β1 = PhysData.dispersion_func(1, nfun)(grid.referenceλ)
    if thg
        β0const = 0.0
    else
        β0const = grid.ω0/PhysData.c * nfun(2π*PhysData.c./grid.ω0)
    end
    make_const_linop(grid, q, n, β1, β0const; thg=thg)
end

function make_const_linop(grid::Grid.EnvGrid, q::Hankel.QDHT,
                          n::AbstractArray, β1::Number, β0ref::Number; thg=false)
    out = Array{ComplexF64}(undef, (length(grid.ω), q.N))
    k2 = @. (n*grid.ω/PhysData.c)^2
    kr2 = q.k.^2
    _fill_linop_r!(out, grid, β1, k2, kr2, q.N, β0ref, thg)
    return out
end

"""
    make_linop(grid, q::QDHT, nfun)

Make z-dependent linear operator for radial free-space propagation. `nfun(ω; z)` should
return the refractive index as a function of frequency `ω` and (kwarg) propagation
distance `z`.
"""
function make_linop(grid::Grid.RealGrid, q::Hankel.QDHT, nfun)
    kr2 = q.k.^2
    k2 = zero(grid.ω)
    nfunλ(z) = λ -> nfun(wlfreq(λ), z=z)
    function linop!(out, z)
        β1 = PhysData.dispersion_func(1, nfunλ(z))(grid.referenceλ)
        k2[grid.sidx] .= (nfun.(grid.ω[grid.sidx]; z=z) .* grid.ω[grid.sidx]./PhysData.c).^2
        _fill_linop_r!(out, grid, β1, k2, kr2, q.N)
    end
end

function _fill_linop_r!(out, grid::Grid.RealGrid, β1, k2, kr2, Nr)
    for ir = 1:Nr
        for iω = 1:length(grid.ω)
            βsq = k2[iω] - kr2[ir]
            if βsq < 0
                # negative βsq -> evanescent fields -> attenuation
                out[iω, ir] = -im*(-β1*grid.ω[iω]) - min(sqrt(abs(βsq)), 200)
            else
                out[iω, ir] = -im*(sqrt(βsq) - β1*grid.ω[iω])
            end
        end
    end
end

function make_linop(grid::Grid.EnvGrid, q::Hankel.QDHT, nfun; thg=false)
    kr2 = q.k.^2
    k2 = zero(grid.ω)
    nfunλ(z) = λ -> nfun(wlfreq(λ), z=z)
    function linop!(out, z)
        β1 = PhysData.dispersion_func(1, nfunλ(z))(grid.referenceλ)
        k2[grid.sidx] .= (nfun.(grid.ω[grid.sidx]; z=z) .* grid.ω[grid.sidx]./PhysData.c).^2
        βref = thg ? 0.0 : grid.ω0/PhysData.c * nfun(grid.ω0; z=z)
        _fill_linop_r!(out, grid, β1, k2, kr2, q.N, βref, thg)
    end
end

function _fill_linop_r!(out, grid::Grid.EnvGrid, β1, k2, kr2, Nr, βref, thg)
    for ir = 1:Nr
        for iω = 1:length(grid.ω)
            βsq = k2[iω] - kr2[ir]
            if βsq < 0
                # negative βsq -> evanescent fields -> attenuation
                out[iω, ir] = -im*(-β1*grid.ω[iω]) - min(sqrt(abs(βsq)), 200)
            else
                out[iω, ir] = -im*(sqrt(βsq) - β1*grid.ω[iω])
            end
            if !thg
                out[iω, ir] -= -im*βref
            end
        end
    end
end

#=================================================#
#===============   MODE AVERAGE   ================#
#=================================================#

"""
    αlim!(α)

Limit α so that we do not get overflow in exp(α*dz)
"""
function αlim!(α)
    # magic number: this is 130 dB/cm
    # a test script sensitive to this is test_main_rect_env.jl
    clamp!(α, 0.0, 3000.0)
end

"""
    conj_clamp(n, ω)

Simultaneously conjugate and clamp the effective index `n` to safe levels.

The real part is lower-bounded at 1e-3 and the imaginary part upper-bounded at an attenuation
coefficient `α` of 3000 (130 dB/cm). The limits are somewhat arbitrary and chosen empirically
from previous bugs. See https://github.com/LupoLab/Luna/pull/142.

See also [`αlim!`](@ref).
"""
conj_clamp(n, ω) = clamp(real(n), 1e-3, Inf) - im*clamp(imag(n), 0, 3000*PhysData.c/ω)

function make_const_linop(grid::Grid.RealGrid, βfun!, αfun!, β1)
    β = similar(grid.ω)
    βfun!(β, 0)
    α = similar(grid.ω)
    αfun!(α, 0)
    αlim!(α)
    linop = @. -im*(β-β1*grid.ω) - α/2
    linop[.!grid.sidx] .= 0
    return linop
end

function make_const_linop(grid::Grid.EnvGrid, βfun!, αfun!, β1, β0ref)
    β = similar(grid.ω)
    βfun!(β, 0)
    α = similar(grid.ω)
    αfun!(α, 0)
    αlim!(α)
    linop = -im.*(β .- β1.*(grid.ω .- grid.ω0) .- β0ref) .- α./2
    linop[.!grid.sidx] .= 0
    return linop
end

"""
    make_const_linop(grid, mode, λ0)

Make constant linear operator for mode-averaged propagation in mode `mode` with a reference
wavelength `λ0`.
"""
function make_const_linop(grid::Grid.EnvGrid, mode::Modes.AbstractMode, λ0; thg=false)
    β1const = Modes.dispersion(mode, 1, wlfreq(λ0))
    if thg
        β0const = 0.0
    else
        β0const = Modes.β(mode, wlfreq(λ0))
    end
    βconst = zero(grid.ω)
    βconst[grid.sidx] = Modes.β.(mode, grid.ω[grid.sidx])
    βconst[.!grid.sidx] .= 1
    function βfun!(out, z)
        out .= βconst
    end
    αconst = zero(grid.ω)
    αconst[grid.sidx] = Modes.α.(mode, grid.ω[grid.sidx])
    function αfun!(out, z)
        out .= αconst
    end
    make_const_linop(grid, βfun!, αfun!, β1const, β0const), βfun!, β1const, αfun!
end

function make_const_linop(grid::Grid.RealGrid, mode::Modes.AbstractMode, λ0)
    β1const = Modes.dispersion(mode, 1, wlfreq(λ0))
    βconst = zero(grid.ω)
    βconst[grid.sidx] = Modes.β.(mode, grid.ω[grid.sidx])
    βconst[.!grid.sidx] .= 1
    function βfun!(out, z)
        out .= βconst
    end
    αconst = zero(grid.ω)
    αconst[grid.sidx] = Modes.α.(mode, grid.ω[grid.sidx])
    function αfun!(out, z)
        out .= αconst
    end
    make_const_linop(grid, βfun!, αfun!, β1const), βfun!, β1const, αfun!
end

"""
    neff_β_grid(grid, mode, λ0; ref_mode=1)

Create closures which return the effective index and propagation constant
as a function of the frequency grid **index**, rather than the frequency itself.
Any [`Modes.AbstractMode`](@ref) may define its own method for `neff_β_grid` to
accelerate repeated calculation on the same frequency grid.
"""
function neff_β_grid(grid, mode, λ0)
    let grid=grid, mode=mode
        _neff(iω; z) = Modes.neff(mode, grid.ω[iω]; z=z)
        _β(iω; z) = Modes.β(mode, grid.ω[iω]; z=z)
        _neff, _β
    end
end

function make_linop(grid::Grid.RealGrid, mode::Modes.AbstractMode, λ0)
    sidcs = (1:length(grid.ω))[grid.sidx]
    neff, β = neff_β_grid(grid, mode, λ0)
    linop! = let neff=neff, ω=grid.ω, mode=mode, ω0=wlfreq(λ0)
        function linop!(out, z)
            fill!(out, 0.0)
            β1 = Modes.dispersion(mode, 1, ω0, z=z)::Float64
            for iω in sidcs
                nc = conj_clamp(neff(iω; z=z), ω[iω])
                out[iω] = -im*(ω[iω]/PhysData.c*nc - ω[iω]*β1)
            end
        end
    end
    βfun! = let β=β, ω=grid.ω
        function βfun!(out, z)
            fill!(out, 1.0)
            for iω in sidcs
                out[iω] = β(iω; z=z)
            end
        end
    end
    return linop!, βfun!
end

function make_linop(grid::Grid.EnvGrid, mode::Modes.AbstractMode, λ0; thg=false)
    sidcs = (1:length(grid.ω))[grid.sidx]
    neff, β = neff_β_grid(grid, mode, λ0)
    linop! = let neff=neff, ω=grid.ω, mode=mode, ω0=wlfreq(λ0), sidcs=sidcs
        function linop!(out, z)
            fill!(out, 0.0)
            β1 = Modes.dispersion(mode, 1, ω0, z=z)::Float64
            if !thg
                βref = Modes.β(mode, ω0, z=z)
            end
            for iω in sidcs
                nc = conj_clamp(neff(iω; z=z), ω[iω])
                out[iω] = -im*(ω[iω]/PhysData.c*nc - (ω[iω] - grid.ω0)*β1)
                if !thg
                    out[iω] -= -im*βref
                end
            end
        end
    end
    βfun! = let β=β, sidcs=sidcs
        function βfun!(out, z)
            fill!(out, 1.0)
            for iω in sidcs
                out[iω] = β(iω, z=z)
            end
        end
    end
    return linop!, βfun!
end

#=================================================#
#=================   MULTIMODE   =================#
#=================================================#
"""
    make_const_linop(grid, modes, λ0; ref_mode=1)

Make constant (z-invariant) linear operator for multimode propagation. The frame velocity is
taken as the group velocity at wavelength `λ0` in the mode given by `ref_mode` (which 
indexes into `modes`)
"""
function make_const_linop(grid::Grid.RealGrid, modes, λ0; ref_mode=1)
    β1 = Modes.dispersion(modes[ref_mode], 1, wlfreq(λ0))
    nmodes = length(modes)
    linops = zeros(ComplexF64, length(grid.ω), nmodes)
    for i = 1:nmodes
        βconst = zero(grid.ω)
        βconst[grid.sidx] = Modes.β.(modes[i], grid.ω[grid.sidx])
        βconst[.!grid.sidx] .= 1
        α = zeros(length(grid.ω))
        α[grid.sidx] .= Modes.α.(modes[i], grid.ω[grid.sidx])
        αlim!(α)
        linops[:,i] = im.*(-βconst .+ grid.ω.*β1) .- α./2
    end
    linops
end

function make_const_linop(grid::Grid.EnvGrid, modes, λ0; ref_mode=1, thg=false)
    β1 = Modes.dispersion(modes[ref_mode], 1, wlfreq(λ0))
    if thg
        βref = 0.0
    else
        βref = Modes.β(modes[ref_mode], wlfreq(λ0))
    end
    nmodes = length(modes)
    linops = zeros(ComplexF64, length(grid.ω), nmodes)
    for i = 1:nmodes
        βconst = zero(grid.ω)
        βconst[grid.sidx] = Modes.β.(modes[i], grid.ω[grid.sidx])
        βconst[.!grid.sidx] .= 1
        α = Modes.α.(modes[i], grid.ω)
        αlim!(α)
        linops[:,i] = -im.*(βconst .- (grid.ω .- grid.ω0).*β1 .- βref) .- α./2
    end
    linops
end

"""
    neff_grid(grid, modes, λ0; ref_mode=1)

Create a closure that returns the effective index as a function of the frequency grid and mode
**index**, rather than the mode and frequency themselves. Any [`Modes.AbstractMode`](@ref)
may define its one method for `neff_grid` to accelerate repeated calculation on the same
frequency grid.
"""
function neff_grid(grid, modes, λ0; ref_mode=1)
    _neff = let grid=grid, modes=modes
        _neff(iω, iim; z) = Modes.neff(modes[iim], grid.ω[iω]; z=z)
    end
    _neff
end

"""
    make_linop(grid, modes, λ0; ref_mode=1[, thg=false])

``z``-dependent modal linear operator `linop!(out, z)` for a collection of modes (tapers,
pressure gradients). For a collection of `Capillary.MarcatiliMode`s whose core and cladding
index functions have a recognisable structure (`Capillary.ConstIndex` /
`Capillary.GradientCoreIndex`, i.e. what the standard constructors and
`Capillary.gradient` produce) `Capillary` adds methods returning a
`Capillary.MarcatiliLinop`, which evaluates all modes and frequencies as one broadcast per
call and can write directly into a device array; for anything else the generic closure
below evaluates `Modes.neff` mode by mode and frequency by frequency on the host.
"""
make_linop(grid::Grid.RealGrid, modes, λ0; ref_mode=1) =
    _make_linop_generic(grid, modes, λ0; ref_mode)
make_linop(grid::Grid.EnvGrid, modes, λ0; ref_mode=1, thg=false) =
    _make_linop_generic(grid, modes, λ0; ref_mode, thg)

function _make_linop_generic(grid::Grid.RealGrid, modes, λ0; ref_mode=1)
    sidcs = (1:length(grid.ω))[grid.sidx]
    neff = neff_grid(grid, modes, λ0; ref_mode=ref_mode)
    linop! = let neff=neff, ω=grid.ω, modes=modes, ω0=wlfreq(λ0), ref_mode=ref_mode
        function linop!(out, z)
            β1 = Modes.dispersion(modes[ref_mode], 1, ω0, z=z)::Float64
            fill!(out, 0.0)
            for i in eachindex(modes)
                for iω in sidcs
                    nc = conj_clamp(neff(iω, i; z=z), ω[iω])
                    out[iω, i] = -im*(ω[iω]/PhysData.c*nc - ω[iω]*β1)
                end
            end
        end
    end
end

function _make_linop_generic(grid::Grid.EnvGrid, modes, λ0; ref_mode=1, thg=false)
    sidcs = (1:length(grid.ω))[grid.sidx]
    neff = neff_grid(grid, modes, λ0; ref_mode=ref_mode)
    linop! = let neff=neff, ω=grid.ω, modes=modes, ω0=wlfreq(λ0), ref_mode=ref_mode
        function linop!(out, z)
            β1 = Modes.dispersion(modes[ref_mode], 1, ω0, z=z)::Float64
            fill!(out, 0.0)
            if !thg
                βref = Modes.β(modes[ref_mode], ω0, z=z)
            end
            for i in eachindex(modes)
                for iω in sidcs
                    nc = conj_clamp(neff(iω, i; z=z), ω[iω])
                    out[iω, i] = -im*(ω[iω]/PhysData.c*nc - (ω[iω] - grid.ω0)*β1)
                    if !thg
                        out[iω, i] -= -im*βref
                    end
                end
            end
        end
    end
end

end
