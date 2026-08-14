module Nonlinear
import Luna
import Luna.PhysData: ε_0, e_ratio
import Luna: Maths, Utils
import FFTW
import LinearAlgebra: mul!, ldiv!

function KerrScalar!(out, E, fac)
    @. out += fac*E^3
end

function KerrVector!(out, E, fac)
    for i = 1:size(E,1)
        Ex = E[i,1]
        Ey = E[i,2]
        Ex2 = Ex^2
        Ey2 = Ey^2
        out[i,1] += fac*(Ex2 + Ey2)*Ex
        out[i,2] += fac*(Ex2 + Ey2)*Ey
    end
end

"""
    KerrField(γ3)

Kerr response for a real field. Callable as `K(out, E, ρ)`, accumulating the nonlinear
polarisation induced by `E` at density `ρ` into `out`. Pointwise for scalar fields
(see [`pointwise`](@ref)).
"""
struct KerrField{T}
    γ3::T
end

function (K::KerrField)(out, E, ρ)
    if size(E,2) == 1
        KerrScalar!(out, E, ρ*ε_0*K.γ3)
    else
        KerrVector!(out, E, ρ*ε_0*K.γ3)
    end
end

"Kerr response for real field"
Kerr_field(γ3) = KerrField(γ3)

"Kerr response for real field but without THG"
function Kerr_field_nothg(γ3, n)
    E = Array{Float64}(undef, n)
    hilbert = Maths.plan_hilbert(E)
    Kerr = let γ3 = γ3, hilbert = hilbert
        function Kerr(out, E, ρ)
            out .+= ρ*3/4*ε_0*γ3.*abs2.(hilbert(E)).*E
        end
    end
end

function KerrScalarEnv!(out, E, fac)
    @. out += 3/4*fac*abs2(E)*E
end

function KerrVectorEnv!(out, E, fac)
    for i = 1:size(E,1)
        Ex = E[i,1]
        Ey = E[i,2]
        Ex2 = abs2(Ex)
        Ey2 = abs2(Ey)
        out[i,1] += 3/4*fac*((Ex2 + 2/3*Ey2)*Ex + 1/3*conj(Ex)*Ey^2)
        out[i,2] += 3/4*fac*((Ey2 + 2/3*Ex2)*Ey + 1/3*conj(Ey)*Ex^2)
    end
end

"""
    KerrEnv(γ3)

Kerr response for an envelope field. Callable as `K(out, E, ρ)`, accumulating the
nonlinear polarisation induced by `E` at density `ρ` into `out`. Pointwise for scalar
fields (see [`pointwise`](@ref)).
"""
struct KerrEnv{T}
    γ3::T
end

function (K::KerrEnv)(out, E, ρ)
    if size(E,2) == 1
        KerrScalarEnv!(out, E, ρ*ε_0*K.γ3)
    else
        KerrVectorEnv!(out, E, ρ*ε_0*K.γ3)
    end
end

"Kerr response for envelope"
Kerr_env(γ3) = KerrEnv(γ3)

"""
    pointwise(resp)

Trait: `true` if `resp` is a pointwise response for scalar fields, i.e. the nonlinear
polarisation at each sample depends only on the field at that same sample. Pointwise
responses implement [`pointwise_P`](@ref) and can be applied to whole (multi-dimensional)
field arrays in a single broadcast instead of column-by-column, which is both faster and
GPU-compatible. Defaults to `false`.
"""
pointwise(resp) = false
pointwise(::KerrField) = true
pointwise(::KerrEnv) = true

"""
    pointwise_P(resp, E, ρ)

The scalar nonlinear polarisation sample induced by the scalar field sample `E` at density
`ρ`. Implementations must perform the same floating-point operations in the same order as
the columnwise call of `resp`, so that both paths produce bit-identical results.
"""
@inline pointwise_P(K::KerrField, E, ρ) = (ρ*ε_0*K.γ3)*E^3
@inline pointwise_P(K::KerrEnv, E, ρ) = 3/4*(ρ*ε_0*K.γ3)*abs2(E)*E

"Kerr response for envelope but with THG"
# see Eq. 4, Genty et al., Opt. Express 15 5382 (2007)
function Kerr_env_thg(γ3, ω0, t)
    C = exp.(2im*ω0.*t)
    Kerr = let γ3 = γ3, C = C
        function Kerr(out, E, ρ)
            @. out += ρ*ε_0*γ3/4*(3*abs2(E) + C*E^2)*E
        end
    end
end

"Response type for cumtrapz-based plasma polarisation, adapted from:
M. Geissler, G. Tempea, A. Scrinzi, M. Schnürer, F. Krausz, and T. Brabec, Physical Review Letters 83, 2930 (1999)."
struct PlasmaCumtrapz{R, EType, tType}
    ratefunc::R # the ionization rate function
    ionpot::Float64 # the ionization potential (for calculation of ionization loss)
    rate::tType # buffer to hold the rate
    fraction::tType # buffer to hold the ionization fraction
    phase::EType # buffer to hold the plasma induced (mostly) phase modulation
    J::EType # buffer to hold the plasma current
    P::EType # buffer to hold the plasma polarisation
    δt::Float64 # the time step
    preionfrac::Float64 # the pre-ionisation fraction
end

"""
    PlasmaCumtrapz(t, E, ratefunc, ionpot)

Construct the Plasma polarisation response for a field on time grid `t`
with example electric field like `E`, an ionization rate callable
`ratefunc` and ionization potential `ionpot`.
"""
function PlasmaCumtrapz(t, E, ratefunc, ionpot; preionfrac=0.0)
    rate = similar(t)
    fraction = similar(t)
    phase = similar(E)
    J = similar(E)
    P = similar(E)
    !(0.0 <= preionfrac <= 1.0) && throw(DomainError(preionfrac, "preionfrac must be between 0 and 1"))
    if preionfrac > 0.0
        @warn("Using preionfrac > 0.0 is not a well founded physical model. Use only after careful consideration.")
    end
    return PlasmaCumtrapz(ratefunc, ionpot, rate, fraction, phase, J, P, t[2]-t[1], preionfrac)
end

"The plasma response for a scalar electric field"
function PlasmaScalar!(Plas::PlasmaCumtrapz, E)
    Plas.ratefunc(Plas.rate, E)
    Maths.cumtrapz!(Plas.fraction, Plas.rate, Plas.δt)
    @. Plas.fraction = Plas.preionfrac + 1 - exp(-Plas.fraction)
    @. Plas.phase = Plas.fraction * e_ratio * E
    Maths.cumtrapz!(Plas.J, Plas.phase, Plas.δt)
    for ii in eachindex(E)
        if abs(E[ii]) > 0
            Plas.J[ii] += Plas.ionpot * Plas.rate[ii] * (1-Plas.fraction[ii])/E[ii]
        end
    end
    Maths.cumtrapz!(Plas.P, Plas.J, Plas.δt)
end

"""
The plasma response for a vector electric field.

We take the magnitude of the electric field to calculate the ionization
rate and fraction, and then solve the plasma polarisation component-wise
for the vector field.

A similar approach was used in: C Tailliez et al 2020 New J. Phys. 22 103038.  
"""
function PlasmaVector!(Plas::PlasmaCumtrapz, E)
    Ex = E[:,1]
    Ey = E[:,2]
    Em = @. hypot.(Ex, Ey)
    Plas.ratefunc(Plas.rate, Em)
    Maths.cumtrapz!(Plas.fraction, Plas.rate, Plas.δt)
    @. Plas.fraction = Plas.preionfrac + 1 - exp(-Plas.fraction)
    @. Plas.phase = Plas.fraction * e_ratio * E
    Maths.cumtrapz!(Plas.J, Plas.phase, Plas.δt)
    for ii in eachindex(Em)
        if abs(Em[ii]) > 0
            pre = Plas.ionpot * Plas.rate[ii] * (1-Plas.fraction[ii])/Em[ii]^2
            Plas.J[ii,1] += pre*Ex[ii]
            Plas.J[ii,2] += pre*Ey[ii]
        end
    end
    Maths.cumtrapz!(Plas.P, Plas.J, Plas.δt)
end

"Handle plasma polarisation routing to `PlasmaVector` or `PlasmaScalar`."
function (Plas::PlasmaCumtrapz)(out, Et, ρ)
    if ndims(Et) > 1
        if size(Et, 2) == 1 # handle scalar case but within modal simulation
            PlasmaScalar!(Plas, reshape(Et, size(Et,1)))
            out .+= ρ .* reshape(Plas.P, size(Et))
        else
            PlasmaVector!(Plas, Et) # vector case
            out .+= ρ .* Plas.P
        end
    else
        PlasmaScalar!(Plas, Et) # straight scalar case
        out .+= ρ .* Plas.P
    end
end

"Raman polarisation response type"
abstract type RamanPolar end

"Raman polarisation response type for a carrier resolved field"
struct RamanPolarField{TR, Tt, Thv, Tω, Tv, FTt, HTt} <: RamanPolar
    r::TR # Raman response
    h::Tt # doubled buffer to hold response + padding 
    ht::Thv # buffer to hold time domain response
    hω::Tω # the frequency domain Raman response function
    Eω2::Tω # buffer to hold the Fourier transform of E^2
    Pω::Tω # buffer to hold the frequency domain polarisation
    E2::Tt # buffer to hold E^2
    E2v::Tv # view into first half of E2
    P::Tt # buffer to hold the time domain polarisation
    Pout::Tt # buffer to hold the output portion of the time domain polarisation
    FT::FTt # Fourier transform plan
    HT::HTt # Hilbert transform
    thg::Bool # do we include third harmonic generation
    dt::Float64 # time step for scaling
end

"Raman polarisation response type for an envelope"
struct RamanPolarEnv{TR, Tt, Thv, Tω, Tv, FTt} <: RamanPolar
    r::TR # Raman response
    h::Tt # doubled buffer to hold response + padding 
    ht::Thv # buffer to hold time domain response
    hω::Tω # the frequency domain Raman response function
    Eω2::Tω # buffer to hold the Fourier transform of E^2
    Pω::Tω # buffer to hold the frequency domain polarisation
    E2::Tω # buffer to hold E^2
    E2v::Tv # view into first half of E2
    P::Tω # buffer to hold the time domain polarisation
    Pout::Tω # buffer to hold the output portion of the time domain polarisation
    FT::FTt # Fourier transform plan
    dt::Float64 # time step for scaling
end

"""
    RamanPolarField(t, ht; thg=true)

Construct Raman polarisation response for a field on time grid `t`
using response function `r`. If `thg=false` then exclude the third
harmonic generation component of the response.
"""
function RamanPolarField(t, r; thg=true)
    h = zeros(length(t)*2) # note double grid size, see explanation below
    ht = view(h, 1:length(t))
    Utils.loadFFTwisdom()
    FT = FFTW.plan_rfft(h, 1, flags=Luna.settings["fftw_flag"])
    inv(FT)
    Utils.saveFFTwisdom()
    hω = FT * h
    Eω2 = similar(hω)
    Pω = similar(hω)
    E2 = similar(h)
    E2v = view(E2, 1:length(t))
    P = similar(h)
    Pout = similar(t)
    HT = Maths.plan_hilbert(Pout)
    fill!(E2, 0.0)
    RamanPolarField(r, h, ht, hω, Eω2, Pω, E2, E2v, P, Pout, FT, HT, thg, t[2] - t[1])
end

"""
    RamanPolarEnv(t, ht)

Construct Raman polarisation response for an envelope on time grid `t`
using response function `r`.
"""
function RamanPolarEnv(t, r)
    h = zeros(length(t)*2) # note double grid size, see explanation below
    ht = view(h, 1:length(t))
    Utils.loadFFTwisdom()
    FT = FFTW.plan_fft(h, 1, flags=Luna.settings["fftw_flag"])
    inv(FT)
    Utils.saveFFTwisdom()
    hω = FT * h
    Eω2 = similar(hω)
    Pω = similar(hω)
    E2 = similar(hω)
    P = similar(hω)
    Pout = Array{ComplexF64,}(undef,size(t))
    E2v = view(E2, 1:length(t))
    fill!(E2, 0.0)
    RamanPolarEnv(r, h, ht, hω, Eω2, Pω, E2, E2v, P, Pout, FT, t[2] - t[1])
end

"Square the field or envelope"
function sqr!(R::RamanPolarField, E)
    if !R.thg
        # see documentation for factor of 1/2 here
        R.E2v .= 1/2 .* abs2.(R.HT(E))
    else
        R.E2v .= E.^2
    end
end

function sqr!(R::RamanPolarEnv, E)
    # see documentation for factor of 1/2 here
    R.E2v .= 1/2 .* abs2.(E)
end

"Calculate Raman polarisation for field/envelope Et"
function (R::RamanPolar)(out, Et, ρ)
    # get the field as a 1D Array
    n = size(Et, 1)
    if ndims(Et) > 1
        if size(Et, 2) == 1 # handle scalar case but within modal simulation
            E = reshape(Et, n)
        else
            # handle vector case
            error("vector Raman not yet implemented")
        end
    else
        E = Et # handle straight scalar case
    end

    # square the field or envelope in first half
    # corresponding to the field/envelope grid size
    sqr!(R, E)

    # update frequency domain response function `hω`.
    # we fill only up to the first half of h (using the view ht)
    # i.e. only the part corresponding to the original time grid
    # note that the response function time 0 is put into the first element of the response array
    # this ensures that causality is maintained, and no artificial delay between the field and
    # the start of the response function occurs, at each convolution point.  
    R.r(R.ht, ρ)
    R.hω .= R.FT * R.h

    # convolution by multiplication in frequency domain
    # The double grid gives us accurate full convolution between the full field grid
    # and full response function. It is unnecessary for highly damped responses, like
    # in glass. But for gases with very long decay times it prevents artefacts due to
    # truncation of the response function. There is likely a more efficient way. But
    # this is safe, until we come up with one.
    # we scale to correct for missing dt*dt*df from IFFT(FFT*FFT)
    # the ifft already scales by 1/n = dt*df, so we need an additional dt
    R.Eω2 .= R.FT * R.E2
    @. R.Pω = R.hω * R.Eω2 * R.dt
    R.P .= R.FT \ R.Pω

    # calculate full polarisation, extracting only the valid
    # grid region, which is the first length(E) part.
    for i = 1:length(E)
        R.Pout[i] = ρ*E[i]*R.P[i]
    end
    
    # copy to output in dimensions requested
    if ndims(Et) > 1
        out .+= reshape(R.Pout, size(Et))
    else
        out .+= R.Pout
    end
end

"""
    batched(resp)

Trait: `true` if `resp` is an array-level (batched) response, called once with the full
multi-dimensional field array (`resp(out, Et, ρ)` with `out`/`Et` of shape
`(nt, ny, nx)`) rather than once per transverse column. Defaults to `false`.
"""
batched(resp) = false

"""
    RamanPolarEnvBatched(t, r)

Batched (array-level) version of [`RamanPolarEnv`](@ref) for 3D free-space propagation:
computes the Raman polarisation for all transverse points at once using batched FFTs
along the time dimension, instead of two small FFTs per transverse column. Also updates
the density-dependent response function once per call instead of once per column (exact:
the density is a scalar, identical for all columns).

Results agree with `RamanPolarEnv` column-for-column to rounding accuracy (~1e-15
relative; the FFT algorithm for batched transforms differs from the single-column one,
so agreement is not bit-exact).

The `(2nt, ny, nx)` work array (the doubled anti-wraparound convolution grid, exactly as
in `RamanPolarEnv`) and its in-place FFT plans are allocated lazily on the first call,
when the transverse grid size is known.
"""
mutable struct RamanPolarEnvBatched{TR, hvT, fT}
    r::TR # Raman response function
    h::Vector{Float64} # doubled buffer to hold response + padding
    ht::hvT # view into first half of h
    hω::Vector{ComplexF64} # the frequency domain Raman response function (host)
    hωd::Any # `hω` on the field's array type; `hω` itself on the host (no copy)
    hFT::fT # 1D Fourier transform plan for h
    dt::Float64 # time step for scaling
    nt::Int # length of the (undoubled) time grid
    B::Any # (2nt, ny, nx) doubled work array, allocated lazily on the field's array type
    FT::Any # in-place batched FFT plan along dim 1 of B (created with B)
    IFT::Any # inverse of FT
end

batched(::RamanPolarEnvBatched) = true

function RamanPolarEnvBatched(t, r)
    h = zeros(length(t)*2) # note double grid size, see explanation in (R::RamanPolar)
    ht = view(h, 1:length(t))
    Utils.loadFFTwisdom()
    hFT = FFTW.plan_fft(h, 1, flags=Luna.settings["fftw_flag"])
    inv(hFT)
    Utils.saveFFTwisdom()
    hω = hFT * h
    RamanPolarEnvBatched(r, h, ht, hω, hω, hFT, t[2] - t[1], length(t),
                         nothing, nothing, nothing)
end

function (R::RamanPolarEnvBatched)(out, Et, ρ)
    nt = R.nt
    size(Et, 1) == nt || error("RamanPolarEnvBatched: field time grid size $(size(Et, 1))"*
                               " does not match response grid size $nt")
    if R.B === nothing
        # The work buffer follows the field: `similar` puts it on the same device, and
        # the plans are made by the matching planner.
        R.B = similar(Et, ComplexF64, (2nt, size(Et, 2), size(Et, 3)))
        Utils.loadFFTwisdom(Utils.backend(Et))
        R.FT = Utils.plan_fft!_backend(R.B, 1)
        R.IFT = Utils.plan_ifft!_backend(R.B, 1)
        Utils.saveFFTwisdom(Utils.backend(Et))
        # The response kernel is computed on the host (it is only 2nt long); on a device
        # it needs a staging copy, which is refreshed per call below.
        Utils.isdevice(Et) && (R.hωd = similar(Et, ComplexF64, length(R.hω)))
    end
    # update the response function and its transform once per call — it depends only on
    # the scalar density, so this is exact (RamanPolarEnv recomputes it per column)
    R.r(R.ht, ρ)
    R.hω .= R.hFT * R.h
    R.hωd === R.hω || copyto!(R.hωd, R.hω)
    _raman_batched!(out, Et, ρ, R.B, R.FT, R.IFT, R.hωd, R.dt, nt)
end

# function barrier: B/FT/IFT fields are loosely typed on the struct
function _raman_batched!(out, Et, ρ, B, FT, IFT, hω, dt, nt)
    ncol = size(Et, 2)*size(Et, 3)
    Etr = reshape(Et, nt, ncol)
    Br = reshape(B, 2nt, ncol)
    outr = reshape(out, nt, ncol)
    Utils.tforeach(ncol; ntotal=length(B)) do i
        @inbounds begin
            Bcol = view(Br, :, i)
            Ecol = view(Etr, :, i)
            # squared envelope in the first half (cf. sqr!), zero padding in the second
            # (the in-place FFTs overwrite the padding, so it is rebuilt every call)
            @views @. Bcol[1:nt] = 1/2 * abs2(Ecol)
            @views fill!(Bcol[nt+1:2nt], 0)
        end
    end
    FT * B # batched (t → ω) in place
    Utils.tforeach(ncol; ntotal=length(B)) do i
        @inbounds begin
            Bcol = view(Br, :, i)
            # convolution by multiplication; dt scaling as in RamanPolar
            @. Bcol = hω * Bcol * dt
        end
    end
    IFT * B # batched (ω → t) in place
    Utils.tforeach(ncol; ntotal=length(out)) do i
        @inbounds begin
            ocol = view(outr, :, i)
            Ecol = view(Etr, :, i)
            Pcol = view(Br, 1:nt, i)
            @. ocol += ρ*Ecol*Pcol
        end
    end
end

end
