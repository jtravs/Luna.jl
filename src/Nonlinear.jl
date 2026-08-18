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
PlasmaScalar!(Plas::PlasmaCumtrapz, E) = _plasma_scalar!(
    Plas.P, Plas.rate, E, Plas.ratefunc, Plas.ionpot, Plas.preionfrac, Plas.δt)

# The scalar plasma kernel on explicit column buffers, shared between the columnwise
# response above and the threaded per-column path of PlasmaCumtrapzBatched: two passes over
# the column. Pass 1 evaluates the ionisation rate (the rate function's array method,
# vectorisable); pass 2 is one serial sweep carrying the three running trapezoid sums
# (∫W dt, the current ∫(ρE) dt and the polarisation ∫J dt) in registers, reading E once and
# writing P once. The arithmetic and its order are exactly those of the multi-pass
# reference below (`_plasma_scalar_multipass!`: cumtrapz!, broadcasts, loss loop,
# cumtrapz!, cumtrapz!), so the results are bit-identical.
function _plasma_scalar!(P, rate, E, ratefunc, ionpot, preionfrac, δt)
    ratefunc(rate, E)
    n = length(E)
    n == 0 && return P
    @inbounds begin
        # i = 1: all cumulative integrals start at zero
        F = 0.0 # ∫W dt
        f = preionfrac + 1 - exp(-F) # ionised fraction
        Ei = E[1]
        ri = rate[1]
        ph_prev = f * e_ratio * Ei # plasma "phase" (∝ ρE), the integrand of the current
        Jint = zero(ph_prev) # ∫phase dt (before the loss term)
        Jl_prev = Jint
        if abs(Ei) > 0
            Jl_prev += ionpot * ri * (1-f)/Ei
        end
        Pacc = zero(ph_prev)
        P[1] = Pacc
        r_prev = ri
        for i in 2:n
            Ei = E[i]
            ri = rate[i]
            Fnew = F + 1//2*(r_prev + ri)*δt
            # the ionised fraction only needs recomputing when the integral changed (it
            # does not over the pulse wings, where the rate is zero — most of the samples);
            # reusing exp(-F) for the same F is exact
            if Fnew != F
                F = Fnew
                f = preionfrac + 1 - exp(-F)
            end
            ph = f * e_ratio * Ei
            Jint = Jint + 1//2*(ph_prev + ph)*δt
            Jl = Jint
            # a zero rate gives a loss term of exactly ±0.0, which leaves Jl unchanged
            # (Jl is never -0.0), so the division is skipped where the rate vanishes
            if ri != 0 && abs(Ei) > 0
                Jl += ionpot * ri * (1-f)/Ei
            end
            Pacc = Pacc + 1//2*(Jl_prev + Jl)*δt
            P[i] = Pacc
            r_prev = ri; ph_prev = ph; Jl_prev = Jl
        end
    end
    P
end

# The original multi-pass form of the scalar kernel, kept as the reference the fused kernel
# is tested against (and for readers: this is the algorithm).
function _plasma_scalar_multipass!(P, J, phase, rate, fraction, E, ratefunc, ionpot,
                                   preionfrac, δt)
    ratefunc(rate, E)
    Maths.cumtrapz!(fraction, rate, δt)
    @. fraction = preionfrac + 1 - exp(-fraction)
    @. phase = fraction * e_ratio * E
    Maths.cumtrapz!(J, phase, δt)
    for ii in eachindex(E)
        if abs(E[ii]) > 0
            J[ii] += ionpot * rate[ii] * (1-fraction[ii])/E[ii]
        end
    end
    Maths.cumtrapz!(P, J, δt)
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
    _plasma_vector!(Plas.P, Plas.rate, Ex, Ey, Em,
                    Plas.ratefunc, Plas.ionpot, Plas.preionfrac, Plas.δt)
end

# The vector plasma kernel on explicit column buffers; `P` is `(nt, 2)`, `Ex`/`Ey` the
# field components and `Em` the field magnitude `hypot(Ex, Ey)`. Two passes as in the
# scalar kernel (rate, then one serial sweep with the running sums for both components);
# bit-identical to `_plasma_vector_multipass!`.
function _plasma_vector!(P, rate, Ex, Ey, Em, ratefunc, ionpot, preionfrac, δt)
    ratefunc(rate, Em)
    n = length(Em)
    n == 0 && return P
    @inbounds begin
        F = 0.0
        f = preionfrac + 1 - exp(-F)
        Exi = Ex[1]; Eyi = Ey[1]; Emi = Em[1]; ri = rate[1]
        phx_prev = f * e_ratio * Exi
        phy_prev = f * e_ratio * Eyi
        Jx = zero(phx_prev); Jy = zero(phy_prev)
        Jlx_prev = Jx; Jly_prev = Jy
        if abs(Emi) > 0
            pre = ionpot * ri * (1-f)/Emi^2
            Jlx_prev += pre*Exi
            Jly_prev += pre*Eyi
        end
        Px = zero(phx_prev); Py = zero(phy_prev)
        P[1, 1] = Px; P[1, 2] = Py
        r_prev = ri
        for i in 2:n
            Exi = Ex[i]; Eyi = Ey[i]; Emi = Em[i]; ri = rate[i]
            Fnew = F + 1//2*(r_prev + ri)*δt
            if Fnew != F # see the scalar kernel
                F = Fnew
                f = preionfrac + 1 - exp(-F)
            end
            phx = f * e_ratio * Exi
            phy = f * e_ratio * Eyi
            Jx = Jx + 1//2*(phx_prev + phx)*δt
            Jy = Jy + 1//2*(phy_prev + phy)*δt
            Jlx = Jx; Jly = Jy
            if ri != 0 && abs(Emi) > 0 # see the scalar kernel
                pre = ionpot * ri * (1-f)/Emi^2
                Jlx += pre*Exi
                Jly += pre*Eyi
            end
            Px = Px + 1//2*(Jlx_prev + Jlx)*δt
            Py = Py + 1//2*(Jly_prev + Jly)*δt
            P[i, 1] = Px; P[i, 2] = Py
            r_prev = ri; phx_prev = phx; phy_prev = phy; Jlx_prev = Jlx; Jly_prev = Jly
        end
    end
    P
end

# The original multi-pass form of the vector kernel (reference for the fused one).
function _plasma_vector_multipass!(P, J, phase, rate, fraction, E, Ex, Ey, Em,
                                   ratefunc, ionpot, preionfrac, δt)
    ratefunc(rate, Em)
    Maths.cumtrapz!(fraction, rate, δt)
    @. fraction = preionfrac + 1 - exp(-fraction)
    @. phase = fraction * e_ratio * E
    # integrate each component with the 1-D routine explicitly: the buffers may be 2-D
    # views, which would otherwise be mistaken for vectors by the 1-D method
    for c in 1:2
        Maths.cumtrapz!(view(J, :, c), view(phase, :, c), δt)
    end
    for ii in eachindex(Em)
        if abs(Em[ii]) > 0
            pre = ionpot * rate[ii] * (1-fraction[ii])/Em[ii]^2
            J[ii,1] += pre*Ex[ii]
            J[ii,2] += pre*Ey[ii]
        end
    end
    for c in 1:2
        Maths.cumtrapz!(view(P, :, c), view(J, :, c), δt)
    end
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
    batched(resp, npol)

Trait: `true` if `resp` has an array-level method for a real-space field with `npol`
polarisation components, i.e. `resp(out, Et, ρ)` with `out`/`Et` of shape
`(nt, npol, npts)` — the layout used by [`Luna.NonlinearRHS.TransModalFixed`](@ref). For a
scalar field (`npol == 1`) this is [`batched(resp)`](@ref); vector fields default to `false`.
"""
batched(resp, npol) = npol == 1 ? batched(resp) : false

# Kerr on a whole (nt, 2, npts) vector field: the same arithmetic as KerrVector!/
# KerrVectorEnv! per sample (so the results are bit-identical), as broadcasts over the
# component slices, threaded on the host and a single kernel each on a device.
batched(::KerrField, npol) = npol == 2
batched(::KerrEnv, npol) = npol == 2

function (K::KerrField)(out::AbstractArray{T, 3}, E::AbstractArray{T, 3}, ρ) where T
    size(E, 2) == 2 || error("array-level KerrField call expects a (nt, 2, npts) vector field")
    fac = ρ*ε_0*K.γ3
    Ex = view(E, :, 1, :); Ey = view(E, :, 2, :)
    Px = view(out, :, 1, :); Py = view(out, :, 2, :)
    Utils.tchunks(Px, Py, Ex, Ey) do Px, Py, Ex, Ey
        @. Px += fac*(Ex^2 + Ey^2)*Ex
        @. Py += fac*(Ex^2 + Ey^2)*Ey
    end
    out
end

function (K::KerrEnv)(out::AbstractArray{T, 3}, E::AbstractArray{T, 3}, ρ) where T
    size(E, 2) == 2 || error("array-level KerrEnv call expects a (nt, 2, npts) vector field")
    fac = ρ*ε_0*K.γ3
    Ex = view(E, :, 1, :); Ey = view(E, :, 2, :)
    Px = view(out, :, 1, :); Py = view(out, :, 2, :)
    Utils.tchunks(Px, Py, Ex, Ey) do Px, Py, Ex, Ey
        @. Px += 3/4*fac*((abs2(Ex) + 2/3*abs2(Ey))*Ex + 1/3*conj(Ex)*Ey^2)
        @. Py += 3/4*fac*((abs2(Ey) + 2/3*abs2(Ex))*Ey + 1/3*conj(Ey)*Ex^2)
    end
    out
end

"""
    PlasmaCumtrapzBatched(ratefunc, ionpot, δt; preionfrac=0.0, arraytype=Array)
    PlasmaCumtrapzBatched(P::PlasmaCumtrapz; arraytype=Array)

Array-level (batched) version of [`PlasmaCumtrapz`](@ref) for a real-space field of shape
`(nt, npol, npts)` (or `(nt, npts)` for scalar fields): the ionisation rate, the
cumulative integrals and the plasma current/polarisation are computed for all transverse
points at once, with the same arithmetic as the columnwise response. On the host the
columns are processed in parallel with the two-pass per-column kernels
(`Utils.tforeach`; only the rate and the polarisation are stored); on a device
everything is whole-array broadcasts with [`Luna.Maths.cumtrapz_scan!`](@ref) for the
cumulative integrals (a native scan where the backend provides one, e.g. `CUDA.cumsum!`,
otherwise the portable doubling scan). Buffers are allocated lazily on the field's array type at the first
call. `ratefunc` is kept as given (host callable, e.g. for `Stats.electrondensity`);
`ratefunc_dev` is its copy adapted to `arraytype` for the device kernels.
"""
mutable struct PlasmaCumtrapzBatched{R, RD}
    ratefunc::R
    ratefunc_dev::RD
    ionpot::Float64
    δt::Float64
    preionfrac::Float64
    rate::Any # (nt, npts)
    fraction::Any # (nt, npts)
    Em::Any # (nt, npts) field magnitude for vector fields, or nothing
    phase::Any # (nt, npol, npts)
    J::Any # (nt, npol, npts)
    P::Any # (nt, npol, npts)
    tmp::Any # scan scratch (nt, npts), device only (`nothing` with a native scan)
    tmp3::Any # scan scratch (nt, npol, npts), device only (`nothing` with a native scan)
end

batched(::PlasmaCumtrapzBatched) = true
batched(::PlasmaCumtrapzBatched, npol) = true

function PlasmaCumtrapzBatched(ratefunc, ionpot, δt; preionfrac=0.0, arraytype=Array)
    !(0.0 <= preionfrac <= 1.0) && throw(DomainError(preionfrac, "preionfrac must be between 0 and 1"))
    ratefunc_dev = arraytype === Array ? ratefunc : Luna.Ionisation.device_ionrate(ratefunc, arraytype)
    PlasmaCumtrapzBatched(ratefunc, ratefunc_dev, ionpot, δt, preionfrac,
                          nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

PlasmaCumtrapzBatched(P::PlasmaCumtrapz; arraytype=Array) =
    PlasmaCumtrapzBatched(P.ratefunc, P.ionpot, P.δt; preionfrac=P.preionfrac, arraytype)

batched_response(P::PlasmaCumtrapz; arraytype=Array) = PlasmaCumtrapzBatched(P; arraytype)

function _plasma_buffers!(B::PlasmaCumtrapzBatched, E3)
    nt, npol, npts = size(E3)
    if B.P === nothing || size(B.P) != size(E3)
        B.rate = similar(E3, real(eltype(E3)), (nt, npts))
        B.Em = npol == 2 ? similar(B.rate) : nothing
        B.P = similar(E3)
        if Utils.isdevice(E3)
            # the device path materialises the intermediate stages as whole arrays
            B.fraction = similar(B.rate)
            B.phase = similar(E3)
            B.J = similar(E3)
            # scratch for the prefix-sum fallback; `nothing` where the backend has a
            # native scan (see `Maths.scan_scratch`)
            B.tmp = Maths.scan_scratch(B.rate)
            B.tmp3 = Maths.scan_scratch(E3)
        else
            # the host path is the two-pass column kernel: only the rate and P are stored
            B.fraction = nothing; B.phase = nothing; B.J = nothing
            B.tmp = nothing; B.tmp3 = nothing
        end
    end
    nothing
end

function (B::PlasmaCumtrapzBatched)(out, Et, ρ)
    E3 = ndims(Et) == 3 ? Et : reshape(Et, size(Et, 1), 1, size(Et, 2))
    out3 = ndims(out) == 3 ? out : reshape(out, size(out, 1), 1, size(out, 2))
    size(E3, 2) in (1, 2) || error("PlasmaCumtrapzBatched: field must have 1 or 2 "*
                                   "polarisation components along dimension 2")
    _plasma_buffers!(B, E3)
    _plasma_batched!(Utils.backend(E3), out3, E3, ρ, B, B.rate, B.fraction, B.Em,
                     B.phase, B.J, B.P, B.tmp, B.tmp3)
    out
end

# Host: one column per task, the columnwise kernels on views into the full-size buffers
# (so no per-thread scratch is needed and the arithmetic is that of PlasmaCumtrapz).
function _plasma_batched!(::Utils.CPUBackend, out3, E3, ρ, B, rate, fraction, Em,
                          phase, J, P, tmp, tmp3)
    nt, npol, npts = size(E3)
    ratefunc = B.ratefunc; ionpot = B.ionpot; preionfrac = B.preionfrac; δt = B.δt
    # each column costs an ionisation-rate evaluation and three cumulative integrals, far
    # more than an elementwise kernel, so thread from a handful of columns upwards
    Utils.tforeach(npts; ntotal=length(E3), minlen=4*nt*npol) do i
        if npol == 1
            Ecol = view(E3, :, 1, i)
            _plasma_scalar!(view(P, :, 1, i), view(rate, :, i), Ecol,
                            ratefunc, ionpot, preionfrac, δt)
            view(out3, :, 1, i) .+= ρ .* view(P, :, 1, i)
        else
            Ex = view(E3, :, 1, i); Ey = view(E3, :, 2, i); Emc = view(Em, :, i)
            @. Emc = hypot(Ex, Ey)
            _plasma_vector!(view(P, :, :, i), view(rate, :, i), Ex, Ey, Emc,
                            ratefunc, ionpot, preionfrac, δt)
            view(out3, :, :, i) .+= ρ .* view(P, :, :, i)
        end
    end
    nothing
end

# Device: whole-array broadcasts and a prefix sum (native where the backend has one,
# otherwise the doubling scan) for the cumulative integrals.
function _plasma_batched!(::Utils.DeviceBackend, out3, E3, ρ, B, rate, fraction, Em,
                          phase, J, P, tmp, tmp3)
    nt, npol, npts = size(E3)
    ionpot = B.ionpot; preionfrac = B.preionfrac; δt = B.δt
    ir = B.ratefunc_dev
    if npol == 1
        Es = reshape(E3, nt, npts)
        Luna.Ionisation.ionrate_device!(rate, ir, Es)
    else
        Ex = view(E3, :, 1, :); Ey = view(E3, :, 2, :)
        @. Em = hypot(Ex, Ey)
        Luna.Ionisation.ionrate_device!(rate, ir, Em)
    end
    Maths.cumtrapz_scan!(fraction, rate, δt, tmp)
    @. fraction = preionfrac + 1 - exp(-fraction)
    frac3 = reshape(fraction, nt, 1, npts)
    @. phase = frac3 * e_ratio * E3
    Maths.cumtrapz_scan!(J, phase, δt, tmp3)
    rate3 = reshape(rate, nt, 1, npts)
    if npol == 1
        @. J += ifelse(abs(E3) > 0, ionpot * rate3 * (1 - frac3) / E3, zero(E3))
    else
        Em3 = reshape(Em, nt, 1, npts)
        @. J += ifelse(Em3 > 0, ionpot * rate3 * (1 - frac3) / Em3^2 * E3, zero(E3))
    end
    Maths.cumtrapz_scan!(P, J, δt, tmp3)
    @. out3 += ρ * P
    nothing
end

"""
    batched_response(resp)

Return the array-level (batched) equivalent of `resp` if one exists, otherwise `resp`
itself. Transforms which evaluate the responses on whole arrays call this on the responses
they are given (see [`batched_responses`](@ref)); the originals are kept for inspection.
"""
batched_response(resp; arraytype=Array) = resp

"""
    batched_responses(responses; arraytype=Array)

Apply [`batched_response`](@ref) to a tuple of responses, or to each tuple of a tuple of
tuples (gas mixtures). `arraytype` is the array type the responses will be evaluated on.
"""
batched_responses(responses::Tuple; arraytype=Array) =
    map(r -> batched_response(r; arraytype), responses)
batched_responses(responses::Tuple{Vararg{Tuple}}; arraytype=Array) =
    map(r -> batched_responses(r; arraytype), responses)
batched_responses(responses; arraytype=Array) =
    Tuple(batched_response(r; arraytype) for r in responses)

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
    lastρ::Float64 # density the response kernel `hω`/`hωd` was last computed for
end

batched(::RamanPolarEnvBatched) = true

# a columnwise envelope Raman response can always be replaced by its batched equivalent
# (same response function and time grid); the batched form allocates its own buffers
batched_response(R::RamanPolarEnv; arraytype=Array) =
    RamanPolarEnvBatched(range(0.0, step=R.dt, length=length(R.ht)), R.r)

function RamanPolarEnvBatched(t, r)
    h = zeros(length(t)*2) # note double grid size, see explanation in (R::RamanPolar)
    ht = view(h, 1:length(t))
    Utils.loadFFTwisdom()
    hFT = FFTW.plan_fft(h, 1, flags=Luna.settings["fftw_flag"])
    inv(hFT)
    Utils.saveFFTwisdom()
    hω = hFT * h
    RamanPolarEnvBatched(r, h, ht, hω, hω, hFT, t[2] - t[1], length(t),
                         nothing, nothing, nothing, NaN)
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
    # update the response function and its transform only when the density changed — it
    # depends on nothing else, so this is exact (RamanPolarEnv recomputes it per column).
    # It is host work (the response kernel, a 2nt-point FFT, and on a device an upload):
    # ~0.5 ms per call on a laptop for nt = 65536 and several ms on a slow host, i.e. a
    # large share of the RHS on a GPU; constant-pressure runs pay it once.
    _update_raman_kernel!(R, ρ)
    _raman_batched!(out, Et, ρ, R.B, R.FT, R.IFT, R.hωd, R.dt, nt)
end

# shared by the batched envelope and field Raman responses (both have r, ht, h, hω, hωd,
# hFT and lastρ)
function _update_raman_kernel!(R, ρ)
    ρ == R.lastρ && return nothing
    R.r(R.ht, ρ)
    R.hω .= R.hFT * R.h
    R.hωd === R.hω || copyto!(R.hωd, R.hω)
    R.lastρ = ρ
    nothing
end

# function barrier: B/FT/IFT fields are loosely typed on the struct
_raman_batched!(out, Et, ρ, B, FT, IFT, hω, dt, nt) =
    _raman_batched!(Utils.backend(out), out, Et, ρ, B, FT, IFT, hω, dt, nt)

function _raman_batched!(::Utils.CPUBackend, out, Et, ρ, B, FT, IFT, hω, dt, nt)
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

# Device: the same four elementwise stages, each as one broadcast over the whole array
# instead of a loop over transverse columns. The doubled-grid layout is unchanged, so
# this computes exactly the same convolution as the host version.
function _raman_batched!(::Utils.DeviceBackend, out, Et, ρ, B, FT, IFT, hω, dt, nt)
    # squared envelope in the first half, zero padding in the second. Views of the
    # leading dimension are contiguous, so these stay device-friendly strided arrays.
    Bfirst = view(B, 1:nt, :, :)
    Bsecond = view(B, nt+1:2nt, :, :)
    @. Bfirst = 1/2 * abs2(Et)
    fill!(Bsecond, 0)
    FT * B # batched (t → ω) in place
    # convolution by multiplication; hω expands along the (doubled) time dimension.
    # Same factor order as the host loop, so the two agree as closely as the differing
    # FFT implementations allow.
    @. B = hω * B * dt
    IFT * B # batched (ω → t) in place
    @. out += ρ*Et*Bfirst
    return nothing
end

"""
    RamanPolarFieldBatched(t, r; thg=true)

Batched (array-level) version of [`RamanPolarField`](@ref) — the Raman polarisation of a
carrier-resolved (real) field for all transverse points at once, on a field of shape
`(nt, npts)` or `(nt, 1, npts)`. The convolution is the same as in `RamanPolarField`
(squared field in the first half of a doubled real work array `B`, zero padding in the
second, one real-to-complex FFT along the time dimension, multiplication by the frequency
domain response `hω`, one complex-to-real inverse FFT), but the FFTs are batched over the
columns and the density-dependent response function is updated once per call instead of
once per column (exact: the density is a scalar). With `thg=false` the squared field is
`|E_a|²/2` from the analytic signal, computed for all columns with one batched complex FFT
pair (the batched form of `Maths.plan_hilbert`).

Results agree with `RamanPolarField` column-for-column to rounding accuracy (~1e-15
relative; batched and single-column FFT algorithms differ, so not bit-exact). On the host
the elementwise stages are threaded over columns and the FFTs are FFTW's batched plans; on
a device they are broadcasts and the device FFT library. Buffers and plans are allocated
lazily on the field's array type at the first call.
"""
mutable struct RamanPolarFieldBatched{TR, hvT, fT}
    r::TR # Raman response function
    h::Vector{Float64} # doubled buffer to hold response + padding
    ht::hvT # view into first half of h
    hω::Vector{ComplexF64} # the frequency domain Raman response function (host)
    hωd::Any # `hω` on the field's array type; `hω` itself on the host (no copy)
    hFT::fT # 1D real-to-complex Fourier transform plan for h
    thg::Bool # include third-harmonic generation (E² rather than |E_a|²/2)
    dt::Float64 # time step for scaling
    nt::Int # length of the (undoubled) time grid
    B::Any # (2nt, npts) real doubled work array, allocated lazily on the field's array type
    Bω::Any # (nt+1, npts) complex spectrum of B
    FT::Any # batched real-to-complex plan B → Bω along dim 1
    IFT::Any # its inverse (stored explicitly: an ldiv! through a device plan rebuilds it per call)
    C::Any # (nt, npts) complex work array for the analytic signal (thg=false only)
    cFT::Any # in-place complex plan along dim 1 of C
    cIFT::Any # its inverse
    lastρ::Float64 # density the response kernel `hω`/`hωd` was last computed for
end

batched(::RamanPolarFieldBatched) = true

# a columnwise real-field Raman response can always be replaced by its batched equivalent
# (same response function, time grid and thg setting); the batched form allocates its own
# buffers
batched_response(R::RamanPolarField; arraytype=Array) =
    RamanPolarFieldBatched(range(0.0, step=R.dt, length=length(R.ht)), R.r; thg=R.thg)

function RamanPolarFieldBatched(t, r; thg=true)
    h = zeros(length(t)*2) # note double grid size, see explanation in (R::RamanPolar)
    ht = view(h, 1:length(t))
    Utils.loadFFTwisdom()
    hFT = FFTW.plan_rfft(h, 1, flags=Luna.settings["fftw_flag"])
    inv(hFT)
    Utils.saveFFTwisdom()
    hω = hFT * h
    RamanPolarFieldBatched(r, h, ht, hω, hω, hFT, thg, t[2] - t[1], length(t),
                           nothing, nothing, nothing, nothing, nothing, nothing, nothing, NaN)
end

function (R::RamanPolarFieldBatched)(out, Et, ρ)
    nt = R.nt
    size(Et, 1) == nt || error("RamanPolarFieldBatched: field time grid size $(size(Et, 1))"*
                               " does not match response grid size $nt")
    ncol = length(Et) ÷ nt
    Etr = reshape(Et, nt, ncol)
    outr = reshape(out, nt, ncol)
    if R.B === nothing || size(R.B, 2) != ncol
        # The work buffers follow the field: `similar` puts them on the same device, and
        # the plans are made by the matching planner.
        R.B = similar(Et, Float64, (2nt, ncol))
        R.Bω = similar(Et, ComplexF64, (nt+1, ncol))
        Utils.loadFFTwisdom(Utils.backend(Et))
        R.FT = Utils.plan_rfft_backend(R.B, 1)
        R.IFT = inv(R.FT)
        if !R.thg
            R.C = similar(Et, ComplexF64, (nt, ncol))
            R.cFT = Utils.plan_fft!_backend(R.C, 1)
            R.cIFT = Utils.plan_ifft!_backend(R.C, 1)
        end
        Utils.saveFFTwisdom(Utils.backend(Et))
        # The response kernel is computed on the host (it is only 2nt long); on a device
        # it needs a staging copy, which is refreshed per call below.
        Utils.isdevice(Et) && (R.hωd = similar(Et, ComplexF64, length(R.hω)))
    end
    # update the response function and its transform only when the density changed (exact;
    # see _update_raman_kernel!)
    _update_raman_kernel!(R, ρ)
    _raman_field_batched!(outr, Etr, ρ, R.B, R.Bω, R.FT, R.IFT, R.hωd, R.dt, nt,
                          R.thg, R.C, R.cFT, R.cIFT)
    out
end

# function barrier: the buffer/plan fields are loosely typed on the struct
_raman_field_batched!(out, Et, ρ, B, Bω, FT, IFT, hω, dt, nt, thg, C, cFT, cIFT) =
    _raman_field_batched!(Utils.backend(out), out, Et, ρ, B, Bω, FT, IFT, hω, dt, nt,
                          thg, C, cFT, cIFT)

# The analytic signal of every column of Et into C (cf. Maths.plan_hilbert!): FFT along
# time, double the positive frequencies, zero the negative ones (and Nyquist), inverse FFT.
function _analytic_batched!(::Utils.CPUBackend, C, Et, cFT, cIFT)
    nt, ncol = size(C)
    n1 = nt ÷ 2
    Utils.tforeach(ncol; ntotal=length(C)) do i
        @inbounds begin
            Ccol = view(C, :, i)
            Ccol .= view(Et, :, i)
        end
    end
    cFT * C
    Utils.tforeach(ncol; ntotal=length(C)) do i
        @inbounds begin
            Ccol = view(C, :, i)
            @views Ccol[2:n1] .*= 2
            @views fill!(Ccol[n1+1:nt], 0)
        end
    end
    cIFT * C
    C
end

function _analytic_batched!(::Utils.DeviceBackend, C, Et, cFT, cIFT)
    nt = size(C, 1)
    n1 = nt ÷ 2
    C .= Et
    cFT * C
    view(C, 2:n1, :) .*= 2
    fill!(view(C, n1+1:nt, :), 0)
    cIFT * C
    C
end

function _raman_field_batched!(::Utils.CPUBackend, out, Et, ρ, B, Bω, FT, IFT, hω, dt,
                               nt, thg, C, cFT, cIFT)
    ncol = size(Et, 2)
    thg || _analytic_batched!(Utils.CPUBackend(), C, Et, cFT, cIFT)
    Utils.tforeach(ncol; ntotal=length(B)) do i
        @inbounds begin
            Bcol = view(B, :, i)
            # squared field in the first half (cf. sqr!), zero padding in the second
            # (the padding is rebuilt every call for uniformity with the envelope version)
            if thg
                Ecol = view(Et, :, i)
                @views @. Bcol[1:nt] = Ecol^2
            else
                Ccol = view(C, :, i)
                @views @. Bcol[1:nt] = 1/2 * abs2(Ccol)
            end
            @views fill!(Bcol[nt+1:2nt], 0)
        end
    end
    mul!(Bω, FT, B) # batched (t → ω)
    Utils.tforeach(ncol; ntotal=length(Bω)) do i
        @inbounds begin
            Bωcol = view(Bω, :, i)
            # convolution by multiplication; dt scaling as in RamanPolar
            @. Bωcol = hω * Bωcol * dt
        end
    end
    mul!(B, IFT, Bω) # batched (ω → t); c2r may destroy Bω, which is scratch
    Utils.tforeach(ncol; ntotal=length(out)) do i
        @inbounds begin
            ocol = view(out, :, i)
            Ecol = view(Et, :, i)
            Pcol = view(B, 1:nt, i)
            @. ocol += ρ*Ecol*Pcol
        end
    end
    nothing
end

# Device: the same stages as whole-array broadcasts. Views of the leading dimension are
# contiguous, so they stay device-friendly strided arrays.
function _raman_field_batched!(::Utils.DeviceBackend, out, Et, ρ, B, Bω, FT, IFT, hω, dt,
                               nt, thg, C, cFT, cIFT)
    Bfirst = view(B, 1:nt, :)
    Bsecond = view(B, nt+1:2nt, :)
    if thg
        @. Bfirst = Et^2
    else
        _analytic_batched!(Utils.DeviceBackend(), C, Et, cFT, cIFT)
        @. Bfirst = 1/2 * abs2(C)
    end
    fill!(Bsecond, 0)
    mul!(Bω, FT, B) # batched (t → ω)
    @. Bω = hω * Bω * dt # hω expands along the (doubled-grid) frequency dimension
    mul!(B, IFT, Bω) # batched (ω → t)
    @. out += ρ*Et*Bfirst
    return nothing
end

end
