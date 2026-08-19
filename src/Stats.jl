module Stats
import Luna
import Luna: Maths, Grid, Modes, Utils, settings, PhysData, Fields, Processing
import Luna.PhysData: wlfreq, c, ε_0
import Luna.NonlinearRHS
import Luna.NonlinearRHS: TransModal, TransModalFixed, TransModeAvg, Erω_to_Prω!, Et_to_Pt!,
                          to_freq!, integral_error!
import Luna.Nonlinear
import Luna.Nonlinear: PlasmaCumtrapz
import Luna.Capillary: MarcatiliMode
import FFTW
import Adapt
import LinearAlgebra: mul!, norm
import Printf: @sprintf
import Logging: @warn

"""
    ω0(grid)

Create stats function to calculate the centre of mass (first moment) of the spectral power
density.
"""
function ω0(grid; like=nothing)
    Ω0Stat(grid.ω, _like(grid.ω, like))
end

struct Ω0Stat{V, VD} <: Function
    ω::V # host frequency axis
    ωd::VD # the same on the array type of the state (device), or the host vector
end
device_capable(::Ω0Stat) = true
function (s::Ω0Stat)(d, Eω, Et, z, dz)
    if Utils.isdevice(Eω)
        I = abs2.(Eω)
        d["ω0"] = squeeze(Array(sum(s.ωd .* I; dims=1) ./ sum(I; dims=1)))
    else
        d["ω0"] = squeeze(Maths.moment(s.ω, abs2.(Eω); dim=1))
    end
end

squeeze(ω0::Array{T, 1}) where T = ω0[1]
squeeze(ω0::Array{T, 2}) where T = ω0[1, :]

"""
    energy(grid, energyfun_ω)

Create stats function to calculate the total energy.
"""
function energy(grid, energyfun_ω; like=nothing)
    w = _energy_weights(grid, energyfun_ω)
    EnergyStat(energyfun_ω, isnothing(w) ? nothing : _like(w, like))
end

"""
    _energy_weights(grid, energyfun_ω)

Quadrature weights `w` such that `energyfun_ω(Eω) == sum(w .* abs2.(Eω))` (to rounding) for
the grid's standard energy functional (`Fields.energyfuncs`: the alternative extended
Simpson rule on a `RealGrid`, a plain sum on an `EnvGrid`), verified against
`energyfun_ω` itself; `nothing` if `energyfun_ω` is not that functional (the statistic is
then host-only).
"""
function _energy_weights(grid, energyfun_ω)
    n = length(grid.ω)
    if grid isa Grid.RealGrid
        h = grid.ω[2] - grid.ω[1]
        w = fill(h, n)
        n >= 8 && (w[1:4] .= h .* [17, 59, 43, 49] ./ 48; w[end-3:end] .= h .* [49, 43, 59, 17] ./ 48)
        w .*= 2π/(grid.ω[end]^2)
    else
        δω = grid.ω[2] - grid.ω[1]
        w = fill(2π*δω/((n*δω)^2), n)
    end
    y = rand(ComplexF64, n) # verify the functional is the standard one
    ref = energyfun_ω(y)
    return isapprox(sum(w .* abs2.(y)), ref; rtol=1e-10) ? w : nothing
end

struct EnergyStat{F, W} <: Function
    energyfun_ω::F
    w::W # weights on the array type of the state, or nothing (host-only)
end
device_capable(s::EnergyStat) = !isnothing(s.w)
function (s::EnergyStat)(d, Eω, Et, z, dz)
    if Utils.isdevice(Eω)
        e = sum(s.w .* abs2.(Eω); dims=1)
        d["energy"] = ndims(Eω) > 1 ? vec(Array(e)) : Array(e)[1]
    elseif ndims(Eω) > 1
        d["energy"] = [s.energyfun_ω(Eω[:, i]) for i=1:size(Eω, 2)]
    else
        d["energy"] = s.energyfun_ω(Eω)
    end
end

"""
    energy_λ(grid, energyfun_ω, λlims; label)

Create stats function to calculate the energy in a wavelength region given by `λlims`.
If `label` is omitted, the stats dataset is named by the wavelength limits.
"""
function energy_λ(grid, energyfun_ω, λlims; label=nothing, winwidth=0, like=nothing)
    λlims = collect(λlims)
    ωmin, ωmax = extrema(wlfreq.(λlims))
    window = Maths.planck_taper(grid.ω, ωmin-winwidth, ωmin, ωmax, ωmax+winwidth)
    if isnothing(label)
        λnm = 1e9.*λlims
        label = @sprintf("%.2fnm_%.2fnm", minimum(λnm), maximum(λnm))
    end
    energy_window(grid, energyfun_ω, window; label=label, like)
end

"""
    energy_window(grid, energyfun_ω, window; label)

Create stats function to calculate the energy filtered by a `window`. The stats dataset will
be named `energy_[label]`.
"""
function energy_window(grid, energyfun_ω, window; label, like=nothing)
    w = _energy_weights(grid, energyfun_ω)
    # the window multiplies the field, so its square multiplies the weights
    EnergyWindowStat("energy_$label", energyfun_ω, window,
                     isnothing(w) ? nothing : _like(w .* abs2.(window), like))
end

struct EnergyWindowStat{F, V, W} <: Function
    key::String
    energyfun_ω::F
    window::V
    w::W # weights × window² on the array type of the state, or nothing (host-only)
end
device_capable(s::EnergyWindowStat) = !isnothing(s.w)
function (s::EnergyWindowStat)(d, Eω, Et, z, dz)
    if Utils.isdevice(Eω)
        e = sum(s.w .* abs2.(Eω); dims=1)
        d[s.key] = ndims(Eω) > 1 ? vec(Array(e)) : Array(e)[1]
    elseif ndims(Eω) > 1
        d[s.key] = [s.energyfun_ω(Eω[:, i].*s.window) for i=1:size(Eω, 2)]
    else
        d[s.key] = s.energyfun_ω(Eω.*s.window)
    end
end

"""
    peakpower(grid)

Create stats function to calculate the peak power.
"""
peakpower(grid; like=nothing) = PeakPowerStat(grid.t)

struct PeakPowerStat{T} <: Function
    t::T
end
device_capable(::PeakPowerStat) = true
function (s::PeakPowerStat)(d, Eω, Et, z, dz)
    if Utils.isdevice(Et)
        I = abs2.(Et)
        if ndims(Et) > 1
            d["peakpower"] = vec(Array(maximum(I, dims=1)))
            d["peakpower_allmodes"] = maximum(sum(I; dims=2))
        else
            d["peakpower"] = maximum(I)
        end
    elseif ndims(Et) > 1
        d["peakpower"] = dropdims(maximum(abs2.(Et), dims=1), dims=1)
        d["peakpower_allmodes"] = maximum(eachindex(s.t)) do ii
            sum(abs2, Et[ii, :])
        end
    else
        d["peakpower"] = maximum(abs2, Et)
    end
end

"""
    peakpower(grid, Eω, window; label)

Create stats function to calculate the peak power within a frequency range defined by the
window function `window`. `window` must have the same length as `grid.ω`. The stats
dataset is labeled as `peakpower_[label]`.
"""
function peakpower(grid, Eω, window::Vector{<:Real}; label)
    Etbuf, analytic! = plan_analytic(grid, Eω) # output buffer and function for inverse FT
    Eωbuf = similar(Eω) # buffer for Eω with window applied
    Pt = zeros((length(grid.t), size(Eωbuf, 2)))
    key = "peakpower_$label"
    function addstat!(d, Eω, Et, z, dz)
        Eωbuf .= Eω .* window
        analytic!(Etbuf, Eωbuf)
        if ndims(Etbuf) > 1
            Pt .= abs2.(Etbuf)
            d[key] = dropdims(maximum(Pt, dims=1), dims=1)
            d[key*"_allmodes"] = maximum(eachindex(grid.t)) do ii
                sum(Pt[ii, :]; dims=2)
            end
        else
            d[key] = maximum(abs2, Etbuf)
        end
    end
end

"""
    peakpower(grid, Eω, λlims; label=nothing)

Create stats function to calculate the peak power within a frequency range defined by the
wavelength limits `λlims`. If `label` is given, the stats dataset is labeled as
`peakpower_[label]`, otherwise `label` is created automatically from `λlims`.
"""
function peakpower(grid, Eω, λlims::NTuple{2, <:Real}; label=nothing, winwidth=:auto)
    window = Processing.ωwindow_λ(grid.ω, λlims; winwidth=winwidth)
    if isnothing(label)
        λnm = 1e9.*λlims
        label = @sprintf("%.2fnm_%.2fnm", minimum(λnm), maximum(λnm))
    end
    peakpower(grid, Eω, window; label=label)
end


"""
    peakintensity(grid, aeff)

Create stats function to calculate the mode-averaged peak intensity given the effective area
`aeff(z)`.
"""
function peakintensity(grid, aeff)
    function addstat!(d, Eω, Et, z, dz)
        d["peakintensity"] = maximum(abs2, Et)/aeff(z)
    end
end

"""
    peakintensity(grid, mode)

Create stats function to calculate the peak intensity for several modes.
"""
function peakintensity(grid, modes::Modes.ModeCollection; components=:y, like=nothing)
    tospace = Modes.ToSpace(modes, components=components)
    PeakIntensityStat(tospace, zeros(ComplexF64, (length(grid.t), tospace.npol)),
                      OnAxisModes(tospace, like), nothing)
end

"""
    OnAxisModes(tospace, like)

The normalised mode fields at the axis `(0, 0)` as a `(nmodes × npol)` matrix on the array
type of `like`, re-evaluated only when `z` changes (and never for `z`-independent modes),
so that the on-axis field `Et * Ems` is one small matrix product on the device.
"""
mutable struct OnAxisModes{T}
    ts::T
    zconstant::Bool
    Ems::Any # (nmodes × npol) on the array type of the state
    zlast::Float64
end
function OnAxisModes(ts::Modes.ToSpace, like)
    Ems = Modes.mode_matrix(ts.ms, ts.indices, [(0.0, 0.0)]; z=0.0)[:, :, 1]
    OnAxisModes(ts, all(Modes.zconstant, ts.ms), _like(Ems, like), 0.0)
end
function (o::OnAxisModes)(z)
    if !o.zconstant && z != o.zlast
        copyto!(o.Ems, Modes.mode_matrix(o.ts.ms, o.ts.indices, [(0.0, 0.0)]; z)[:, :, 1])
        o.zlast = z
    end
    o.Ems
end

mutable struct PeakIntensityStat{T, E, O} <: Function
    tospace::T
    Et0::E # host (nt × npol) buffer
    onaxis::O
    Et0d::Any # device (nt × npol) buffer, allocated on first use
end
device_capable(::PeakIntensityStat) = true
function (s::PeakIntensityStat)(d, Eω, Et, z, dz)
    npol = s.tospace.npol
    if Utils.isdevice(Et)
        isnothing(s.Et0d) && (s.Et0d = similar(Et, ComplexF64, (size(Et, 1), npol)))
        mul!(s.Et0d, Et, s.onaxis(z))
        I = abs2.(s.Et0d)
        d["peakintensity"] = c*ε_0/2 * (npol > 1 ? maximum(sum(I; dims=2)) : maximum(I))
        return
    end
    Modes.to_space!(s.Et0, Et, (0, 0), s.tospace; z=z)
    if npol > 1
        d["peakintensity"] = c*ε_0/2 * maximum(axes(s.Et0, 1)) do ii
            sum(abs2, s.Et0[ii, :])
        end
    else
        d["peakintensity"] = c*ε_0/2 * maximum(abs2, s.Et0)
    end
end

"""
    fwhm_t(grid)

Create stats function to calculate the temporal FWHM (pulse duration) for mode average.
"""
fwhm_t(grid; like=nothing) = FWHMtStat(grid.t, nothing)

mutable struct FWHMtStat{T} <: Function
    t::T
    Pth::Any # host buffer for |Et|² of a device field
end
device_capable(::FWHMtStat) = true
function (s::FWHMtStat)(d, Eω, Et, z, dz)
    if Utils.isdevice(Et)
        # the crossing search is a host scalar routine; |Et|² is formed on the device and
        # copied down (nt × nmodes reals)
        (s.Pth isa Array && size(s.Pth) == size(Et)) || (s.Pth = Array{Float64}(undef, size(Et)))
        Pt = copyto!(s.Pth, abs2.(Et))
    else
        Pt = abs2.(Et)
    end
    if ndims(Et) > 1
        Ptsum = dropdims(sum(Pt; dims=2); dims=2)
        d["fwhm_t_min"] = [Maths.fwhm(s.t, Pt[:, i], method=:linear)
                          for i=1:size(Et, 2)]
        d["fwhm_t_max"] = [Maths.fwhm(s.t, Pt[:, i], method=:linear, minmax=:max)
                          for i=1:size(Et, 2)]
        d["fwhm_t_min_allmodes"] = Maths.fwhm(s.t, Ptsum, method=:linear)
        d["fwhm_t_max_allmodes"] = Maths.fwhm(s.t, Ptsum, method=:linear, minmax=:max)
    else
        d["fwhm_t_min"] = Maths.fwhm(s.t, Pt, method=:linear, minmax=:min)
        d["fwhm_t_max"] = Maths.fwhm(s.t, Pt, method=:linear, minmax=:max)
    end
end

"""
    fwhm_r(grid, modes; components=:y)

Create stats function to calculate the radial FWHM (aka beam size) in a modal propagation.

The radial profile of the spectrally integrated intensity, ``Σ_ω |Σ_m E_m(ω) e_m(r)|² =
Σ_p e_p(r)ᵀ G e_p(r)`` with the mode coherency matrix ``G = Eᴴ E`` (``M × M``), is formed
from one small matrix product per evaluation (on the device for a device state) and the
half-maximum radius is then found on the host by evaluating only the mode fields — instead
of re-synthesising the whole spectrum at every trial radius.
"""
function fwhm_r(grid, modes; components=:y, like=nothing)
    tospace = Modes.ToSpace(modes, components=components)
    FWHMrStat(tospace, Matrix{ComplexF64}(undef, tospace.nmodes, tospace.nmodes), nothing)
end

mutable struct FWHMrStat{T} <: Function
    tospace::T
    G::Matrix{ComplexF64} # host coherency matrix
    Gd::Any # device coherency matrix buffer
end
device_capable(::FWHMrStat) = true
function (s::FWHMrStat)(d, Eω, Et, z, dz)
    if Utils.isdevice(Eω)
        isnothing(s.Gd) && (s.Gd = similar(Eω, ComplexF64, size(s.G)))
        mul!(s.Gd, Eω', Eω)
        copyto!(s.G, s.Gd)
    else
        mul!(s.G, Eω', Eω)
    end
    ts = s.tospace
    function f(r)
        e = Modes.mode_matrix(ts.ms, ts.indices, [(r, 0.0)]; z)[:, :, 1] # (nmodes × npol)
        real(sum(e[:, p]' * s.G * e[:, p] for p in axes(e, 2)))
    end
    d["fwhm_r"] = 2*Maths.hwhm(f)
end

"""
    electrondensity(grid, ionrate, dfun, aeff; oversampling=1)

Create stats function to calculate the maximum electron density in mode average.

If oversampling > 1, the field is oversampled before the calculation
!!! warning
    Oversampling can lead to a significant performance hit
"""
function electrondensity(grid::Grid.RealGrid, ionrate!, dfun, aeff; oversampling=1)
    to, Eto = Maths.oversample(grid.t, complex(grid.t), factor=oversampling)
    δt = to[2] - to[1]
    # ionfrac! stores the time-dependent ionisation fraction in out and returns the max
    # ionisation rate
    function ionfrac!(out, Et)
        ionrate!(out, Et)
        ratemax = maximum(out)
        Maths.cumtrapz!(out, δt) # in-place cumulative integration
        @. out = 1 - exp(-out)
        return ratemax
    end
    frac = similar(to)
    function addstat!(d, Eω, Et, z, dz)
        # note: oversampling returns its arguments without any work done if factor==1
        to, Eto = Maths.oversample(grid.t, Et, factor=oversampling)
        @. Eto /= sqrt(ε_0*c*aeff(z)/2)
        ratemax = ionfrac!(frac, real(Eto))
        d["electrondensity"] = frac[end]*dfun(z)
        d["peak_ionisation_rate"] = ratemax
    end
end

"""
    electrondensity(grid, ionrate, dfun, modes; oversampling=1)

Create stats function to calculate the maximum electron density for multimode simulations.

If oversampling > 1, the field is oversampled before the calculation
!!! warning
    Oversampling can lead to a significant performance hit
"""
function electrondensity(grid::Grid.RealGrid, ionrate!, dfun,
                         modes::Modes.ModeCollection;
                         components=:y, oversampling=1, like=nothing, ionrate_dev=nothing)
    to, Eto = Maths.oversample(grid.t, complex(grid.t), factor=oversampling)
    δt = to[2] - to[1]
    tospace = Modes.ToSpace(modes, components=components)
    npol = tospace.npol
    # the device path needs the rate's tables on the device: take the caller's adapted
    # copy (the transform's batched plasma response has one) or adapt it here
    if isnothing(ionrate_dev) && Utils.isdevice(like) && oversampling == 1
        ionrate_dev = Luna.Ionisation.device_ionrate(ionrate!, Base.typename(typeof(like)).wrapper)
    end
    ElectronDensityStat(grid.t, oversampling, δt, ionrate!, ionrate_dev, dfun, tospace,
                        similar(to), zeros(ComplexF64, (length(to), npol)),
                        OnAxisModes(tospace, like), nothing, nothing, nothing, nothing)
end

mutable struct ElectronDensityStat{T, R, RD, D, TS, O} <: Function
    t::T
    oversampling::Int
    δt::Float64
    ionrate!::R
    ionrate_dev::RD # the rate adapted to the device, or nothing (host-only)
    dfun::D
    tospace::TS
    frac::Vector{Float64} # host buffers
    Et0::Matrix{ComplexF64}
    onaxis::O
    Et0d::Any # device buffers, allocated on first use
    Ed::Any
    rated::Any
    tmpd::Any
end
device_capable(s::ElectronDensityStat) = !isnothing(s.ionrate_dev) && s.oversampling == 1

# ionfrac! stores the time-dependent ionisation fraction in out and returns the max
# ionisation rate
function _ionfrac!(out, Et, ionrate!, δt)
    ionrate!(out, Et)
    ratemax = maximum(out)
    Maths.cumtrapz!(out, δt) # in-place cumulative integration
    @. out = 1 - exp(-out)
    return ratemax
end

function (s::ElectronDensityStat)(d, Eω, Et, z, dz)
    npol = s.tospace.npol
    if Utils.isdevice(Et)
        nt = size(Et, 1)
        if isnothing(s.Et0d)
            s.Et0d = similar(Et, ComplexF64, (nt, npol))
            s.Ed = similar(Et, Float64, nt)
            s.rated = similar(s.Ed)
            s.tmpd = Maths.scan_scratch(s.Ed)
        end
        mul!(s.Et0d, Et, s.onaxis(z)) # on-axis analytic field
        if npol > 1
            s.Ed .= hypot.(real.(view(s.Et0d, :, 1)), real.(view(s.Et0d, :, 2)))
        else
            s.Ed .= real.(view(s.Et0d, :, 1))
        end
        Luna.Ionisation.ionrate_device!(s.rated, s.ionrate_dev, s.Ed)
        ratemax = maximum(s.rated)
        # only the final value of the cumulative integral is needed: the plain sum with the
        # same trapezoid end corrections, one reduction instead of a scan
        F = s.δt*(sum(s.rated) - 0.5*(sum(view(s.rated, 1:1)) + sum(view(s.rated, nt:nt))))
        d["electrondensity"] = (1 - exp(-F))*s.dfun(z)
        d["peak_ionisation_rate"] = ratemax
        return
    end
    # note: oversampling returns its arguments without any work done if factor==1
    to, Eto = Maths.oversample(s.t, Et, factor=s.oversampling)
    Modes.to_space!(s.Et0, Eto, (0, 0), s.tospace; z=z)
    if npol > 1
        ratemax = _ionfrac!(s.frac, hypot.(real(s.Et0[:, 1]), real(s.Et0[:, 2])), s.ionrate!, s.δt)
    else
        ratemax = _ionfrac!(s.frac, real(s.Et0[:, 1]), s.ionrate!, s.δt)
    end
    d["electrondensity"] = s.frac[end]*s.dfun(z)
    d["peak_ionisation_rate"] = ratemax
end

"""
    mode_reconstruction_error(t::TransModal)

Create a stats function to calculate and collect the mode reconstruction error in the
induced polarisation on axis at every step.
"""
function mode_reconstruction_error(t::TransModal)
    Prω_recon = similar(t.Prω)
    difference = similar(Prω_recon)
    nl = similar(t.Emω)
    function addstat!(d, Eω, Et, z, dz)
        t(nl, Eω, z)
        x = (0.0, 0.0) # on-axis coordinate
        # reconstruct 
        Modes.to_space!(Prω_recon, nl, x, t.ts, z=z)
        # in going to modes and back we've picked up two factors of the mode normalisation
        Prω_recon .*= 1/2*sqrt(PhysData.ε_0/PhysData.μ_0)
        Erω_to_Prω!(t, x)
        difference .= Prω_recon .- t.Prω
        d["mode_reconstruction_error"] = sqrt(sum(abs2, difference))/sqrt(sum(abs2, Prω_recon))
        d["transverse_points"] = float(t.ncalls) # convert to Float64 to enable NaN padding
        d["transverse_integral_error_abs"] = sqrt(sum(abs2, t.err)/length(t.err))
        d["transverse_integral_error_rel"] = d["transverse_integral_error_abs"]/sqrt(sum(abs2, nl)/length(nl))
    end
end

"""
    mode_reconstruction_error(t::TransModalFixed; window=nothing)

Create a stats function for a fixed-quadrature modal transform which collects, at every
step, the mode reconstruction error of the on-axis nonlinear polarisation (the modal
expansion of the polarisation projected back to `r=0` against the polarisation evaluated
directly there), the number of quadrature nodes, and the embedded quadrature error
estimate of the transform ([`Luna.NonlinearRHS.integral_error!`](@ref)) as an RMS value, relative
to the RMS polarisation, and — if `window=(λmin, λmax)` is given — relative to the
polarisation within that wavelength window (`"transverse_integral_error_rel_window"`), which
is the relevant measure for weak spectral features such as dispersive waves.

The transform is evaluated once per step (cheap for the fixed rule). On a device the
statistic asks for the propagating array itself (see [`wants_state`](@ref)), so the
transform runs on the device state directly and only the small results are copied back;
if it is nevertheless handed a host array (e.g. an `HDF5Output` with `cache=true`, which
needs the host copy anyway), the modal field is uploaded for the evaluation.
"""
function mode_reconstruction_error(t::TransModalFixed; window=nothing)
    grid = t.grid
    nω, nmodes = size(t.err)
    nto = length(grid.to)
    npol = t.ts.npol
    tT = eltype(t.Emt)
    ondevice = Utils.isdevice(t.err)
    nl = similar(t.err)
    Eωd = ondevice ? similar(t.err) : nothing # upload buffer, only used for a host state
    nlh = Array{ComplexF64}(undef, nω, nmodes)
    errh = Array{ComplexF64}(undef, nω, nmodes)
    Emth = Array{tT}(undef, nto, nmodes)
    Er = Array{tT}(undef, nto, npol)
    Pr = similar(Er)
    Prω = Array{ComplexF64}(undef, nω, npol)
    Prωo = Array{ComplexF64}(undef, length(grid.ωo), npol)
    Prω_recon = similar(Prω)
    Utils.loadFFTwisdom()
    FTo = tT <: Real ? FFTW.plan_rfft(Pr, 1, flags=settings["fftw_flag"]) :
                       FFTW.plan_fft(Pr, 1, flags=settings["fftw_flag"])
    Utils.saveFFTwisdom()
    # the normalisation factor of the transform as a host vector (it may live on a device)
    nfac = similar(t.err, nω, 1); fill!(nfac, 1); t.norm!(nfac)
    nfach = Adapt.adapt(Array, nfac)
    windowidcs = isnothing(window) ? nothing :
                 (window[1] .<= wlfreq.(grid.ω) .<= window[2]) .& grid.sidx
    # device path: the on-axis reference polarisation is evaluated on the device with the
    # batched responses (own copies — the transform's are sized for all nodes), and every
    # norm is a device reduction; nothing but scalars comes back
    if ondevice
        arraytype = Base.typename(typeof(t.err)).wrapper
        respd = Nonlinear.batched_responses(t.resp; arraytype)
        onaxis = OnAxisModes(t.ts, t.err)
        Erd = similar(t.Emt, tT, (nto, npol)); Prd = similar(Erd)
        Prωd = similar(t.err, ComplexF64, (nω, npol)); Prωod = similar(t.err, ComplexF64, (length(grid.ωo), npol))
        Prω_recond = similar(Prωd)
        FTd = NonlinearRHS._plan_forward(Prd, 1)
        maskd = isnothing(windowidcs) ? nothing : _like(Float64.(windowidcs), t.err)
        recon_fac = 1/2*sqrt(PhysData.ε_0/PhysData.μ_0)
    end
    function addstat!(d, Eω, Et, z, dz, state)
        if ondevice && Utils.isdevice(state)
            t(nl, state, z)
            Ems0 = onaxis(z)
            mul!(Prω_recond, nl, Ems0); Prω_recond .*= recon_fac
            mul!(Erd, isnothing(t.Emt_noise) ? t.Emt : t.Emt_nl, Ems0)
            NonlinearRHS.apply_responses!(Prd, Erd, respd, t.density, npol)
            Prd .*= t.gv.towin
            to_freq!(Prωd, Prωod, Prd, FTd)
            Prωd .*= t.gv.ωwin
            t.norm!(Prωd)
            d["mode_reconstruction_error"] = sqrt(sum(abs2, Prω_recond .- Prωd)/sum(abs2, Prω_recond))
            d["transverse_points"] = float(t.ncalls)
            err = integral_error!(t)
            nlsq = sum(abs2, nl)
            d["transverse_integral_error_abs"] = sqrt(sum(abs2, err)/length(err))
            d["transverse_integral_error_rel"] = d["transverse_integral_error_abs"]/sqrt(nlsq/length(nl))
            if !isnothing(maskd)
                d["transverse_integral_error_rel_window"] = sqrt(sum(abs2, err .* maskd)/sum(abs2, nl .* maskd))
            end
            return
        end
        if ondevice # a host copy was handed over: upload it
            copyto!(Eωd, state)
            t(nl, Eωd, z)
        else
            t(nl, state, z) # the propagating array itself
        end
        copyto!(nlh, nl)
        # reconstruct the on-axis polarisation from its modal expansion
        Modes.to_space!(Prω_recon, nlh, (0.0, 0.0), t.ts, z=z)
        # in going to modes and back we've picked up two factors of the mode normalisation
        Prω_recon .*= 1/2*sqrt(PhysData.ε_0/PhysData.μ_0)
        # and evaluate it directly on axis with the columnwise responses
        copyto!(Emth, isnothing(t.Emt_noise) ? t.Emt : t.Emt_nl)
        Ems0 = Modes.mode_matrix(t.ts.ms, t.ts.indices, [(0.0, 0.0)]; z)[:, :, 1]
        mul!(Er, Emth, Ems0)
        Et_to_Pt!(Pr, Er, t.resp, t.density)
        Pr .*= grid.towin
        to_freq!(Prω, Prωo, Pr, FTo)
        Prω .*= grid.ωwin
        Prω .*= nfach
        d["mode_reconstruction_error"] = norm(Prω_recon .- Prω)/norm(Prω_recon)
        d["transverse_points"] = float(t.ncalls)
        copyto!(errh, integral_error!(t))
        d["transverse_integral_error_abs"] = sqrt(sum(abs2, errh)/length(errh))
        d["transverse_integral_error_rel"] = d["transverse_integral_error_abs"]/sqrt(sum(abs2, nlh)/length(nlh))
        if !isnothing(windowidcs)
            d["transverse_integral_error_rel_window"] = norm(errh[windowidcs, :])/norm(nlh[windowidcs, :])
        end
    end
    StateStat(addstat!)
end

"""
    StateStat(f)

A statistics function which, besides the usual `(d, Eω, Et, z, dz)` (host arrays), also
receives the propagating array itself as a sixth argument — on a device run the device
array, otherwise the same host array — so it can run computations on the device state
without a host round trip. `f` is called as `f(d, Eω, Et, z, dz, state)`.
See [`wants_state`](@ref).
"""
struct StateStat{F} <: Function
    f::F
end
(s::StateStat)(d, Eω, Et, z, dz, state) = s.f(d, Eω, Et, z, dz, state)
(s::StateStat)(d, Eω, Et, z, dz) = s.f(d, Eω, Et, z, dz, Eω) # standalone use: host state
# the mode-error statistic (the only StateStat) ignores Eω/Et for a device state
device_capable(::StateStat) = true

"""
    wants_state(f) -> Bool

Trait: `true` if the statistics function `f` takes the propagating array as a sixth
argument (see [`StateStat`](@ref)); `collect_stats` then calls it as
`f(d, Eω, Et, z, dz, state)`.
"""
wants_state(f) = false
wants_state(::StateStat) = true

"""
    density(dfun)

Create stats function to capture the gas density as defined by `dfun(z)`
"""
function density(dfun)
    FieldFree() do d, Eω, Et, z, dz
        d["density"] = dfun(z)
    end
end

"""
    pressure(dfun, gas)

Create stats function to capture the pressure. Like [`density`](@ref) but converts to
pressure.
"""
function pressure(dfun, gas)
    FieldFree() do d, Eω, Et, z, dz
        d["pressure"] = PhysData.pressure(gas, dfun(z))
    end
end

function pressure(dfun, gases::Tuple)
    FieldFree() do d, Eω, Et, z, dz
        dens = dfun(z)
        for (di, gi) in zip(dens, gases)
            d["pressure_$gi"] = PhysData.pressure(gi, di)
        end
    end
end

"""
    core_radius(a)

Create stats function to capture core radius as defined by `a` (either a `Number` or a 
callable `a(z)`)
"""
function core_radius(a::Number)
    FieldFree() do d, Eω, Et, z, dz
        d["core_radius"] = a
    end
end

function core_radius(afun)
    FieldFree() do d, Eω, Et, z, dz
        d["core_radius"] = afun(z)
    end
end

"""
    zdw(mode)

Create stats function to capture the zero-dispersion wavelength (ZDW).

!!! warning
    Since [`Modes.zdw`](@ref) is based on root-finding of a derivative, this can be slow!
"""
function zdw(mode::Modes.AbstractMode; λmin=100e-9, λmax=3000e-9)
    λ00 = Modes.zdw(mode; λmin=λmin, λmax=λmax, z=0)
    FieldFree() do d, Eω, Et, z, dz
        d["zdw"] = missnan(Modes.zdw(mode, λ00; z=z))
        λ00 = d["zdw"]
    end
end

"""
    zdw(mode)

Create stats function to capture the zero-dispersion wavelength (ZDW).

!!! warning
    Since [`Modes.zdw`](@ref) is based on root-finding of a derivative, this can be slow!
"""
function zdw(modes; λmin=100e-9, λmax=3000e-9)
    λ00 = zeros(length(modes))
    for (ii, mode) in enumerate(modes)
        tmp = Modes.zdw(mode; λmin=λmin, λmax=λmax, z=0)
        if ismissing(tmp)
            λ00[ii] = λmin
        else
            λ00[ii] = tmp
        end
    end
    FieldFree() do d, Eω, Et, z, dz
        d["zdw"] = [missnan(Modes.zdw(modes[ii], λ00[ii]; z=z)) for ii in eachindex(modes)]
        λ00 .= d["zdw"]
    end
end

# convert missing to NaN
missnan(x) = ismissing(x) ? NaN : x

function zdz!(d, Eω, Et, z, dz)
    d["z"] = z
    d["dz"] = dz
end

"""
    plan_analytic(grid, Eω)

Plan a transform from the frequency-domain field `Eω` to the analytic time-domain field.

Returns both a buffer for the analytic field and a closure to do the transform.
"""
function plan_analytic(grid::Grid.EnvGrid, Eω)
    Utils.isdevice(Eω) && return _plan_analytic_device(grid, Eω)
    Eta = similar(Eω)
    Utils.loadFFTwisdom()
    iFT = FFTW.plan_ifft(copy(Eω), 1, flags=settings["fftw_flag"])
    Utils.saveFFTwisdom()
    function analytic!(Eta, Eω)
        mul!(Eta, iFT, Eω) # for envelope fields, we only need to do the inverse transform
    end
    return Eta, analytic!
end

function plan_analytic(grid::Grid.RealGrid, Eω)
    Utils.isdevice(Eω) && return _plan_analytic_device(grid, Eω)
    s = collect(size(Eω))
    s[1] = (length(grid.ω) - 1)*2 # e.g. for 4097 rFFT samples, we need 8192 FFT samples
    Eta = Array{ComplexF64, ndims(Eω)}(undef, Tuple(s))
    Eωa = zero(Eta)
    idxhi = CartesianIndices(size(Eω)[2:end]) # index over all other dimensions
    Utils.loadFFTwisdom()
    iFT = FFTW.plan_ifft(Eωa, 1, flags=settings["fftw_flag"])
    Utils.saveFFTwisdom()
    function analytic!(Eta, Eω)
        copyto_fft!(Eωa, Eω, idxhi) # copy across to FFT-sampled buffer
        mul!(Eta, iFT, Eωa) # now do the inverse transform
    end
    return Eta, analytic!
end

# Device versions: the same transforms as broadcasts and a device FFT plan (the inverse is
# stored explicitly — see NonlinearRHS.TransFree for why)
function _plan_analytic_device(grid::Grid.EnvGrid, Eω)
    Eta = similar(Eω)
    FT = Utils.plan_fft_backend(Eta, 1)
    IFT = inv(FT)
    analytic!(Eta, Eω) = mul!(Eta, IFT, Eω)
    return Eta, analytic!
end

function _plan_analytic_device(grid::Grid.RealGrid, Eω)
    s = collect(size(Eω))
    s[1] = (length(grid.ω) - 1)*2
    Eta = similar(Eω, ComplexF64, Tuple(s))
    Eωa = similar(Eta)
    fill!(Eωa, 0)
    FT = Utils.plan_fft_backend(Eta, 1)
    IFT = inv(FT)
    n = length(grid.ω) - 1 # cf. copyto_fft!: the rFFT's +fs/2 sample has no FFT counterpart
    function analytic!(Eta, Eω)
        selectdim(Eωa, 1, 1:n) .= 2 .* selectdim(Eω, 1, 1:n)
        mul!(Eta, IFT, Eωa)
    end
    return Eta, analytic!
end

"""
    copyto_fft!(Eωa, Eω, idxhi)

Copy the rFFT-sampled field `Eω` to the FFT-sampled buffer `Eωa`, ready for inverse FFT
"""
function copyto_fft!(Eωa, Eω, idxhi)
    n = size(Eω, 1)-1 # rFFT has sample at +fs/2, but FFT does not (only at -fs/2)
    for idx in idxhi
        for i in 1:n
            Eωa[i, idx] = 2*Eω[i, idx]
        end
    end
end

"""
    collect_stats(grid, Eω, funcs...)

Create a closure which collects statistics from the individual functions in `funcs`.

Each function given will be called with the arguments `(d, Eω, Et, z, dz)`, where
- d -> dictionary to store statistics values. each `func` should **mutate** this
- Eω -> frequency-domain field
- Et -> analytic time-domain field
- z -> current propagation distance
- dz -> current stepsize
"""
function collect_stats(grid, Eω, funcs...)
    # make sure z and dz are recorded
    if !(zdz! in funcs)
        funcs = (funcs..., zdz!)
    end
    if Utils.isdevice(Eω)
        # The state lives on a device. Statistics which are device-capable (see
        # `device_capable`) receive the device array and the analytic field computed on
        # the device; only if some statistic is host-only does the collector also keep a
        # host copy and a host analytic transform (copied every call).
        Etd, analyticd! = plan_analytic(grid, Eω)
        hostneeded = !all(device_capable, funcs)
        if hostneeded
            Eωh = Adapt.adapt(Array, Eω)
            Et, analytic! = plan_analytic(grid, Eωh)
        else
            Eωh = nothing; Et = nothing; analytic! = nothing
        end
        return StatsCollector(funcs, Eωh, Et, analytic!, Etd, analyticd!, hostneeded)
    end
    Et, analytic! = plan_analytic(grid, Eω)
    StatsCollector(funcs, nothing, Et, analytic!, nothing, nothing, false)
end

"""
    StatsCollector

The callable returned by [`collect_stats`](@ref): `(c::StatsCollector)(Eω, z, dz)` returns
the statistics `Dict`. Accepts the propagating array on the host or on a device. On a
device the analytic field is computed there and [`device_capable`](@ref) statistics run
on the device arrays (returning only their scalar results); host-only statistics receive
a host copy of the state and a host analytic field, which the collector maintains only if
such statistics are present. Statistics with [`wants_state`](@ref) additionally receive
the propagating array itself.
"""
mutable struct StatsCollector{F} <: Function
    funcs::F
    Eωh::Any # host copy of a device state (only if a host-only statistic is present)
    Et::Any # host analytic field buffer
    analytic!::Any # host analytic transform
    Etd::Any # device analytic field buffer (device states)
    analyticd!::Any # device analytic transform
    hostneeded::Bool # whether a host copy is made for a device state
end

function (c::StatsCollector)(Eω, z, dz)
    d = Dict{String, Any}()
    if Utils.isdevice(Eω)
        c.analyticd!(c.Etd, Eω)
        if c.hostneeded
            c.analytic!(c.Et, copyto!(c.Eωh, Eω))
        end
        for func in c.funcs
            if device_capable(func)
                wants_state(func) ? func(d, Eω, c.Etd, z, dz, Eω) : func(d, Eω, c.Etd, z, dz)
            else
                wants_state(func) ? func(d, c.Eωh, c.Et, z, dz, Eω) : func(d, c.Eωh, c.Et, z, dz)
            end
        end
        return d
    end
    c.analytic!(c.Et, Eω)
    for func in c.funcs
        wants_state(func) ? func(d, Eω, c.Et, z, dz, Eω) : func(d, Eω, c.Et, z, dz)
    end
    return d
end

# the collector handles device arrays itself (see Luna.device_stats / HostOutput)
Luna.device_stats(::StatsCollector) = true

"""
    device_capable(f) -> Bool

Trait: `true` if the statistics function `f` accepts the propagating field and the analytic
field as **device** arrays (writing its results as host scalars/small arrays into the
dictionary). The default statistics are device-capable; an unknown (user) function is
assumed host-only, and the collector then keeps a host copy of the state for it.
"""
device_capable(f) = false

"""
    FieldFree(f)

Mark a statistics function `f(d, Eω, Et, z, dz)` as not reading the field arrays at all
(e.g. density, pressure, ZDW), so that it is trivially [`device_capable`](@ref).
"""
struct FieldFree{F} <: Function
    f::F
end
(s::FieldFree)(d, Eω, Et, z, dz) = s.f(d, Eω, Et, z, dz)
device_capable(::FieldFree) = true
device_capable(::typeof(zdz!)) = true

# a host array on the array type of `like` (a copy on a device, the array itself on the host)
_like(x::AbstractArray, like) = Utils.isdevice(like) ? copyto!(similar(like, eltype(x), size(x)), x) : x
_like(x::AbstractArray, ::Nothing) = x

function default(grid, Eω, mode::Modes.AbstractMode, linop, transform;
                 windows=nothing, gas=nothing, onaxis=false, userfuns=Any[])
    Eω = Adapt.adapt(Array, Eω)
    _, energyfunω = Fields.energyfuncs(grid)
    funs = [ω0(grid), energy(grid, energyfunω), peakpower(grid),
            fwhm_t(grid), zdw_linop(mode, linop),
            density(transform.densityfun)]
    if !isnothing(gas)
        push!(funs, pressure(transform.densityfun, gas))
    end
    if onaxis
        push!(funs, peakintensity(grid, (mode,)))
    else
        push!(funs, peakintensity(grid, transform.aeff))
    end
    for resp in transform.resp
        if resp isa PlasmaCumtrapz
            ir = resp.ratefunc
            if onaxis
                push!(funs, electrondensity(grid, resp.ratefunc, transform.densityfun, (mode,)))
            else
                push!(funs, electrondensity(grid, resp.ratefunc, transform.densityfun, transform.aeff))
            end
        end
    end
    if !isnothing(windows)
        for win in windows
            push!(funs, energy_λ(grid, energyfunω, win))
        end
    end
    for (idx, uf) in enumerate(userfuns)
        if uf in funs
            @warn("userfun $idx is already present in the default set and will be ignored")
        else
            push!(funs, uf)
        end
    end
    collect_stats(grid, Eω, funs...)
end

"""
    default(grid, Eω, modes, linop, transform; kwargs...)

Default statistics for multimode propagation.

# Keyword arguments
- `windows`: wavelength windows `(λmin, λmax)` for which to record the energy.
- `gas`: gas species, to record the pressure.
- `mode_error=true`: record the mode reconstruction error and transverse integral error
  (see [`mode_reconstruction_error`](@ref)).
- `error_window=nothing`: wavelength window `(λmin, λmax)` for which to record the
  transverse integral error relative to the polarisation inside that window (fixed
  quadrature transforms only).
- `userfuns`: additional statistics functions.
"""
function default(grid, Eω, modes::Modes.ModeCollection, linop, transform;
                 windows=nothing, gas=nothing, mode_error=true, error_window=nothing,
                 userfuns=Any[])
    # `like`: the propagating array — on a device the default statistics keep their
    # constant vectors there and run on the device (see `device_capable`)
    like = Eω
    _, energyfunω = Fields.energyfuncs(grid)
    pol = transform.ts.indices == 1:2 ? :xy : transform.ts.indices == 1 ? :x : :y
    funs = [ω0(grid; like), energy(grid, energyfunω; like), peakpower(grid; like),
            peakintensity(grid, modes; components=pol, like), fwhm_t(grid; like),
            zdw_linop(modes, linop), density(transform.densityfun),
            fwhm_r(grid, modes; components=pol, like)]
    if !isnothing(gas)
        push!(funs, pressure(transform.densityfun, gas))
    end
    if mode_error
        if transform isa TransModalFixed
            push!(funs, mode_reconstruction_error(transform; window=error_window))
        else
            push!(funs, mode_reconstruction_error(transform))
        end
    end
    for resp in transform.resp
        if resp isa PlasmaCumtrapz
            # the transform's batched plasma response already holds the rate on the device
            ionrate_dev = nothing
            if transform isa TransModalFixed && Utils.isdevice(like)
                for r in transform.resp_eval
                    r isa Nonlinear.PlasmaCumtrapzBatched && (ionrate_dev = r.ratefunc_dev)
                end
            end
            ed = electrondensity(grid, resp.ratefunc, transform.densityfun, modes;
                                 components=pol, like, ionrate_dev)
            push!(funs, ed)
        end
    end
    if !isnothing(windows)
        for win in windows
            push!(funs, energy_λ(grid, energyfunω, win; like))
        end
    end
    for (idx, uf) in enumerate(userfuns)
        if uf in funs
            @warn("userfun $uf is already present in the default set and will be ignored")
        else
            push!(funs, uf)
        end
    end
    collect_stats(grid, Eω, funs...)
end

# For constant linop, ZDW is also constant
function zdw_linop(mode::Modes.AbstractMode, linop::AbstractArray)
    zdw = missnan(Modes.zdw(mode))
    FieldFree((d, Eω, Et, z, dz) -> d["zdw"] = zdw)
end

function zdw_linop(modes, linop::AbstractArray)
    zdw = [missnan(Modes.zdw(mode)) for mode in modes]
    FieldFree((d, Eω, Et, z, dz) -> d["zdw"] = zdw)
end

zdw_linop(mode_s, linop) = zdw(mode_s)

end