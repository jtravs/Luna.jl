module Processing
import FFTW
import Luna: Maths, Fields
import Luna.PhysData: wlfreq
import Luna.Grid: RealGrid, EnvGrid

"""
    arrivaltime(grid, Eω; λlims, winwidth=0, method=:moment, oversampling=1)

Extract the arrival time of the pulse in the wavelength limits `λlims`.

# Arguments
    - `λlims::Tuple{Number, Number}` : wavelength limits (λmin, λmax)
    - `winwidth` : If a `Number`, set smoothing width (in rad/s) of the window function
                   used to bandpass.
                   If `:auto`, automatically set the width to 128 frequency samples.
    - `method::Symbol` : `:moment` to use 1st moment to extract arrival time, `:peak` to use
                        the time of peak power
    - `oversampling::Int` : If >1, oversample the time-domain field before extracting delay
"""
function arrivaltime(grid, Eω; λlims, winwidth=:auto, kwargs...)
    ωmin, ωmax = extrema(wlfreq.(λlims))
    winwidth == :auto && (winwidth = 128*abs(grid.ω[2] - grid.ω[1]))
    window = Maths.planck_taper(grid.ω, ωmin-winwidth, ωmin, ωmax, ωmax+winwidth)
    arrivaltime(grid, Eω, window; kwargs...)
end

function arrivaltime(grid::RealGrid, Eω, ωwindow::Vector{<:Real}; method=:moment, oversampling=1)
    Et = FFTW.irfft(Eω .* ωwindow, length(grid.t), 1)
    to, Eto = Maths.oversample(grid.t, Et)
    arrivaltime(to, abs2.(Maths.hilbert(Eto)); method=method)
end

function arrivaltime(grid::EnvGrid, Eω, ωwindow::Vector{<:Real}; method=:moment, oversampling=1)
    Et = FFTW.ifft(Eω .* ωwindow, 1)
    to, Eto = Maths.oversample(grid.t, Et)
    arrivaltime(to, abs2.(Eto); method=method)
end

function arrivaltime(t, It::Vector{<:Real}; method)
    if method == :moment
        Maths.moment(t, It)
    elseif method == :peak
        t[argmax(It)]
    else
        error("Unknown arrival time method $method")
    end
end

function arrivaltime(t, It::Array{<:Real, N}; method) where N
    if method == :moment
        dropdims(Maths.moment(t, It; dim=1); dims=1)
    elseif method == :peak
        out = Array{Float64, ndims(It)-1}(undef, size(It)[2:end])
        cidcs = CartesianIndices(size(It)[2:end])
        for ii in cidcs
            out[ii] = arrivaltime(t, It[:, ii]; method=:peak)
        end
        out
    else
        error("Unknown arrival time method $method")
    end
end

"""
    energy_λ(grid, Eω; λlims, winwidth=:auto)

Extract energy within a wavelength band given by `λlims`. `winwidth` can be a `Number` in 
rad/fs to set the smoothing edges of the frequency window, or `:auto`, in which case the
width defaults to 128 frequency samples.
"""
function energy_λ(grid, Eω; λlims, winwidth=:auto)
    ωmin, ωmax = extrema(wlfreq.(λlims))
    winwidth == :auto && (winwidth = 128*abs(grid.ω[2] - grid.ω[1]))
    window = Maths.planck_taper(grid.ω, ωmin-winwidth, ωmin, ωmax, ωmax+winwidth)
    energy_window(grid, Eω, window)
end

function energy_window(grid, Eω, ωwindow)
    _, energyω = Fields.energyfuncs(grid)
    _energy_window(Eω, ωwindow, energyω)
end

_energy_window(Eω::Vector, ωwindow, energyω) = energyω(Eω .* ωwindow)
function _energy_window(Eω, ωwindow, energyω)
    out = Array{Float64, ndims(Eω)-1}(undef, size(Eω)[2:end])
    cidcs = CartesianIndices(size(Eω)[2:end])
    for ii in cidcs
        out[ii] = _energy_window(Eω[:, ii], ωwindow, energyω)
    end
    out
end

"Square window centred at `x0` with full width `xfw`"
function swin(x, x0, xfw)
    abs(x - x0) < xfw/2 ? 1.0 : 0.0
end

"Gaussian window centred at `x0` with full width `xfw`"
function gwin(x, x0, xfw)
    σ = xfw * 0.42
    exp(-0.5 * ((x-x0)/σ)^2)
end

"""
    Eω_to_Pλ(ω, Eω, λrange, resolution; window=gwin, nsamples=4)

Calculate the spectral power, defined by frequency domain field `Eω` over
angular frequency grid `ω`, on a wavelength scale over the range `λrange` taking account
of spectral `resolution`. The `window` function to use defaults to a Gaussian function with
FWHM of `resolution`, and by default we sample `nsamples=4` times within each `resolution`.

This works for both fields and envelopes and it is assumed that `Eω` is suitably fftshifted
if necessary before this function is called, and that `ω` is monotonically increasing.
"""
function Eω_to_Pλ(ω, Eω, λrange, resolution; window=gwin, nsamples=4)
    _Eω_to_Px(ω, Eω, λrange, resolution, window, nsamples, PhysData.wlfreq, PhysData.wlfreq)
end

"""
    Eω_to_Pf(ω, Eω, Frange, resolution; window=gwin, nsamples=4)

Calculate the spectral power, defined by frequency domain field `Eω` over
angular frequency grid `ω`, on a frequency scale over the range `Frange` taking account
of spectral `resolution`. The `window` function to use defaults to a Gaussian function with
FWHM of `resolution`, and by default we sample `nsamples=4` times within each `resolution`.

This works for both fields and envelopes and it is assumed that `Eω` is suitably fftshifted
if necessary before this function is called, and that `ω` is monotonically increasing.
"""
function Eω_to_Pf(ω, Eω, Frange, resolution; window=gwin, nsamples=4)
    _Eω_to_Px(ω, Eω, Frange, resolution, window, nsamples, x -> x/(2π), x -> x*(2π))
end

"""
Convolution kernel for each output point. We simply loop over all `z` and output points.
The inner loop adds up the contributions from the specified window around
the target point. Note that this works without scaling also for wavelength ranges
because the integral is still over a frequency grid (with appropriate frequency dependent
integration bounds).
"""
function _Eω_to_Px_kernel!(Px, cidcs, istart, iend, Eω, window, x, xg, resolution)
    for ii in cidcs
        for j in 1:size(Px, 1)
            for k in istart[j]:iend[j]
                Px[j,ii] += abs2(Eω[k,ii]) * window(x[k], xg[j], resolution)
            end
        end
    end
    Px[Px .<= 0.0] .= maximum(Px) * 1e-20
end

function _Eω_to_Px(ω, Eω, xrange, resolution, window, nsamples, ωtox, xtoω)
    # build output grid and array
    x = ωtox.(ω)
    nxg = ceil(Int, (xrange[2] - xrange[1])/resolution*nsamples)
    xg = collect(range(xrange[1], xrange[2], length=nxg))
    rdims = size(Eω)[2:end]
    Px = Array{Float64, ndims(Eω)}(undef, ((nxg,)..., rdims...))
    fill!(Px, 0.0)
    cidcs = CartesianIndices(rdims)
    # we find a suitable nspan
    nspan = 1
    while window(nspan*resolution, 0.0, resolution) > 1e-8
        nspan += 1
    end
    # now we build arrays of start and end indices for the relevent frequency
    # band for each output. For a frequency grid this is a little (tiny) inefficient
    # but for a wavelength grid, which has varying index ranges, this is essential
    # and I think having a common code is simpler/cleaner.
    istart = Array{Int,1}(undef,nxg)
    iend = Array{Int,1}(undef,nxg)
    δω = ω[2] - ω[1]
    i0 = argmin(abs.(ω))
    for i in 1:nxg
        i1 = i0 + round(Int, xtoω(xg[i] + resolution*nspan)/δω)
        i2 = i0 + round(Int, xtoω(xg[i] - resolution*nspan)/δω)
        # we want increasing indices
        if i1 > i2
            i1,i2 = i2,i1
        end
        # handle boundaries
        if i2 > length(ω)
            i2 = length(ω)
        end
        if i1 < i0
            i1 = i0
        end
        istart[i] = i1
        iend[i] = i2
    end
    # run the convolution kernel - the function barrier massively improves performance
    _Eω_to_Px_kernel!(Px, cidcs, istart, iend, Eω, window, x, xg, resolution)
    xg, Px
end

end