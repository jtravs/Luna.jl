module Plotting
import Luna: Grid, Maths, PhysData, Processing
import Luna.PhysData: wlfreq, c, ε_0
import Luna.Output: AbstractOutput
import Luna.Processing: makegrid, getIω, getEω, getEt, nearest_z
import FFTW
import Printf: @sprintf
import Base: display

function getext()
    ext = Base.get_extension(@__MODULE__, :PythonPlotExt)
    !isnothing(ext) && return ext
    ext = Base.get_extension(@__MODULE__, :PyPlotExt)
    !isnothing(ext) && return ext
    ext = Base.get_extension(@__MODULE__, :GLMakieExt)
    !isnothing(ext) && return ext
    error("Please load a plotting backend: PythonPlot, PyPlot, or GLMakie.")
end

"""
    cmap_white(cmap, N=512, n=8)

Replace the lowest colour stop of `cmap` (after splitting into `n` stops) with white and
create a new colourmap with `N` stops.
"""
function cmap_white(cmap; N=2^12, n=8)
    getext().cmap_white(cmap; N, n)
end

"""
    get_modes(output)

Determine whether `output` contains a multimode simulation, and if so, return the names
of the modes.
"""
function get_modes(output)
    t = output["simulation_type"]["transform"]
    !startswith(t, "TransModal") && return false, nothing
    lines = split(t, "\n")
    modeline = findfirst(li -> startswith(li, "  modes:"), lines)
    endline = findnext(li -> !startswith(li, " "^4), lines, modeline+1)
    mlines = lines[modeline+1 : endline-1]
    labels = [match(r"{([^,]*),", li).captures[1] for li in mlines]
    angles = zeros(length(mlines))
    for (ii, li) in enumerate(mlines)
        m = match(r"ϕ=(-?[0-9]+.[0-9]+)π", li)
        isnothing(m) && continue # no angle information in mode label)
        angles[ii] = parse(Float64, m.captures[1])
    end
    if !all(angles .== 0)
        for i in eachindex(labels)
            if startswith(labels[i], "HE")
                if angles[i] == 0
                    θs = "x"
                elseif angles[i] == 0.5
                    θs = "y"
                else
                    θs = "$(angles[i])π"
                end
                labels[i] *= " ($θs)"
            end
        end
    end
    return true, labels
end

"""
    stats(output; kwargs...)

Plot all statistics available in `output`. Additional `kwargs` are passed onto `pyplot.plot()`
"""
function stats(output; kwargs...)
    stats = output["stats"]

    pstats = [] # pulse statistics
    haskey(stats, "energy") && push!(pstats, (1e6*stats["energy"], "Energy (μJ)"))
    for (k, v) in pairs(stats)
        startswith(k, "energy_") || continue
        str = "Energy "*replace(k[8:end], "_" => " ")*" (μJ)"
        push!(pstats, (1e6*stats[k], str))
    end
    for (k, v) in pairs(stats)
        startswith(k, "peakpower_") || continue
        Pfac, unit = power_unit(stats[k])
        str = "Peak power "*replace(k[11:end], "_" => " ")*" ($unit)"
        push!(pstats, (Pfac*stats[k], str))
    end
    if haskey(stats, "peakpower")
        Pfac, unit = power_unit(stats["peakpower"])
        push!(pstats, (Pfac*stats["peakpower"], "Peak power ($unit)"))
    end
    haskey(stats, "peakintensity") && push!(
        pstats, (1e-16*stats["peakintensity"], "Peak Intensity (TW/cm\$^2\$)"))
    haskey(stats, "fwhm_t_min") && push!(pstats, (1e15*stats["fwhm_t_min"], "min FWHM (fs)"))
    haskey(stats, "fwhm_t_max") && push!(pstats, (1e15*stats["fwhm_t_max"], "max FWHM (fs)"))
    haskey(stats, "fwhm_r") && push!(pstats, (1e6*stats["fwhm_r"], "Radial FWHM (μm)"))
    haskey(stats, "ω0") && push!(pstats, (1e9*wlfreq.(stats["ω0"]), "Central wavelength (nm)"))

    fstats = [] # fibre/waveguide/propagation statistics
    if haskey(stats, "electrondensity")
        push!(fstats, (1e-6*stats["electrondensity"], "Electron density (cm\$^{-3}\$)"))
        if haskey(stats, "density")
            push!(fstats,
                 (100*stats["electrondensity"]./stats["density"], "Ionisation fraction (%)"))
        end
    end
    haskey(stats, "density") && push!(
        fstats, (1e-6*stats["density"], "Density (cm\$^{-3}\$)"))
    haskey(stats, "pressure") && push!(
        fstats, (stats["pressure"], "Pressure (bar)"))
    haskey(stats, "dz") && push!(fstats, (1e6*stats["dz"], "Stepsize (μm)"))
    haskey(stats, "core_radius") && push!(fstats, (1e6*stats["core_radius"], "Core radius (μm)"))
    haskey(stats, "zdw") && push!(fstats, (1e9*stats["zdw"], "ZDW (nm)"))

    z = stats["z"]*1e2

    multimode, modes = get_modes(output)
    modes = isnothing(modes) ? [""] : modes

    getext().stats(pstats, fstats, multimode, modes; kwargs...)
end

"""
    should_log10(A, tolfac=10)

For multi-line plots, determine whether data for different lines contained in `A` spans
a sufficiently large range that a logarithmic scale should be used. By default, this is the
case when there is any point where the lines are different by more than a factor of 10.
"""
function should_log10(A, tolfac=10)
    mi = minimum(A; dims=2)
    ma = maximum(A; dims=2)
    any(ma./mi .> 10)
end

window_str(::Nothing) = ""
window_str(win::NTuple{4, Number}) = @sprintf("%.1f nm to %.1f nm", 1e9.*win[2:3]...)
window_str(win::NTuple{2, Number}) = @sprintf("%.1f nm to %.1f nm", 1e9.*win...)
window_str(window) = "custom bandpass"

modeidcs(m::Int, ml) = [m]
modeidcs(m::Symbol, ml) = (m == :sum) ? [] : error("modes must be :sum, a single integer, or iterable")
modeidcs(m::Nothing, ml) = 1:length(ml)
modeidcs(m, ml) = m

# Helper function to convert λrange to the correct numbers depending on specaxis
function getspeclims(λrange, specaxis)
    if specaxis == :f
        specxfac = 1e-15
        speclims = (specxfac*c/maximum(λrange), specxfac*c/minimum(λrange))
        speclabel = "Frequency (PHz)"
    elseif specaxis == :ω
        specxfac = 1e-15
        speclims = (specxfac*wlfreq(maximum(λrange)), specxfac*wlfreq(minimum(λrange)))
        speclabel = "Angular frequency (rad/fs)"
    elseif specaxis == :λ
        specxfac = 1e9
        speclims = λrange .* specxfac
        speclabel = "Wavelength (nm)"
    else
        error("Unknown specaxis $specaxis")
    end
    return speclims, speclabel, specxfac
end

# Automatically find power unit depending on scale of electric field.
function power_unit(Pt, y=:Pt)
    units = ["kW", "MW", "GW", "TW", "PW"]
    Pmax = maximum(Pt)
    oom = clamp(floor(Int, log10(Pmax)/3), 1, 5) # maximum unit is PW
    powerfac = 1/10^(oom*3)
    if y == :Et
        sqrt(powerfac), "$(units[oom])\$^{1/2}\$"
    else
        return powerfac, units[oom]
    end
end   

"""
    prop_2D(output, specaxis=:f)

Make false-colour propagation plots for `output`, using spectral x-axis `specaxis` (see
[`getIω`](@ref)). For multimode simulations, create one figure for each mode plus one for
the sum of all modes.

# Keyword arguments
- `λrange::Tuple(Float64, Float64)` : x-axis limits for spectral plot (wavelength in metres)
- `trange::Tuple(Float64, Float64)` : x-axis limits for time-domain plot (time in seconds)
- `dBmin::Float64` : lower colour-scale limit for logarithmic spectral plot
- `resolution::Real` smooth the spectral energy density as defined by [`getIω`](@ref).
"""
function prop_2D(output, specaxis=:f;
                 trange=(-50e-15, 50e-15), bandpass=nothing,
                 λrange=(150e-9, 2000e-9), dBmin=-60,
                 resolution=nothing, modes=nothing, oversampling=4,
                 kwargs...)
    getext().prop_2D(output, specaxis; trange, bandpass,
                     λrange, dBmin, resolution, modes, oversampling, kwargs...)
end

"""
    time_1D(output, zslice, y=:Pt, kwargs...)

Create lineplots of time-domain slice(s) of the propagation.

The keyword argument `y` determines
what is plotted: `:Pt` (power, default), `:Esq` (squared electric field) or `:Et` (electric field).

The keyword argument `modes` selects which modes (if present) are to be plotted, and can be
a single index, a `range` or `:sum`. In the latter case, the sum of modes is plotted.

The keyword argument `oversampling` determines the amount of oversampling done before plotting.

Other `kwargs` are passed onto `pyplot.plot`.
"""
function time_1D(output, zslice=maximum(output["z"]);
                y=:Pt, modes=nothing,
                oversampling=4, trange=(-50e-15, 50e-15), bandpass=nothing,
                FTL=false, propagate=nothing,
                kwargs...)
    getext().time_1D(output, zslice; y, modes, oversampling, trange, bandpass,
                     FTL, propagate, kwargs...)
end

"""
    spec_1D(output, zslice, specaxis=:λ, log10=true, log10min=1e-6)

Create lineplots of spectral-domain slices of the propagation.

The x-axis is determined by `specaxis` (see [`getIω`](@ref)).

If `log10` is true, plot on a logarithmic scale, with a y-axis range of `log10min`. 

The keyword argument `modes` selects which modes (if present) are to be plotted, and can be
a single index, a `range` or `:sum`. In the latter case, the sum of modes is plotted.

Other `kwargs` are passed onto `pyplot.plot`.
"""
function spec_1D(output, zslice=maximum(output["z"]), specaxis=:λ;
                 modes=nothing, λrange=(150e-9, 1200e-9),
                 log10=true, log10min=1e-6, resolution=nothing,
                 kwargs...)
    getext().spec_1D(output, zslice, specaxis;
                     modes, λrange, log10, log10min, resolution, kwargs...)
end


spectrogram(output::AbstractOutput, args...; kwargs...) = spectrogram(
    makegrid(output), output, args...; kwargs...)

function spectrogram(grid::Grid.AbstractGrid, Eω::AbstractArray, specaxis=:λ;
                     propagate=nothing, kwargs...)
    t, Et = getEt(grid, Eω; propagate=propagate, oversampling=1)
    spectrogram(t, Et, specaxis; kwargs...)
end

function spectrogram(grid::Grid.AbstractGrid, output, zslice, specaxis=:λ;
                     propagate=nothing, kwargs...)
    t, Et, zactual = getEt(output, zslice; oversampling=1, propagate=propagate)
    Et = Et[:, 1]
    spectrogram(t, Et, specaxis; kwargs...)
end

function spectrogram(t::AbstractArray, Et::AbstractArray, specaxis=:λ;
    trange, N, fw, λrange=(150e-9, 2000e-9), log=false, dBmin=-40,
    kwargs...)
    getext().spectrogram(t, Et, specaxis;
                         trange, N, fw, λrange, log, dBmin, kwargs...)
end

function energy(output; modes=nothing, bandpass=nothing, figsize=(7, 5))
    getext().energy(output; modes, bandpass, figsize)
end

end