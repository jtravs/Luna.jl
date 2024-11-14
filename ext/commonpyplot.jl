import Luna: Grid, Maths, PhysData, Processing
import Luna.PhysData: wlfreq, c, ε_0
import Luna.Output: AbstractOutput
import Luna.Processing: makegrid, getIω, getEω, getEt, nearest_z
import FFTW
import Printf: @sprintf
import Base: display

"""
    displayall()

`display` all currently open figures.
"""
function displayall()
    for fign in plt.get_fignums()
        fig = plt.figure(fign)
        display(fig)
    end

end

display(figs::AbstractArray{Figure, N}) where N = [display(fig) for fig in figs]

"""
    cmap_white(cmap, N=512, n=8)

Replace the lowest colour stop of `cmap` (after splitting into `n` stops) with white and
create a new colourmap with `N` stops.
"""
function cmap_white(cmap; N=2^12, n=8)
    vals = collect(range(0, 1, length=n))
    vals_i = collect(range(0, 1, length=N))
    cm = ColorMap(cmap)
    clist = cm(vals)
    clist[1, :] = [1, 1, 1, 1]
    clist_i = Array{Float64}(undef, (N, 4))
    for ii in 1:4
        clist_i[:, ii] .= Maths.BSpline(vals, clist[:, ii]).(vals_i)
    end
    ColorMap(clist_i)
end

"""
    cmap_colours(num, cmap="viridis"; cmin=0, cmax=0.8)

Make an array of `num` different colours that follow the colourmap `cmap` between the values
`cmin` and `cmax`.
"""
function cmap_colours(num, cmap="viridis"; cmin=0, cmax=0.8)
    cm = ColorMap(cmap)
    n = collect(range(cmin, cmax; length=num))
    cm.(n)
end

"""
    subplotgrid(N, portrait=true, kwargs...)

Create a figure with `N` subplots laid out in a grid that is as close to square as possible.
If `portrait` is `true`, try to lay out the grid in portrait orientation (taller than wide),
otherwise landscape (wider than tall).
"""
function subplotgrid(N, portrait=true; colw=4, rowh=2.5, title=nothing)
    cols = ceil(Int, sqrt(N))
    rows = ceil(Int, N/cols)
    portrait && ((rows, cols) = (cols, rows))
    fig, axs = pyplot.subplots(rows, cols, num=title)
    axs = convertany(axs)
    ndims(axs) > 1 && (axs = permutedims(axs, (2, 1)))
    if cols*rows > N
        for axi in axs[N+1:end]
            axi.remove()
        end
    end
    fig.set_size_inches(cols*colw, rows*rowh)
    fig, N > 1 ? axs : [axs]
end

function stats(pstats, fstats, multimode, modes; kwargs...)
    Npl = length(pstats)
    if Npl > 0
        pfig, axs = subplotgrid(Npl, title="Pulse stats")
        for n in 1:Npl
            ax = axs[n]
            data, label = pstats[n]
            multimode && (ndims(data) > 1) && (data = data')
            ax.plot(z, data; kwargs...)
            ax.set_xlabel("Distance (cm)")
            ax.set_ylabel(label)
            multimode && (ndims(data) > 1) && ax.semilogy()
            multimode && (ndims(data) > 1) && ax.legend(modes, frameon=false)
        end
        pfig.tight_layout()
    end
    
    Npl = length(fstats)
    if Npl > 0
        ffig, axs = subplotgrid(Npl, title="Other stats")
        for n in 1:Npl
            ax = axs[n]
            data, label = fstats[n]
            multimode && (ndims(data) > 1) && (data = data')
            ax.plot(z, data; kwargs...)
            ax.set_xlabel("Distance (cm)")
            ax.set_ylabel(label)
            multimode && (ndims(data) > 1) && should_log10(data) && ax.semilogy()
            multimode && (ndims(data) > 1) && ax.legend(modes, frameon=false)
        end
        ffig.tight_layout()
    end
    [pfig, ffig]
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
    z = output["z"]*1e2
    if specaxis == :λ
            specx, Iω = getIω(output, specaxis, specrange=λrange, resolution=resolution)
    else
            specx, Iω = getIω(output, specaxis, resolution=resolution)
    end

    t, Et = getEt(output; trange, bandpass, oversampling)
    It = abs2.(Et)

    speclims, speclabel, specxfac = getspeclims(λrange, specaxis)
    specx .*= specxfac

    multimode, modelabels = get_modes(output)

    if multimode
        fig = _prop2D_mm(modelabels, modeidcs(modes, modelabels), t, z, specx, It, Iω,
                         speclabel, speclims, trange, dBmin, window_str(bandpass);
                         kwargs...)
    else
        fig = _prop2D_sm(t, z, specx, It, Iω,
                         speclabel, speclims, trange, dBmin, window_str(bandpass);
                         kwargs...)
    end
    fig
end

function spectrogram(t::AbstractArray, Et::AbstractArray, specaxis=:λ;
                     trange, N, fw, λrange=(150e-9, 2000e-9), log=false, dBmin=-40,
                     kwargs...)
    ω = Maths.rfftfreq(t)[2:end]
    tmin, tmax = extrema(trange)
    tg = collect(range(tmin, tmax, length=N))
    g = Maths.gabor(t, real(Et), tg, fw)
    g = g[2:end, :]

    specy, Ig = getIω(ω, g*Maths.rfftnorm(t[2]-t[1]), specaxis)
    speclims, speclabel, specyfac = getspeclims(λrange, specaxis)

    log && (Ig = 10*log10.(Maths.normbymax(Ig)))

    fig = plt.figure()
    plt.pcolormesh(tg.*1e15, specyfac*specy, Ig; shading="auto", kwargs...)
    plt.ylim(speclims...)
    plt.ylabel(speclabel)
    plt.xlabel("Time (fs)")
    log && plt.clim(dBmin, 0)
    plt.colorbar()
    fig
end

function energy(output; modes=nothing, bandpass=nothing, figsize=(7, 5))
    e = Processing.energy(output; bandpass=bandpass)
    eall = Processing.energy(output)

    multimode, modestrs = get_modes(output)
    if multimode
        e0 = sum(eall[:, 1])
        modes = isnothing(modes) ? (1:size(e, 1)) : modes
        if modes == :sum
            e = dropdims(sum(e, dims=1), dims=1)
            modestrs = join(modestrs, "+")
            nmodes = 1
        else
            isnothing(modes) && (modes = 1:length(modestrs))
            e = e[modes, :]
            modestrs = modestrs[modes]
            nmodes = length(modes)
        end
    else
        e0 = eall[1]
    end

    z = output["z"]*100

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(z, 1e6*e')
    ax.set_xlim(extrema(z)...)
    ax.set_ylim(ymin=0)
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel("Energy (μJ)")
    rax = ax.twinx()
    rax.plot(z, 100*(e/e0)', linewidth=0)
    lims = ax.get_ylim()
    rax.set_ylim(100/(1e6*e0).*lims)
    rax.set_ylabel("Conversion efficiency (%)")
    fig.set_size_inches(figsize...)
    fig
end

function auto_fwhm_arrows(ax, x, y; color="k", arrowlength=nothing, hpad=0, linewidth=1,
                                    text=nothing, units="fs", kwargs...)
    left, right = Maths.level_xings(x, y; kwargs...)
    fw = abs(right - left)
    halfmax = maximum(y)/2
    arrowlength = isnothing(arrowlength) ? 2*fw : arrowlength

    ax.annotate("", xy=(left-hpad, halfmax),
                xytext=(left-hpad-arrowlength, halfmax),
                arrowprops=Dict("arrowstyle" => "->",
                                "color" => color,
                                "linewidth" => linewidth))
    ax.annotate("", xy=(right+hpad, halfmax),
                xytext=(right+hpad+arrowlength, halfmax),
                arrowprops=Dict("arrowstyle" => "->",
                                "color" => color,
                                "linewidth" => linewidth))

    if text == :left
        ax.text(left-arrowlength/2, 1.1*halfmax, @sprintf("%.2f %s", fw, units),
                ha="right", color=color)
    elseif text == :right
        ax.text(right+arrowlength/2, 1.1*halfmax, @sprintf("%.2f %s", fw, units),
                color=color)
    end
end

function add_fwhm_legends(ax, unit)
    leg = ax.get_legend()
    texts = leg.get_texts()
    handles, labels = ax.get_legend_handles_labels()
    handles = convertany(handles)
    for (ii, line) in enumerate(handles)
        xy = line.get_xydata()
        xy = convertany(xy)
        fw = Maths.fwhm(xy[:, 1], xy[:, 2])
        t = texts[ii-1]
        s = convertany(t.get_text())
        s *= @sprintf(" [%.2f %s]", fw, unit)
        t.set_text(s)
    end
end

"""
    cornertext(ax, text;
               corner="ul", pad=0.02, xpad=nothing, ypad=nothing, kwargs...)

Place a `text` in the axes `ax` in the corner defined by `corner`. Padding can be
defined for `x` and `y` together via `pad` or separately via `xpad` and `ypad`. Further
keyword arguments are passed to `pyplot.text`. 

Possible values for `corner` are `ul`, `ur`, `ll`, `lr` where the first letter
defines upper/lower and the second defines left/right.
"""
function cornertext(ax, text; corner="ul", pad=0.02, xpad=nothing, ypad=nothing, kwargs...)
    xpad = isnothing(xpad) ? pad : xpad
    ypad = isnothing(ypad) ? pad : ypad
    if corner[1] == 'u'
        val = "top"
        y = 1 - ypad
    elseif corner[1] == 'l'
        val = "bottom"
        y = ypad
    else
        error("Invalid corner $corner. Must be one of ul, ur, ll, lr")
    end
    if corner[2] == 'l'
        hal = "left"
        x = xpad
    elseif corner[2] == 'r'
        hal = "right"
        x = 1 - xpad
    else
        error("Invalid corner $corner. Must be one of ul, ur, ll, lr")
    end
    ax.text(x, y, text; horizontalalignment=hal, verticalalignment=val,
                 transform=ax.transAxes, kwargs...)
end

