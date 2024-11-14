module PythonPlotExt
import PythonPlot: ColorMap, Figure, pyplot
import PythonCall: pyconvert

plt = pyplot
convertany(x) = pyconvert(Any, x)
converarray(x) = pyconvert(Array, x)

# single-mode 2D propagation plots
function _prop2D_sm(t, z, specx, It, Iω, speclabel, speclims, trange, dBmin, bpstr; kwargs...)
    id = "($(string(hash(gensym()); base=16)[1:4])) "
    num = id * "Propagation" * ((length(bpstr) > 0) ? ", $bpstr" : "")
    pfig, axs = pyplot.subplots(1, 2, num=num)
    axs = pyconvert(Any, axs)
    pfig.set_size_inches(12, 4)
    Iω = Maths.normbymax(Iω)
    _spec2D_log(axs[1], specx, z, Iω, dBmin, speclabel, speclims; kwargs...)

    _time2D(axs[2], t, z, It, trange; kwargs...)
    pfig.tight_layout()
    return pfig
end

# multi-mode 2D propagation plots
function _prop2D_mm(modelabels, modes, t, z, specx, It, Iω,
                    speclabel, speclims, trange, dBmin, bpstr;
                    kwargs...)
    pfigs = []
    Iω = Maths.normbymax(Iω)
    id = "($(string(hash(gensym()); base=16)[1:4])) "
    for mi in modes
        num = id * "Propagation ($(modelabels[mi]))" * ((length(bpstr) > 0) ? ", $bpstr" : "")
        pfig, axs = pyplot.subplots(1, 2, num=num)
        axs = pyconvert(Any, axs)
        pfig.set_size_inches(12, 4)
        _spec2D_log(axs[1], specx, z, Iω[:, mi, :], dBmin, speclabel, speclims; kwargs...)

        _time2D(axs[2], t, z, It[:, mi, :], trange; kwargs...)
        push!(pfigs, pfig)
    end

    num = id * "Propagation (all modes)" * ((length(bpstr) > 0) ? ", $bpstr" : "")
    pfig, axs = pyplot.subplots(1, 2, num=num)
    axs = pyconvert(Any, axs)
    pfig.set_size_inches(12, 4)
    Iωall = dropdims(sum(Iω, dims=2), dims=2)
    _spec2D_log(axs[1], specx, z, Iωall, dBmin, speclabel, speclims; kwargs...)

    Itall = dropdims(sum(It, dims=2), dims=2)
    _time2D(axs[2], t, z, Itall, trange; kwargs...)
    pfig.tight_layout()
    push!(pfigs, pfig)

    return pfigs
end

# a single logarithmic colour-scale spectral domain plot
function _spec2D_log(ax, specx, z, I, dBmin, speclabel, speclims; kwargs...)
    im = ax.pcolormesh(specx, z, 10*log10.(transpose(I)); shading="auto", kwargs...)
    im.set_clim(dBmin, 0)
    cb = pyplot.colorbar(im, ax=ax)
    cb.set_label("SED (dB)")
    ax.set_ylabel("Distance (cm)")
    ax.set_xlabel(speclabel)
    ax.set_xlim(speclims...)
end

# a single time-domain propagation plot
function _time2D(ax, t, z, I, trange; kwargs...)
    Pfac, unit = power_unit(I)
    im = ax.pcolormesh(t*1e15, z, Pfac*transpose(I); shading="auto", kwargs...)
    cb = pyplot.colorbar(im, ax=ax)
    cb.set_label("Power ($unit)")
    ax.set_xlim(trange.*1e15)
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Distance (cm)")
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
    t, Et, zactual = getEt(output, zslice,
                           trange=trange, oversampling=oversampling, bandpass=bandpass,
                           FTL=FTL, propagate=propagate)
    if y == :Pt
        yt = abs2.(Et)
    elseif y == :Et
        yt = real(Et)
    elseif y == :Esq
        yt = real(Et).^2
    else
        error("unknown time plot variable $y")
    end
    multimode, modestrs = get_modes(output)
    if multimode
        if modes == :sum
            y == :Pt || error("Modal sum can only be plotted for power!")
            yt = dropdims(sum(yt, dims=2), dims=2)
            modestrs = join(modestrs, "+")
            nmodes = 1
        else
            isnothing(modes) && (modes = 1:length(modestrs))
            yt = yt[:, modes, :]
            modestrs = modestrs[modes]
            nmodes = length(modes)
        end
    end

    yfac, unit = power_unit(abs2.(Et), y)

    sfig = pyplot.figure()
    if multimode && nmodes > 1
        _plot_slice_mm(pyplot.gca(), t*1e15, yfac*yt, zactual, modestrs; kwargs...)
    else
        zs = [@sprintf("%.2f cm", zi*100) for zi in zactual]
        label = multimode ? zs.*" ($modestrs)" : zs
        for iz in eachindex(zactual)
            pyplot.plot(t*1e15, yfac*yt[:, iz]; label=label[iz], kwargs...)
        end
    end
    pyplot.legend(frameon=false)
    add_fwhm_legends(pyplot.gca(), "fs")
    pyplot.xlabel("Time (fs)")
    pyplot.xlim(1e15.*trange)
    ylab = y == :Et ?  "Field ($unit)" : "Power ($unit)"
    pyplot.ylabel(ylab)
    y == :Et || pyplot.ylim(ymin=0)
    sfig.set_size_inches(8.5, 5)
    sfig.tight_layout()
    sfig
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
    if specaxis == :λ
        specx, Iω, zactual = getIω(output, specaxis, zslice, specrange=λrange, resolution=resolution)
    else
        specx, Iω, zactual = getIω(output, specaxis, zslice, resolution=resolution)
    end
    speclims, speclabel, specxfac = getspeclims(λrange, specaxis)
    multimode, modestrs = get_modes(output)
    if multimode
        modes = isnothing(modes) ? (1:size(Iω, 2)) : modes
        if modes == :sum
            Iω = dropdims(sum(Iω, dims=2), dims=2)
            modestrs = join(modestrs, "+")
            nmodes = 1
        else
            isnothing(modes) && (modes = 1:length(modestrs))
            Iω = Iω[:, modes, :]
            modestrs = modestrs[modes]
            nmodes = length(modes)
        end
    end

    specx .*= specxfac

    sfig = pyplot.figure()
    if multimode && nmodes > 1
        _plot_slice_mm(pyplot.gca(), specx, Iω, zactual, modestrs, log10; kwargs...)
    else
        zs = [@sprintf("%.2f cm", zi*100) for zi in zactual]
        label = multimode ? zs.*" ($modestrs)" : zs
        for iz in eachindex(zactual)
            (log10 ? pyplot.semilogy : pyplot.plot)(specx, Iω[:, iz]; label=label[iz], kwargs...)
        end
    end
    pyplot.legend(frameon=false)
    pyplot.xlabel(speclabel)
    pyplot.ylabel("Spectral energy density")
    log10 && pyplot.ylim(3*maximum(Iω)*log10min, 3*maximum(Iω))
    pyplot.xlim(speclims...)
    sfig.set_size_inches(8.5, 5)
    sfig.tight_layout()
    sfig
end

dashes = [(0, (10, 1)),
          (0, (5, 1)),
          (0, (1, 0.5)),
          (0, (1, 0.5, 1, 0.5, 3, 1)),
          (0, (5, 1, 1, 1))]

function _plot_slice_mm(ax, x, y, z, modestrs, log10=false, fwhm=false; kwargs...)
    pfun = (log10 ? ax.semilogy : ax.plot)
    for sidx = 1:size(y, 3) # iterate over z-slices
        zs = @sprintf("%.2f cm", z[sidx]*100)
        line = pfun(x, y[:, 1, sidx]; label="$zs ($(modestrs[1]))", kwargs...)[0]
        for midx = 2:size(y, 2) # iterate over modes
            pfun(x, y[:, midx, sidx], linestyle=dashes[midx], color=line.get_color(),
                 label="$zs ($(modestrs[midx]))"; kwargs...)
        end
    end
end

end # module