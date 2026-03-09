module PyPlotExt
import Luna: Maths, PhysData, Processing
import Luna.PhysData: wlfreq, c, ε_0
import Luna.Output: AbstractOutput
import Luna.Processing: makegrid, getIω, getEω, getEt, nearest_z
import Luna.Plotting: get_modes, power_unit, getspeclims, modeidcs, window_str, should_log10
import PyPlot: ColorMap, plt, Figure
import FFTW
import Printf: @sprintf
import Base: display

convertany(x) = x
convertarray(x) = x

include("commonpyplot.jl")

end # module
