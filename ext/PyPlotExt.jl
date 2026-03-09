module PyPlotExt
import PyPlot: ColorMap, plt, Figure
convertany(x) = x
convertarray(x) = x
include("commonpyplot.jl")
end # module
