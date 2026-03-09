module PythonPlotExt
import Luna: Maths, PhysData, Processing
import Luna.PhysData: wlfreq, c, ε_0
import Luna.Output: AbstractOutput
import Luna.Processing: makegrid, getIω, getEω, getEt, nearest_z
import Luna.Plotting: get_modes, power_unit, getspeclims, modeidcs, window_str, should_log10
import PythonPlot: ColorMap, Figure, pyplot
import PythonCall: pyconvert
import FFTW
import Printf: @sprintf
import Base: display

const plt = pyplot
convertany(x) = pyconvert(Any, x)
convertarray(x) = pyconvert(Array, x)

include("commonpyplot.jl")

end # module
