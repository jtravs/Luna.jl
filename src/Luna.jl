module Luna
import FFTW
import Hankel
import Logging
import Adapt
import LinearAlgebra: mul!, ldiv!
Logging.disable_logging(Logging.BelowMinLevel)

"""
    Luna.settings

Dictionary of global settings for `Luna`.
"""
settings = Dict{String, Any}("fftw_flag" => FFTW.PATIENT,
                             "fftw_threads" => 0)

"""
    set_fftw_mode(mode)

Set FFTW planning mode for all FFTW transform planning in `Luna`.

Possible values for `mode` are `:estimate`, `:measure`, `:patient`, and `:exhaustive`.
The initial value upon loading `Luna` is `:patient`

# Examples
```jldoctest
julia> Luna.set_fftw_mode(:patient)
0x00000020
```
"""
function set_fftw_mode(mode)
    s = uppercase(string(mode))
    flag = getfield(FFTW, Symbol(s))
    settings["fftw_flag"] = flag
end

"""
    set_fftw_threads(nthr)

Set number of threads to be used by FFTW. If set to `0`, the number of threads used by
FFTW is determined automatically (see [`Utils.FFTWthreads()`](@ref)).

An explicit `nthr > 0` takes effect even when Julia itself runs single-threaded, in
which case libfftw3 uses its own native pthreads pool rather than Julia tasks.

On Julia ≥ 1.12 with more than one Julia thread, FFTW.jl's Julia-task threading callback
(which can segfault there) is deregistered automatically and libfftw3's native pthreads
pool is used instead — so `JULIA_NUM_THREADS > 1` (for Luna's threaded elementwise
kernels) combines safely with threaded FFTs. On a single Julia thread the callback is
never installed and nothing needs deregistering.
"""
function set_fftw_threads(nthr=0)
    settings["fftw_threads"] = nthr
    FFTW.set_num_threads(Utils.FFTWthreads())
    if VERSION >= v"1.12" && Threads.nthreads() > 1
        if Utils.use_native_fftw_threads() && !_native_fftw_logged[]
            _native_fftw_logged[] = true
            Logging.@info("Julia ≥ 1.12 with $(Threads.nthreads()) threads: " *
                          "using libfftw3's native pthreads for FFTs " *
                          "(FFTW.jl's Julia-task callback deregistered).")
        end
    end
end
const _native_fftw_logged = Ref(false)

function __init__()
    set_fftw_threads()
end

include("Utils.jl")
include("Scans.jl")
include("Output.jl")
include("Maths.jl")
include("PhysData.jl")
include("Grid.jl")
include("Device.jl") # device (GPU) scaffolding; needs Utils, Output and Grid
include("Modes.jl")
include("Fields.jl")
include("RK45.jl")
include("LinearOps.jl")
include("Capillary.jl")
include("Antiresonant.jl")
include("RectModes.jl")
include("StepIndexFibre.jl")
include("SimpleFibre.jl")
include("Nonlinear.jl")
include("Ionisation.jl")
include("NonlinearRHS.jl")
include("Processing.jl")
include("Stats.jl")
include("Polarisation.jl")
include("Tools.jl")
include("Plotting.jl")
include("Raman.jl")
include("SFA.jl")
include("Interface.jl")

prop_capillary = Interface.prop_capillary
prop_gnlse = Interface.prop_gnlse
Pulses = Interface.Pulses

Scan = Scans.Scan
runscan = Scans.runscan
makefilename = Scans.makefilename
addvariable! = Scans.addvariable!

export Utils, Scans, Output, Maths, PhysData, Grid, RK45, Modes, Capillary, RectModes,
       Nonlinear, Ionisation, NonlinearRHS, LinearOps, Stats, Polarisation,
       Tools, Plotting, Raman, Antiresonant, Fields, Processing, Interface, SFA,
       prop_capillary, prop_gnlse, Pulses, Scan, runscan, makefilename, addvariable!,
       StepIndexFibre, SimpleFibre

# for a tuple of TimeFields we assume all inputs are for mode 1
function doinput_sm(grid, inputs::Tuple{Vararg{T} where T <: Fields.TimeField}, FT)
    out = fill(0.0 + 0.0im, length(grid.ω))
    for field in inputs
        out .+= field(grid, FT)
    end
    return out
end

# for a single Fields.TimeField we assume a single input for mode 1
function doinput_sm(grid, inputs::Fields.TimeField, FT)
    doinput_sm(grid, (inputs,), FT)
end

function doinput_sm(grid, inputs::Tuple{Vararg{T} where T <: NamedTuple{<:Any, <:Tuple{Vararg{Any}}}}, FT)
    if any([i.mode ≠ 1 for i in inputs])
        error("For mode-averaged propagation, all inputs must be in 1st mode.")
    end
    inputs_flat = Tuple(Iterators.flatten([i.fields for i in inputs]))
    doinput_sm(grid, inputs_flat, FT)
end

function setup(grid::Grid.RealGrid, densityfun, responses, inputs, βfun!, aeff;
               norm! = NonlinearRHS.norm_mode_average(grid, βfun!, aeff),
               noise_field=nothing)
    Logging.@info("Setting up and planning FFTs...")
    flush(stderr)
    Utils.loadFFTwisdom()
    xo = Array{Float64}(undef, length(grid.to))
    FTo = FFTW.plan_rfft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransModeAvg(grid, FTo, responses, densityfun, norm!, aeff;
                                          noise_field)
    x = Array{Float64}(undef, length(grid.t))
    FT = FFTW.plan_rfft(x, 1, flags=settings["fftw_flag"])
    Eω = doinput_sm(grid, inputs, FT)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Logging.@info("Setup finished.")
    flush(stderr)
    Eω, transform, FT
end

function setup(grid::Grid.EnvGrid, densityfun, responses, inputs, βfun!, aeff;
               norm! = NonlinearRHS.norm_mode_average(grid, βfun!, aeff),
               noise_field=nothing)
    Logging.@info("Setting up and planning FFTs...")
    flush(stderr)
    Utils.loadFFTwisdom()
    x = Array{ComplexF64}(undef, length(grid.t))
    FT = FFTW.plan_fft(x, 1, flags=settings["fftw_flag"])
    xo = Array{ComplexF64}(undef, length(grid.to))
    FTo = FFTW.plan_fft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransModeAvg(grid, FTo, responses, densityfun, norm!, aeff;
                                          noise_field)
    Eω = doinput_sm(grid, inputs, FT)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Logging.@info("Setup finished.")
    flush(stderr)
    Eω, transform, FT
end

# for a tuple of NamedTuple's with tuple fields we assume all is well
function doinput_mm!(Eω, grid, inputs::Tuple{Vararg{T} where T <: NamedTuple{<:Any, <:Tuple{Vararg{Any}}}}, FT)
    for input in inputs
        out = @view Eω[:, input.mode]
        for field in input.fields
            out .+= field(grid, FT)
        end
    end
end

# for a tuple of TimeFields we assume all inputs are for mode 1
function doinput_mm!(Eω, grid, inputs::Tuple{Vararg{T} where T <: Fields.TimeField}, FT)
    doinput_mm!(Eω, grid, ((mode=1, fields=inputs),), FT)
end

# for a single Fields.TimeField we assume a single input for mode 1
function doinput_mm!(Eω, grid, inputs::Fields.TimeField, FT)
    doinput_mm!(Eω, grid, ((mode=1, fields=(inputs,)),), FT)
end

function setup(grid::Grid.RealGrid, densityfun, responses, inputs,
               modes::Modes.ModeCollection, components;
               full=false, norm! = NonlinearRHS.norm_modal(grid),
               rtol=1e-3, atol=0.0, mfcn=512, noise_field=nothing)
    Logging.@info("Setting up and planning FFTs...")
    flush(stderr)
    ts = Modes.ToSpace(modes, components=components)
    Utils.loadFFTwisdom()
    xt = Array{Float64}(undef, length(grid.t))
    FTt = FFTW.plan_rfft(xt, 1, flags=settings["fftw_flag"])
    Eω = zeros(ComplexF64, length(grid.ω), length(modes))
    doinput_mm!(Eω, grid, inputs, FTt)
    x = Array{Float64}(undef, length(grid.t), length(modes))
    FT = FFTW.plan_rfft(x, 1, flags=settings["fftw_flag"])
    xo = Array{Float64}(undef, length(grid.to), ts.npol)
    FTo = FFTW.plan_rfft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransModal(grid, ts, FTo,
                                 responses, densityfun, norm!;
                                 rtol, atol, mfcn, full, noise_field)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Logging.@info("Setup finished.")
    flush(stderr)
    Eω, transform, FT
end

function setup(grid::Grid.EnvGrid, densityfun, responses, inputs,
               modes::Modes.ModeCollection, components;
               full=false, norm! = NonlinearRHS.norm_modal(grid),
               rtol=1e-3, atol=0.0, mfcn=512, noise_field=nothing)
    Logging.@info("Setting up and planning FFTs...")
    flush(stderr)
    ts = Modes.ToSpace(modes, components=components)
    Utils.loadFFTwisdom()
    xt = Array{ComplexF64}(undef, length(grid.t))
    FTt = FFTW.plan_fft(xt, 1, flags=settings["fftw_flag"])
    Eω = zeros(ComplexF64, length(grid.ω), length(modes))
    doinput_mm!(Eω, grid, inputs, FTt)
    x = Array{ComplexF64}(undef, length(grid.t), length(modes))
    FT = FFTW.plan_fft(x, 1, flags=settings["fftw_flag"])
    xo = Array{ComplexF64}(undef, length(grid.to), ts.npol)
    FTo = FFTW.plan_fft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransModal(grid, ts, FTo,
                                 responses, densityfun, norm!;
                                 rtol, atol, mfcn, full, noise_field)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Logging.@info("Setup finished.")
    flush(stderr)
    Eω, transform, FT
end

function doinputs_fs!(Eωk, grid, spacegrid::Union{Hankel.QDHT,Grid.FreeGrid}, FT,
                   inputs::Tuple{Vararg{T} where T <: Fields.SpatioTemporalField})
    for field in inputs
        Eωk .+= field(grid, spacegrid, FT)
    end
end

function doinputs_fs!(Eωk, grid, spacegrid::Union{Hankel.QDHT,Grid.FreeGrid}, FT,
                   inputs::Fields.SpatioTemporalField)
    doinputs_fs!(Eωk, grid, spacegrid, FT, (inputs,))
end

function setup(grid::Grid.RealGrid, q::Hankel.QDHT,
               densityfun, normfun, responses, inputs; noise_field=nothing)
    Logging.@info("Setting up and planning FFTs...")
    flush(stderr)
    Utils.loadFFTwisdom()
    xt = zeros(Float64, length(grid.t), length(q.r))
    FT = FFTW.plan_rfft(xt, 1, flags=settings["fftw_flag"])
    Eω = zeros(ComplexF64, length(grid.ω), length(q.k))
    Eωk = q * Eω
    doinputs_fs!(Eωk, grid, q, FT, inputs)
    xo = Array{Float64}(undef, length(grid.to), length(q.r))
    FTo = FFTW.plan_rfft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransRadial(grid, q, FTo, responses, densityfun, normfun;
                                         noise_field)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Logging.@info("Setup finished.")
    flush(stderr)
    Eωk, transform, FT
end

function setup(grid::Grid.EnvGrid, q::Hankel.QDHT,
               densityfun, normfun, responses, inputs; noise_field=nothing)
    Logging.@info("Setting up and planning FFTs...")
    flush(stderr)
    Utils.loadFFTwisdom()
    xt = zeros(ComplexF64, length(grid.t), length(q.r))
    FT = FFTW.plan_fft(xt, 1, flags=settings["fftw_flag"])
    Eω = zeros(ComplexF64, length(grid.ω), length(q.k))
    Eωk = q * Eω
    doinputs_fs!(Eωk, grid, q, FT, inputs)
    xo = Array{ComplexF64}(undef, length(grid.to), length(q.r))
    FTo = FFTW.plan_fft(xo, 1, flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransRadial(grid, q, FTo, responses, densityfun, normfun;
                                         noise_field)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Logging.@info("Setup finished.")
    flush(stderr)
    Eωk, transform, FT
end

function setup(grid::Grid.RealGrid, xygrid::Grid.FreeGrid,
               densityfun, normfun, responses, inputs; noise_field=nothing)
    Logging.@info("Setting up and planning FFTs...")
    flush(stderr)
    Utils.loadFFTwisdom()
    x = xygrid.x
    y = xygrid.y
    xr = Array{Float64}(undef, length(grid.t), length(y), length(x))
    FT = FFTW.plan_rfft(xr, (1, 2, 3), flags=settings["fftw_flag"])
    Eωk = zeros(ComplexF64, length(grid.ω), length(y), length(x))
    doinputs_fs!(Eωk, grid, xygrid, FT, inputs)
    xo = Array{Float64}(undef, length(grid.to), length(y), length(x))
    FTo = FFTW.plan_rfft(xo, (1, 2, 3), flags=settings["fftw_flag"])
    transform = NonlinearRHS.TransFree(grid, xygrid, FTo,
                                       responses, densityfun, normfun;
                                       noise_field)
    inv(FT) # create inverse FT plans now, so wisdom is saved
    inv(FTo)
    Utils.saveFFTwisdom()
    Logging.@info("Setup finished.")
    flush(stderr)
    Eωk, transform, FT
end

"""
    setup(grid::EnvGrid, xygrid::FreeGrid, densityfun, normfun, responses, inputs;
          noise_field=nothing, fastpath=true, arraytype=Array)

Set up a 3D free-space envelope propagation, returning `(Eωk, transform, FT)`.

`arraytype` selects where the propagation runs. The default `Array` is the host. Passing
a GPU array type (or use [`Luna.resolve_arraytype`](@ref) to name one lazily) allocates
the transform buffers on that device, plans the FFTs with the device's own planner, and
returns a device-resident `Eωk` — `Luna.run` then executes the whole propagation there.
The input field is always built on the host (`Fields` is host-only) with a host plan and
uploaded once, which costs one transfer at setup.
"""
function setup(grid::Grid.EnvGrid, xygrid::Grid.FreeGrid,
               densityfun, normfun, responses, inputs;
               noise_field=nothing, fastpath=true, arraytype=Array)
    Logging.@info("Setting up and planning FFTs...")
    flush(stderr)
    ondevice = arraytype !== Array
    Utils.loadFFTwisdom() # host wisdom: the input field is always built on the host
    x = xygrid.x
    y = xygrid.y
    xr = Array{ComplexF64}(undef, length(grid.t), length(y), length(x))
    FTh = FFTW.plan_fft(xr, (1, 2, 3), flags=settings["fftw_flag"])
    Eωk = zeros(ComplexF64, length(grid.ω), length(y), length(x))
    doinputs_fs!(Eωk, grid, xygrid, FTh, inputs)
    if ondevice
        # Upload the input, then plan against it: at production sizes the device plan's
        # prototype is the state array itself, so no host prototype is allocated.
        Eωk = Adapt.adapt(arraytype, Eωk)
        FT = Utils.plan_fft_backend(Eωk, (1, 2, 3))
        length(grid.to) == length(grid.t) || error(
            "device propagation requires a non-oversampled grid (length(grid.to) == "*
            "length(grid.t)); got $(length(grid.to)) vs $(length(grid.t))")
        FTo = FT
    else
        FT = FTh
        if length(grid.to) == length(grid.t)
            FTo = FT # no oversampling: the two plans would be identical, so share one
        else
            xo = Array{ComplexF64}(undef, length(grid.to), length(y), length(x))
            FTo = FFTW.plan_fft(xo, (1, 2, 3), flags=settings["fftw_flag"])
        end
    end
    transform = NonlinearRHS.TransFree(grid, xygrid, FTo,
                                       responses, densityfun, normfun;
                                       noise_field, fastpath, arraytype)
    inv(FTh) # create inverse FT plans now, so wisdom is saved
    ondevice || inv(FTo)
    Utils.saveFFTwisdom()
    Logging.@info("Setup finished.")
    flush(stderr)
    Eωk, transform, FT
end

linoptype(l::AbstractArray) = "constant"
linoptype(l) = "variable"

gridtype(g::Grid.RealGrid) = "field-resolved"
gridtype(g::Grid.EnvGrid) = "envelope"
gridtype(g) = "unknown"

simtype(g, t, l) = Dict("field" => gridtype(g),
                        "transform" => string(t),
                        "linop" => linoptype(l))

function save_modeinfo_maybe(output, t::NonlinearRHS.TransModal)
    pol = t.ts.indices == 1:2 ? "xy" : t.ts.indices == 1 ? "x" : "y"
    modeinfos = unnest([Modes.modeinfo(m) for m in t.ts.ms])
    output(modeinfos; group="modes")
    output("polarisation", pol)
end

function unnest(dicts)
    out = Dict{String, Any}()
    for k in keys(dicts[1]) # assuming all dicts have the same keys
        out[sym2string(k)] = [sym2string(di[k]) for di in dicts]
    end
    out
end

sym2string(sym::Symbol) = string(sym)
sym2string(other) = other

save_modeinfo_maybe(output, t) = nothing

function run(Eω, grid,
             linop, transform, FT, output;
             min_dz=0, max_dz=grid.zmax/2, init_dz=1e-4, z0=0.0,
             rtol=1e-6, atol=1e-10, safety=0.9, norm=RK45.weaknorm,
             status_period=1, step_on=nothing, preserve_input=true,
             twin_period::Int=1)

    # reuse the transform's time-domain buffer (dead between RHS evaluations) as the
    # window-application scratch where possible, instead of allocating another field
    scr = NonlinearRHS.scratch(transform)
    Et = scr === nothing ? FT \ Eω : scr

    # twin_period > 1 applies the spectral/temporal windows only on every twin_period-th
    # accepted step (and always immediately before a save). NOTE: the windowing feeds
    # back into the propagation, so this changes results at the apodisation-leakage
    # level — the default of 1 reproduces the historical behavior exactly.
    nstep = 0
    function stepfun(Eω, z, dz, interpolant)
        nstep += 1
        if twin_period == 1 || nstep % twin_period == 0 ||
           Output.willsave(output, Eω, z, dz)
            Eω .*= grid.ωwin
            ldiv!(Et, FT, Eω)
            Et .*= grid.twin
            mul!(Eω, FT, Et)
        end
        output(Eω, z, dz, interpolant)
    end

    # check_cache does nothing except for HDF5Outputs
    Eωc, zc, dzc = Output.check_cache(output, Eω, z0, init_dz)
    if zc > z0
        Logging.@info("Found cached propagation. Resuming...")
        Eω, z0, init_dz = Eωc, zc, dzc
    end

    output(Grid.to_dict(grid), group="grid")
    output(simtype(grid, transform, linop), group="simulation_type")
    save_modeinfo_maybe(output, transform)

    flush(stderr) # flush std error once before starting to show setup steps
    RK45.solve_precon(
        transform, linop, Eω, z0, init_dz, grid.zmax, stepfun=stepfun,
        max_dt=max_dz, min_dt=min_dz,
        rtol=rtol, atol=atol, safety=safety, norm=norm,
        status_period=status_period, step_on=step_on,
        preserve_input=preserve_input)
end

# run some code for precompilation
Logging.with_logger(Logging.NullLogger()) do
    prop_capillary(125e-6, 0.3, :He, 1.0; λ0=800e-9, energy=1e-9,
                    τfwhm=10e-15, λlims=(150e-9, 4e-6), trange=1e-12, saveN=11,
                    PPT_options=Dict(:cache => false))
    prop_capillary(125e-6, 0.3, :He, (1.0, 0); λ0=800e-9, energy=1e-9,
                    τfwhm=10e-15, λlims=(150e-9, 4e-6), trange=1e-12, saveN=11,
                    PPT_options=Dict(:cache => false))
    prop_capillary(125e-6, 0.3, :He, 1.0; λ0=800e-9, energy=1e-9,
                    τfwhm=10e-15, λlims=(150e-9, 4e-6), trange=1e-12, saveN=11,
                    modes=4, PPT_options=Dict(:cache => false))
    p = Tools.capillary_params(120e-6, 10e-15, 800e-9, 125e-6, :He, P=1.0)

    # gnlse_sol.jl example but with N=1 and 100th of the fibre length
    γ = 0.1
    β2 = -1e-26
    τ0 = 280e-15
    τfwhm = (2*log(1 + sqrt(2)))*τ0
    fr = 0.18
    P0 = abs(β2)/((1-fr)*γ*τ0^2)
    flength = π*τ0^2/abs(β2)/100
    βs =  [0.0, 0.0, β2]
    λ0 = 835e-9
    λlims = [450e-9, 8000e-9]
    trange = 4e-12
    output = prop_gnlse(γ, flength, βs; λ0, τfwhm, power=P0, pulseshape=:sech, λlims, trange,
                        raman=true, shock=true, fr, shotnoise=true, saveN=11)
end

end # module
