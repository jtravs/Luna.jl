module Luna
import FFTW
import Hankel
import Logging
import LinearAlgebra: mul!, ldiv!
import Printf: @sprintf
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
FFTW is determined automatically (see [`Utils.FFTWthreads()`](@ref))
"""
function set_fftw_threads(nthr=0)
    settings["fftw_threads"] = nthr
    FFTW.set_num_threads(Utils.FFTWthreads())
end

function __init__()
    set_fftw_threads()
end

include("Utils.jl")
include("Scans.jl")
include("Output.jl")
include("Maths.jl")
include("PhysData.jl")
include("Grid.jl")
include("Boundaries.jl")
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
       Nonlinear, Ionisation, NonlinearRHS, LinearOps, Boundaries, Stats, Polarisation,
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

function setup(grid::Grid.EnvGrid, xygrid::Grid.FreeGrid,
               densityfun, normfun, responses, inputs; noise_field=nothing)
    Logging.@info("Setting up and planning FFTs...")
    flush(stderr)
    Utils.loadFFTwisdom()
    x = xygrid.x
    y = xygrid.y
    xr = Array{ComplexF64}(undef, length(grid.t), length(y), length(x))
    FT = FFTW.plan_fft(xr, (1, 2, 3), flags=settings["fftw_flag"])
    Eωk = zeros(ComplexF64, length(grid.ω), length(y), length(x))
    doinputs_fs!(Eωk, grid, xygrid, FT, inputs)
    xo = Array{ComplexF64}(undef, length(grid.to), length(y), length(x))
    FTo = FFTW.plan_fft(xo, (1, 2, 3), flags=settings["fftw_flag"])
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

linoptype(l::AbstractArray) = "constant"
linoptype(l) = "variable"

"""
    linop_array(linop, Eω, z)

The linear operator as an array, materialising the z-dependent (closure) form at `z`. The
linop has the same shape as the field, which is how `RK45.make_prop!` allocates its own
buffer.
"""
linop_array(linop::AbstractArray, Eω, z) = linop
function linop_array(linop!, Eω, z)
    out = similar(Eω)
    linop!(out, z)
    out
end

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

"Report the absorbing-boundary configuration."
function log_boundary(grid, N, ℓ, collar)
    l = Boundaries.reflength(grid, N, ℓ)
    w = Boundaries.tcollarwidth(grid, collar)
    trange = maximum(grid.t) - minimum(grid.t)
    #= The clamp in Boundaries.rate is not worth reporting: it bites only where the profile
       is already below exp(-30), and even there the attenuation over the whole propagation
       is exp(-30 zmax/ℓ). What is worth reporting is the strength the user actually chose,
       so quote the attenuation over the propagation at the half-way point of a taper. =#
    Logging.@info(@sprintf(
        "Absorbing boundaries: rate-based, reference length %.3g m (%.3g applications of \
         the window profile over %.3g m; a %.0f%% point of the taper attenuates by %.1e \
         over the propagation). Temporal collar %.3g fs, %.1f%% of the time window.",
        l, grid.zmax/l, grid.zmax, 50, 0.5^(grid.zmax/l), w*1e15, 100*w/trange))
end

"""
    run(Eω, grid, linop, transform, FT, output; kwargs...)

Run the propagation.

# Absorbing boundaries
- `boundary::Symbol=:rate`: how the absorbing boundaries at the edges of the frequency and
    time windows are applied.
    - `:rate` applies them as an absorption *rate per unit distance*: the spectral absorber
      is folded into `linop` (so the propagator applies it exactly) and the temporal
      absorber is applied as `exp(-α_t Δz)`. The total absorption over a distance depends
      only on that distance, not on how many steps the solver took.
    - `:legacy` reproduces the historical behaviour — multiplying the solution by
      `grid.ωwin` and `grid.twin` after every accepted step. This makes the cumulative
      absorption depend on the step count and hence on `rtol`, so the result does not
      converge as the tolerance is tightened. Provided only to reproduce older results.
    - `:none` disables the graded absorbers. The hard band limit outside `grid.sidx` is
      still applied (it is a 0/1 mask, so it never had a step-count dependence), and the
      nonlinear polarisation is still band-limited in `NonlinearRHS`, but nothing stops
      energy wrapping around the time window or piling up at the edge of the frequency
      window.
- `boundary_N::Real=$(Boundaries.DEFAULT_N)`: absorber strength, expressed as the number of
    times the historical window profile is applied over the whole propagation length. The
    reference length is `ℓ = zmax/boundary_N`.
- `boundary_length=nothing`: reference length `ℓ` in metres, overriding `boundary_N`.
- `tcollar::Real=$(Boundaries.DEFAULT_TCOLLAR)`: width of the temporal absorber collar as a
    fraction of the time window. Only used if it is wider than the collar `grid.twin`
    already has (which can be nearly zero, depending on how `trange` rounds up).

See [`Luna.Boundaries`](@ref) for the rationale.
"""
function run(Eω, grid,
             linop, transform, FT, output;
             min_dz=0, max_dz=grid.zmax/2, init_dz=1e-4, z0=0.0,
             rtol=1e-6, atol=1e-10, safety=0.9, norm=RK45.weaknorm,
             status_period=1,
             boundary=:rate, boundary_N=Boundaries.DEFAULT_N, boundary_length=nothing,
             tcollar=Boundaries.DEFAULT_TCOLLAR)

    boundary in (:rate, :legacy, :none) || error(
        "boundary must be :rate, :legacy or :none, not $boundary")

    Et = FT \ Eω

    # check_cache does nothing except for HDF5Outputs
    Eωc, zc, dzc = Output.check_cache(output, Eω, z0, init_dz)
    if zc > z0
        Logging.@info("Found cached propagation. Resuming...")
        Eω, z0, init_dz = Eωc, zc, dzc
    end

    #= NOTE: everything below must come after check_cache, which can move z0 and init_dz:
       zprev seeds the distance the temporal absorber is applied over. =#
    ℓabs = Boundaries.reflength(grid, boundary_N, boundary_length)
    ωmask = Boundaries.ωmask(grid)

    stepfun = if boundary === :rate
        #= The stepper must resolve the absorber's reference length. RK45's fbar! amplifies
           the RHS by exp(+α Δz) while the nonlinear drive it acts on is ∝ exp(-α ℓ), so the
           amplified band-edge elements stay bounded — and stay small in weaknorm's global
           sum — only while Δz ≤ ℓ. This costs nothing in practice: it just requires at
           least boundary_N steps, and real runs take far more. =#
        if max_dz > ℓabs
            Logging.@info(@sprintf(
                "Reducing max_dz from %.3g m to the absorber reference length %.3g m.",
                max_dz, ℓabs))
            max_dz = ℓabs
        end
        init_dz = min(init_dz, max_dz)

        αω = Boundaries.spectral_rate(grid; N=boundary_N, ℓ=boundary_length)
        αt = Boundaries.temporal_rate(grid; N=boundary_N, ℓ=boundary_length,
                                      collar=tcollar)
        Boundaries.walkoff_check(grid, linop_array(linop, Eω, z0), αt,
                                 Boundaries.tcollarwidth(grid, tcollar))
        linop = Boundaries.addloss(linop, αω)
        log_boundary(grid, boundary_N, boundary_length, tcollar)

        # only the collar entries are ever ≠ 1, so only those need updating each step
        tidcs = findall(>(0), αt)
        tfac = ones(Float64, length(grid.t))
        zprev = Ref(float(z0))
        function stepfun_rate(Eω, z, dz, interpolant)
            #= The absorption applied here is exp(-αt*Δz) over the distance actually
               travelled, so successive factors telescope to exp(-αt*L) no matter how the
               solver subdivides the propagation. =#
            Δz = z - zprev[]
            zprev[] = z
            Eω .*= ωmask # idempotent hard band limit, so no rate treatment needed
            if Δz > 0
                @inbounds for i in tidcs
                    tfac[i] = exp(-αt[i]*Δz)
                end
                ldiv!(Et, FT, Eω) # NB: for a RealGrid rfft plan this destroys Eω
                Et .*= tfac
                mul!(Eω, FT, Et)
            end
            output(Eω, z, dz, interpolant)
        end
    elseif boundary === :legacy
        Logging.@warn(
            "boundary=:legacy applies the absorbing boundaries once per accepted step, " *
            "so the absorption depends on the step count and the result depends on rtol.")
        function stepfun_legacy(Eω, z, dz, interpolant)
            Eω .*= grid.ωwin
            ldiv!(Et, FT, Eω)
            Et .*= grid.twin
            mul!(Eω, FT, Et)
            output(Eω, z, dz, interpolant)
        end
    else
        #= Even with no absorber the hard band limit has to stay: TransModeAvg leaves the
           out-of-band part of nl unnormalised and unwindowed, and the linear operator is
           zero there, so nothing else removes it. =#
        function stepfun_none(Eω, z, dz, interpolant)
            Eω .*= ωmask
            output(Eω, z, dz, interpolant)
        end
    end

    output(Grid.to_dict(grid), group="grid")
    st = simtype(grid, transform, linop)
    st["boundary"] = string(boundary)
    boundary === :rate && (st["boundary_length"] = string(ℓabs))
    output(st, group="simulation_type")
    save_modeinfo_maybe(output, transform)

    flush(stderr) # flush std error once before starting to show setup steps
    RK45.solve_precon(
        transform, linop, Eω, z0, init_dz, grid.zmax, stepfun=stepfun,
        max_dt=max_dz, min_dt=min_dz,
        rtol=rtol, atol=atol, safety=safety, norm=norm,
        status_period=status_period)
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
