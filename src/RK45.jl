module RK45
import Dates
import Logging
import Printf: @sprintf
import Luna: Utils
import Luna.Utils: format_elapsed, tchunks

#Get Butcher tableau etc from separate file (for convenience of changing if wanted)
include("dopri.jl")

function solve(f!, y0, t, dt, tmax;
               rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0, locextrap=true,
               norm=weaknorm, preserve_input=true,
               kwargs...)
    stepper = Stepper(f!, y0, t, dt,
                      rtol=rtol, atol=atol, safety=safety, max_dt=max_dt, min_dt=min_dt, locextrap=locextrap, norm=norm,
                      preserve_input=preserve_input)
    return solve(stepper, tmax; kwargs...)
end

function solve_precon(f!, linop, y0, t, dt, tmax;
                    rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0, locextrap=true, norm=weaknorm,
                    preserve_input=true,
                    kwargs...)
    stepper = PreconStepper(f!, linop, y0, t, dt,
                      rtol=rtol, atol=atol, safety=safety, max_dt=max_dt, min_dt=min_dt, locextrap=locextrap, norm=norm,
                      preserve_input=preserve_input)
    return solve(stepper, tmax; kwargs...)
end

function solve(s, tmax; stepfun=donothing!, output=false, outputN=201,
                        status_period=1, repeat_limit=10, step_on=nothing)
    # `step_on`: strictly increasing positions the stepper must land on
    # exactly (clipping the adaptive step, never enlarging it). Use for save
    # positions: the dense-output interpolant is one order less accurate than
    # the stepped solution (4th vs 5th) and shares the error norm's blind spot
    # (the step size is set by the dominant field components, not the weak
    # ones), so interpolated saves of a weak signal can scatter between
    # otherwise identical runs. Landing on the position instead makes the save
    # an exact step endpoint. Costs a few extra steps.
    step_on = isnothing(step_on) ? Float64[] :
        sort(collect(Float64, filter(z -> z > 0, step_on)))
    next_on = 1
    if output
        yout = Array{eltype(s.y)}(undef, (size(s.y)..., outputN))
        tout = range(s.t, stop=tmax, length=outputN)
        saved = 1
        yout[fill(:, ndims(s.y))..., 1] = s.y
    end

    steps = 0
    repeated = 0
    repeated_tot = 0

    Logging.@info "Starting propagation"
    start = Dates.now()
    tic = Dates.now()
    while s.tn <= tmax
        # advance past positions already reached (within relative eps), then
        # clip the trial step so it lands exactly on the next one
        while next_on <= length(step_on) &&
              step_on[next_on] <= s.tn + 1e-12 * tmax
            next_on += 1
        end
        if next_on <= length(step_on)
            gap = step_on[next_on] - s.tn
            gap < s.dtn && (s.dtn = gap)
        end
        ok = step!(s)
        steps += 1
        if Dates.value(Dates.now()-tic) > 1000*status_period
            speed = s.tn/(Dates.value(Dates.now()-start)/1000)
            eta_in_s = (tmax-s.tn)/(speed)
            if eta_in_s > 356400
                Logging.@info @sprintf("Progress: %.2f %%, ETA: XX:XX:XX, stepsize %.2e, err %.2f, repeated %d/%d steps",
                s.tn/tmax*100, s.dt, s.err, repeated_tot, steps)
            else
                eta_in_ms = Dates.Millisecond(ceil(eta_in_s*1000))
                etad = Dates.DateTime(Dates.UTInstant(eta_in_ms))
                Logging.@info @sprintf("Progress: %.2f %%, ETA: %s, stepsize %.2e, err %.2f, repeated %d/%d steps",
                    s.tn/tmax*100, Dates.format(etad, "HH:MM:SS"), s.dt, s.err, repeated_tot, steps)
            end
            flush(stderr)
            tic = Dates.now()
        end
        if ok
            if output
                while (saved<outputN) && tout[saved+1] < s.tn
                    ti = tout[saved+1]
                    yout[fill(:, ndims(s.y))..., saved+1] .= interpolate(s, ti)
                    saved += 1
                end
            end
            stepfun(s.yn, s.tn, s.dtn, t -> interpolate(s, t))
            repeated = 0
        else
            repeated += 1
            repeated_tot += 1
            if repeated > repeat_limit
                error("Reached limit for step repetition ($repeat_limit)")
            end
        end
    end
    totaltime = Dates.now()-start
    dtstring = format_elapsed(totaltime)
    Logging.@info @sprintf("Propagation finished in %s, %d steps",
                           dtstring, steps)

    if output
        return collect(tout), yout, steps
    else      
        return nothing
    end
end


mutable struct Stepper{T<:AbstractArray, F, nT}
    f!::F  # RHS function
    y::T  # Solution at current t
    yn::T  # Solution at t+dt
    yi::Union{Nothing, T}  # Interpolant array, allocated on first use (see interpolate())
    yerr::Union{Nothing, T}  # error estimate, allocated on first use (see errnorm)
    ks::NTuple{7, T}  # k values (intermediate solutions for Runge-Kutta method)
    t::Float64  # current time (propagation variable)
    tn::Float64  # next time
    dt::Float64  # time step
    dtn::Float64  # time step for next step
    rtol::Float64  # relative tolerance on error
    atol::Float64  # absolute tolerance on error
    safety::Float64  # safety factor for stepsize control
    max_dt::Float64  # maximum value for dt (default Inf)
    min_dt::Float64  # minimum value for dt (default 0)
    locextrap::Bool  # true if using local extrapolation
    ok::Bool  # true if current step was successful. Also means "an FSAL move is owed"
              # (see step!), so nothing may write it after step! has returned.
    err::Float64  # error metric to be compared to tol
    errlast::Float64  # error of the most recent successful step
    norm::nT # function to calculate error metric, defaults to RK45.weaknorm
end

function Stepper(f!, y0, t, dt;
                 rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0,
                 locextrap=true, norm=weaknorm, preserve_input=true)
    k1 = similar(y0)
    f!(k1, y0, t)
    ks = (k1, similar(k1), similar(k1), similar(k1), similar(k1), similar(k1), similar(k1))
    # with preserve_input=false the stepper adopts y0 as its yn buffer (the caller's array
    # is written into during stepping), saving one field-sized allocation
    yn = preserve_input ? copy(y0) : y0
    return Stepper(f!, copy(y0), yn, nothing, nothing, ks,
        float(t), float(t), float(dt), float(dt),
        float(rtol), float(atol), float(safety), float(max_dt), float(min_dt),
        locextrap, false, 0.0, 0.0, norm)
end

mutable struct PreconStepper{T<:AbstractArray, F, P, nT}
    fbar!::F  # RHS callable
    prop!::P # linear propagator callable
    y::T  # Solution at current t
    yn::T  # Solution at t+dt
    yi::Union{Nothing, T}  # Interpolant array, allocated on first use (see interpolate())
    yerr::Union{Nothing, T}  # error estimate, allocated on first use (see errnorm)
    ks::NTuple{7, T}  # k values (intermediate solutions for Runge-Kutta method)
    t::Float64  # current time (propagation variable)
    tn::Float64  # next time
    dt::Float64  # time step
    dtn::Float64  # time step for next step
    rtol::Float64  # relative tolerance on error
    atol::Float64  # absolute tolerance on error
    safety::Float64  # safety factor for stepsize control
    max_dt::Float64  # maximum value for dt (default Inf)
    min_dt::Float64  # minimum value for dt (default 0)
    locextrap::Bool  # true if using local extrapolation
    ok::Bool  # true if current step was successful. Also means "an FSAL move is owed"
              # (see step!), so nothing may write it after step! has returned.
    err::Float64  # error metric to be compared to tol
    errlast::Float64  # error of the most recent successful step
    norm::nT  # function to calculate error metric, defaults to RK45.weaknorm
end

function PreconStepper(f!, linop, y0, t, dt;
                       rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0,
                       locextrap=true, norm=weaknorm, preserve_input=true)
    prop! = make_prop!(linop, y0)
    fbar! = make_fbar!(f!, prop!, y0)
    k1 = similar(y0)
    # fbar! propagates (clobbers) its input in place; pass a transient copy so that y0 is
    # untouched and k1 is computed through a scratch copy exactly as before
    fbar!(k1, copy(y0), t, t)
    ks = (k1, similar(k1), similar(k1), similar(k1), similar(k1), similar(k1), similar(k1))
    # with preserve_input=false the stepper adopts y0 as its yn buffer (the caller's array
    # is written into during stepping), saving one field-sized allocation
    yn = preserve_input ? copy(y0) : y0

    return PreconStepper(fbar!, prop!, copy(y0), yn, nothing, nothing, ks,
        float(t), float(t), float(dt), float(dt), float(rtol), float(atol), float(safety),
        float(max_dt), float(min_dt), locextrap, false, 0.0, 0.0, norm)
end

function step!(s)
    # FSAL: k7 of the previous step is k1 of this one. The move is deferred to here
    # (rather than done at the end of step!) because the completed step's dense output is
    # consumed between step! returning and this call, and the interpolant's k1 weight is
    # nonzero (interpC column 1; b1(1) = 35/384) -- moving it any earlier degrades every
    # interpolated save from 4th to 2nd order. s.ok is false before the first step (ks[2:7]
    # are `similar`, i.e. undef) and after a rejected step (k1 must survive for the retry).
    # For the PreconStepper this must also stay ahead of evaluate!'s rebase of ks[1] into
    # the new anchor frame, which it does by construction.
    s.ok && (s.ks[1] .= s.ks[end])
    evaluate!(s)

    # Propagate the 5th-order solution (local extrapolation, the default) or the embedded
    # 4th-order one. Single fused pass; the n-ary + is left-associated, so per element this
    # accumulates in exactly the same order as the sequential .+= version (zero weights are
    # skipped, as before: b5[2] == b5[7] == 0, b4[2] == 0). Both are formed explicitly
    # rather than reusing the stage-6 accumulation evaluate! leaves in yn -- which is the
    # 5th-order solution -- so this does not depend on fbar! leaving its input alone.
    k1, k2, k3, k4, k5, k6, k7 = s.ks
    if s.locextrap
        c1 = s.dt*b5[1]; c3 = s.dt*b5[3]; c4 = s.dt*b5[4]
        c5 = s.dt*b5[5]; c6 = s.dt*b5[6]
        tchunks(s.yn, s.y, k1, k3, k4, k5, k6) do yn, y, k1, k3, k4, k5, k6
            @. yn = y + c1*k1 + c3*k3 + c4*k4 + c5*k5 + c6*k6
        end
    else
        c1 = s.dt*b4[1]; c3 = s.dt*b4[3]; c4 = s.dt*b4[4]
        c5 = s.dt*b4[5]; c6 = s.dt*b4[6]; c7 = s.dt*b4[7]
        tchunks(s.yn, s.y, k1, k3, k4, k5, k6, k7) do yn, y, k1, k3, k4, k5, k6, k7
            @. yn = y + c1*k1 + c3*k3 + c4*k4 + c5*k5 + c6*k6 + c7*k7
        end
    end

    s.err = errnorm(s)
    s.ok = s.err <= 1
    stepcontrolPI!(s)
    if s.ok
        s.tn = s.t + s.dt
    else
        s.yn .= s.y
    end
    prop!_maybe(s) # propagate to new time to pass correct solution to stepfun
    return s.ok
end

# Accumulate the ii-th Butcher stage state into yn in a single fused pass. The n-ary +
# is left-associated, so per element this matches the sequential yn .= y; yn .+= ...
# accumulation bit-for-bit (zero tableau entries are skipped, as before: B[6][2] == 0).
function stage!(yn, y, dt, ks, ii)
    k1, k2, k3, k4, k5, k6, k7 = ks
    if ii == 1
        c1 = dt*B[1][1]
        tchunks(yn, y, k1) do yn, y, k1
            @. yn = y + c1*k1
        end
    elseif ii == 2
        c1 = dt*B[2][1]; c2 = dt*B[2][2]
        tchunks(yn, y, k1, k2) do yn, y, k1, k2
            @. yn = y + c1*k1 + c2*k2
        end
    elseif ii == 3
        c1 = dt*B[3][1]; c2 = dt*B[3][2]; c3 = dt*B[3][3]
        tchunks(yn, y, k1, k2, k3) do yn, y, k1, k2, k3
            @. yn = y + c1*k1 + c2*k2 + c3*k3
        end
    elseif ii == 4
        c1 = dt*B[4][1]; c2 = dt*B[4][2]; c3 = dt*B[4][3]; c4 = dt*B[4][4]
        tchunks(yn, y, k1, k2, k3, k4) do yn, y, k1, k2, k3, k4
            @. yn = y + c1*k1 + c2*k2 + c3*k3 + c4*k4
        end
    elseif ii == 5
        c1 = dt*B[5][1]; c2 = dt*B[5][2]; c3 = dt*B[5][3]; c4 = dt*B[5][4]; c5 = dt*B[5][5]
        tchunks(yn, y, k1, k2, k3, k4, k5) do yn, y, k1, k2, k3, k4, k5
            @. yn = y + c1*k1 + c2*k2 + c3*k3 + c4*k4 + c5*k5
        end
    else
        c1 = dt*B[6][1]; c3 = dt*B[6][3]; c4 = dt*B[6][4]; c5 = dt*B[6][5]; c6 = dt*B[6][6]
        tchunks(yn, y, k1, k3, k4, k5, k6) do yn, y, k1, k3, k4, k5, k6
            @. yn = y + c1*k1 + c3*k3 + c4*k4 + c5*k5 + c6*k6
        end
    end
end

function evaluate!(s::Stepper)
    # Set new time and stepsize values -- this happens at the beginning because
    # the interpolant still requires the old values after the step has finished
    s.dt = s.dtn
    s.t = s.tn
    s.y .= s.yn
    for ii = 1:6
        stage!(s.yn, s.y, s.dt, s.ks, ii)
        s.f!(s.ks[ii+1], s.yn, s.t+nodes[ii]*s.dt)
    end
end

function evaluate!(s::PreconStepper)
    # Set new time and stepsize values -- this happens at the beginning because
    # the interpolant still requires the old values after the step has finished
    s.y .= s.yn
    s.prop!(s.ks[1], s.t, s.tn)
    s.dt = s.dtn
    s.t = s.tn
    for ii = 1:6
        stage!(s.yn, s.y, s.dt, s.ks, ii)
        s.fbar!(s.ks[ii+1], s.yn, s.t, s.t+nodes[ii]*s.dt)
    end
    # fbar! propagated (clobbered) yn in place during the last stage, but that no longer
    # matters: step! rebuilds yn from the weight vector in both locextrap branches.
end

prop!_maybe(s::PreconStepper) = s.prop!(s.yn, s.t, s.tn)
prop!_maybe(s) = nothing

"Interpolate solution, aka dense output."
function interpolate(s::Stepper, ti::Float64)
    # Snap queries within round-off of the step endpoint to the stepped solution. A
    # `step_on` landing is t + (target - t), which can differ from target by 1 ulp, so
    # bitwise equality with tn is not a sufficient endpoint test. The polynomial's σ=1
    # weights are b5 (sum(interpC, dims=1) == b5), i.e. exactly what locextrap
    # propagation uses, so just outside the window it agrees with yn to round-off —
    # but a save asked for at the endpoint should be the stepped solution itself.
    if abs(ti - s.tn) <= 4*eps(abs(s.tn))
        return s.yn
    end
    if ti > s.tn
        error("Attempting to extrapolate!")
    end
    if ti == s.t
        return s.y
    end
    interpolant!(s, ti)
    return @. s.y + s.dt.*s.yi
end

"""
Accumulate the dense-output interpolant at `ti` into `s.yi` (allocated on first use).
Single fused pass; per element this matches the sequential fill!/.+= accumulation
bit-for-bit (including the zero-weight `ks[2]` term, which the loop also included).
"""
function interpolant!(s, ti::Float64)
    σ = (ti - s.t)/s.dt
    σp = map(p -> σ^p, range(1, stop=4))
    b = sum(σp.*interpC, dims=1)
    if s.yi === nothing
        s.yi = similar(s.y)
    end
    k1, k2, k3, k4, k5, k6, k7 = s.ks
    w1 = b[1]; w2 = b[2]; w3 = b[3]; w4 = b[4]; w5 = b[5]; w6 = b[6]; w7 = b[7]
    tchunks(s.yi, k1, k2, k3, k4, k5, k6, k7) do yi, k1, k2, k3, k4, k5, k6, k7
        @. yi = 0 + k1*w1 + k2*w2 + k3*w3 + k4*w4 + k5*w5 + k6*w6 + k7*w7
    end
end

"Interpolate solution, aka dense output."
function interpolate(s::PreconStepper, ti::Float64)
    # Near-endpoint snap: see interpolate(s::Stepper, ti) for the rationale.
    if abs(ti - s.tn) <= 4*eps(abs(s.tn))
        return s.yn
    end
    if ti > s.tn
        error("Attempting to extrapolate!")
    end
    if ti == s.t
        return s.y
    end
    interpolant!(s, ti)
    out =  @. s.y + s.dt.*s.yi
    s.prop!(out, s.t, ti)
    return out
end

"Make propagator for the case of constant linear operator"
function make_prop!(linop::AbstractArray, y0)
    prop! = let linop=linop
        function prop!(y, t1, t2, bwd=false)
            dt = bwd ? (t1-t2) : (t2-t1)
            tchunks(y, linop) do y, linop
                @. y *= exp(linop*dt)
            end
        end
    end
end

"""
    device_capable(linop!)

Trait for `z`-dependent linear operators `linop!(out, z)`: `true` if `linop!` can write
directly into a device array. The default `false` makes [`make_prop!`](@ref) evaluate the
operator into a host buffer and upload it when the state lives on a device — the modal and
mode-averaged operators of `LinearOps` are host scalar loops.
"""
device_capable(linop!) = false

"Make propagator for the case of non-constant linear operator"
function make_prop!(linop!, y0)
    linop_int = similar(y0)
    # a host-only linop! is evaluated on the host and uploaded (small for modal states)
    hostbuf = (Utils.isdevice(y0) && !device_capable(linop!)) ?
              Array{eltype(y0)}(undef, size(y0)) : nothing
    lastt2 = [typemin(Float64)]
    function prop!(y, t1, t2, bwd=false)
        #= linop is always evaluated at later time, even for backward propagation
            therefore, linop is often evaluated at the same t2 twice in a row=#
        if lastt2[1] != t2
            if isnothing(hostbuf)
                linop!(linop_int, t2)
            else
                linop!(hostbuf, t2)
                copyto!(linop_int, hostbuf)
            end
        end
        lastt2[1] = t2
        dt = bwd ? (t1-t2) : (t2-t1)
        tchunks(y, linop_int) do y, linop_int
            @. y *= exp(linop_int*dt)
        end
    end
    return prop!
end

"""
Make closure for the pre-conditioned RHS function.

!!! note
    `fbar!(out, ybar, t1, t2)` propagates `ybar` to `t2` **in place** — the caller must
    not rely on the contents of `ybar` afterwards. `evaluate!(::PreconStepper)` rebuilds
    `yn` from `y` after every stage, so this is safe there and saves one field-sized
    scratch buffer.
"""
function make_fbar!(f!, prop!, y0)
    fbar! = let f! = f!, prop! = prop!
        function fbar!(out, ybar, t1, t2)
            prop!(ybar, t1, t2) # propagate to t2 (in place)
            f!(out, ybar, t2) # evaluate RHS function
            prop!(out, t1, t2, true) # propagate back to t1
        end
    end
end

"Max-ish norm (from Dane Austin's code, no idea where he got it from)."
function maxnorm(yerr, y, yn, rtol, atol)
    maxerr = 0
    maxy = 0
    for ii in eachindex(yerr)
        maxerr = max(maxerr, abs(yerr[ii]))
        maxy = max(maxy, max(abs(y[ii]), abs(yn[ii])))
    end
    return maxerr/(atol + rtol*maxy)
end

"Alternative form of max-ish norm."
function maxnorm_ratio(yerr, y, yn, rtol, atol)
    m = 0
    for ii in eachindex(yerr)
        den = atol + rtol*max(abs(y[ii]), abs(yn[ii]))
        m = max(abs(yerr[ii])/den, m)
    end
    return m
end

"Semi-norm as used in DifferentialEquations.jl, see Hairer, Solving Ordinary Differential
Equations: Nonstiff Problems, eq. (4.11) (p.168 of the second revised edition)."
function normnorm(yerr, y, yn, rtol, atol)
    s = 0
    for ii in eachindex(yerr)
        s += abs2(yerr[ii]/(atol + rtol*max(abs(y[ii]), abs(yn[ii]))))
    end
    sqrt(s/length(yerr))
end

"'Weak' norm as used in fnfep."
function weaknorm(yerr, y, yn, rtol, atol)
    sy = 0
    syn = 0
    syerr = 0
    for ii in eachindex(yerr)
        sy += abs2(y[ii])
        syn += abs2(yn[ii])
        syerr += abs2(yerr[ii])
    end
    errwt = max(max(sqrt(sy), sqrt(syn)), atol)
    return sqrt(syerr)/rtol/errwt
end

"""
    fused_errnorm(norm)

Return a function `f(s)` which computes the error metric for stepper `s` without
materialising the error-estimate array, or `nothing` if no fused version exists for
`norm`. In the latter case the error estimate is accumulated into `s.yerr` (allocated on
first use) and `norm(yerr, y, yn, rtol, atol)` is called as before. Custom norms can opt
in by adding a method for their own type.
"""
fused_errnorm(norm) = nothing
fused_errnorm(::typeof(weaknorm)) = weaknorm_fused

"""
Fused version of [`weaknorm`](@ref): the error estimate is computed element-by-element on
the fly instead of being accumulated into a separate array.

On the host this performs the same floating-point operations in the same order as the
materialised version, so the result is bit-identical. On a device it is a single
reduction over the same expression; the accumulation order of a parallel reduction
differs, so the result agrees only to rounding — which is expected and unavoidable, and
is why device runs are validated against the host to a tolerance rather than bitwise.
"""
weaknorm_fused(s) = _weaknorm_fused(Utils.backend(s.y), s)

function _weaknorm_fused(::Utils.CPUBackend, s)
    dt = s.dt
    k1, k2, k3, k4, k5, k6, k7 = s.ks
    e1 = errest[1]; e3 = errest[3]; e4 = errest[4] # errest[2] == 0 and is skipped,
    e5 = errest[5]; e6 = errest[6]; e7 = errest[7] # matching the materialised loop
    y = s.y
    yn = s.yn
    sy = 0
    syn = 0
    syerr = 0
    @inbounds for ii in eachindex(y)
        yerr = 0 + dt*k1[ii]*e1 + dt*k3[ii]*e3 + dt*k4[ii]*e4 +
                   dt*k5[ii]*e5 + dt*k6[ii]*e6 + dt*k7[ii]*e7
        sy += abs2(y[ii])
        syn += abs2(yn[ii])
        syerr += abs2(yerr)
    end
    errwt = max(max(sqrt(sy), sqrt(syn)), s.atol)
    return sqrt(syerr)/s.rtol/errwt
end

# One reduction over eight arrays, accumulating the three sums as a tuple, so the error
# estimate is never materialised: on a device that would be a tenth field-sized buffer
# (2 GiB at production size) plus an extra pass. `mapreduce` over several arrays fuses
# them into a single lazy broadcast, so nothing intermediate is allocated either.
function _weaknorm_fused(::Utils.DeviceBackend, s)
    dt = s.dt
    k1, k2, k3, k4, k5, k6, k7 = s.ks
    e1 = errest[1]; e3 = errest[3]; e4 = errest[4]
    e5 = errest[5]; e6 = errest[6]; e7 = errest[7]
    f = @inline function (y, yn, a1, a3, a4, a5, a6, a7)
        yerr = 0 + dt*a1*e1 + dt*a3*e3 + dt*a4*e4 + dt*a5*e5 + dt*a6*e6 + dt*a7*e7
        (abs2(y), abs2(yn), abs2(yerr))
    end
    sy, syn, syerr = mapreduce(f, _add3, s.y, s.yn, k1, k3, k4, k5, k6, k7;
                               init=(0.0, 0.0, 0.0))
    errwt = max(max(sqrt(sy), sqrt(syn)), s.atol)
    return sqrt(syerr)/s.rtol/errwt
end

# Associative reduction operator for the three partial sums above. Defined as a named
# function rather than a closure so it is isbits and can be passed into a device kernel.
@inline _add3(a, b) = (a[1]+b[1], a[2]+b[2], a[3]+b[3])

"""
Compute the error metric for the current step. Uses a fused, allocation-free version if
one exists for `s.norm` (see [`fused_errnorm`](@ref)); otherwise materialises the error
estimate into `s.yerr` and calls `s.norm`.
"""
function errnorm(s)
    f = fused_errnorm(s.norm)
    f === nothing || return f(s)
    return _errnorm_materialised(Utils.backend(s.y), s)
end

function _errnorm_materialised(::Utils.CPUBackend, s)
    if s.yerr === nothing
        s.yerr = similar(s.y)
    end
    k1, k2, k3, k4, k5, k6, k7 = s.ks
    dt = s.dt
    e1 = errest[1]; e3 = errest[3]; e4 = errest[4] # errest[2] == 0 and is skipped
    e5 = errest[5]; e6 = errest[6]; e7 = errest[7]
    # single fused pass; left-associated + matches the sequential .+= accumulation
    tchunks(s.yerr, k1, k3, k4, k5, k6, k7) do yerr, k1, k3, k4, k5, k6, k7
        @. yerr = 0 + dt*k1*e1 + dt*k3*e3 + dt*k4*e4 + dt*k5*e5 + dt*k6*e6 + dt*k7*e7
    end
    return s.norm(s.yerr, s.y, s.yn, s.rtol, s.atol)
end

# Refuse rather than crawl: the materialising path would allocate a tenth field-sized
# device array and then hand it to a norm which almost certainly indexes it scalar-wise.
_errnorm_materialised(::Utils.DeviceBackend, s) = error(
    "the error norm $(s.norm) has no `RK45.fused_errnorm` method, so it would need the "*
    "error estimate materialised into an extra field-sized device array and evaluated "*
    "by a host-side scalar loop. Define `RK45.fused_errnorm(::typeof(yournorm))` "*
    "returning a function of the stepper (see `weaknorm_fused`).")

"Simple proportional error controller, see e.g. Hairer eq. (4.13)."
function stepcontrolP!(s)
    if s.ok
        # if error is zero, there is no nonlinearity: increase step size by a lot
        s.dtn = s.err == 0 ? 1.5*s.dt : s.dt * min(5, s.safety*(s.err)^(-1/5))
    else
        if !isfinite(s.err) # check for NaN or Inf
            s.dtn = s.dt/2  # if we have one then we're in big trouble so halve the step size
        else
            s.dtn = s.dt * max(0.1, s.safety*(s.err)^(-1/5))
        end
    end
    steplims!(s)
end

"Proportional-integral error controller, aka Lund stabilisation.
See G. Söderlind and L. Wang, J. Comput. Appl. Math. 185, 225 (2006).
"
function stepcontrolPI!(s)
    β1 = 3/5 / 5
    β2 = -1/5 / 5
    ε = 0.8
    if s.ok
        s.errlast == 0 && (s.errlast = s.err) # if last error is zero, use current error instead
        if s.err == 0
            fac = 1.5 # zero error means no nonlinearity: increase step size by a lot 
        else
            fac = s.safety * (ε/s.err)^β1 * (ε/s.errlast)^β2
        end
        # (0.99 <= fac <= 1.01) && (fac = 1.0)
        s.dtn = fac * s.dt
        s.errlast = s.err
    else
        if !isfinite(s.err) # check for NaN or Inf
            s.dtn = s.dt/2  # if we have one then we're in big trouble so halve the step size
        else
            s.dtn = s.dt * max(0.1, s.safety*(s.err)^(-1/5))
        end
    end
    steplims!(s)
end

"Apply user-defined limits on step size."
function steplims!(s)
    if s.dtn > s.max_dt
        s.dtn = s.max_dt
    elseif s.dtn < s.min_dt
        s.dtn = s.min_dt
        s.ok = true
    end
end

function donothing!(y, z, dz, interpolant)
end

end