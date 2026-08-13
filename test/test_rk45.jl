import FFTW
import Logging
import Luna: RK45
import Test: @test

function testinit()
    trange = 32
    ωrange = 160
    samples = 2^(ceil(Int, log2(trange*ωrange/2π)))
    δt = trange/samples
    δω = 2π/trange
    n = collect(range(0, length=samples))
    t = @. (n-samples/2)*δt
    ω = @. (n-samples/2)*δω
    N = 5
    z0 = π/2
    zmax = z0*2
    
    At = @. N*sech(t)
    Aω = FFTW.fftshift(FFTW.fft(At))

    At = zero(Aω)
    nl = zero(Aω)
    nlω = zero(Aω)

    FT = FFTW.plan_fft!(Aω)
    IFT = FFTW.plan_ifft!(Aω)
    f! = let FT=FT, IFT=IFT, ω=ω
        function f!(out, Aω, z)
                out .= Aω
                IFT*out
                out .= abs2.(out).*out
                FT*out
                @. out = im*out - im/2*ω^2*Aω
                return out
        end
    end

    Lin = @. -im/2*ω^2
    
    linout = similar(Aω)
    Linfunc = let ω=ω
        function Linfunc(out, z)
            @. out = -im/2*ω^2
        end
    end

    fnl! = let FT=FT, IFT=IFT
        function fnl!(out, Aω, z)
                out .= Aω
                IFT*out
                out .= abs2.(out).*out
                FT*out
                @. out = im*out
                return out
        end
    end

    return t, ω, zmax, Aω, f!, Lin, fnl!, Linfunc
end


function test_precon(plot=false)
    t, ω, zmax, Aω, f!, Lin, fnl!, Linfunc = testinit()
    z = 0
    dz = 1e-3

    saveN = 501

    zarr, Aarr = RK45.solve_precon(fnl!, Lin, Aω, z, dz, zmax, saveN, rtol=1e-6, status_period=2)

    if plot
        Atarr = FFTW.ifft(FFTW.ifftshift(Aarr, 1), 1)
        energy = dropdims(sum(abs2.(Atarr), dims=1), dims=1)
        pygui(true)
        plt.figure()
        plt.pcolormesh(t, zarr, abs2.(transpose(Atarr)))
        plt.colorbar()
        plt.figure()
        plt.pcolormesh(ω, zarr, abs2.(transpose(Aarr)))
        plt.figure()
        plt.plot(zarr, 1 .- energy/energy[1])
        plt.figure()
        plt.plot(t, abs2.(Atarr[:, 1]))
        plt.plot(t, abs2.(Atarr[:, end]))
    end

    return zarr, Aarr
end

function test_noprecon(plot=false)
    t, ω, zmax, Aω, f!, Lin, fnl!, Linfunc = testinit()
    z = 0
    dz = 1e-3

    saveN = 201

    zarr, Aarr = RK45.solve(f!, Aω, z, dz, zmax, saveN)

    if plot
        Atarr = FFTW.ifft(FFTW.ifftshift(Aarr, 1), 1)
        energy = dropdims(sum(abs2.(Atarr), dims=1), dims=1)
        pygui(true)
        plt.figure()
        plt.pcolormesh(t, zarr, abs2.(transpose(Atarr)))
        plt.colorbar()
        plt.figure()
        plt.pcolormesh(ω, zarr, abs2.(transpose(Aarr)))
        plt.figure()
        plt.plot(zarr, 1 .- energy/energy[1])
        plt.figure()
        plt.plot(t, abs2.(Atarr[:, 1]))
        plt.plot(t, abs2.(Atarr[:, end]))
    end

    return zarr, Aarr
end

t, ω, zmax, Aω, f!, Lin, fnl!, Linfunc = testinit()
z = 0
dz = 1e-3

zarr, Aarr = RK45.solve(f!, copy(Aω), z, dz, zmax, rtol=1e-8, output=true, outputN=501)
zarrp, Aarrp = RK45.solve_precon(fnl!, Lin, copy(Aω), z, dz, zmax,
                                 rtol=1e-8, output=true, outputN=501)
zarrpf, Aarrpf = RK45.solve_precon(fnl!, Linfunc, copy(Aω), z, dz, zmax,
                                   rtol=1e-8, output=true, outputN=501)
# Is the initial spectrum restored after 2 soliton periods? (without preconditioner)
@test isapprox(abs2.(Aarr[:, 1]), abs2.(Aarr[:, end]), rtol=1e-4)
# Is the initial spectrum restored after 2 soliton periods? (with preconditioner)
@test isapprox(abs2.(Aarrp[:, 1]), abs2.(Aarrp[:, end]), rtol=1e-3)
# Is the initial spectrum restored after 2 soliton periods?
# (with preconditioner and z-dependent linear part)
@test isapprox(abs2.(Aarrpf[:, 1]), abs2.(Aarrpf[:, end]), rtol=1e-3)
# Is there a difference if the linear part is a function (but constant)?
@test all(abs2.(Aarrp) .== abs2.(Aarrpf))

# --- step_on: force the stepper to land exactly on prescribed positions ------
# The dense-output interpolant is accurate only for the dominant components of
# the solution (it shares the error norm's blind spot), so saves of weak
# components must be exact step endpoints, not interpolations. `step_on` clips
# the adaptive step (never enlarges it) so every listed position is reached
# exactly. These tests assert that contract on the same soliton problem.

# Record every successful step endpoint via stepfun.
function landings(step_on; precon=false, kwargs...)
    reached = Float64[]
    recorder = (y, tn, dtn, interp) -> push!(reached, tn)
    if precon
        RK45.solve_precon(fnl!, Lin, copy(Aω), 0, 1e-3, zmax; rtol=1e-8,
                          stepfun=recorder, step_on=step_on, kwargs...)
    else
        RK45.solve(f!, copy(Aω), 0, 1e-3, zmax; rtol=1e-8,
                   stepfun=recorder, step_on=step_on, kwargs...)
    end
    reached
end

# Landing tolerance must match the "already reached" test inside RK45.solve
# (1e-12 * tmax): a landed position is t + (target - t), which is within a few
# ulps of target but not necessarily bitwise equal.
landtol = 1e-12*zmax
# Deliberately non-round interior positions.
targets = zmax .* [0.137, 0.319, 0.5, 0.681, 0.926]

# Plain stepper: every requested position is an exact step endpoint, and the
# propagation still runs to completion.
reached = landings(targets)
for zt in targets
    @test minimum(abs.(reached .- zt)) <= landtol
end
@test maximum(reached) >= zmax

# Preconditioned stepper (the path used by Luna.run): same contract.
reachedp = landings(targets; precon=true)
for zt in targets
    @test minimum(abs.(reachedp .- zt)) <= landtol
end
@test maximum(reachedp) >= zmax

# Input hygiene: an unsorted list containing duplicates, zero, negative values
# and positions beyond tmax must neither error nor hang; the valid interior
# positions still land, and out-of-range entries are ignored.
messy = zmax .* [0.681, -1.0, 0.0, 0.319, 0.319, 2.0]
reachedm = landings(messy)
for zt in zmax .* [0.319, 0.681]
    @test minimum(abs.(reachedm .- zt)) <= landtol
end
@test maximum(reachedm) >= zmax

# Positions closer together than the landing tolerance: the second is treated
# as already reached after landing on the first (no zero-size step, no hang).
cluster = [0.5*zmax, 0.5*zmax + 0.1*landtol]
reachedc = landings(cluster)
@test minimum(abs.(reachedc .- 0.5*zmax)) <= landtol

# step_on composes with max_dt: the clip only ever shrinks the step, so the
# max_dt cap is respected AND every position still lands. The gaps between
# successive successful endpoints are the actual step sizes taken.
reachedmax = landings(targets; max_dt=zmax/40)
for zt in targets
    @test minimum(abs.(reachedmax .- zt)) <= landtol
end
@test maximum(diff(reachedmax)) <= zmax/40*(1 + 1e-12)

# Accuracy is unaffected: forcing the landings changes the step sequence but
# not the solution. Compare the final spectrum against the free-running
# reference run from above (both rtol=1e-8; agreement far tighter than the
# soliton-restoration tolerance).
zso, Aso = RK45.solve(f!, copy(Aω), z, dz, zmax, rtol=1e-8, output=true,
                      outputN=501, step_on=targets)
@test isapprox(abs2.(Aso[:, end]), abs2.(Aarr[:, end]), rtol=1e-4)
# Interpolated output saves on the regular tout grid keep working alongside
# step_on (all columns filled, same grid as the reference run).
@test zso == zarr
@test all(isfinite, abs2.(Aso))

# The property step_on relies on: querying the interpolant AT the step
# endpoint returns the stepped solution yn, not a polynomial value. A landing
# is t + (target - t), which can sit ±1 ulp from the requested position, so
# bitwise equality with tn is not a sufficient endpoint test — interpolate()
# snaps queries within 4 ulps of tn to yn. Pin both behaviours: inside the
# window the mismatch is exactly zero, and just outside it the polynomial
# agrees with yn to round-off (its σ=1 weights are b5, exactly what locextrap
# propagates — before that was corrected the gap here was the full embedded
# error estimate, ~1e-2 relative).
function endpoint_consistency(; precon=false)
    inside = Ref(0.0)   # query 1 ulp inside the endpoint: snapped to yn
    outside = Ref(0.0)  # query just outside the snap window: polynomial path
    checker = function (y, tn, dtn, interp)
        yi = interp(prevfloat(tn))
        inside[] = max(inside[], maximum(abs.(yi .- y))/maximum(abs.(y)))
        yo = interp(tn - 10*eps(abs(tn)))
        outside[] = max(outside[], maximum(abs.(yo .- y))/maximum(abs.(y)))
    end
    if precon
        RK45.solve_precon(fnl!, Lin, copy(Aω), 0, 1e-3, zmax; rtol=1e-8,
                          stepfun=checker, step_on=targets)
    else
        RK45.solve(f!, copy(Aω), 0, 1e-3, zmax; rtol=1e-8,
                   stepfun=checker, step_on=targets)
    end
    (inside[], outside[])
end
in_plain, out_plain = endpoint_consistency()
@test in_plain == 0.0
@test out_plain < 1e-9
in_pre, out_pre = endpoint_consistency(precon=true)
@test in_pre == 0.0
@test out_pre < 1e-9

# Document the coefficient identities behind the snap: the dense-output
# polynomial at σ=1 reproduces b5 — which is also the final Butcher row (the
# FSAL stage) and what locextrap propagates — and the error estimate is the
# embedded 4th-order vector minus it. If a future edit to dopri.jl changes any
# of this, the endpoint snap in interpolate() must be revisited.
@test vec(sum(RK45.interpC, dims=1)) ≈ collect(RK45.b5)
@test all(RK45.B[6] .≈ RK45.b5[1:6])
@test collect(RK45.errest) ≈ collect(RK45.b4) .- collect(RK45.b5)

# --- dense output (interpolant) accuracy ------------------------------------
# DOPRI5 is FSAL: k7 of a step is k1 of the next. That move must not happen
# until the completed step's dense output has been consumed, because the
# interpolant's k1 weight is nonzero (interpC column 1, b1(1) = 35/384). Doing
# it inside the step poisons every interpolated save by dt*b1(σ)*(k7 - k1),
# which is O(h²) -- the dense output drops from 4th to 2nd order while the
# stepped endpoints, and therefore almost every other test, stay untouched.
#
# Test problem: y' = i(a + |y|²)y on a single element. |y| is conserved, so
#     y(t) = y(t₀)*exp(i(a + |y(t₀)|²)(t - t₀))
# exactly -- and the same holds for the preconditioned split with
# linop = [i*a] and nonlinear part i|y|²y. Setting min_dt == max_dt == h pins
# every step to exactly h (steplims! clamps from both sides), so the
# convergence studies below are fixed-step and cannot be disturbed by the
# step-size controller.

dns_a = 3.0
dns_f!(o, y, t) = (o[1] = im*(dns_a + abs2(y[1]))*y[1]; o)
dns_fnl!(o, y, t) = (o[1] = im*abs2(y[1])*y[1]; o)
dns_lin = ComplexF64[im*dns_a]
dns_y0() = ComplexF64[1.0]
"Exact solution at `t` of the trajectory passing through `y` at `t0`."
dns_exact(y, t0, t) = y*exp(im*(dns_a + abs2(y))*(t - t0))

dns_stepper(h; precon=false, rtol=1e-2, locextrap=true) = precon ?
    RK45.PreconStepper(dns_fnl!, dns_lin, dns_y0(), 0.0, h;
                       rtol=rtol, min_dt=h, max_dt=h, locextrap=locextrap) :
    RK45.Stepper(dns_f!, dns_y0(), 0.0, h;
                 rtol=rtol, min_dt=h, max_dt=h, locextrap=locextrap)

"""
Run the model problem to `tmax` with every step pinned to exactly `h`, probing
the dense output of each completed step. Returns
    eloc:   max over steps of the dense-output error at the step midpoint,
            measured against the exact solution continued from that step's own
            starting value, so the trajectory's accumulated error cannot mask it
    eslope: max over steps of the relative error in the interpolant's slope at
            the start of the step, which must be f(y(tₙ)). This sees the FSAL
            corruption directly, without any convergence study
    eglob:  max over steps of the error of the stepped solution itself
    egap:   max over steps of the mismatch between the interpolant just inside
            the step endpoint and the stepped endpoint yn
    nsteps, dterr: step count, and max deviation of the realised step from `h`
"""
function dns_probe(h; precon=false, tmax=2.0, rtol=1e-2, locextrap=true)
    y0 = dns_y0()
    tprev = 0.0
    yprev = copy(y0)
    eloc = 0.0; eslope = 0.0; eglob = 0.0; egap = 0.0; nsteps = 0; dterr = 0.0
    probe = function (yn, tn, dtn, interp)
        dtk = tn - tprev  # the 3rd argument is the *next* step size, not this one
        dterr = max(dterr, abs(dtk - h))
        y0k = yprev[1]
        tm = tprev + dtk/2
        eloc = max(eloc, abs(interp(tm)[1] - dns_exact(y0k, tprev, tm)))
        δ = 1e-6*dtk
        slope = (interp(tprev + δ)[1] - y0k)/δ
        exslope = im*(dns_a + abs2(y0k))*y0k
        eslope = max(eslope, abs(slope - exslope)/abs(exslope))
        egap = max(egap, abs(interp(tn - 10*eps(abs(tn)))[1] - yn[1])/abs(yn[1]))
        eglob = max(eglob, abs(yn[1] - dns_exact(y0[1], 0.0, tn)))
        nsteps += 1
        tprev = tn
        yprev .= yn  # yn is the stepper's live buffer, so copy it
    end
    Logging.with_logger(Logging.NullLogger()) do
        RK45.solve(dns_stepper(h; precon=precon, rtol=rtol, locextrap=locextrap),
                   tmax; stepfun=probe)
    end
    (eloc=eloc, eslope=eslope, eglob=eglob, egap=egap, nsteps=nsteps, dterr=dterr)
end

# A step must leave ks[1] alone: the FSAL move belongs to the *next* step, after
# the dense output of this one has been consumed. (For the PreconStepper the
# first step's rebase of ks[1] is prop!(., t, t), an exact identity, so the
# comparison is bitwise for both stepper types.)
for dns_precon in (false, true)
    s = dns_stepper(0.1; precon=dns_precon)
    dns_k1 = copy(s.ks[1])
    @test RK45.step!(s)
    @test s.ks[1] == dns_k1  # still pending...
    @test RK45.step!(s)
    @test s.ks[1] != dns_k1  # ...and performed by the next step
end

for dns_precon in (false, true)
    r1 = dns_probe(0.1; precon=dns_precon)
    r2 = dns_probe(0.025; precon=dns_precon)
    # the steps really were pinned to h, so the convergence study is meaningful
    @test r1.dterr < 1e-9
    @test r2.dterr < 1e-9
    @test r1.nsteps >= 20
    @test r2.nsteps >= 80
    # The interpolant must leave the step start along f(y(tₙ)). With the FSAL
    # move done too early this is wrong at the 40% (plain) / 2.5% (precon)
    # level; correct, it is 2e-5 / 2e-7 at h = 0.1.
    @test r1.eslope < 1e-3
    @test r2.eslope < 1e-5
    # Convergence order of the pure dense-output error: 4 or better. With the
    # FSAL move done too early it is exactly 2.
    @test log(r1.eloc/r2.eloc)/log(4) > 3.5
    @test r1.eloc < 1e-4  # buggy: 1.6e-2 (plain), 1.0e-3 (precon)
    # With min_dt == max_dt every step is accepted only because steplims!
    # forces ok = true; the FSAL move must be made on those steps too, and the
    # result must not depend on the tolerance that is being overridden.
    @test dns_probe(0.1; precon=dns_precon, rtol=1e-14) == r1
end

# --- which weight vector is which ------------------------------------------
# The two vectors of the Dormand-Prince RK5(4)7M pair, as exact rationals. They
# are told apart by the quadrature order conditions Σᵢbᵢcᵢᵖ = 1/(p+1): the
# 5th-order vector satisfies them through p = 4, the embedded 4th-order one only
# through p = 3. dopri.jl had the two labelled the wrong way round, so
# locextrap=true propagated the *lower*-order solution; these tests pin the
# correction down.
dns_c = Rational{BigInt}[0, 1//5, 3//10, 4//5, 8//9, 1, 1]
dns_v5 = Rational{BigInt}[35//384, 0, 500//1113, 125//192, -2187//6784, 11//84, 0]
dns_v4 = Rational{BigInt}[5179//57600, 0, 7571//16695, 393//640, -92097//339200,
                          187//2100, 1//40]
dns_quad(b, p) = sum(b .* dns_c.^p) == 1//(p + 1)
@test all(p -> dns_quad(dns_v5, p), 0:4)
@test all(p -> dns_quad(dns_v4, p), 0:3)
@test !dns_quad(dns_v4, 4)
@test RK45.b5 == Float64.(dns_v5)
@test RK45.b4 == Float64.(dns_v4)
# DOPRI5 is FSAL: the final Butcher row is the 5th-order solution, which is
# therefore where stage 7 is evaluated, and which the dense output reproduces
# at σ = 1.
@test RK45.b5[1:6] == RK45.B[6]
@test RK45.b5[7] == 0
@test vec(sum(RK45.interpC, dims=1)) ≈ RK45.b5
# The error estimate is untouched by the relabelling: it is still the standard
# DOPRI5 E vector, (4th order) - (5th order).
@test RK45.errest ≈ [-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40]

# The stepped solution converges at 5th order with local extrapolation (the
# default) and at 4th without it; and with local extrapolation the interpolant
# is a continuous extension of the solution actually propagated, so at the step
# endpoint it reproduces yn to round-off. Without it the two differ by the full
# embedded error estimate -- which is why interpolate() cannot simply be trusted
# at σ ≈ 1.
for dns_precon in (false, true)
    r1 = dns_probe(0.1; precon=dns_precon)
    r2 = dns_probe(0.025; precon=dns_precon)
    n1 = dns_probe(0.1; precon=dns_precon, locextrap=false)
    n2 = dns_probe(0.025; precon=dns_precon, locextrap=false)
    @test log(r1.eglob/r2.eglob)/log(4) > 4.6
    @test 3.5 < log(n1.eglob/n2.eglob)/log(4) < 4.6
    @test r1.egap < 1e-12
    @test n1.egap > 1e-9
end

# FSAL is exact: k7 is the RHS evaluated at the solution that is propagated, so
# after a step ks[1] is the RHS at that step's own endpoint.
dns_s = dns_stepper(0.1)
RK45.step!(dns_s)
RK45.step!(dns_s)  # the second step performs the deferred FSAL move
dns_fk = similar(dns_s.y)
dns_f!(dns_fk, dns_s.y, dns_s.t)
@test dns_s.ks[1] ≈ dns_fk rtol=1e-13

