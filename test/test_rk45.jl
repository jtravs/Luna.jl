import FFTW
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
# snaps queries within 4 ulps of tn to yn. This matters because the
# interpolation polynomial at σ = 1 does NOT reproduce yn: its σ=1 weights
# equal the b4-labelled (FSAL-row) vector rather than the b5-labelled weights
# used by locextrap propagation, so just outside the snap window the
# interpolant differs from yn by the full embedded error estimate (which is
# large relative to weak solution components). Pin both behaviours: inside
# the window the mismatch is exactly zero; outside it is finite.
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
@test out_plain > 0.0
in_pre, out_pre = endpoint_consistency(precon=true)
@test in_pre == 0.0
@test out_pre > 0.0

# Document the coefficient identities behind the snap: the dense-output
# polynomial at σ=1 reproduces the b4-labelled weights — which are also the
# final Butcher row (the FSAL stage) — and differs from the b5-labelled
# locextrap weights by exactly the error-estimate vector. If a future edit to
# dopri.jl changes any of this, the endpoint snap in interpolate() must be
# revisited.
@test vec(sum(RK45.interpC, dims=1)) ≈ collect(RK45.b4)
@test all(RK45.B[6] .≈ RK45.b4[1:6])
@test collect(RK45.errest) ≈ collect(RK45.b5) .- collect(RK45.b4)

