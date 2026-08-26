import Test: @test, @testset, @test_throws
import Luna
import Luna: Utils, Output, Grid, LinearOps, NonlinearRHS, PhysData, Nonlinear, Fields,
             Raman, RK45
import LinearAlgebra
import Random

# =============================================================================
# Hardware-gated CUDA tests.
#
# Run with LUNA_TEST_CUDA=1 on a machine with a working GPU; skipped everywhere else,
# so this never runs in normal CI or on the macOS dev machines.
#
#     LUNA_TEST_CUDA=1 julia --project test/runtests.jl test_cuda
#
# test_device.jl already validates the device LOGIC using JLArrays (a CPU-backed
# AbstractGPUArray). What it cannot check, and what this file exists for:
#
#  * CUDA code generation — JLArrays interprets kernels on the host, so boxed closures,
#    dynamic dispatch and non-isbits captures pass there and fail here;
#  * cuFFT semantics: input preservation for out-of-place c2c, the ScaledPlan route
#    taken by the inverse, in-place batched plans, and plan workspace memory;
#  * that the lazy operator structs really are rejected by a device broadcast (JLArrays
#    tolerates what CUDA rejects — see test_device.jl);
#  * the resident memory footprint, which decides which campaigns fit which card.
# =============================================================================

if get(ENV, "LUNA_TEST_CUDA", "") == "1"
    import CUDA

    if !CUDA.functional()
        @warn "LUNA_TEST_CUDA=1 but CUDA is not functional; skipping"
    else
        @testset "CUDA" begin
            CuArray = CUDA.CuArray
            # Any scalar indexing of a device array is a bug in the device path, and on
            # real hardware it is also catastrophically slow. Make it an error.
            CUDA.allowscalar(false)

            @testset "environment" begin
                @test Utils.backend(CuArray(zeros(4))) isa Utils.DeviceBackend
                @test Luna.resolve_arraytype(:cuda) === CuArray
                # The extension must have installed its hooks. These are function-valued
                # rather than methods precisely so the extension can precompile: an
                # extension cannot overwrite an identically-signatured stub, and the
                # resulting precompilation failure is easy to miss because the extension
                # still loads (just recompiled in every process — once per delay point
                # under `instances`).
                @test Luna._DEVICE_RECLAIM[] !== nothing
                @test Luna._DEVICE_MEMORY_STATUS[] !== nothing
                @test Luna._DEVICE_SELECT[] !== nothing
                @test Luna.device_memory_status() !== nothing
                free, total = Luna.device_memory_status()
                @test 0 < free <= total
                @test Luna.device_reclaim() === nothing   # runs, returns nothing
                @info "CUDA device" name=CUDA.name(CUDA.device()) free_GiB=free/2^30 total_GiB=total/2^30
            end

            @testset "cuFFT semantics" begin
                x = CuArray(randn(ComplexF64, 16, 8, 8))
                xc = copy(x)
                p = Utils.plan_fft_backend(x, (1, 2, 3))
                y = similar(x)
                LinearAlgebra.mul!(y, p, x)
                # Out-of-place c2c must preserve its input: the TransFree fast path
                # transforms the solver's state array directly and relies on this.
                @test Array(x) == Array(xc)
                # The explicitly stored inverse plan must round-trip
                ip = inv(p)
                z = similar(x)
                LinearAlgebra.mul!(z, ip, y)
                @test isapprox(Array(z), Array(xc); rtol=1e-12)
                # inv should be memoised rather than re-planned per call
                @test inv(p) === ip

                # r2c/c2r, which the field-resolved (RealGrid) path uses. The forward
                # transform must preserve its real input (`Luna.run` windows in a scratch
                # buffer and transforms it back), and the INVERSE is the one that does
                # not — which is exactly why a RealGrid can never take the TransFree fast
                # path, where the input would be the solver's state. Assert the
                # round-trip rather than the destruction: destroying the input is
                # permitted, not promised.
                xr = CuArray(randn(16, 8, 8))
                xrc = copy(xr)
                pr = Utils.plan_rfft_backend(xr, (1, 2, 3))
                yr = similar(xr, ComplexF64, size(pr * xr))
                LinearAlgebra.mul!(yr, pr, xr)
                @test Array(xr) == Array(xrc)
                ipr = inv(pr)
                zr = similar(xr)
                LinearAlgebra.mul!(zr, ipr, yr)
                @test isapprox(Array(zr), Array(xrc); rtol=1e-12)
            end

            @testset "factor broadcast vs adapted-struct broadcast" begin
                # The device propagator broadcasts the operator's separable FACTORS. An
                # adapted operator struct ALSO works inside a CUDA broadcast, because
                # CUDA adapts non-native AbstractArray operands and the struct carries
                # an Adapt rule — so check the two agree. Luna uses the factor form so
                # as not to depend on that behaviour (it is backend-specific), and to
                # hoist the reference-phase test out of the per-element work.
                grid = Grid.EnvGrid(5e-3, 800e-9, (400e-9, 2000e-9), 100e-15)
                xygrid = Grid.FreeGrid(400e-6, 8)
                nfun = PhysData.ref_index_fun(:Ar, 4)
                l = LinearOps.make_const_linop(grid, xygrid, nfun;
                                               factored=true, arraytype=CuArray)
                y0 = randn(ComplexF64, size(l))
                dz = 3.7e-5

                yk = CuArray(copy(y0))                        # factor kernel
                RK45.make_prop!(l, yk)(yk, 0.0, dz)
                ys = CuArray(copy(y0))                        # adapted struct
                ys .*= exp.(l .* dz)
                @test isapprox(Array(yk), Array(ys); rtol=1e-12)

                # ...and both agree with the host
                lh = LinearOps.make_const_linop(grid, xygrid, nfun; factored=true)
                yh = copy(y0)
                RK45.make_prop!(lh, yh)(yh, 0.0, dz)
                @test isapprox(Array(yk), yh; rtol=1e-12)
            end

            @testset "end-to-end propagation vs host" begin
                function propagate(arraytype; raman=false, N=16, saveN=5)
                    λ0 = 800e-9
                    L = 5e-3
                    grid = Grid.EnvGrid(L, λ0, (400e-9, 2000e-9), 100e-15)
                    xygrid = Grid.FreeGrid(400e-6, N)
                    gas = raman ? :N2 : :Ar
                    densityfun = let d = PhysData.density(gas, 4); z -> d end
                    nfun = PhysData.ref_index_fun(gas, 4)
                    responses = if raman
                        rr = Raman.raman_response(grid.to, gas)
                        (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),
                         Nonlinear.RamanPolarEnvBatched(grid.to, rr))
                    else
                        (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)
                    end
                    linop = LinearOps.make_const_linop(grid, xygrid, nfun;
                                                       factored=true, arraytype)
                    normfun = NonlinearRHS.const_norm_free(grid, xygrid, nfun;
                                                           factored=true, arraytype)
                    inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=30e-15, energy=30e-6,
                                                    w0=100e-6)
                    Eω, transform, FT = Luna.setup(grid, xygrid, densityfun, normfun,
                                                   responses, inputs; arraytype)
                    output = Output.MemoryOutput(0, grid.zmax, saveN)
                    Luna.run(Eω, grid, linop, transform, FT, output;
                             max_dz=Inf, init_dz=L/50, rtol=1e-8,
                             step_on=collect(range(0, grid.zmax, saveN)))
                    output.data["Eω"]
                end

                for raman in (false, true)
                    Eh = propagate(Array; raman)
                    Ed = propagate(CuArray; raman)
                    @test Ed isa Array          # saves come back on the host
                    rel = sqrt(sum(abs2, Ed .- Eh) / sum(abs2, Eh))
                    @info "CUDA vs host" raman relL2=rel
                    @test rel < 1e-8
                end
            end

            @testset "RealGrid free-space end to end" begin
                # The field-resolved 3-D path, which always takes TransFree's GENERAL
                # route. JLArrays cannot catch what this can: its c2r shim transforms
                # out of place and so silently tolerates an aliasing mistake that cuFFT
                # would not, and it interprets kernels on the host rather than compiling
                # them.
                function propagate_real(arraytype; nothg=true, N=16, saveN=3)
                    λ0, L, gas, pres = 800e-9, 5e-3, :Ar, 4
                    grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 100e-15)
                    xygrid = Grid.FreeGrid(400e-6, N)
                    densityfun = let d = PhysData.density(gas, pres); z -> d end
                    nfun = PhysData.ref_index_fun(gas, pres)
                    γ3 = PhysData.γ3_gas(gas)
                    # a fresh response per transform: a batched response owns buffers on
                    # the array type of the field it first saw
                    responses = nothg ? (Nonlinear.Kerr_field_nothg(γ3),) :
                                        (Nonlinear.Kerr_field(γ3),)
                    linop = LinearOps.make_const_linop(grid, xygrid, nfun;
                                                       factored=true, arraytype)
                    normfun = NonlinearRHS.const_norm_free(grid, xygrid, nfun;
                                                           factored=true, arraytype)
                    inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=30e-15, energy=30e-6,
                                                    w0=100e-6)
                    Eω, transform, FT = Luna.setup(grid, xygrid, densityfun, normfun,
                                                   responses, inputs; arraytype)
                    zs = collect(range(0, grid.zmax, saveN))
                    output = Output.MemoryOutput(Output.GridCondition(zs, saveN), "Eω", "z")
                    Luna.run(Eω, grid, linop, transform, FT, output;
                             max_dz=Inf, init_dz=L/50, rtol=1e-8, step_on=zs)
                    output.data["Eω"]
                end
                for nothg in (true, false)      # batched response / pointwise response
                    Eh = propagate_real(Array; nothg)
                    Ed = propagate_real(CuArray; nothg)
                    @test Ed isa Array
                    rel = sqrt(sum(abs2, Ed .- Eh) / sum(abs2, Eh))
                    @info "CUDA vs host (RealGrid free space)" nothg relL2=rel
                    @test rel < 1e-8
                end
            end

            @testset "memory footprint" begin
                # Measure what a propagation actually holds on the device, to validate
                # the campaign budget (resident field count) and — the one quantity the
                # plan could not predict — the cuFFT plan workspace.
                λ0 = 800e-9
                grid = Grid.EnvGrid(5e-3, λ0, (400e-9, 2000e-9), 100e-15)
                N = 128
                xygrid = Grid.FreeGrid(400e-6, N)
                field_bytes = 16 * length(grid.ω) * N * N
                nfun = PhysData.ref_index_fun(:Ar, 4)
                densityfun = let d = PhysData.density(:Ar, 4); z -> d end
                responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(:Ar)),)

                CUDA.reclaim()
                free0, _ = Luna.device_memory_status()
                linop = LinearOps.make_const_linop(grid, xygrid, nfun;
                                                   factored=true, arraytype=CuArray)
                normfun = NonlinearRHS.const_norm_free(grid, xygrid, nfun;
                                                       factored=true, arraytype=CuArray)
                inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=30e-15, energy=30e-6,
                                                w0=100e-6)
                Eω, transform, FT = Luna.setup(grid, xygrid, densityfun, normfun,
                                               responses, inputs; arraytype=CuArray)
                free1, _ = Luna.device_memory_status()
                setup_fields = (free0 - free1) / field_bytes
                @info "device memory after setup" field_MiB=field_bytes/2^20 setup_fields
                # state + transform buffer + plan workspace; the factored operators are
                # only a vector and a small matrix
                @test setup_fields < 6

                output = Output.MemoryOutput(0, grid.zmax, 3)
                Luna.run(Eω, grid, linop, transform, FT, output;
                         max_dz=Inf, init_dz=5e-4, rtol=1e-8,
                         step_on=collect(range(0, grid.zmax, 3)))
                free2, _ = Luna.device_memory_status()
                run_fields = (free0 - free2) / field_bytes
                @info "device memory after run" run_fields
                # 9 solver registers + 1 transform buffer + workspace. The budget in the
                # plan assumes <= 1 field of cuFFT workspace; flag if that is wrong.
                @test run_fields < 14
            end

            @testset "native scan" begin
                # The extension replaces the portable doubling scan by CUDA's cumsum! and
                # drops the scratch buffers. Same result as the sequential trapezoid rule
                # to rounding, for real and complex, 2-D and 3-D, and for reshaped views.
                import Luna: Maths
                @test Maths.scan_scratch(CUDA.zeros(4, 3)) === nothing
                relerr(x, y) = sqrt(sum(abs2, x .- y)/sum(abs2, y))
                δt = 1e-17
                for T in (Float64, ComplexF64), sz in ((1000, 3), (1000, 2, 3))
                    y = randn(T, sz...)
                    ref = similar(y); Maths.cumtrapz!(ref, y, δt)
                    yd = CuArray(y); outd = similar(yd)
                    Maths.cumtrapz_scan!(outd, yd, δt, Maths.scan_scratch(yd))
                    @test relerr(Array(outd), ref) < 1e-13
                    @test all(iszero, Array(selectdim(outd, 1, 1)))
                    # the plasma response scans (nt, npts) buffers reshaped from (nt, 1, npts)
                    if length(sz) == 2
                        y3 = reshape(yd, sz[1], 1, sz[2]); out3 = reshape(similar(yd), sz[1], 1, sz[2])
                        Maths.cumtrapz_scan!(out3, y3, δt, nothing)
                        @test relerr(Array(reshape(out3, sz)), ref) < 1e-13
                    end
                end
            end

            @testset "RK45 error norm host vs device (cancellation)" begin
                # Ported from ModelPNPS/examples/check_device_norm.jl. The step-size
                # controller sets dz from this norm, so a device norm that is
                # systematically smaller than the host one would run at an effectively
                # looser rtol (step count ∝ ratio^(1/5)). test_device.jl checks
                # `weaknorm_fused` host vs device only under JLArrays; here it is the real
                # CUDA tree reduction with FMA contraction versus the sequential host loop.
                # The test must reproduce the cancellation in the error estimate:
                # Dormand–Prince's `errest` weights sum to zero, so with identical stages
                # yerr vanishes and the estimate is built from the differences BETWEEN
                # stages. Stages are therefore a common base field (with the ~10-decade
                # dynamic range a real state has) plus a relative perturbation `delta`,
                # swept towards zero: a backend disagreement driven by cancellation grows
                # as delta shrinks; rounding-level agreement stays at ~eps/delta.
                mutable struct FakeStepperCUDA{T, N}
                    y::T
                    yn::T
                    ks::NTuple{7, T}
                    yerr::Union{Nothing, T}
                    dt::Float64
                    rtol::Float64
                    atol::Float64
                    norm::N
                end
                rng = Random.Xoshiro(20260815)
                DT, RTOL, ATOL = 1.7e-4, 1e-7, 1e-10
                for sz in ((64, 64, 64), (4097, 6)) # 3-D free-space and modal (nω × nmodes) shapes
                    dyn = length(sz) == 3 ? reshape(10.0 .^ range(0, -10; length=sz[2]), 1, :, 1) :
                                            reshape(10.0 .^ range(0, -10; length=sz[2]), 1, :)
                    base = randn(rng, ComplexF64, sz) .* dyn
                    for delta in (1e-2, 1e-4, 1e-6, 1e-8)
                        ks = ntuple(_ -> base .* (1 .+ delta .* randn(rng, ComplexF64, sz)), 7)
                        yn = base .* (1 .+ delta .* randn(rng, ComplexF64, sz))
                        host = FakeStepperCUDA(base, yn, ks, nothing, DT, RTOL, ATOL, RK45.weaknorm)
                        dev = FakeStepperCUDA(CuArray(base), CuArray(yn), map(CuArray, ks),
                                              nothing, DT, RTOL, ATOL, RK45.weaknorm)
                        eh = RK45.weaknorm_fused(host)
                        ed = RK45.weaknorm_fused(dev)
                        # materialised reference (what a custom, non-fused norm receives)
                        yerr = @. DT*(ks[1]*RK45.errest[1] + ks[3]*RK45.errest[3] +
                                      ks[4]*RK45.errest[4] + ks[5]*RK45.errest[5] +
                                      ks[6]*RK45.errest[6] + ks[7]*RK45.errest[7])
                        em = RK45.weaknorm(yerr, base, yn, RTOL, ATOL)
                        reldiff = abs(ed - eh)/abs(eh)
                        @info "device norm" sz delta ratio=ed/eh reldiff hostmaterialised=em/eh
                        # rounding: eps per operation, amplified by the cancellation ~1/delta
                        rtol = max(1e-12, 1e3*eps(Float64)/delta)
                        @test isapprox(ed, eh; rtol)
                        @test isapprox(em, eh; rtol)
                        @test dev.yerr === nothing # the fused device path never materialises yerr
                    end
                end
                GC.gc(); Luna.device_reclaim()
            end

            @testset "TransModalFixed on CUDA" begin
                # The modal path on hardware: real-to-complex batched cuFFT plans, CUBLAS
                # GEMMs, and — the parts JLArrays cannot prove — the cached-PPT spline and
                # ADK rate compiled into device kernels, and the native prefix sum.
                import Luna: Capillary, Ionisation, Stats, Interface
                λ0 = 800e-9
                a = 100e-6
                grid = Grid.RealGrid(4e-3, λ0, (200e-9, 3000e-9), 200e-15)
                modes = [Capillary.MarcatiliMode(a, :Ar, 3.0; n=1, m=m) for m in 1:3]
                densityfun = let d = PhysData.density(:Ar, 3.0); z -> d end
                Ip = PhysData.ionisation_potential(:Ar)
                inputs = ((mode=1, fields=(Fields.GaussField(λ0=λ0, τfwhm=15e-15, energy=100e-6),)),
                          (mode=2, fields=(Fields.GaussField(λ0=λ0, τfwhm=15e-15, energy=10e-6),)))
                relerr(x, y) = sqrt(sum(abs2, x .- y)/sum(abs2, y))
                function make(arraytype, ionrate)
                    resp = (Nonlinear.Kerr_field(PhysData.γ3_gas(:Ar)),
                            Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, Ip))
                    Luna.setup(grid, densityfun, resp, inputs, modes, :y;
                               modal_integral=:fixed, nr=33, kronrod=true, arraytype)
                end
                for ionrate in (Ionisation.IonRateADK(:Ar),
                                Ionisation.IonRatePPTAccel(:Ar, λ0; N=2^14, cache=false))
                    Eωh, th, FTh = make(Array, ionrate)
                    Eωd, td, FTd = make(CuArray, ionrate)
                    @test Eωd isa CuArray && td.Et isa CuArray
                    nlh = similar(Eωh); th(nlh, Eωh, 0.0)
                    nld = similar(Eωd); td(nld, Eωd, 0.0)
                    rel = relerr(Array(nld), nlh)
                    @info "CUDA modal transform vs host" ionrate=typeof(ionrate).name.name rel
                    @test rel < 1e-10
                    errh = Array(NonlinearRHS.integral_error!(td))
                    @test all(isfinite, errh)
                end
                # real-field Raman (H2), batched on the device: r2c/c2r batched cuFFT plans
                # on the doubled convolution grid, and the analytic-signal path (thg=false)
                import Luna: Raman
                for thg in (true, false)
                    function make_raman(arraytype)
                        rr = Raman.raman_response(grid.to, :H2)
                        resp = (Nonlinear.Kerr_field(PhysData.γ3_gas(:H2)),
                                Nonlinear.RamanPolarField(grid.to, rr; thg),
                                Nonlinear.PlasmaCumtrapz(grid.to, grid.to,
                                                         Ionisation.IonRateADK(:H2),
                                                         PhysData.ionisation_potential(:H2)))
                        Luna.setup(grid, densityfun, resp, inputs, modes, :y;
                                   modal_integral=:fixed, nr=17, arraytype)
                    end
                    Eωh, th, _ = make_raman(Array)
                    Eωd, td, _ = make_raman(CuArray)
                    @test td.resp_eval[2] isa Nonlinear.RamanPolarFieldBatched
                    nlh = similar(Eωh); th(nlh, Eωh, 0.0)
                    nld = similar(Eωd); td(nld, Eωd, 0.0)
                    rel = relerr(Array(nld), nlh)
                    @info "CUDA modal transform with Raman vs host" thg rel
                    @test rel < 1e-10
                end
                # end-to-end with the pressure gradient (z-dependent operator evaluated on
                # the host and uploaded) through the simple interface, versus the host
                oh = Interface.prop_capillary(a, 5e-3, :Ar, (3.0, 1.0); λ0, τfwhm=15e-15,
                        energy=100e-6, modes=3, trange=200e-15, λlims=(200e-9, 3000e-9),
                        shotnoise=false, saveN=3, plasma=:ADK, modal_integral=:fixed, nr=33)
                od = Interface.prop_capillary(a, 5e-3, :Ar, (3.0, 1.0); λ0, τfwhm=15e-15,
                        energy=100e-6, modes=3, trange=200e-15, λlims=(200e-9, 3000e-9),
                        shotnoise=false, saveN=3, plasma=:ADK, modal_integral=:fixed, nr=33,
                        arraytype=:cuda)
                rel = relerr(od["Eω"], oh["Eω"])
                @info "CUDA modal propagation vs host" rel
                @test rel < 1e-8
                @test isapprox(od["stats"]["energy"], oh["stats"]["energy"]; rtol=1e-6)
                @test isapprox(od["stats"]["electrondensity"], oh["stats"]["electrondensity"];
                               rtol=1e-4)
            end

            @testset "lazy :cuda inside one top-level call (world age)" begin
                # The scan-script use case: a login node submits without loading CUDA, and
                # on the compute node the whole propagation happens inside ONE call of the
                # scan closure. `prop_capillary(...; arraytype=:cuda)` then loads CUDA
                # *during* the call, whose methods would be too new for that frame; the
                # interface re-enters through invokelatest. This must run in a fresh
                # process where CUDA is not yet loaded (this test file imports it), so
                # spawn one with the same project and no `import CUDA`.
                script = """
                    import Luna
                    import Luna: Interface
                    function run_inside_one_call()
                        o = Interface.prop_capillary(100e-6, 2e-3, :Ar, 1.0; λ0=800e-9,
                                τfwhm=15e-15, energy=50e-6, modes=2, trange=200e-15,
                                λlims=(200e-9, 3000e-9), shotnoise=false, saveN=3,
                                plasma=:ADK, nr=17, arraytype=:cuda)
                        # ...and the same through the args form + Luna.run: the tuple comes
                        # back into this (old-world) frame, so `run` must be re-entered too
                        args = Interface.prop_capillary_args(100e-6, 2e-3, :Ar, 1.0; λ0=800e-9,
                                τfwhm=15e-15, energy=50e-6, modes=2, trange=200e-15,
                                λlims=(200e-9, 3000e-9), shotnoise=false, saveN=3,
                                plasma=:ADK, nr=17, arraytype=:cuda)
                        Base.invokelatest(Luna.run, args...; status_period=30,
                                          allow_device_stats=true)
                        o["Eω"] isa Array && all(isfinite, o["Eω"])
                    end
                    ok = run_inside_one_call()
                    println("LAZY_CUDA_OK=", ok)
                    exit(ok ? 0 : 1)
                    """
                cmd = `$(Base.julia_cmd()) --project=$(dirname(Base.active_project())) -e $script`
                out = IOBuffer()
                proc = run(pipeline(ignorestatus(cmd); stdout=out, stderr=out))
                txt = String(take!(out))
                success(proc) || @info "lazy :cuda subprocess output" txt
                @test success(proc)
                @test occursin("LAZY_CUDA_OK=true", txt)
            end
        end
    end
else
    @info "CUDA tests skipped (set LUNA_TEST_CUDA=1 on a GPU machine to run them)"
end
