import Test: @test, @testset, @test_throws
import Luna
import Luna: Utils, Output, Grid, LinearOps, NonlinearRHS, PhysData, Nonlinear, Fields,
             Raman, RK45
import LinearAlgebra

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

            @testset "TransModalFixed on CUDA" begin
                # The modal path on hardware: real-to-complex batched cuFFT plans, CUBLAS
                # GEMMs, and — the parts JLArrays cannot prove — the cached-PPT spline and
                # ADK rate compiled into device kernels, and the doubling scan.
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
        end
    end
else
    @info "CUDA tests skipped (set LUNA_TEST_CUDA=1 on a GPU machine to run them)"
end
