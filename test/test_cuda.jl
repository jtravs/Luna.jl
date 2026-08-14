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
                # the extension must have replaced the no-op stubs
                @test Luna.device_memory_status() !== nothing
                free, total = Luna.device_memory_status()
                @test 0 < free <= total
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

            @testset "lazy operators are rejected by a device broadcast" begin
                # The reason the device kernels broadcast the separable FACTORS rather
                # than the operator struct. JLArrays does not reproduce this rejection.
                grid = Grid.EnvGrid(5e-3, 800e-9, (400e-9, 2000e-9), 100e-15)
                xygrid = Grid.FreeGrid(400e-6, 8)
                nfun = PhysData.ref_index_fun(:Ar, 4)
                l = LinearOps.make_const_linop(grid, xygrid, nfun;
                                               factored=true, arraytype=CuArray)
                y = CuArray(randn(ComplexF64, size(l)))
                @test_throws Exception (y .*= l)
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
        end
    end
else
    @info "CUDA tests skipped (set LUNA_TEST_CUDA=1 on a GPU machine to run them)"
end
