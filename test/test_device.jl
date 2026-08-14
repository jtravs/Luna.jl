import Test: @test, @testset, @test_throws, @inferred
import Luna
import Luna: Utils, Output
import GPUArraysCore
import FFTW
import AbstractFFTs
import Adapt
import Random

# =============================================================================
# Tests for Luna's device (GPU) support.
#
# Luna's device code paths are ordinary broadcasts and `mapreduce`s over arrays that
# follow the GPUArrays interface, so they can be exercised WITHOUT GPU hardware:
#
#  * the `Utils.backend` trait is purely type-level, so a dummy `AbstractGPUArray`
#    subtype is enough to test dispatch — these tests always run;
#  * `JLArrays.JLArray` is a CPU-backed `AbstractGPUArray` which runs the same generic
#    GPUArrays code path CUDA does and enforces the same no-scalar-indexing contract,
#    so it can execute the real device kernels. It is a test-only dependency: these
#    tests are skipped if it is not loadable (e.g. when running
#    `julia --project test/runtests.jl` directly rather than via `Pkg.test`).
#
# What this file CANNOT catch, and what hardware tests must (see test_cuda.jl):
# JLArrays interprets kernels on the host, so CUDA code-generation limits (boxed
# closures, dynamic dispatch, non-isbits captures) pass here and could still fail on a
# real GPU. Luna's device code therefore sticks to broadcasts and reductions over
# concrete isbits types, and this file additionally checks type stability where that
# matters.
# =============================================================================

# A minimal device-array type: enough to test the `backend` trait's dispatch, which
# never touches the data.
struct DummyGPUArray{T, N} <: GPUArraysCore.AbstractGPUArray{T, N}
    a::Array{T, N}
end
Base.size(x::DummyGPUArray) = size(x.a)

@testset "Device support" begin

@testset "backend trait" begin
    host = zeros(ComplexF64, 4, 3, 2)
    dev = DummyGPUArray(host)

    @test Utils.backend(host) isa Utils.CPUBackend
    @test Utils.backend(dev) isa Utils.DeviceBackend
    @test !Utils.isdevice(host)
    @test Utils.isdevice(dev)

    # Types dispatch identically to instances
    @test Utils.backend(typeof(host)) isa Utils.CPUBackend
    @test Utils.backend(typeof(dev)) isa Utils.DeviceBackend

    # Wrappers report their parent's backend. This is load-bearing: `tchunks` hands
    # views to its callback, so a wrapper misreporting as CPU would silently serialise
    # a device kernel (or worse, index it scalar-wise).
    @test Utils.backend(view(host, :, 1, 1)) isa Utils.CPUBackend
    @test Utils.backend(view(dev, :, 1, 1)) isa Utils.DeviceBackend
    @test Utils.backend(reshape(host, 24)) isa Utils.CPUBackend
    @test Utils.backend(reshape(dev, 24)) isa Utils.DeviceBackend
    @test Utils.backend(PermutedDimsArray(host, (3, 2, 1))) isa Utils.CPUBackend
    @test Utils.backend(PermutedDimsArray(dev, (3, 2, 1))) isa Utils.DeviceBackend

    # Unknown array types are treated as host: a host wrapper misread as CPU is a
    # no-op, whereas the reverse would break it.
    @test Utils.backend(1:10) isa Utils.CPUBackend
    @test Utils.backend(reshape(1:24, 4, 6)) isa Utils.CPUBackend

    # The trait resolves at compile time, so it costs nothing in the hot loop
    @test @inferred(Utils.backend(host)) isa Utils.CPUBackend
    @test @inferred(Utils.backend(dev)) isa Utils.DeviceBackend
end

@testset "tchunks dispatch" begin
    # On the host, `tchunks` may split into per-thread chunks; on a device it must call
    # the closure exactly once with the whole arrays (the broadcast is the kernel).
    dest = zeros(8)
    src = collect(1.0:8.0)
    ncalls = Ref(0)
    Utils.tchunks(dest, src) do d, s
        ncalls[] += 1
        @. d = 2s
    end
    @test dest == 2 .* src
    @test ncalls[] >= 1

    devdest = DummyGPUArray(zeros(8))
    devsrc = DummyGPUArray(collect(1.0:8.0))
    devcalls = Ref(0)
    got = Ref{Any}(nothing)
    Utils.tchunks(devdest, devsrc) do d, s
        devcalls[] += 1
        got[] = (d, s)
    end
    @test devcalls[] == 1                    # exactly one kernel launch
    @test got[][1] === devdest               # whole arrays, not views
    @test got[][2] === devsrc
end

@testset "FFT planner dispatch" begin
    # The host planner takes Luna's FFTW flags; the device planner must not (device FFT
    # libraries reject them), so these route through different code.
    x = zeros(ComplexF64, 8)
    v = rand(ComplexF64, 8)
    p = Utils.plan_fft_backend(x, 1)
    @test p isa AbstractFFTs.Plan
    @test p * v ≈ FFTW.fft(v)
    @test Utils.plan_fft!_backend(copy(x), 1) isa AbstractFFTs.Plan
    @test Utils.plan_ifft!_backend(copy(x), 1) isa AbstractFFTs.Plan
    @test Utils.plan_rfft_backend(zeros(8), 1) isa AbstractFFTs.Plan

    # Wisdom is FFTW-only (and does filesystem locking): device planning must skip it
    @test Utils.loadFFTwisdom(Utils.DeviceBackend()) === nothing
    @test Utils.saveFFTwisdom(Utils.DeviceBackend()) === nothing
end

@testset "array type resolution" begin
    @test Luna.resolve_arraytype(Array) === Array
    @test Luna.resolve_arraytype(nothing) === Array
    @test Luna.resolve_arraytype(:cpu) === Array
    @test_throws ArgumentError Luna.resolve_arraytype(:nonsense)
    # `:cuda` must fail with a clear error rather than a load-time crash when CUDA is
    # unavailable (as on the macOS dev machines and on GPU-less login nodes).
    if !haskey(ENV, "LUNA_TEST_CUDA")
        @test_throws Exception Luna.resolve_arraytype(:cuda)
    end
end

@testset "device_zeros" begin
    z = Luna.device_zeros(Array, ComplexF64, (3, 2))
    @test z isa Array{ComplexF64, 2}
    @test size(z) == (3, 2)
    @test all(iszero, z)
    @test isequal(z, zeros(ComplexF64, 3, 2))
end

@testset "GridVectors" begin
    grid = Luna.Grid.EnvGrid(1e-3, 800e-9, (400e-9, 2000e-9), 100e-15)
    gv = Luna.gridvectors(grid)
    # On the host these alias the grid's own vectors: no copy, no extra memory, and the
    # CPU path is therefore untouched.
    @test gv.ω === grid.ω
    @test gv.ωwin === grid.ωwin
    @test gv.twin === grid.twin
    @test gv.towin === grid.towin
    @test Luna.gridvectors(grid, Array).ω === grid.ω
end

@testset "HostOutput" begin
    grid = Luna.Grid.EnvGrid(1e-3, 800e-9, (400e-9, 2000e-9), 100e-15)
    y = zeros(ComplexF64, length(grid.ω), 2, 2)

    # By default only the interpolated (saved) value is copied to the host: copying `y`
    # as well would cost a full transfer on every step rather than every save.
    mem = Output.MemoryOutput(0, 1e-3, 3)
    h = Luna.HostOutput(mem, y)
    @test h.ybuf === nothing
    @test Luna.needs_host_y(mem) == false
    @test size(h.ibuf) == size(y)
    @test h.ibuf isa Array

    # Metadata calls and indexing pass through to the wrapped handler
    h(Dict("a" => 1); group="meta")
    @test haskey(h, "meta")
    @test h["meta"]["a"] == 1

    # The saving contract still works through the wrapper
    h(y, 0.0, 1e-4, t -> y)
    @test haskey(h, "Eω")
end

@testset "arraytype plumbing (host default)" begin
    import Luna: Grid, LinearOps, NonlinearRHS, PhysData, Nonlinear, Fields
    grid = Grid.EnvGrid(5e-3, 800e-9, (400e-9, 2000e-9), 100e-15)
    xygrid = Grid.FreeGrid(400e-6, 16)
    nfun = PhysData.ref_index_fun(:Ar, 4)

    # `arraytype=Array` must be exactly the previous behaviour, not merely equivalent
    l = LinearOps.make_const_linop(grid, xygrid, nfun; factored=true)
    la = LinearOps.make_const_linop(grid, xygrid, nfun; factored=true, arraytype=Array)
    @test typeof(l) === typeof(la)
    @test isequal(collect(l), collect(la))
    n = NonlinearRHS.const_norm_free(grid, xygrid, nfun; factored=true)(0.0)
    na = NonlinearRHS.const_norm_free(grid, xygrid, nfun; factored=true,
                                      arraytype=Array)(0.0)
    @test isequal(collect(n), collect(na))

    # The lazy operators are Adapt-able, which is how their factors reach a device.
    # Adapting to `Array` is the identity, so this also checks the rule is wired up.
    @test Adapt.adapt(Array, l) isa LinearOps.FactoredFreeLinop
    @test Adapt.adapt(Array, n) isa NonlinearRHS.FreeNorm
    @test isequal(collect(Adapt.adapt(Array, l)), collect(l))
    @test isequal(collect(Adapt.adapt(Array, n)), collect(n))

    # A materialised operator is a whole extra field-sized device array, so a device
    # request must be refused rather than silently doubling the footprint
    @test_throws ErrorException LinearOps.make_const_linop(
        grid, xygrid, nfun; factored=false, arraytype=DummyGPUArray)
    @test_throws ErrorException NonlinearRHS.const_norm_free(
        grid, xygrid, nfun; factored=false, arraytype=DummyGPUArray)

    # The transform keeps grid-vector mirrors and (host) has no separate inverse plan
    densityfun = z -> 1.0
    responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(:Ar)),)
    normfun = NonlinearRHS.const_norm_free(grid, xygrid, nfun; factored=true)
    inputs = Fields.GaussGaussField(λ0=800e-9, τfwhm=30e-15, energy=1e-6, w0=100e-6)
    Eω, transform, FT = Luna.setup(grid, xygrid, densityfun, normfun, responses, inputs)
    @test Eω isa Array{ComplexF64, 3}
    @test transform.gv.ω === grid.ω          # aliased on the host: no copy
    @test transform.IFT === nothing          # host keeps using ldiv! through FT
    @test Utils.backend(transform.Eto) isa Utils.CPUBackend

    # The transform call is the hot path: the added type parameters must not cost
    # inference (a dynamic dispatch here would be a large silent slowdown)
    nl = similar(Eω)
    @test (@inferred transform(nl, Eω, 0.0); true)
end

# --- Functional device-path tests, skipped without JLArrays ------------------
have_jlarrays = try
    @eval import JLArrays
    true
catch
    false
end

if have_jlarrays
    @testset "JLArray backend" begin
        JLArray = JLArrays.JLArray
        @test Utils.backend(JLArray(zeros(4))) isa Utils.DeviceBackend
        @test Utils.backend(view(JLArray(zeros(4)), 1:2)) isa Utils.DeviceBackend

        # tchunks must launch exactly one kernel over the whole array, and the fused
        # broadcast inside it must run without scalar indexing (JLArray throws on it).
        d = JLArray(zeros(64))
        s = JLArray(collect(1.0:64.0))
        n = Ref(0)
        Utils.tchunks(d, s) do dd, ss
            n[] += 1
            @. dd = 2ss
        end
        @test n[] == 1
        @test Array(d) == 2 .* collect(1.0:64.0)
    end

    @testset "device error norms" begin
        import Luna: RK45
        JLArray = JLArrays.JLArray

        # A stand-in for the stepper: `weaknorm_fused`/`errnorm` only read these fields.
        mutable struct FakeStepper{T, N}
            y::T
            yn::T
            ks::NTuple{7, T}
            yerr::Union{Nothing, T}
            dt::Float64
            rtol::Float64
            atol::Float64
            norm::N
        end

        rng = Random.Xoshiro(20260814)
        n = 512
        mk() = randn(rng, ComplexF64, n)
        y, yn = mk(), mk()
        ks = ntuple(_ -> mk(), 7)
        host = FakeStepper(y, yn, ks, nothing, 1.7e-4, 1e-8, 1e-10, RK45.weaknorm)
        dev = FakeStepper(JLArray(y), JLArray(yn), map(JLArray, ks), nothing,
                          host.dt, host.rtol, host.atol, RK45.weaknorm)

        ehost = RK45.weaknorm_fused(host)
        edev = RK45.weaknorm_fused(dev)
        # Not bitwise: a parallel reduction sums in a different order. This tolerance is
        # what the whole device path is validated to.
        @test isapprox(ehost, edev; rtol=1e-12)

        # The device path must not allocate the error estimate (a tenth field-sized
        # buffer at production size)
        @test dev.yerr === nothing

        # ...and the fused version must agree with the materialised reference, which is
        # what a custom norm would receive
        yerr = @. host.dt*(ks[1]*RK45.errest[1] + ks[3]*RK45.errest[3] +
                           ks[4]*RK45.errest[4] + ks[5]*RK45.errest[5] +
                           ks[6]*RK45.errest[6] + ks[7]*RK45.errest[7])
        @test isapprox(ehost, RK45.weaknorm(yerr, y, yn, host.rtol, host.atol);
                       rtol=1e-12)

        # A norm without a fused version must fail loudly on a device rather than
        # silently allocating and running a host scalar loop
        othernorm(yerr, y, yn, rtol, atol) = RK45.weaknorm(yerr, y, yn, rtol, atol)
        devother = FakeStepper(dev.y, dev.yn, dev.ks, nothing,
                               host.dt, host.rtol, host.atol, othernorm)
        @test_throws ErrorException RK45.errnorm(devother)
        # the same norm on the host still works, via the materialising path
        hostother = FakeStepper(y, yn, ks, nothing, host.dt, host.rtol, host.atol,
                                othernorm)
        @test isapprox(RK45.errnorm(hostother), ehost; rtol=1e-12)
        @test hostother.yerr !== nothing   # materialised, as documented
    end
else
    @info "JLArrays not available — skipping functional device tests " *
          "(run via `Pkg.test(\"Luna\")` to include them)"
end

end
