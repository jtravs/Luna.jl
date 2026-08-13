import Test: @test, @testset

# Bit-identity harness for the optimised free-space code paths.
#
# Runs a small envelope free-space Kerr propagation and checks that optimised
# code paths (TransFree fast path, RK45 register reduction, threaded kernels,
# factored linear operator) reproduce the reference implementation exactly.
# "Exactly" means isequal on every element: the fast paths perform the same
# floating-point operations in the same order as the reference, so any
# difference at all indicates a real behavioural change.

@testset "Performance fast paths" begin
using Luna
Luna.set_fftw_mode(:estimate)

function propagate_free3d(; N=16, R=400e-6, saveN=5, factored=false,
                            setup_kwargs=(;), run_kwargs=(;))
    gas = :Ar
    pres = 4
    τ = 30e-15
    λ0 = 800e-9
    w0 = 100e-6
    energy = 30e-6
    L = 5e-3
    grid = Grid.EnvGrid(L, λ0, (400e-9, 2000e-9), 100e-15)
    xygrid = Grid.FreeGrid(R, N)
    densityfun = let dens0=PhysData.density(gas, pres)
        z -> dens0
    end
    responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)
    nfun = PhysData.ref_index_fun(gas, pres)
    linop = LinearOps.make_const_linop(grid, xygrid, nfun; factored)
    normfun = NonlinearRHS.const_norm_free(grid, xygrid, nfun; factored)
    inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0)
    Eω, transform, FT = Luna.setup(grid, xygrid, densityfun, normfun, responses,
                                   inputs; setup_kwargs...)
    output = Output.MemoryOutput(0, grid.zmax, saveN)
    Luna.run(Eω, grid, linop, transform, FT, output;
             max_dz=Inf, init_dz=L/50, rtol=1e-8, run_kwargs...)
    output.data["Eω"], output.data["z"]
end

@testset "determinism" begin
    Eω1, z1 = propagate_free3d()
    Eω2, z2 = propagate_free3d()
    @test isequal(Eω1, Eω2)
    @test isequal(z1, z2)
end

@testset "TransFree fast path" begin
    Eref, zref = propagate_free3d(setup_kwargs=(; fastpath=false))
    Efast, zfast = propagate_free3d()
    @test isequal(Eref, Efast)
    @test isequal(zref, zfast)
end

@testset "RK45 register reduction" begin
    Eref, zref = propagate_free3d()
    # adopting the caller's input array must not change results
    Ead, zad = propagate_free3d(run_kwargs=(; preserve_input=false))
    @test isequal(Eref, Ead)
    @test isequal(zref, zad)
    # a custom norm uses the materialised error estimate; wrapping weaknorm must
    # therefore reproduce the fused fast path exactly
    wnorm(yerr, y, yn, rtol, atol) = Luna.RK45.weaknorm(yerr, y, yn, rtol, atol)
    Emat, zmat = propagate_free3d(run_kwargs=(; norm=wnorm))
    @test isequal(Eref, Emat)
    @test isequal(zref, zmat)
end

@testset "threaded kernels" begin
    import Luna: Utils
    # lower the size threshold so the small test arrays actually exercise the threaded
    # code paths (they are dormant below THREADING_MINLEN); with a single Julia thread
    # both runs are serial and this test is trivial
    minlen_old = Utils.THREADING_MINLEN[]
    Utils.THREADING_MINLEN[] = 1
    try
        Eth, zth = propagate_free3d()
        Efacth, zfacth = propagate_free3d(factored=true)
        Utils.set_threading(false)
        Eser, zser = propagate_free3d()
        Efacser, zfacser = propagate_free3d(factored=true)
        @test isequal(Eth, Eser)
        @test isequal(zth, zser)
        @test isequal(Efacth, Efacser)
        @test isequal(zfacth, zfacser)
    finally
        Utils.set_threading(true)
        Utils.THREADING_MINLEN[] = minlen_old
    end
end

@testset "factored linop and norm" begin
    Eref, zref = propagate_free3d()
    Efac, zfac = propagate_free3d(factored=true)
    @test isequal(Eref, Efac)
    @test isequal(zref, zfac)
    # elementwise identity of the lazy operators vs the materialised arrays
    grid = Grid.EnvGrid(5e-3, 800e-9, (400e-9, 2000e-9), 100e-15)
    rgrid = Grid.RealGrid(5e-3, 800e-9, (400e-9, 2000e-9), 100e-15)
    xygrid = Grid.FreeGrid(400e-6, 16)
    nfun = PhysData.ref_index_fun(:Ar, 4)
    for g in (grid, rgrid)
        lmat = LinearOps.make_const_linop(g, xygrid, nfun)
        lfac = LinearOps.make_const_linop(g, xygrid, nfun; factored=true)
        @test size(lfac) == size(lmat)
        @test isequal(collect(lfac), lmat)
    end
    nmat = NonlinearRHS.const_norm_free(grid, xygrid, nfun)(0.0)
    nfac = NonlinearRHS.const_norm_free(grid, xygrid, nfun; factored=true)(0.0)
    @test isequal(collect(nfac), nmat)
end

@testset "pointwise Kerr agreement" begin
    import Luna: Nonlinear, NonlinearRHS
    for (resp, TT) in ((Nonlinear.Kerr_env(1e-25), ComplexF64),
                       (Nonlinear.Kerr_field(1e-25), Float64))
        @test Nonlinear.pointwise(resp)
        Et = TT <: Complex ? (randn(64, 4, 4) .+ im .* randn(64, 4, 4)) : randn(64, 4, 4)
        Et = TT.(1e8 .* Et)
        ρ = 2.5e25
        Pref = zero(Et)
        NonlinearRHS.Et_to_Pt!(Pref, Et, (resp,), ρ, CartesianIndices((4, 4)))
        Pord = zero(Et)
        NonlinearRHS.Et_to_Pt_ordered!(Pord, Et, (resp,), ρ, CartesianIndices((4, 4)))
        @test isequal(Pref, Pord)
        Ppw = zero(Et)
        NonlinearRHS.pointwise_Pt!(Ppw, Et, (resp,), ρ)
        @test isequal(Pref, Ppw)
        # in-place (aliased) application
        Pal = copy(Et)
        NonlinearRHS.pointwise_Pt!(Pal, Pal, (resp,), ρ)
        @test isequal(Pref, Pal)
    end
end

end
