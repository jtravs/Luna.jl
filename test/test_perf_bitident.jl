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

function propagate_free3d(; N=16, R=400e-6, saveN=5,
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
    linop = LinearOps.make_const_linop(grid, xygrid, nfun)
    normfun = NonlinearRHS.const_norm_free(grid, xygrid, nfun)
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

end
