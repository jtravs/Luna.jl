import Test: @test, @testset

@testset "Full 3D propagation" begin
using Luna
Luna.set_fftw_mode(:estimate)
import FFTW
import Luna.PhysData: wlfreq
import LinearAlgebra: norm

gas = :Ar
pres = 1

τ = 10e-15
λ0 = 800e-9

w0 = 500e-6
energy = 1e-12
L = 2

R = 4e-3
N = 128

grid = Grid.RealGrid(L, 800e-9, (400e-9, 2000e-9), 80e-15)
xygrid = Grid.FreeGrid(R, N)

x = xygrid.x
y = xygrid.y
energyfun, energyfunω = Fields.energyfuncs(grid, xygrid)

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0

responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)

linop = LinearOps.make_const_linop(grid, xygrid, PhysData.ref_index_fun(gas, pres))
normfun = NonlinearRHS.const_norm_free(grid, xygrid, PhysData.ref_index_fun(gas, pres))

inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0)

Eω, transform, FT = Luna.setup(grid, xygrid, densityfun, normfun, responses, inputs)

output = Output.MemoryOutput(0, grid.zmax, 21)

Luna.run(Eω, grid, linop, transform, FT, output, max_dz=Inf, init_dz=1e-1)

Eout = output.data["Eω"] # (ω, ky, kx, z)

ω0idx = argmin(abs.(grid.ω .- wlfreq(λ0)))
λ0 = 2π*PhysData.c/grid.ω[ω0idx]
w1 = w0*sqrt(1+(L*λ0/(π*w0^2))^2)
Iω0_analytic = Maths.gauss.(xygrid.x, w1/2) # analytical solution (in paraxial approx)

Eω0yx = FFTW.ifft(Eout[ω0idx, :, :, end], (1, 2))
Iω0yx = abs2.(Eω0yx)
Iω0y = Maths.normbymax(dropdims(sum(Iω0yx, dims=2), dims=2))
Iω0x = Maths.normbymax(dropdims(sum(Iω0yx, dims=1), dims=1))

@test maximum(abs.(Iω0x .- Iω0_analytic)/norm(Iω0x)) < 5e-5
@test maximum(abs.(Iω0y .- Iω0_analytic)/norm(Iω0y)) < 5e-5

@testset "RealGrid ffac" begin
    # The default reproduces the historical grid exactly.
    gdef = Grid.RealGrid(L, 800e-9, (400e-9, 2000e-9), 80e-15)
    g6 = Grid.RealGrid(L, 800e-9, (400e-9, 2000e-9), 80e-15; ffac=6)
    for f in fieldnames(Grid.RealGrid)
        @test isequal(getfield(gdef, f), getfield(g6, f))
    end
    @test_throws ArgumentError Grid.RealGrid(L, 800e-9, (400e-9, 2000e-9), 80e-15; ffac=1)

    # At the TG-FROG production shape, ffac = 4 removes the oversampling entirely: the
    # crop lands on the last fine sample, so the fine and propagated grids coincide.
    p6 = Grid.RealGrid(40e-6, 260e-9, (143e-9, 600e-9), 110e-15)
    p4 = Grid.RealGrid(40e-6, 260e-9, (143e-9, 600e-9), 110e-15; ffac=4)
    @test length(p6.to) == 2*length(p6.t)          # 2x oversampled
    @test length(p4.to) == length(p4.t)            # not oversampled
    @test length(p4.ωo) == length(p4.ω)
    @test length(p4.to) == length(p6.t)            # half the fine grid of ffac = 6
    # ...but it is a DIFFERENT grid, not a cheaper version of the same one
    @test p4.ω != p6.ω
end

@testset "ffac convergence for the no-THG response" begin
    # ffac = 4 samples the nonlinear grid at 4x fmax rather than 6x, which is enough for
    # |E_a|²E (reaching 2ωmax - ωmin) but not for E³ (reaching 3ωmax). If that argument is
    # wrong the missing headroom aliases straight back into the propagated band, and the
    # error grows with distance. Run both and compare.
    #
    # NOTE the grid: the spectrum must be well resolved for this comparison to mean
    # anything. `ffac` changes δω and the realised time window, so the two runs live on
    # different grids and are compared as physical spectral densities through a spline —
    # on a grid with only a handful of samples across the spectral FWHM that interpolation
    # alone contributes several percent, swamping any aliasing signal.
    τ_c, λ0_c, w0_c, e_c, L_c, N_c, R_c = 5e-15, 800e-9, 100e-6, 30e-6, 5e-3, 16, 400e-6
    function run_ffac(ffac)
        g = Grid.RealGrid(L_c, λ0_c, (400e-9, 2000e-9), 400e-15; ffac)
        xy = Grid.FreeGrid(R_c, N_c)
        dfun = let d0 = PhysData.density(gas, pres); z -> d0 end
        resp = (Nonlinear.Kerr_field_nothg(PhysData.γ3_gas(gas)),)
        nf = PhysData.ref_index_fun(gas, pres)
        lo = LinearOps.make_const_linop(g, xy, nf; factored=true)
        nfn = NonlinearRHS.const_norm_free(g, xy, nf; factored=true)
        inp = Fields.GaussGaussField(λ0=λ0_c, τfwhm=τ_c, energy=e_c, w0=w0_c)
        Eω, tr, FTc = Luna.setup(g, xy, dfun, nfn, resp, inp)
        o = Output.MemoryOutput(0, g.zmax, 2)
        Luna.run(Eω, g, lo, tr, FTc, o; max_dz=Inf, init_dz=L_c/50, rtol=1e-8)
        (g, xy, o.data["Eω"])
    end
    # Physical spectral energy density, using energyfuncs' own prefactor so that
    # sum(density)*δω is the pulse energy on either grid.
    function density(g, xy, E, iz)
        dω = g.ω[2] - g.ω[1]
        dkx = xy.kx[2] - xy.kx[1]; Δkx = length(xy.kx)*dkx
        dky = xy.ky[2] - xy.ky[1]; Δky = length(xy.ky)*dky
        pre = PhysData.c*PhysData.ε_0/2 * 2π*dω/g.ω[end]^2 *
              2π*dkx/Δkx^2 * 2π*dky/Δky^2
        pre .* dropdims(sum(abs2, E[:, :, :, iz]; dims=(2, 3)); dims=(2, 3)) ./ dω
    end
    function relL2(ga, xya, Ea, gb, xyb, Eb, iz)
        Da = density(ga, xya, Ea, iz); Db = density(gb, xyb, Eb, iz)
        ωc = collect(range(max(ga.ω[2], gb.ω[2]), min(ga.ω[end], gb.ω[end]), 500))
        A = Maths.CSpline(ga.ω, Da).(ωc); B = Maths.CSpline(gb.ω, Db).(ωc)
        m = A .> 1e-6*maximum(A)
        sqrt(sum(abs2, (A .- B)[m]) / sum(abs2, A[m]))
    end
    g6, xy6, E6 = run_ffac(6)
    g4, xy4, E4 = run_ffac(4)
    @test length(g4.to) == length(g4.t)   # the point of the exercise
    r_in  = relL2(g6, xy6, E6, g4, xy4, E4, 1)
    r_out = relL2(g6, xy6, E6, g4, xy4, E4, 2)
    # `r_in` is the floor: it compares the same input pulse discretised on the two grids,
    # so it contains no propagation at all. Aliasing would make `r_out` grow away from it.
    @test r_in < 1e-6
    @test r_out < 3*r_in
end

end