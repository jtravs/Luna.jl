import Luna: PhysData, Grid, LinearOps, Modes, Capillary
import Test: @testset, @test
import Luna.PhysData: wlfreq
import Luna: Hankel

R = 5e-3
Nr = 256
Nx = 128
Ny = 64
gas = :Ar
pres = 1
nfun = let rif=PhysData.ref_index_fun(gas, pres)
    (ω; z) -> rif(wlfreq(ω))
end

@testset "radial, field" begin
grid = Grid.RealGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12)
q = Hankel.QDHT(R, Nr, dim=2)

linop = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
linopf = LinearOps.make_linop(grid, q, nfun)
out = similar(linop)

@test size(linop) == (length(grid.ω), q.N)

linopf(out, 0.0)
@test all(imag(out) .≈ imag(linop))
@test all(real(out) .≈ real(linop))
linopf(out, 0.5)
@test all(imag(out) .≈ imag(linop))
@test all(real(out) .≈ real(linop))
end

@testset "radial, env" begin
grid = Grid.EnvGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12)
grid_thg = Grid.EnvGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12; thg=true)
q = Hankel.QDHT(R, Nr, dim=2)

for gi in (grid, grid_thg)
    linop = LinearOps.make_const_linop(gi, q, PhysData.ref_index_fun(gas, pres))
    linopf = LinearOps.make_linop(gi, q, nfun)
    out = similar(linop)

    linopf(out, 0.0)
    @test all(imag(out) .≈ imag(linop))
    @test all(real(out) .≈ real(linop))
    linopf(out, 0.5)
    @test all(imag(out) .≈ imag(linop))
    @test all(real(out) .≈ real(linop))
end
end

@testset "3D, field" begin
grid = Grid.RealGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12)
xygrid = Grid.FreeGrid(R, Nx, R, Ny)

linop = LinearOps.make_const_linop(grid, xygrid, PhysData.ref_index_fun(gas, pres))
linopf = LinearOps.make_linop(grid, xygrid, nfun)
out = similar(linop)

@test size(linop) == (length(grid.ω), Ny, Nx)

linopf(out, 0.0)
@test all(imag(out) .≈ imag(linop))
@test all(real(out) .≈ real(linop))
linopf(out, 0.5)
@test all(imag(out) .≈ imag(linop))
@test all(real(out) .≈ real(linop))
end

@testset "3D, env" begin
grid = Grid.EnvGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12)
grid_thg = Grid.EnvGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12; thg=true)
xygrid = Grid.FreeGrid(R, Nx, R, Ny)

for gi in (grid, grid_thg)
    linop = LinearOps.make_const_linop(gi, xygrid, PhysData.ref_index_fun(gas, pres))
    linopf = LinearOps.make_linop(gi, xygrid, nfun)
    out = similar(linop)

    @test size(linop) == (length(gi.ω), Ny, Nx)

    linopf(out, 0.0)
    @test all(imag(out) .≈ imag(linop))
    @test all(real(out) .≈ real(linop))
    linopf(out, 0.5)
    @test all(imag(out) .≈ imag(linop))
    @test all(real(out) .≈ real(linop))
end
end

@testset "3D, frozen transverse" begin
# frozen_transverse=true replaces k_z(ω, k⊥) by k_z(ω, 0): the operator must become
# transverse-uniform and equal the unfrozen operator's k⊥ = 0 column everywhere, the
# default (off) must stay bit-identical to the pre-existing behaviour, and the
# factored/materialised bit-identity guarantee must survive the flag.
grid = Grid.EnvGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12)
grid_thg = Grid.EnvGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12; thg=true)
rgrid = Grid.RealGrid(1, 800e-9, (400e-9, 2000e-9), 0.2e-12)
xygrid = Grid.FreeGrid(R, Nx, R, Ny)
rif = PhysData.ref_index_fun(gas, pres)
iy0 = findfirst(iszero, xygrid.ky)
ix0 = findfirst(iszero, xygrid.kx)
for gi in (grid, grid_thg, rgrid)
    lref = LinearOps.make_const_linop(gi, xygrid, rif)
    loff = LinearOps.make_const_linop(gi, xygrid, rif; frozen_transverse=false)
    @test isequal(loff, lref)
    lfrz = LinearOps.make_const_linop(gi, xygrid, rif; frozen_transverse=true)
    lfrzfac = LinearOps.make_const_linop(gi, xygrid, rif; frozen_transverse=true,
                                         factored=true)
    @test all(iszero, lfrzfac.kperp2)
    @test isequal(collect(lfrzfac), lfrz)
    # transverse-uniform, and equal to the unfrozen k⊥ = 0 column
    @test all(lfrz .== lfrz[:, iy0:iy0, ix0:ix0])
    @test isequal(lfrz[:, iy0, ix0], lref[:, iy0, ix0])
end
end

@testset "equivalence for fast z-dependent linops" begin
a = 125e-6
L = 1
grid = Grid.RealGrid(L, 800e-9, (400e-9, 2000e-9), 0.5e-12)
coren, densityfun = Capillary.gradient(gas, L, pres, 0)
m = Capillary.MarcatiliMode(a, coren)
dm = Modes.delegated(m) # delegated mode tricks make_linop into using the generic version

lom!, βm! = LinearOps.make_linop(grid, m, 800e-9)
lodm!, βdm! = LinearOps.make_linop(grid, dm, 800e-9)
@assert typeof(lom!) != typeof(lodm!) # ...but best to check

outm = complex(similar(grid.ω))
outdm = complex(similar(grid.ω))
for zi in range(0, L, length=10)
    lom!(outm, zi)
    lodm!(outdm, zi)
    @test outm == outdm
    βm!(outm, zi)
    βdm!(outdm, zi)
    @test outm == outdm
end

a = 125e-6
L = 1
# NO THG
thg = false
grid = Grid.EnvGrid(L, 800e-9, (400e-9, 2000e-9), 0.5e-12; thg=thg)
coren, densityfun = Capillary.gradient(gas, L, pres, 0)
m = Capillary.MarcatiliMode(a, coren)
dm = Modes.delegated(m) # delegated mode tricks make_linop into using the generic version...

lom!, βm! = LinearOps.make_linop(grid, m, 800e-9; thg=thg)
lodm!, βdm! = LinearOps.make_linop(grid, dm, 800e-9; thg=thg)
@assert typeof(lom!) != typeof(lodm!) # ...but best to check

outm = complex(similar(grid.ω))
outdm = complex(similar(grid.ω))
for zi in range(0, L, length=10)
    lom!(outm, zi)
    lodm!(outdm, zi)
    @test outm == outdm
    βm!(outm, zi)
    βdm!(outdm, zi)
    @test outm == outdm
end
# WITH THG
thg = true
grid = Grid.EnvGrid(L, 800e-9, (400e-9, 2000e-9), 0.5e-12; thg=thg)
coren, densityfun = Capillary.gradient(gas, L, pres, 0)
m = Capillary.MarcatiliMode(a, coren)
dm = Modes.delegated(m) # delegated mode tricks make_linop into using the generic version...

lom!, βm! = LinearOps.make_linop(grid, m, 800e-9; thg=thg)
lodm!, βdm! = LinearOps.make_linop(grid, dm, 800e-9; thg=thg)
@assert typeof(lom!) != typeof(lodm!) # ...but best to check

outm = complex(similar(grid.ω))
outdm = complex(similar(grid.ω))
for zi in range(0, L, length=10)
    lom!(outm, zi)
    lodm!(outdm, zi)
    @test outm == outdm
    βm!(outm, zi)
    βdm!(outdm, zi)
    @test outm == outdm
end
end
@testset "MarcatiliLinop (multimode, tapers and gradients)" begin
    import Luna: RK45
    import LinearAlgebra: norm
    λ0 = 800e-9
    grid = Grid.RealGrid(1.0, λ0, (200e-9, 3000e-9), 300e-15)
    egrid = Grid.EnvGrid(1.0, λ0, (400e-9, 2000e-9), 300e-15)
    coren, dens = Capillary.gradient(:He, 1.0, 0.8, 0.0)
    @test coren isa Capillary.GradientCoreIndex
    taper(z) = 125e-6*(1 - 0.3z)
    cases = [
        [Capillary.MarcatiliMode(125e-6, coren; n=1, m=k) for k in 1:4],           # gradient
        [Capillary.MarcatiliMode(taper, :Ar, 1.0; n=1, m=k) for k in 1:4],         # taper
        [Capillary.MarcatiliMode(taper, coren; n=1, m=k) for k in 1:4],            # both
        [Capillary.MarcatiliMode(taper, coren; kind=:HE, n=1, m=1),                # mixed kinds
         Capillary.MarcatiliMode(taper, coren; kind=:TE, n=0, m=1),
         Capillary.MarcatiliMode(taper, coren; kind=:TM, n=0, m=1),
         Capillary.MarcatiliMode(taper, coren; kind=:HE, n=2, m=1)],
        [Capillary.MarcatiliMode(125e-6, coren; n=1, m=k, loss=false) for k in 1:3],
        [Capillary.MarcatiliMode(125e-6, coren; n=1, m=k, model=:reduced) for k in 1:3],
        Tuple(Capillary.MarcatiliMode(125e-6, coren; n=1, m=k, model=:reduced) for k in 1:3),
    ]
    for modes in cases
        @test Capillary.marcatili_linop_ok(modes)
        for (g, thgs) in ((grid, (false,)), (egrid, (false, true)))
            for thg in thgs
                new = g isa Grid.EnvGrid ? LinearOps.make_linop(g, modes, λ0; thg) :
                                          LinearOps.make_linop(g, modes, λ0)
                old = g isa Grid.EnvGrid ? LinearOps._make_linop_generic(g, modes, λ0; thg) :
                                          LinearOps._make_linop_generic(g, modes, λ0)
                @test new isa Capillary.MarcatiliLinop
                @test RK45.device_capable(new) && !RK45.device_capable(old)
                for z in (0.0, 0.3, 0.77, 1.0)
                    on = zeros(ComplexF64, length(g.ω), length(modes)); oo = similar(on)
                    new(on, z); old(oo, z)
                    @test on == oo # bit-identical: same scalar kernel, same operation order
                    @test all(iszero, on[.!g.sidx, :])
                end
            end
        end
    end
    # a user-supplied core index (arbitrary closure) or a wrapped mode falls back to the
    # generic closure
    modes = [Capillary.MarcatiliMode(125e-6, (ω; z) -> 1.0 + 1e-4*z; n=1, m=k) for k in 1:2]
    @test !Capillary.marcatili_linop_ok(modes)
    @test !(LinearOps.make_linop(grid, modes, λ0) isa Capillary.MarcatiliLinop)
    dmodes = [Modes.delegated(m) for m in cases[1]]
    @test !(LinearOps.make_linop(grid, dmodes, λ0) isa Capillary.MarcatiliLinop)
    # the fixed-core acceleration path (neff_wg) agrees with Modes.neff for every model
    # (the :reduced loss sign used to be reversed there)
    for model in (:full, :reduced), loss in (true, false)
        m = Capillary.MarcatiliMode(100e-6, :Ar, 1.0; n=1, m=2, model, loss)
        for ω in grid.ω[grid.sidx][1:100:end]
            @test Modes.neff(m, ω; z=0) == Capillary.neff(m, m.coren(ω, z=0)^2, Capillary.neff_wg(m, ω; z=0))
        end
    end
end
