import Test: @test, @testset, @test_throws
import Luna: Modes, Capillary, RectModes, Antiresonant, PhysData
import Luna.Modes: transverse_quadrature, quadrature_nodes, quadrature_weights, mode_matrix
import LinearAlgebra: norm, I
import QuadGK

# discrete inner product on the rule: (nmodes × nmodes) matrix
function gram(Ems, w)
    nm, npol, np = size(Ems)
    G = zeros(nm, nm)
    for p in 1:np, c in 1:npol
        G .+= w[p] .* Ems[:, c, p] .* Ems[:, c, p]'
    end
    G
end
# ∫|Exy|² dA for Luna's normalisation
const Nnorm = 2/(PhysData.ε_0*PhysData.c)

@testset "Rules" begin
    a = 100e-6
    dl = (:polar, (0.0, 0.0), (a, 2π))
    for kronrod in (false, true)
        q = transverse_quadrature(dl, false; nr=24, kronrod)
        @test q.kind == :polar
        @test length(q) == q.nr
        @test all(0 .< q.ξ1 .< 1)
        w = quadrature_weights(q, dl)
        # area of the disc
        @test sum(w) ≈ π*a^2
        # embedded rule also integrates constants
        @test sum(quadrature_weights(q, dl; coarse=true)) ≈ π*a^2
        # ∫ r^4 dA = 2π a^6/6
        xs = quadrature_nodes(q, dl)
        @test sum(w .* [x[1]^4 for x in xs]) ≈ 2π*a^6/6
        if kronrod
            @test isodd(q.nr)
            @test count(!iszero, q.wc1) == (q.nr - 1)÷2
        end
    end
    # 2-D polar: ∫ r² cos²θ dA = πa⁴/4
    q = transverse_quadrature(dl, true; nr=16, nθ=8)
    xs = quadrature_nodes(q, dl); w = quadrature_weights(q, dl)
    @test sum(w .* [x[1]^2*cos(x[2])^2 for x in xs]) ≈ π*a^4/4
    wc = quadrature_weights(q, dl; coarse=true)
    @test sum(wc .* [x[1]^2*cos(x[2])^2 for x in xs]) ≈ π*a^4/4
    @test count(!iszero, q.wc2) == 4
    # cartesian
    dlc = (:cartesian, (-2.0, -1.0), (2.0, 1.0))
    q = transverse_quadrature(dlc, true; nr=8, nθ=6)
    xs = quadrature_nodes(q, dlc); w = quadrature_weights(q, dlc)
    @test sum(w) ≈ 8.0
    @test sum(w .* [x[1]^2*x[2]^2 for x in xs]) ≈ (2*8/3)*(2/3)
    @test_throws ErrorException transverse_quadrature(dlc, false; nr=8)
end

@testset "Orthonormality HE1m radial" begin
    a = 125e-6
    ms = [Capillary.MarcatiliMode(a, :He, 1.0; n=1, m=m) for m in 1:6]
    dl = Modes.dimlimits(ms[1])
    for nr in (24, 32)
        q = transverse_quadrature(dl, false; nr)
        Ems = mode_matrix(ms, 2, q)
        @test size(Ems) == (6, 1, nr)
        G = gram(Ems, quadrature_weights(q, dl))
        @test maximum(abs.(G ./ Nnorm .- I(size(G,1)))) < 1e-12
    end
end

@testset "Orthonormality mixed 2-D" begin
    a = 125e-6
    ms = [Capillary.MarcatiliMode(a, :He, 1.0; n=1, m=1),
          Capillary.MarcatiliMode(a, :He, 1.0; n=1, m=2, ϕ=π/2),
          Capillary.MarcatiliMode(a, :He, 1.0; n=2, m=1, ϕ=-π/4),
          Capillary.MarcatiliMode(a, :He, 1.0; n=0, m=1, kind=:TE),
          Capillary.MarcatiliMode(a, :He, 1.0; n=0, m=2, kind=:TM),
          Capillary.MarcatiliMode(a, :He, 1.0; n=3, m=1)]
    dl = Modes.dimlimits(ms[1])
    q = transverse_quadrature(dl, true; nr=32, nθ=16)
    Ems = mode_matrix(ms, 1:2, q)
    G = gram(Ems, quadrature_weights(q, dl))
    @test maximum(abs.(G ./ Nnorm .- I(size(G,1)))) < 1e-12
    @test [Modes.azimuthal_order(m) for m in ms] == [0, 0, 1, 1, 1, 2]
    @test all(Modes.zconstant, ms)
    @test all(Modes.scale_invariant, ms)
end

@testset "Kerr overlap tensor convergence" begin
    a = 125e-6
    ms = [Capillary.MarcatiliMode(a, :He, 1.0; n=1, m=m) for m in 1:4]
    dl = Modes.dimlimits(ms[1])
    function gamma(nr; kronrod=false)
        q = transverse_quadrature(dl, false; nr, kronrod)
        E = mode_matrix(ms, 2, q)[:, 1, :]
        w = quadrature_weights(q, dl)
        Γ = zeros(4, 4, 4, 4)
        for m=1:4, n=1:4, p=1:4, r=1:4
            Γ[m,n,p,r] = sum(w .* E[m,:] .* E[n,:] .* E[p,:] .* E[r,:])
        end
        Γ
    end
    Γref = gamma(2048)
    @test maximum(abs.(gamma(32) .- Γref))/maximum(abs.(Γref)) < 1e-13
    @test maximum(abs.(gamma(41; kronrod=true) .- Γref))/maximum(abs.(Γref)) < 1e-13
    @test maximum(abs.(gamma(12) .- Γref))/maximum(abs.(Γref)) > 1e-6 # not yet converged
end

@testset "Rectangular modes" begin
    ms = [RectModes.RectMode(50e-6, 30e-6, :He, 1.0, :SiO2; n=n, m=m, pol=:x)
          for (n, m) in ((1, 1), (2, 1), (1, 2), (3, 2))]
    dl = Modes.dimlimits(ms[1])
    q = transverse_quadrature(dl, true; nr=16, nθ=16)
    Ems = mode_matrix(ms, 1, q)
    G = gram(Ems, quadrature_weights(q, dl))
    @test maximum(abs.(G ./ Nnorm .- I(size(G,1)))) < 1e-12
    @test !Modes.zconstant(ms[1]) # RectMode always wraps callables
end

@testset "Taper scaling and delegation" begin
    afun(z) = 100e-6*(1 - 0.5*z)
    m = Capillary.MarcatiliMode(afun, :He, 1.0; n=1, m=2)
    @test !Modes.zconstant(m)
    @test Modes.scale_invariant(m)
    dl0 = Modes.dimlimits(m, z=0.0); dl1 = Modes.dimlimits(m, z=1.0)
    q = transverse_quadrature(dl0, true; nr=12, nθ=8)
    E0 = mode_matrix((m,), 1:2, quadrature_nodes(q, dl0); z=0.0)
    E1 = mode_matrix((m,), 1:2, quadrature_nodes(q, dl1); z=1.0)
    a0 = dl0[3][1]; a1 = dl1[3][1]
    @test E1 .* a1 ≈ E0 .* a0
    @test quadrature_weights(q, dl1) ≈ quadrature_weights(q, dl0) .* (a1/a0)^2
    # antiresonant and delegated modes inherit the traits
    zm = Antiresonant.ZeisbergerMode(m; wallthickness=1e-6)
    @test Modes.scale_invariant(zm) && !Modes.zconstant(zm)
    dm = Modes.delegated(Capillary.MarcatiliMode(100e-6, :He, 1.0); neff=(m, ω; z=0) -> 1.0)
    @test Modes.zconstant(dm) && Modes.scale_invariant(dm) && Modes.azimuthal_order(dm) == 0
    dm2 = Modes.delegated(Capillary.MarcatiliMode(100e-6, :He, 1.0);
                          field=(m, xs; z=0) -> Modes.field(m, xs; z))
    @test !Modes.zconstant(dm2) && !Modes.scale_invariant(dm2)
end
