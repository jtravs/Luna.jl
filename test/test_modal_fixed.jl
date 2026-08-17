import Test: @test, @testset, @test_throws, @test_logs
import Luna
import Luna: Grid, Modes, Capillary, Fields, PhysData, Nonlinear, NonlinearRHS, Ionisation
import Luna.NonlinearRHS: TransModal, TransModalFixed
import LinearAlgebra: norm
import Random: MersenneTwister

# Frozen-field comparison of the fixed-quadrature modal transform against the adaptive
# cubature TransModal at tight tolerance. Kerr is a quartic in the mode fields, so both are
# exact to rounding; a smooth (ADK) plasma converges to ~1e-9 on a few tens of nodes.

a = 100e-6
λ0 = 800e-9

function modal_setup(grid, modes, components, resp; energies, kwargs...)
    inputs = Tuple((mode=i, fields=(Fields.GaussField(λ0=λ0, τfwhm=15e-15, energy=e),))
                   for (i, e) in enumerate(energies))
    Luna.setup(grid, z -> PhysData.density(:Ar, 3.0), resp, inputs, modes, components; kwargs...)
end

# make the field spectrally rich: propagate linearly a little and add a chirp
function nontrivial!(Eω, grid, seed=1)
    rng = MersenneTwister(seed)
    for i in axes(Eω, 2)
        Eω[:, i] .*= exp.(1im .* (0.3 .* randn(rng) .* (grid.ω .- 2π*PhysData.c/λ0).^2 .* 1e-30))
    end
    Eω
end

relerr(x, y) = norm(x .- y)/norm(y)

function reference(grid, ts, resp, densityfun, norm!; full, rtol=1e-10, mfcn=10^7)
    xo = Array{grid isa Grid.RealGrid ? Float64 : ComplexF64}(undef, length(grid.to), ts.npol)
    FTo = grid isa Grid.RealGrid ? Luna.FFTW.plan_rfft(xo, 1) : Luna.FFTW.plan_fft(xo, 1)
    TransModal(grid, ts, FTo, resp, densityfun, norm!; rtol, mfcn, full)
end

@testset "Kerr, HE1m radial, scalar" begin
    grid = Grid.RealGrid(1.0, λ0, (200e-9, 3000e-9), 200e-15)
    modes = [Capillary.MarcatiliMode(a, :Ar, 3.0; n=1, m=m) for m in 1:4]
    resp = (Nonlinear.Kerr_field(PhysData.γ3_gas(:Ar)),)
    ρ = z -> PhysData.density(:Ar, 3.0)
    Eω, tf, FT = modal_setup(grid, modes, :y, resp; energies=(1e-6, 4e-7, 2e-7, 1e-7),
                             modal_integral=:fixed, nr=32)
    nontrivial!(Eω, grid)
    tref = reference(grid, tf.ts, resp, ρ, tf.norm!; full=false)
    nlref = similar(Eω); tref(nlref, Eω, 0.0)
    nl = similar(Eω); tf(nl, Eω, 0.0)
    @test relerr(nl, nlref) < 1e-10
    # Kronrod rule and error estimate
    Eω2, tk, _ = modal_setup(grid, modes, :y, resp; energies=(1e-6, 4e-7, 2e-7, 1e-7),
                             modal_integral=:fixed, nr=41, kronrod=true)
    tk(nl, Eω, 0.0)
    @test relerr(nl, nlref) < 1e-10
    err = NonlinearRHS.integral_error!(tk)
    @test norm(err)/norm(nl) < 1e-8 # embedded Gauss(20) is also converged for Kerr
    # too few nodes: not converged, and the estimate says so
    _, tc, _ = modal_setup(grid, modes, :y, resp; energies=(1e-6, 4e-7, 2e-7, 1e-7),
                           modal_integral=:fixed, nr=9, kronrod=true)
    tc(nl, Eω, 0.0)
    e = relerr(nl, nlref)
    @test e > 1e-6
    @test norm(NonlinearRHS.integral_error!(tc))/norm(nl) > e/100
    @test startswith(string(tf), "TransModal")
    @test occursin("  modes: 4\n    ", string(tf))
    @test NonlinearRHS.scratch(tf) === tf.Et_scratch
    @test size(tf.Et_scratch) == (length(grid.t), 4)
end

@testset "Kerr, HE11 x+y, vector" begin
    grid = Grid.RealGrid(1.0, λ0, (200e-9, 3000e-9), 200e-15)
    modes = [Capillary.MarcatiliMode(a, :Ar, 3.0; n=1, m=1, ϕ=0.0),
             Capillary.MarcatiliMode(a, :Ar, 3.0; n=1, m=1, ϕ=π/2),
             Capillary.MarcatiliMode(a, :Ar, 3.0; n=1, m=2, ϕ=0.0)]
    resp = (Nonlinear.Kerr_field(PhysData.γ3_gas(:Ar)),)
    ρ = z -> PhysData.density(:Ar, 3.0)
    Eω, tf, FT = modal_setup(grid, modes, :xy, resp; energies=(1e-6, 3e-7, 2e-7),
                             modal_integral=:fixed, nr=32)
    nontrivial!(Eω, grid, 2)
    tref = reference(grid, tf.ts, resp, ρ, tf.norm!; full=false)
    nlref = similar(Eω); tref(nlref, Eω, 0.0)
    nl = similar(Eω); tf(nl, Eω, 0.0)
    @test relerr(nl, nlref) < 1e-10
    @test size(tf.Et) == (length(grid.to), 2*32)
end

@testset "Kerr, TM01+HE21+HE11, full 2-D" begin
    grid = Grid.RealGrid(1.0, λ0, (200e-9, 3000e-9), 200e-15)
    modes = [Capillary.MarcatiliMode(a, :Ar, 3.0; n=0, m=1, kind=:TM),
             Capillary.MarcatiliMode(a, :Ar, 3.0; n=2, m=1, ϕ=-π/4),
             Capillary.MarcatiliMode(a, :Ar, 3.0; n=1, m=1)]
    resp = (Nonlinear.Kerr_field(PhysData.γ3_gas(:Ar)),)
    ρ = z -> PhysData.density(:Ar, 3.0)
    Eω, tf, FT = modal_setup(grid, modes, :xy, resp; energies=(1e-6, 1e-6, 5e-7),
                             modal_integral=:fixed, full=true, nr=32, nθ=12)
    nontrivial!(Eω, grid, 3)
    nl = similar(Eω); tf(nl, Eω, 0.0)
    # a much finer fixed rule as reference: exactness in θ needs only nθ >= 4*1+1
    _, tfine, _ = modal_setup(grid, modes, :xy, resp; energies=(1e-6, 1e-6, 5e-7),
                              modal_integral=:fixed, full=true, nr=96, nθ=32)
    nlfine = similar(Eω); tfine(nlfine, Eω, 0.0)
    @test relerr(nl, nlfine) < 1e-10
    # adaptive 2-D cubature at moderate tolerance agrees to its tolerance
    tref = reference(grid, tf.ts, resp, ρ, tf.norm!; full=true, rtol=1e-6, mfcn=10^6)
    nlref = similar(Eω); tref(nlref, Eω, 0.0)
    @test relerr(nl, nlref) < 1e-5
    # LP11 (TM01 + HE21) does not couple to HE11 through the Kerr effect on the fixed rule
    Eω[:, 3] .= 0
    tf(nl, Eω, 0.0)
    @test norm(nl[:, 3])/norm(nl[:, 1:2]) < 1e-14
    # nθ below the exactness bound warns
    @test_logs (:warn, r"exactness bound") match_mode=:any modal_setup(grid, modes, :xy, resp; energies=(1e-6, 1e-6, 5e-7), modal_integral=:fixed, full=true,
        nr=16, nθ=4)
end

@testset "Kerr, envelope grid" begin
    grid = Grid.EnvGrid(1.0, λ0, (400e-9, 2000e-9), 300e-15)
    modes = [Capillary.MarcatiliMode(a, :Ar, 3.0; n=1, m=m) for m in 1:3]
    resp = (Nonlinear.Kerr_env(PhysData.γ3_gas(:Ar)),)
    ρ = z -> PhysData.density(:Ar, 3.0)
    Eω, tf, FT = modal_setup(grid, modes, :y, resp; energies=(1e-6, 4e-7, 2e-7),
                             modal_integral=:fixed, nr=24)
    nontrivial!(Eω, grid, 4)
    tref = reference(grid, tf.ts, resp, ρ, tf.norm!; full=false)
    nlref = similar(Eω); tref(nlref, Eω, 0.0)
    nl = similar(Eω); tf(nl, Eω, 0.0)
    @test relerr(nl, nlref) < 1e-10
    @test eltype(tf.S) == ComplexF64 # zgemm rather than a mixed-eltype fallback
end

@testset "Kerr + ADK plasma, scalar and vector" begin
    grid = Grid.RealGrid(1.0, λ0, (200e-9, 3000e-9), 200e-15)
    ρ = z -> PhysData.density(:Ar, 3.0)
    ionrate = Ionisation.IonRateADK(:Ar)
    Ip = PhysData.ionisation_potential(:Ar)
    for (label, modes, comps, npol) in (
            ("scalar", [Capillary.MarcatiliMode(a, :Ar, 3.0; n=1, m=m) for m in 1:3], :y, 1),
            ("vector", [Capillary.MarcatiliMode(a, :Ar, 3.0; n=1, m=1, ϕ=0.0),
                        Capillary.MarcatiliMode(a, :Ar, 3.0; n=1, m=1, ϕ=π/2)], :xy, 2))
        Et_ex = npol == 1 ? grid.to : Array{Float64}(undef, length(grid.to), 2)
        resp = (Nonlinear.Kerr_field(PhysData.γ3_gas(:Ar)),
                Nonlinear.PlasmaCumtrapz(grid.to, Et_ex, ionrate, Ip))
        energies = npol == 1 ? (100e-6, 10e-6, 3e-6) : (100e-6, 50e-6)
        Eω, tf, FT = modal_setup(grid, modes, comps, resp; energies, modal_integral=:fixed, nr=48)
        # fresh plasma responses for the reference (the response objects hold buffers)
        respref = (Nonlinear.Kerr_field(PhysData.γ3_gas(:Ar)),
                   Nonlinear.PlasmaCumtrapz(grid.to, Et_ex, ionrate, Ip))
        tref = reference(grid, tf.ts, respref, ρ, tf.norm!; full=false, rtol=1e-9)
        nlref = similar(Eω); tref(nlref, Eω, 0.0)
        nl = similar(Eω); tf(nl, Eω, 0.0)
        # ionisation must actually be happening for this to be a meaningful test
        Eω0 = copy(Eω); nlk = similar(Eω)
        tk = reference(grid, tf.ts, (respref[1],), ρ, tf.norm!; full=false, rtol=1e-9)
        tk(nlk, Eω, 0.0)
        @test relerr(nlk, nlref) > 1e-3
        @test relerr(nl, nlref) < 1e-7
    end
end

@testset "Noise field and gas mixtures" begin
    grid = Grid.RealGrid(1.0, λ0, (200e-9, 3000e-9), 200e-15)
    modes = [Capillary.MarcatiliMode(a, :Ar, 3.0; n=1, m=m) for m in 1:2]
    ρ = z -> PhysData.density(:Ar, 3.0)
    noise = Fields.generate_noise_field(grid; rng=MersenneTwister(5), nmodes=2)
    resp = (Nonlinear.Kerr_field(PhysData.γ3_gas(:Ar)),)
    Eω, tf, FT = modal_setup(grid, modes, :y, resp; energies=(1e-6, 4e-7),
                             modal_integral=:fixed, nr=24, noise_field=noise)
    tref = reference(grid, tf.ts, resp, ρ, tf.norm!; full=false)
    tref = TransModal(grid, tf.ts, tref.FT, resp, ρ, tf.norm!; rtol=1e-10, mfcn=10^7,
                      noise_field=noise)
    nlref = similar(Eω); tref(nlref, Eω, 0.0)
    nl = similar(Eω); tf(nl, Eω, 0.0)
    @test relerr(nl, nlref) < 1e-10
    # mixture: two gases as a tuple of response tuples with a vector density
    respmix = ((Nonlinear.Kerr_field(PhysData.γ3_gas(:Ar)),),
               (Nonlinear.Kerr_field(PhysData.γ3_gas(:He)),))
    ρmix = z -> [PhysData.density(:Ar, 2.0), PhysData.density(:He, 1.0)]
    Eω, tf, FT = Luna.setup(grid, ρmix, respmix,
                            ((mode=1, fields=(Fields.GaussField(λ0=λ0, τfwhm=15e-15, energy=1e-6),)),),
                            modes, :y; modal_integral=:fixed, nr=24)
    tref = reference(grid, tf.ts, respmix, ρmix, tf.norm!; full=false)
    nlref = similar(Eω); tref(nlref, Eω, 0.0)
    nl = similar(Eω); tf(nl, Eω, 0.0)
    @test relerr(nl, nlref) < 1e-10
end

@testset "Batched plasma equals columnwise" begin
    import Luna: Maths, Utils
    nt = 2048; npts = 7
    t = collect(range(-40e-15, 40e-15, length=nt)); δt = t[2] - t[1]
    ω0 = 2π*PhysData.c/λ0
    rng = MersenneTwister(11)
    ρ = PhysData.density(:Ar, 3.0)
    Ip = PhysData.ionisation_potential(:Ar)
    rates = (Ionisation.IonRateADK(:Ar),
             Ionisation.IonRatePPTAccel(:Ar, λ0; N=2^12, cache=false))
    for npol in (1, 2), ir in rates
        E3 = zeros(nt, npol, npts)
        for i in 1:npts, p in 1:npol
            A = 6e10*rand(rng)*exp.(-t.^2 ./ (2*(8e-15)^2))
            E3[:, p, i] .= A .* cos.(ω0 .* t .+ 2π*rand(rng))
        end
        Eex = npol == 1 ? t : Array{Float64}(undef, nt, 2)
        colwise = Nonlinear.PlasmaCumtrapz(t, Eex, ir, Ip)
        outc = zeros(nt, npol, npts)
        for i in 1:npts
            Ecol = npol == 1 ? view(E3, :, 1, i) : view(E3, :, :, i)
            Pcol = npol == 1 ? view(outc, :, 1, i) : view(outc, :, :, i)
            colwise(Pcol, Ecol, ρ)
        end
        @test maximum(abs.(outc)) > 0
        batched = Nonlinear.batched_response(colwise)
        @test batched isa Nonlinear.PlasmaCumtrapzBatched
        @test Nonlinear.batched(batched, npol)
        outb = zeros(nt, npol, npts)
        batched(outb, E3, ρ)
        @test outb == outc # same kernels on views: bit-identical
        # 2-D (nt, npts) call for scalar fields
        if npol == 1
            outb2 = zeros(nt, npts)
            batched(outb2, E3[:, 1, :], ρ)
            @test outb2 == outc[:, 1, :]
        end
        # the host path stores only the rate and P (two-pass column kernel)
        @test batched.fraction === nothing && batched.J === nothing && batched.phase === nothing
        # the device path (whole-array broadcasts + doubling scan) on host arrays, which
        # needs the intermediate-stage buffers the host path no longer allocates
        outd = zeros(nt, npol, npts)
        Nonlinear._plasma_buffers!(batched, E3)
        fraction = similar(batched.rate); phase = similar(E3); J = similar(E3)
        tmp = similar(batched.rate); tmp3 = similar(E3)
        Nonlinear._plasma_batched!(Utils.DeviceBackend(), outd, E3, ρ, batched,
                                   batched.rate, fraction, batched.Em,
                                   phase, J, batched.P, tmp, tmp3)
        @test relerr(outd, outc) < 1e-12
        # the fused two-pass column kernels are bit-identical to the multi-pass reference
        Pf = zeros(nt, npol); Pm = zeros(nt, npol)
        ratef = zeros(nt); ratem = zeros(nt); frac = zeros(nt); ph = zeros(nt, npol); Jm = zeros(nt, npol)
        for i in 1:npts
            if npol == 1
                Ecol = view(E3, :, 1, i)
                Nonlinear._plasma_scalar!(view(Pf, :, 1), ratef, Ecol, ir, Ip, 0.0, δt)
                Nonlinear._plasma_scalar_multipass!(view(Pm, :, 1), view(Jm, :, 1), view(ph, :, 1),
                                                    ratem, frac, Ecol, ir, Ip, 0.0, δt)
            else
                Ex = view(E3, :, 1, i); Ey = view(E3, :, 2, i); Em = hypot.(Ex, Ey)
                Nonlinear._plasma_vector!(Pf, ratef, Ex, Ey, Em, ir, Ip, 0.0, δt)
                Nonlinear._plasma_vector_multipass!(Pm, Jm, ph, ratem, frac, view(E3, :, :, i),
                                                    Ex, Ey, Em, ir, Ip, 0.0, δt)
            end
            @test Pf == Pm
            @test maximum(abs.(Pf)) > 0
        end
    end
    # the doubling scan against the sequential trapezoid rule
    y = randn(rng, 1000, 2, 3)
    ref = similar(y); Maths.cumtrapz!(ref, y, δt)
    out = similar(y); tmp = similar(y)
    Maths.cumtrapz_scan!(out, y, δt, tmp)
    @test relerr(out, ref) < 1e-13
    @test out[1, :, :] == zeros(2, 3)
end

@testset "Batched real-field Raman equals columnwise" begin
    import Luna: Raman, Utils
    nt = 2048; npts = 6
    t = collect(range(-200e-15, 200e-15, length=nt))
    ω0 = 2π*PhysData.c/λ0
    rng = MersenneTwister(13)
    ρ = PhysData.density(:H2, 5.0)
    rr = Raman.raman_response(t, :H2)
    E3 = zeros(nt, 1, npts)
    for i in 1:npts
        A = 3e9*rand(rng)*exp.(-t.^2 ./ (2*(30e-15)^2))
        E3[:, 1, i] .= A .* cos.(ω0 .* t .+ 2π*rand(rng))
    end
    for thg in (true, false)
        colwise = Nonlinear.RamanPolarField(t, rr; thg)
        outc = zeros(nt, 1, npts)
        for i in 1:npts
            colwise(view(outc, :, 1, i), view(E3, :, 1, i), ρ)
        end
        @test maximum(abs.(outc)) > 0
        batched = Nonlinear.batched_response(colwise)
        @test batched isa Nonlinear.RamanPolarFieldBatched
        @test batched.thg == thg
        @test Nonlinear.batched(batched) && Nonlinear.batched(batched, 1)
        outb = zeros(nt, 1, npts)
        batched(outb, E3, ρ)
        @test relerr(outb, outc) < 1e-12 # batched vs single-column FFTs: rounding only
        # 2-D (nt, npts) call and a second call (buffers reused, padding rebuilt)
        outb2 = zeros(nt, npts)
        batched(outb2, E3[:, 1, :], ρ)
        @test relerr(outb2, outc[:, 1, :]) < 1e-12
        # the device path (whole-array broadcasts) on host arrays, with the same buffers
        outd = zeros(nt, npts)
        Nonlinear._raman_field_batched!(Utils.DeviceBackend(), outd, E3[:, 1, :], ρ,
                                        batched.B, batched.Bω, batched.FT, batched.IFT,
                                        batched.hω, batched.dt, nt, thg,
                                        batched.C, batched.cFT, batched.cIFT)
        @test relerr(outd, outc[:, 1, :]) < 1e-12
    end
    # the transform picks the batched form up automatically for a Raman gas
    grid = Grid.RealGrid(1e-3, λ0, (200e-9, 3000e-9), 400e-15)
    modes = [Capillary.MarcatiliMode(a, :H2, 5.0; n=1, m=m) for m in 1:2]
    resp = (Nonlinear.Kerr_field(PhysData.γ3_gas(:H2)),
            Nonlinear.RamanPolarField(grid.to, Raman.raman_response(grid.to, :H2)))
    Eω, tf, FT = modal_setup(grid, modes, :y, resp; energies=(5e-6, 1e-6),
                             modal_integral=:fixed, nr=17)
    @test tf.resp_eval[2] isa Nonlinear.RamanPolarFieldBatched
    @test tf.resp[2] isa Nonlinear.RamanPolarField # the original is kept
    # Raman is cubic in the mode fields like Kerr, so the fixed rule is exact and the
    # adaptive cubature (nested Clenshaw–Curtis, exact for the cubic too) agrees to
    # rounding even at a loose tolerance. NB every new (2nto × npts) shape costs a
    # one-time PATIENT FFTW planning of the doubled-grid batched transform.
    _, ta, _ = modal_setup(grid, modes, :y, resp; energies=(5e-6, 1e-6),
                           modal_integral=:adaptive, rtol=1e-6, mfcn=10^4)
    nontrivial!(Eω, grid)
    nlf = similar(Eω); tf(nlf, Eω, 0.0)
    nla = similar(Eω); ta(nla, Eω, 0.0)
    @test relerr(nlf, nla) < 1e-12
end

@testset "Vector Kerr array-level equals columnwise" begin
    nt = 512; npts = 5
    rng = MersenneTwister(12)
    for (K, T) in ((Nonlinear.Kerr_field(1e-24), Float64), (Nonlinear.Kerr_env(1e-24), ComplexF64))
        E3 = randn(rng, T, nt, 2, npts) .* 1e9
        outc = zeros(T, nt, 2, npts)
        for i in 1:npts
            K(view(outc, :, :, i), view(E3, :, :, i), 2e25)
        end
        outb = zeros(T, nt, 2, npts)
        @test Nonlinear.batched(K, 2) && !Nonlinear.batched(K, 1)
        K(outb, E3, 2e25)
        @test outb == outc
    end
end
