import Test: @test, @testset, @test_broken
import Luna: Maths, Nonlinear, PhysData, Utils
import FFTW
import Random

# test full / nothg / envelope kerr effect
Nt = collect(range(0, length=2^16))
t = @. (Nt - 2^16/2)*3.430944979182369e-16/4
E = exp.(-0.5.*(t./10e-15).^2).*cos.(2π*PhysData.c/800e-9.*t)
kerrfield = Nonlinear.Kerr_field(1.0)
outf = similar(E)
fill!(outf, 0.0)
kerrfield(outf, E, 1.0)
kerrfieldn = Nonlinear.Kerr_field_nothg(1.0, 2^16)
outn = similar(E)
fill!(outn, 0.0)
kerrfieldn(outn, E, 1.0)
kerrfieldenv = Nonlinear.Kerr_env(1.0)
Eenv = Maths.hilbert(E)
oute = similar(Eenv)
fill!(oute, 0.0)
kerrfieldenv(oute, Eenv, 1.0)

outfω = FFTW.rfft(outf)
outnω = FFTW.rfft(outn)
outeω = FFTW.rfft(real.(oute))
# we compare only low (non THG) frequencies
# note that these are not expected to be exact, because we have dropped not just the THG term
# but also cross terms between positive and negative frequencies
@test isapprox(abs.(outnω[1800:2400]), abs.(outfω[1800:2400]), rtol=1e-15)
@test isapprox(abs.(outeω[1800:2400]), abs.(outfω[1800:2400]), rtol=1e-15)

@testset "batched no-THG Kerr" begin
    # `KerrFieldNoTHG` is the array-level form of the `Kerr_field_nothg(γ3, n)` closure:
    # same expression, same factor order, but the analytic signal of every column comes
    # from one batched FFT pair instead of one per column.
    Kbat = Nonlinear.Kerr_field_nothg(1.0)
    @test Kbat isa Nonlinear.KerrFieldNoTHG
    @test Nonlinear.batched(Kbat)
    @test !Nonlinear.pointwise(Kbat)
    # `batched_responses` leaves it alone — it is already the batched form
    @test Nonlinear.batched_responses((Kbat,))[1] === Kbat

    # single column: exactly the field the columnwise test above uses
    outb = zero(E)
    Kbat(outb, E, 1.0)
    @test isapprox(outb, outn; rtol=1e-12)

    # many columns at once, against the columnwise closure applied one at a time
    rng = Random.Xoshiro(2718)
    nt = 128
    Et = 1e8 .* randn(rng, nt, 3, 5)
    ρ, γ3 = 2.5e25, 1e-25
    Kcol = Nonlinear.Kerr_field_nothg(γ3, nt)
    Pcol = zeros(nt, 3, 5)
    for i in CartesianIndices((3, 5))
        Kcol(view(Pcol, :, i), view(Et, :, i), ρ)
    end
    Pbat = zeros(nt, 3, 5)
    Nonlinear.Kerr_field_nothg(γ3)(Pbat, Et, ρ)
    # not bit-exact: the batched and single-column FFT algorithms differ
    @test isapprox(Pcol, Pbat; rtol=1e-12)

    # accumulates into `out` rather than overwriting it, like every other response
    Pacc = copy(Pbat)
    Nonlinear.Kerr_field_nothg(γ3)(Pacc, Et, ρ)
    @test isapprox(Pacc, 2 .* Pbat; rtol=1e-12)

    # one response object reapplied to a differently-typed field must reallocate its
    # buffer rather than reuse the first one (a host/device A/B does exactly this)
    Kreuse = Nonlinear.Kerr_field_nothg(γ3)
    P1 = zeros(nt, 3, 5); Kreuse(P1, Et, ρ)
    Etbig = 1e8 .* randn(rng, nt, 3, 5)
    P2 = zeros(nt, 3, 5); Kreuse(P2, Etbig, ρ)
    Pfresh = zeros(nt, 3, 5); Nonlinear.Kerr_field_nothg(γ3)(Pfresh, Etbig, ρ)
    @test isequal(P2, Pfresh)

    # threaded and serial host paths must agree exactly (the kernel is elementwise, so
    # the chunking cannot change the result)
    minlen_old = Utils.THREADING_MINLEN[]
    Utils.THREADING_MINLEN[] = 1
    try
        Pth = zeros(nt, 3, 5); Nonlinear.Kerr_field_nothg(γ3)(Pth, Et, ρ)
        Utils.set_threading(false)
        Pser = zeros(nt, 3, 5); Nonlinear.Kerr_field_nothg(γ3)(Pser, Et, ρ)
        @test isequal(Pth, Pser)
    finally
        Utils.set_threading(true)
        Utils.THREADING_MINLEN[] = minlen_old
    end
end
