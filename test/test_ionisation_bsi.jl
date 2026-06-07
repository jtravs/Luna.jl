import Test: @test, @testset, @test_throws
using Luna
import Luna.PhysData: au_Efield

# Tests for the barrier-suppression (over-the-barrier) corrections to the PPT rate.
#
# Expected behaviour (He @ 800 nm is the primary case):
#  - `:none` reproduces the current PPT rate exactly (regression guard).
#  - The correction factor is ≈ 1 (within a few %) far below E_b, then falls below 1 as the
#    field approaches and exceeds E_b ≈ 0.204 a.u. (1.05e11 V/m) for He.
#  - At E_b: Tong–Lin → 0.52×, Zhang → 0.49× the bare rate.
#  - Tong–Lin and Zhang coincide to <10% for E < 2 E_b (their common validity region). If
#    this test fails, suspect a wrong/sign-flipped Zhang coefficient or the Eq. (8) typo.
#  - Above 2 E_b they diverge: Tong–Lin keeps suppressing (its corrected rate eventually
#    turns over and decreases with field — unphysical), while Zhang flattens and the rate
#    keeps rising. Hence `zh > tl` beyond ~2 E_b.

@testset "PPT barrier-suppression corrections" begin
    λ0 = 800e-9
    Ip = PhysData.ionisation_potential(:He)
    Ip_au = Ip / PhysData.au_energy
    Z = 1.0
    κ  = sqrt(2Ip_au)
    Eb_au = Ip_au^2 / (4Z)              # = 0.204 a.u.
    Eb = Eb_au * au_Efield              # field in V/m

    none = Ionisation.IonRatePPT(:He, λ0; bsi=:none)
    tl   = Ionisation.IonRatePPT(:He, λ0; bsi=:tonglin)
    zh   = Ionisation.IonRatePPT(:He, λ0; bsi=:zhang)

    factor(rate, E) = rate(E) / none(E)

    # (a) Regression: :none must equal the unmodified PPT exactly.
    for x in (0.3, 0.8, 1.5)
        @test none(x*Eb) == Ionisation.IonRatePPT(:He, λ0)(x*Eb)
    end

    # (b) Mild correction well below E_b (factor close to 1).
    @test 0.88 ≤ factor(tl, 0.15Eb) ≤ 1.0
    @test 0.88 ≤ factor(zh, 0.15Eb) ≤ 1.0

    # (c) Known He values at the barrier-suppression field E_b.
    @test isapprox(factor(tl, Eb), 0.522; atol=0.01)   # Tong–Lin α=7
    @test isapprox(factor(zh, Eb), 0.494; atol=0.01)   # Zhang

    # (d) Tong–Lin and Zhang must AGREE below ~2 E_b (the overlap of their validity).
    for x in (0.5, 0.8, 1.0, 1.3, 1.8)
        @test isapprox(tl(x*Eb), zh(x*Eb); rtol=0.10)
    end

    # (e) Both must SUPPRESS (never enhance) over the whole range; factor ≤ 1.
    for x in (0.2, 0.5, 1.0, 2.0, 3.0, 4.5)
        @test factor(tl, x*Eb) ≤ 1.0 + 1e-9
        @test factor(zh, x*Eb) ≤ 1.0 + 1e-9
        @test tl(x*Eb) ≤ none(x*Eb)
        @test zh(x*Eb) ≤ none(x*Eb)
    end

    # (f) Beyond 2 E_b, Tong–Lin over-suppresses → Zhang gives the LARGER (more physical) rate.
    for x in (2.5, 3.0, 4.0)
        @test zh(x*Eb) > tl(x*Eb)
    end

    # (g) Independent self-consistency of the Tong–Lin analytic factor.
    αHe = 7.0
    f_expected = exp(-αHe * (Z^2/Ip_au) * (Eb_au/κ^3))
    @test isapprox(factor(tl, Eb), f_expected; rtol=1e-6)

    # (h) Zhang argument is capped at 4.5 E_b (factor frozen beyond).
    @test isapprox(factor(zh, 5.0Eb), factor(zh, 4.5Eb); rtol=1e-6)

    # (i) Unknown-species guard for :zhang via raw-ip constructor (no coefficients given).
    @test_throws ErrorException Ionisation.IonRatePPT(Ip, λ0, Z, 0; bsi=:zhang)

    # (extra) Molecular species must be rejected for any correction.
    @test_throws ErrorException Ionisation.IonRatePPT(:N2, λ0; bsi=:tonglin)
    @test_throws ErrorException Ionisation.IonRatePPT(:O2, λ0; bsi=:zhang)
    @test_throws ErrorException Ionisation.ionrate_PPT(:N2, λ0, Eb; bsi=:tonglin)
    @test_throws ErrorException Ionisation.IonRatePPTAccel(:O2, λ0; cache=false, bsi=:tonglin)

    # (j) The correction must propagate through the accelerated/cached rate path, and the
    #     three settings must yield different interpolants (→ different cache files).
    #     cache=false avoids polluting the cache dir; Emax covers the tested fields.
    Emax = 4.5Eb
    acc_none = Ionisation.IonRatePPTAccel(:He, λ0; cache=false, bsi=:none, Emax)
    acc_tl   = Ionisation.IonRatePPTAccel(:He, λ0; cache=false, bsi=:tonglin, Emax)
    acc_zh   = Ionisation.IonRatePPTAccel(:He, λ0; cache=false, bsi=:zhang, Emax)
    @test acc_tl.spline.y != acc_none.spline.y
    @test acc_zh.spline.y != acc_none.spline.y
    @test acc_tl.spline.y != acc_zh.spline.y
    # accelerated correction roughly matches the direct functor at E_b
    @test isapprox(acc_tl(Eb), tl(Eb); rtol=1e-2)
    @test isapprox(acc_zh(Eb), zh(Eb); rtol=1e-2)
end
