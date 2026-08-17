# Manual validation: energy conservation of the Kerr term under the fixed-quadrature modal
# transform versus the adaptive cubature.
#
# For an instantaneous Kerr response the nonlinear term conserves the total energy: with
# Luna's normalisation the rate of change of Σ_m ∫|E_m(ω)|² dω along z due to the
# nonlinear term is D = 2 Re Σ_m ∫ conj(E_m) nl_m dω, and for P = χ E³ this is
# ∝ ∫dA ∫dt ∂_t E · E³ = 0. A symmetric fixed quadrature rule keeps the spatial part of that
# identity exact (the effective overlap tensor is permutation-symmetric); an adaptive rule
# whose mesh changes between evaluations only to its tolerance. The time-domain part holds
# up to the aliasing of the E'·E³ product on the oversampled grid and the apodisation
# windows, identically for both transforms, so the *difference* between them isolates the
# spatial quadrature.
#
# 1. Frozen fields at several z of a short Kerr-only RDW-type propagation: |D|/E_tot per
#    unit length for fixed nr=17/33/65 and adaptive rtol=1e-3/1e-6.
# 2. Energy drift over a 20 cm loss-free Kerr-only propagation, both transforms.
#
#     julia --project -t 8 test/manual/modal_fixed_conservation.jl
using Luna
import Luna: Interface, Fields, Grid, PhysData
using Printf, LinearAlgebra
Luna.set_fftw_mode(:measure)

args = (125e-6, 0.2, :He, 1.0)
common = (λ0=800e-9, τfwhm=7.5e-15, energy=275e-6, modes=4, trange=400e-15,
          λlims=(90e-9, 4e-6), shotnoise=false, saveN=11, plasma=false, loss=false,
          stats_kwargs=Dict(:mode_error => false))

# --- 1. frozen-field energy derivative ------------------------------------------------
Eω, grid, linop, transform, FT, output = Interface.prop_capillary_args(args...; common..., nr=32)
Luna.run(Eω, grid, linop, transform, FT, output; status_period=60)
_, energyfunω = Fields.energyfuncs(grid)
runs = [("fixed nr=17", (modal_integral=:fixed, nr=17)),
        ("fixed nr=33", (modal_integral=:fixed, nr=33)),
        ("fixed nr=65", (modal_integral=:fixed, nr=65)),
        ("adaptive rtol=1e-3", (modal_integral=:adaptive, radial_integral_rtol=1e-3)),
        ("adaptive rtol=1e-6", (modal_integral=:adaptive, radial_integral_rtol=1e-6))]
transforms = [(label, Interface.prop_capillary_args(args...; common..., kw...)[4]) for (label, kw) in runs]
println("Frozen-field energy derivative of the Kerr term, |D|/E_tot (per metre):")
@printf("%-8s", "z [m]")
for (label, _) in transforms; @printf("  %-20s", label); end
println()
for iz in 2:2:length(output["z"])
    z = output["z"][iz]
    Ez = output["Eω"][:, :, iz]
    Etot = sum(energyfunω(Ez[:, m]) for m in axes(Ez, 2))
    @printf("%-8.3f", z)
    for (label, t) in transforms
        nl = similar(Ez); t(nl, Ez, z)
        # d/dz Σ_m ∫|E_m|² ∝ 2 Re Σ_m ∫ conj(E_m) nl_m: use the same spectral weights as
        # the energy functional by differentiating it. The functional is quadratic, so the
        # central difference [E(Eω + ε nl) - E(Eω - ε nl)]/(2ε) is exact up to rounding;
        # ε is chosen so that the perturbation is 1e-3 of the field (rounding floor ~1e-13).
        ε = 1e-3*norm(Ez)/norm(nl)
        D = sum(energyfunω(Ez[:, m] .+ ε .* nl[:, m]) - energyfunω(Ez[:, m] .- ε .* nl[:, m])
                for m in axes(Ez, 2))/(2ε)
        @printf("  %-20.2e", abs(D)/Etot)
    end
    println()
end

# --- 2. energy drift over the propagation ---------------------------------------------
println("\nEnergy drift over 20 cm, loss-free Kerr only (relative to the input):")
for (label, kw) in [("fixed nr=33", (modal_integral=:fixed, nr=33)),
                    ("adaptive rtol=1e-3", (modal_integral=:adaptive, radial_integral_rtol=1e-3))]
    o = prop_capillary(args...; common..., kw..., status_period=60)
    E = vec(sum(o["stats"]["energy"]; dims=1))
    @printf("  %-20s  max |ΔE|/E = %.2e   final ΔE/E = %+.2e   (%d steps)\n", label,
            maximum(abs.(E .- E[1]))/E[1], (E[end] - E[1])/E[1], length(E))
end
