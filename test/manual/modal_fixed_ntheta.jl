# Manual validation: convergence of the full 2-D (r, θ) fixed rule with the number of
# azimuthal nodes nθ, for a fully vectorial mode set (HE11 x+y, TE01, TM01, HE21 x+y) with
# Kerr and plasma, on a frozen field from a short strongly ionising propagation. Kerr is a
# trigonometric polynomial in θ and exact once nθ ≥ 4 h_max + 1 (= 5 here, h_max = 1);
# the plasma polarisation is not polynomial, so its θ convergence is only spectral for a
# smooth rate — this is the study that justifies the nθ default. Reference: nθ = 128.
#
#     julia --project -t 8 test/manual/modal_fixed_ntheta.jl
using Luna
import Luna: Interface, NonlinearRHS, Nonlinear, PhysData
import Luna.Interface: Pulses
using Printf, LinearAlgebra
Luna.set_fftw_mode(:measure)

# A field with strong θ dependence: linearly polarised HE11 plus HE21 and TM01 excited
# directly (a circularly polarised HE11 alone has a θ-independent intensity, and then even
# nθ=4 is exact — not a test of anything).
args = (125e-6, 5e-3, :Ar, 1.0)
pulses = [Pulses.GaussPulse(λ0=800e-9, τfwhm=10e-15, energy=90e-6, mode=:HE11),
          Pulses.GaussPulse(λ0=800e-9, τfwhm=10e-15, energy=50e-6, mode=:HE21),
          Pulses.GaussPulse(λ0=800e-9, τfwhm=10e-15, energy=40e-6, mode=:TM01)]
common = (λ0=800e-9, pulses=pulses, modes=(:HE11, :TE01, :TM01, :HE21),
          trange=300e-15, λlims=(150e-9, 3e-6), shotnoise=false,
          saveN=3, stats_kwargs=Dict(:mode_error => false))
relerr(x, y) = norm(x .- y)/norm(y)

# frozen field: the end of a 5 mm propagation at the default rule
Eω, grid, linop, transform, FT, output = Interface.prop_capillary_args(args...; common..., nr=64, nθ=16)
Luna.run(Eω, grid, linop, transform, FT, output; status_period=60)
Ez = output["Eω"][:, :, end]; z = output["z"][end]
# the θ-dependent modes (everything but HE11) — error on those columns is the sensitive
# metric; the total is dominated by the HE11 column
modenames = string.(transform.ts.ms)
θmodes = [i for (i, m) in enumerate(transform.ts.ms) if !(m.kind == :HE && m.n == 1)]
println("modes: ", join(modenames, " | "))
println("θ-dependent columns: ", θmodes, "; input energies: HE11 90 µJ, HE21 50 µJ, TM01 40 µJ")

function projections(; nr, nθ, kerr, plasma)
    _, _, _, t, _, _ = Interface.prop_capillary_args(args...; common..., nr, nθ, kerr, plasma)
    nl = similar(Ez); t(nl, Ez, z)
    nl
end

for (label, kw) in [("Kerr only", (kerr=true, plasma=false)), ("plasma only", (kerr=false, plasma=true)),
                    ("Kerr + plasma", (kerr=true, plasma=true))]
    ref = projections(; nr=64, nθ=128, kw...)
    println("$label — relative error vs nθ=128 (nr=64), all modes / θ-dependent modes:")
    for nθ in (4, 6, 8, 12, 16, 24, 32, 48, 64)
        nl = projections(; nr=64, nθ, kw...)
        @printf("  nθ=%3d  %.2e   %.2e\n", nθ, relerr(nl, ref), relerr(nl[:, θmodes], ref[:, θmodes]))
    end
    # and the radial direction at nθ=32
    ref = projections(; nr=256, nθ=32, kw...)
    print("  radial (nθ=32) vs nr=256:")
    for nr in (16, 24, 32, 48, 64, 96, 128)
        nl = projections(; nr, nθ=32, kw...)
        @printf("  nr=%d %.1e", nr, relerr(nl, ref))
    end
    println()
    flush(stdout)
end
println("peak field on the frozen field: ", maximum(abs.(transform.Et)), " V/m; peak electron density in the run: ",
        maximum(output["stats"]["electrondensity"]))
