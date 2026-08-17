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
using Printf, LinearAlgebra
Luna.set_fftw_mode(:measure)

args = (125e-6, 5e-3, :Ar, 1.0)
common = (λ0=800e-9, τfwhm=10e-15, energy=150e-6, modes=(:HE11, :TE01, :TM01, :HE21),
          polarisation=:circular, trange=300e-15, λlims=(150e-9, 3e-6), shotnoise=false,
          saveN=3, stats_kwargs=Dict(:mode_error => false))
relerr(x, y) = norm(x .- y)/norm(y)

# frozen field: the end of a 5 mm propagation at the default rule
Eω, grid, linop, transform, FT, output = Interface.prop_capillary_args(args...; common..., nr=64, nθ=16)
Luna.run(Eω, grid, linop, transform, FT, output; status_period=60)
Ez = output["Eω"][:, :, end]; z = output["z"][end]
λ = PhysData.wlfreq.(grid.ω)
band = 150e-9 .<= λ .<= 250e-9

function projections(; nr, nθ, kerr, plasma)
    _, _, _, t, _, _ = Interface.prop_capillary_args(args...; common..., nr, nθ, kerr, plasma)
    nl = similar(Ez); t(nl, Ez, z)
    nl
end

for (label, kw) in [("Kerr only", (kerr=true, plasma=false)), ("plasma only", (kerr=false, plasma=true)),
                    ("Kerr + plasma", (kerr=true, plasma=true))]
    ref = projections(; nr=64, nθ=128, kw...)
    println("$label — relative error vs nθ=128 (nr=64), all / 150–250 nm band:")
    for nθ in (4, 6, 8, 12, 16, 24, 32, 48, 64)
        nl = projections(; nr=64, nθ, kw...)
        @printf("  nθ=%3d  %.2e   %.2e\n", nθ, relerr(nl, ref), relerr(nl[band, :], ref[band, :]))
    end
    # and the radial direction at nθ=32
    ref = projections(; nr=256, nθ=32, kw...)
    print("  radial (nθ=32) vs nr=256:")
    for nr in (16, 24, 32, 48, 64, 96, 128)
        nl = projections(; nr, nθ=32, kw...)
        @printf("  nr=%d %.1e", nr, relerr(nl, ref))
    end
    println()
end
println("peak field on the frozen field: ", maximum(abs.(transform.Et)), " V/m; peak electron density in the run: ",
        maximum(output["stats"]["electrondensity"]))
