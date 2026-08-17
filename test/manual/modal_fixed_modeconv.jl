# Manual validation: convergence with the number of modes (4, 6, 8 HE1m) of the RDW VUV
# example, with and without plasma, on the fixed-quadrature transform. With the spatial
# quadrature error now fixed and small (smooth PPT rate), differences between mode counts
# are mode truncation, not quadrature noise. Compares the dispersive-wave band energy
# (100–140 nm), the band spectrum, total energy, peak electron density and wall-clock.
# nr scales with the highest radial order (default 64 is used throughout — ~8 nodes per
# radial order at 8 modes; the embedded error estimate is printed as a check).
#
#     julia --project -t 8 test/manual/modal_fixed_modeconv.jl [flength]
using Luna
import Luna: Fields, PhysData
using Printf, LinearAlgebra, Statistics
Luna.set_fftw_mode(:measure)

flength = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.5
args = (125e-6, flength, :He, (0.8, 0))
band = (100e-9, 140e-9)
common = (λ0=800e-9, τfwhm=7.5e-15, energy=275e-6, trange=400e-15, λlims=(90e-9, 4e-6),
          shotnoise=false, saveN=11, stats_kwargs=Dict(:error_window => band))

function observables(o)
    grid = o["grid"]
    Eω = o["Eω"][:, :, end]
    _, energyfunω = Fields.energyfuncs(Luna.Grid.RealGrid(grid))
    λ = PhysData.wlfreq.(grid["ω"])
    idcs = band[1] .<= λ .<= band[2]
    Eband = sum(energyfunω(Eω[:, i] .* idcs) for i in axes(Eω, 2))
    Etot = sum(energyfunω(Eω[:, i]) for i in axes(Eω, 2))
    # spectrum in the band summed over modes (a mode-count independent observable)
    Iband = vec(sum(abs2.(Eω[idcs, :]); dims=2))
    (; Eband, Etot, Iband)
end

results = Dict{Tuple{Int, Bool}, Any}()
for plasma in (false, true), modes in (4, 6, 8)
    t0 = time()
    o = prop_capillary(args...; common..., modes, plasma, status_period=120)
    dt = time() - t0
    st = o["stats"]
    obs = observables(o)
    results[(modes, plasma)] = (; obs, dt, nsteps=length(st["z"]),
        err=maximum(st["transverse_integral_error_rel"]),
        errwin=maximum(filter(isfinite, st["transverse_integral_error_rel_window"])),
        peakρ=plasma ? maximum(st["electrondensity"]) : 0.0)
    @printf("modes=%d plasma=%-5s  %6.1f s  %5d steps  E_band %.4e J (%.3f%%)  peak ρe %.2e  quad err %.1e (band %.1e)\n",
            modes, plasma, dt, length(st["z"]), obs.Eband, 100obs.Eband/obs.Etot,
            results[(modes, plasma)].peakρ, results[(modes, plasma)].err, results[(modes, plasma)].errwin)
end
println("\nRelative to 8 modes:")
for plasma in (false, true)
    ref = results[(8, plasma)].obs
    for modes in (4, 6)
        ob = results[(modes, plasma)].obs
        @printf("  plasma=%-5s modes=%d: ΔE_band/E_band=%+.2e  |ΔI_band|/|I_band|=%.2e  ΔE_tot/E_tot=%+.2e\n",
                plasma, modes, (ob.Eband - ref.Eband)/ref.Eband, norm(ob.Iband .- ref.Iband)/norm(ref.Iband),
                (ob.Etot - ref.Etot)/ref.Etot)
    end
end
