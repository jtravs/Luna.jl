# Manual validation: the RDW VUV example (examples/simple_interface/RDWemission_VUV_gradient.jl)
# with the adaptive-cubature modal transform versus the fixed-quadrature one at several node
# counts, and with a smooth (sum_integral) PPT rate. Compares weak-feature observables
# (VUV energy and duration in the 100–140 nm band, spectral phase there), step counts and
# wall-clock. Not part of the test suite: takes minutes.
#
#     julia --project -t 8 test/manual/modal_fixed_rdw.jl [flength]
using Luna
import Luna: Processing, Maths, PhysData
using Printf, LinearAlgebra, Statistics

flength = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.5
common = (λ0=800e-9, τfwhm=7.5e-15, energy=275e-6, modes=4, trange=400e-15,
          λlims=(90e-9, 4e-6), shotnoise=false, saveN=51,
          stats_kwargs=Dict(:error_window => (100e-9, 140e-9)))
band = (100e-9, 140e-9)

function observables(o)
    grid = o["grid"]
    Eω = o["Eω"][:, :, end]
    # VUV band energy (all modes) and duration in the band (fundamental mode)
    _, energyfunω = Luna.Fields.energyfuncs(Luna.Grid.RealGrid(grid))
    ω = grid["ω"]; λ = PhysData.wlfreq.(ω)
    idcs = band[1] .<= λ .<= band[2]
    Eband = sum(energyfunω(Eω[:, i] .* idcs) for i in axes(Eω, 2))
    Etot = sum(energyfunω(Eω[:, i]) for i in axes(Eω, 2))
    # spectral phase in the band (mode 1), relative to the band centre
    ϕ = unwrap(angle.(Eω[idcs, 1]))
    ϕ .-= ϕ[end÷2]
    (; Eband, Etot, ϕ, Iband=abs2.(Eω[idcs, 1]), Iall=abs2.(Eω))
end
unwrap(x) = Luna.Modes.unwrap(x)

results = Dict{String, Any}()
runs = [
    ("adaptive", (modal_integral=:adaptive,)),
    ("fixed nr=64", (modal_integral=:fixed, nr=64)),
    ("fixed nr=128", (modal_integral=:fixed, nr=128)),
    ("fixed nr=256", (modal_integral=:fixed, nr=256)),
    ("fixed nr=512", (modal_integral=:fixed, nr=512)),
    ("fixed nr=32 smoothPPT", (modal_integral=:fixed, nr=32, PPT_options=Dict(:sum_integral=>true))),
    ("fixed nr=128 smoothPPT", (modal_integral=:fixed, nr=128, PPT_options=Dict(:sum_integral=>true))),
    ("adaptive smoothPPT", (modal_integral=:adaptive, PPT_options=Dict(:sum_integral=>true))),
]
for (label, kw) in runs
    t0 = time()
    o = prop_capillary(125e-6, flength, :He, (0.8, 0); common..., kw...)
    dt = time() - t0
    obs = observables(o)
    st = o["stats"]
    results[label] = (; obs, dt, nsteps=length(st["z"]),
                        errrel=st["transverse_integral_error_rel"],
                        errwin=get(st, "transverse_integral_error_rel_window", [NaN]), o)
    @printf("%-24s  %6.1f s  %5d steps  VUV energy %.4e J (%.3f%% of total)\n",
            label, dt, length(st["z"]), obs.Eband, 100*obs.Eband/obs.Etot)
end

println("\nComparison of end-of-fibre observables (relative to 'fixed nr=512'):")
ref = results["fixed nr=512"].obs
for (label, _) in runs
    r = results[label]
    ob = r.obs
    @printf("%-24s  ΔE_VUV/E_VUV=%+.2e  |ΔI_band|/|I_band|=%.2e  |ΔI_all|/|I_all|=%.2e  rms Δϕ_band=%.3f rad  max err_win(stat)=%.2e\n",
            label, (ob.Eband - ref.Eband)/ref.Eband, norm(ob.Iband .- ref.Iband)/norm(ref.Iband),
            norm(ob.Iall .- ref.Iall)/norm(ref.Iall), sqrt(mean(abs2, ob.ϕ .- ref.ϕ)),
            maximum(filter(isfinite, r.errwin); init=NaN))
end
println("\nSmooth-PPT runs relative to 'fixed nr=128 smoothPPT':")
ref = results["fixed nr=128 smoothPPT"].obs
for label in ("fixed nr=32 smoothPPT", "adaptive smoothPPT")
    ob = results[label].obs
    @printf("%-24s  ΔE_VUV/E_VUV=%+.2e  |ΔI_band|/|I_band|=%.2e  |ΔI_all|/|I_all|=%.2e\n",
            label, (ob.Eband - ref.Eband)/ref.Eband, norm(ob.Iband .- ref.Iband)/norm(ref.Iband),
            norm(ob.Iall .- ref.Iall)/norm(ref.Iall))
end
