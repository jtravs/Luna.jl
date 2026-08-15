# Manual validation: the RDW examples (examples/simple_interface/RDWemission_VUV_gradient.jl
# and RDWemission_DUV.jl) with the adaptive-cubature modal transform versus the
# fixed-quadrature one at several node counts, and with the smooth (sum_integral) PPT rate
# (Luna's default) versus the literal channel-sum ("kinky") one. Compares weak-feature
# observables (dispersive-wave energy and spectrum in its band, spectral phase there), step
# counts and wall-clock.
# Not part of the test suite: takes tens of minutes.
#
#     julia --project -t 8 test/manual/modal_fixed_rdw.jl [vuv|duv] [full|quick] [flength]
#
# `quick` runs only kinky nr=256/512 and smooth nr=32/128 (the model comparison);
# `full` adds the adaptive cubature and kinky nr=64/128.
using Luna
import Luna: Processing, Maths, PhysData
using Printf, LinearAlgebra, Statistics

case = length(ARGS) >= 1 ? ARGS[1] : "vuv"
mode = length(ARGS) >= 2 ? ARGS[2] : "full"
if case == "vuv"
    flength = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.5
    args = (125e-6, flength, :He, (0.8, 0))
    common = (λ0=800e-9, τfwhm=7.5e-15, energy=275e-6, modes=4, trange=400e-15,
              λlims=(90e-9, 4e-6), shotnoise=false, saveN=51,
              stats_kwargs=Dict(:error_window => (100e-9, 140e-9)))
    band = (100e-9, 140e-9)
elseif case == "duv"
    flength = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 3.0
    args = (125e-6, flength, :Ar, 80e-3)
    common = (λ0=800e-9, τfwhm=10e-15, energy=60e-6, modes=4, trange=400e-15,
              λlims=(150e-9, 4e-6), shotnoise=false, saveN=51,
              stats_kwargs=Dict(:error_window => (220e-9, 270e-9)))
    band = (220e-9, 270e-9)
else
    error("unknown case $case")
end

function observables(o)
    grid = o["grid"]
    Eω = o["Eω"][:, :, end]
    _, energyfunω = Luna.Fields.energyfuncs(Luna.Grid.RealGrid(grid))
    ω = grid["ω"]; λ = PhysData.wlfreq.(ω)
    idcs = band[1] .<= λ .<= band[2]
    Eband = sum(energyfunω(Eω[:, i] .* idcs) for i in axes(Eω, 2))
    Etot = sum(energyfunω(Eω[:, i]) for i in axes(Eω, 2))
    ϕ = unwrap(angle.(Eω[idcs, 1]))
    ϕ .-= ϕ[end÷2]
    (; Eband, Etot, ϕ, Iband=abs2.(Eω[idcs, 1]), Iall=abs2.(Eω))
end
unwrap(x) = Luna.Modes.unwrap(x)

# the smooth (sum_integral) PPT rate is Luna's default; the "kinky" runs use the literal sum
smooth = Dict(:sum_integral => true)
kinky = Dict(:sum_integral => false)
runs = [
    ("fixed nr=512 kinkyPPT", (modal_integral=:fixed, nr=512, PPT_options=kinky)),
    ("fixed nr=256 kinkyPPT", (modal_integral=:fixed, nr=256, PPT_options=kinky)),
    ("fixed nr=32 smoothPPT", (modal_integral=:fixed, nr=32, PPT_options=smooth)),
    ("fixed nr=128 smoothPPT", (modal_integral=:fixed, nr=128, PPT_options=smooth)),
]
if mode == "full"
    append!(runs, [
        ("adaptive kinkyPPT", (modal_integral=:adaptive, PPT_options=kinky)),
        ("fixed nr=64 kinkyPPT", (modal_integral=:fixed, nr=64, PPT_options=kinky)),
        ("fixed nr=128 kinkyPPT", (modal_integral=:fixed, nr=128, PPT_options=kinky)),
        ("adaptive smoothPPT", (modal_integral=:adaptive, PPT_options=smooth)),
    ])
end

results = Dict{String, Any}()
for (label, kw) in runs
    t0 = time()
    o = prop_capillary(args...; common..., kw...)
    dt = time() - t0
    obs = observables(o)
    st = o["stats"]
    results[label] = (; obs, dt, nsteps=length(st["z"]),
                        errwin=get(st, "transverse_integral_error_rel_window", [NaN]),
                        peakρ=maximum(st["electrondensity"]))
    @printf("%-24s  %6.1f s  %5d steps  band energy %.4e J (%.3f%% of total)  peak ρe %.2e m⁻³\n",
            label, dt, length(st["z"]), obs.Eband, 100*obs.Eband/obs.Etot,
            results[label].peakρ)
end

println("\nEnd-of-fibre observables relative to 'fixed nr=512 kinkyPPT':")
ref = results["fixed nr=512 kinkyPPT"].obs
for (label, _) in runs
    ob = results[label].obs
    @printf("%-24s  ΔE_band/E_band=%+.2e  |ΔI_band|/|I_band|=%.2e  |ΔI_all|/|I_all|=%.2e  rms Δϕ_band=%.3f rad\n",
            label, (ob.Eband - ref.Eband)/ref.Eband, norm(ob.Iband .- ref.Iband)/norm(ref.Iband),
            norm(ob.Iall .- ref.Iall)/norm(ref.Iall), sqrt(mean(abs2, ob.ϕ .- ref.ϕ)))
end
println("\nSmooth-PPT runs relative to 'fixed nr=128 smoothPPT':")
ref = results["fixed nr=128 smoothPPT"].obs
for label in filter(l -> occursin("smoothPPT", l) && l != "fixed nr=128 smoothPPT", first.(runs))
    ob = results[label].obs
    @printf("%-24s  ΔE_band/E_band=%+.2e  |ΔI_band|/|I_band|=%.2e  |ΔI_all|/|I_all|=%.2e\n",
            label, (ob.Eband - ref.Eband)/ref.Eband, norm(ob.Iband .- ref.Iband)/norm(ref.Iband),
            norm(ob.Iall .- ref.Iall)/norm(ref.Iall))
end
