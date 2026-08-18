# Full-length production-like propagations of the two core cases on one array type, with
# wall time, step count, ms/step, resident device memory and a physics checksum (VUV /
# UV band energy), for the "how long does a real run take on this GPU" question:
#   vuv   RDW VUV example, 1.5 m, He gradient, 4 modes, PPT plasma — at nr=32 and nr=64,
#         with lean statistics (no mode-error statistic, stats every 10th step) and, for
#         nr=32, once more with the default statistics (shows the host statistics' share)
#   ctc   CtC-type H2 supercontinuum, 515 nm, 40 bar, 6 modes, Kerr + ADK + Raman:
#         CTC_LENGTH m (default 0.3; the full case is 1.5 m and ~50 steps/mm — FULL=1)
#
#   LUNA_ARRAYTYPE=cuda julia --project test/manual/gpu_production_timing.jl
#   LUNA_ARRAYTYPE=cuda CTC_LENGTH=1.5 julia --project test/manual/gpu_production_timing.jl
using Luna
import Luna: Fields, PhysData
using Printf

arraytype = Symbol(get(ENV, "LUNA_ARRAYTYPE", "cuda"))
ctclength = parse(Float64, get(ENV, "CTC_LENGTH", "0.3"))
Luna.set_fftw_mode(:measure)
gib(x) = x/2^30
devfree() = (s = Luna.device_memory_status(); isnothing(s) ? NaN : s[1])

function bandenergy(o, band)
    grid = o["grid"]; Eω = o["Eω"][:, :, end]
    _, energyfunω = Fields.energyfuncs(Luna.Grid.RealGrid(grid))
    λ = PhysData.wlfreq.(grid["ω"]); idcs = band[1] .<= λ .<= band[2]
    sum(energyfunω(Eω[:, i] .* idcs) for i in axes(Eω, 2))
end

function timed(label, band; args, kw)
    Luna.device_reclaim(); free0 = devfree()
    t0 = time()
    o = prop_capillary(args...; kw..., arraytype, status_period=600)
    Luna.device_synchronize()
    wall = time() - t0
    period = get(kw, :stats_period, 1)
    nsteps = (length(o["stats"]["z"]) - 1)*period + 1
    mem = isnan(free0) ? NaN : gib(free0 - devfree())
    @printf("%-40s %7.1f s  ≈%6d steps  %6.2f ms/step  band energy %.4e J  peak ρe %.2e m^-3%s\n",
            label, wall, nsteps, 1e3wall/nsteps, bandenergy(o, band), maximum(o["stats"]["electrondensity"]),
            isnan(mem) ? "" : @sprintf("  device %.2f GiB", mem))
    flush(stdout)
end

println("production timing on $(arraytype) (first run of each case includes compilation)")
vuvargs = (125e-6, 1.5, :He, (0.8, 0))
vuvcommon = (λ0=800e-9, τfwhm=7.5e-15, energy=275e-6, modes=4, trange=400e-15,
             λlims=(90e-9, 4e-6), shotnoise=false, saveN=11)
lean = (stats_kwargs=Dict(:mode_error => false), stats_period=10)
timed("vuv 1.5 m nr=32 lean stats (warm-up)", (100e-9, 140e-9); args=vuvargs, kw=(; vuvcommon..., nr=32, lean...))
timed("vuv 1.5 m nr=32 lean stats", (100e-9, 140e-9); args=vuvargs, kw=(; vuvcommon..., nr=32, lean...))
timed("vuv 1.5 m nr=64 lean stats", (100e-9, 140e-9); args=vuvargs, kw=(; vuvcommon..., nr=64, lean...))
timed("vuv 1.5 m nr=32 default stats", (100e-9, 140e-9); args=vuvargs,
      kw=(; vuvcommon..., nr=32, stats_kwargs=Dict(:error_window => (100e-9, 140e-9))))
ctcargs = (16e-6, ctclength, :H2, 40.0)
ctccommon = (λ0=515e-9, τfwhm=400e-15, energy=10e-6, modes=6, trange=4e-12,
             λlims=(170e-9, 4e-6), loss=false, shotnoise=false, saveN=11)
timed("ctc $(ctclength) m nr=64 lean stats", (170e-9, 300e-9); args=ctcargs, kw=(; ctccommon..., lean...))
timed("ctc $(ctclength) m nr=32 lean stats", (170e-9, 300e-9); args=ctcargs, kw=(; ctccommon..., nr=32, lean...))
println("done")
