# Rehearsal of a fast parameter scan (energy × pressure) of the RDW VUV example through
# Luna's real scan machinery, with the settings that keep host work per point minimal
# ("lean": no mode-error statistic, statistics every 10th step, saveN=2), and per-point
# timing split into setup / propagation / output. Prints the mean time per point and an
# extrapolation to a 20 × 20 scan, and finally extracts the reduced results (VUV band energy,
# peak electron density) from the HDF5 outputs with Processing.scanproc — the "collected"
# quantities one actually wants off the device.
#
# Configuration by environment variables (so that Luna's own command-line scan options
# `--batch k,i` / `--range a:b` / `--queue n` still work — that is how several processes
# share one GPU):
#   LUNA_ARRAYTYPE = cpu | cuda      (default cpu)
#   SCAN_N         = points per axis  (default 2 → 2 × 2 = 4 points)
#   SCAN_FLENGTH   = fibre length [m] (default 1.5; the full example)
#   SCAN_STATS     = lean | default   (default lean)
#   SCAN_NR        = radial nodes     (default 32)
#   SCAN_OUT       = output directory (default ./scan_rehearsal_<arraytype>)
#
#   LUNA_ARRAYTYPE=cuda julia --project test/manual/gpu_scan_rehearsal.jl
#   # 4 processes sharing the GPU, each a quarter of the points:
#   for i in 1 2 3 4; do LUNA_ARRAYTYPE=cuda julia --project test/manual/gpu_scan_rehearsal.jl --batch 4,$i & done; wait
using Luna
import Luna: Processing, Fields, PhysData, Interface, Utils
using Printf, Statistics

arraytype = Symbol(get(ENV, "LUNA_ARRAYTYPE", "cpu"))
N = parse(Int, get(ENV, "SCAN_N", "2"))
flength = parse(Float64, get(ENV, "SCAN_FLENGTH", "1.5"))
statsmode = get(ENV, "SCAN_STATS", "lean")
nr = parse(Int, get(ENV, "SCAN_NR", "32"))
outputdir = get(ENV, "SCAN_OUT", joinpath(pwd(), "scan_rehearsal_$(arraytype)"))
Luna.set_fftw_mode(:measure)

# the RDW VUV example, scanned in energy and (input) pressure of the gradient
energies = collect(range(200e-6, 300e-6; length=N))
pressures = collect(range(0.6, 1.0; length=N))
scan = Scan("gpu_scan_rehearsal"; energy=energies, pressure=pressures)
band = (100e-9, 140e-9)
lean = statsmode == "lean"
common = (λ0=800e-9, τfwhm=7.5e-15, modes=4, trange=400e-15, λlims=(90e-9, 4e-6),
          shotnoise=false, saveN=lean ? 2 : 51, nr,
          stats_kwargs=lean ? Dict(:mode_error => false) : Dict(:error_window => band),
          stats_period=lean ? 10 : 1)

println("scan rehearsal: $(arraytype), $(N)×$(N) points, flength=$(flength) m, stats=$(statsmode), nr=$(nr), out=$(outputdir)")
flush(stdout)
tsetup = Float64[]; tprop = Float64[]; tout = Float64[]; nsteps = Int[]
runscan(scan) do scanidx, energy, pressure
    t0 = time()
    # setup and propagation timed separately (prop_capillary = prop_capillary_args + run)
    args = Interface.prop_capillary_args(125e-6, flength, :He, (pressure, 0); common..., energy,
                                         arraytype, scan, scanidx, filepath=outputdir)
    Luna.device_synchronize()
    t1 = time()
    Base.invokelatest(Luna.run, args...; status_period=600,
                      allow_device_stats=Utils.isdevice(args[1]))
    Luna.device_synchronize()
    t2 = time()
    output = args[end]
    # statistics are recorded every stats_period-th step, so the step count is estimated
    push!(tsetup, t1 - t0); push!(tprop, t2 - t1)
    push!(nsteps, (length(output["stats"]["z"]) - 1)*common.stats_period + 1)
    push!(tout, time() - t2)
    @printf("  point %d (E=%.0f µJ, p=%.2f bar): setup %.2f s, propagation %.1f s (≈%d steps, %.2f ms/step), output %.2f s\n",
            scanidx, 1e6energy, pressure, tsetup[end], tprop[end], nsteps[end], 1e3tprop[end]/nsteps[end], tout[end])
    flush(stdout)
    nothing
end
if !isempty(tprop)
    per = mean(tsetup .+ tprop .+ tout)
    @printf("\nper point: setup %.2f s + propagation %.1f s + output %.2f s = %.1f s (first point includes compilation)\n",
            mean(tsetup), mean(tprop), mean(tout), per)
    if length(tprop) > 1
        per2 = mean((tsetup .+ tprop .+ tout)[2:end])
        @printf("excluding the first point: %.1f s per point → 400 points ≈ %.1f min in one process\n", per2, 400per2/60)
    end
end

# the reduced results, off the HDF5 outputs (this is what a scan is for). With several
# --batch processes writing the same directory this may run before the others finish, so
# a failure here is only a warning.
if isdir(outputdir)
  try
    Eband, ρmax = Processing.scanproc(outputdir) do o
        grid = o["grid"]; Eω = o["Eω"][:, :, end]
        _, energyfunω = Fields.energyfuncs(Luna.Grid.RealGrid(grid))
        λ = PhysData.wlfreq.(grid["ω"]); idcs = band[1] .<= λ .<= band[2]
        sum(energyfunω(Eω[:, i] .* idcs) for i in axes(Eω, 2)), maximum(o["stats"]["electrondensity"])
    end
    println("VUV band energy [µJ] (energy × pressure):"); display(1e6 .* Eband); println()
    println("peak electron density [m^-3]:"); display(ρmax); println()
  catch e
    @warn "could not collect the scan outputs yet (other processes still writing?): $e"
  end
end
