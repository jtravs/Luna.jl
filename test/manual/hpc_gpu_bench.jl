# =============================================================================
# CPU-vs-GPU benchmark of the fixed-quadrature modal transform on production-like cases.
#
# For every case and every requested array type this script
#   1. builds the propagation through the simple interface (prop_capillary_args),
#   2. times ONE isolated nonlinear transform call (the RHS) — the field-independent cost
#      of the fixed rule — as the median of `nrep` synchronised calls,
#   3. runs a (short) propagation and reports wall-clock, steps, ms/step, the transform's
#      share of a step (6 RK45 stages + 1 call for the mode-error statistic = 7 calls) and
#      the peak device memory,
#   4. checks the GPU result against the CPU one (relative L2 of the final field), and
#   5. times the computational primitives of one RHS at the same shapes (batched FFTs,
#      synthesis/projection GEMMs, an ionisation-rate-like broadcast, the prefix-sum scan,
#      the doubled-grid Raman FFTs) so the measured RHS can be compared with the sum of its
#      parts: a large gap is launch latency / host syncs (GPU) or threading overhead (CPU).
#
# Cases (see `CASES` below; select with --cases=name,name):
#   vuv     the RDW VUV example (He gradient, 800 nm, 4 HE1m modes, PPT plasma, nt≈8k)
#   ctc     a CtC-type H2 supercontinuum (16 µm, 515 nm, 40 bar, 6 modes, 4 ps window:
#           Raman + ADK plasma + Kerr on a large time grid, ~2^15 samples)
#   vector  a fully vectorial set (HE11 x+y, TE01, TM01, HE21) with plasma, 2-D quadrature
#   bignt   the VUV case on a 4× longer time window (nt ≈ 2^15): the bandwidth-bound regime
#
# Usage:
#   julia --project=<env> -t <threads> test/manual/hpc_gpu_bench.jl [options]
#     --arraytypes=cpu,cuda    (default: cpu, plus cuda if CUDA is functional)
#     --cases=vuv,ctc,vector,bignt   (default: all)
#     --lengths=vuv:0.3,ctc:0.02,vector:0.1,bignt:0.3   fibre length [m] propagated per
#                              case for the ms/step measurement (defaults as shown: a few
#                              hundred to ~1000 steps each; the CtC case takes ~50 steps/mm).
#                              The per-step numbers do not depend on it. (--scale=f, the
#                              old form, is still accepted: min(f × full length, default).)
#     --nrep=20                repetitions for the isolated RHS timing
#     --fftw=measure|patient|estimate   FFTW planning rigour on the CPU (default measure:
#                              PATIENT plans a fresh (2·nto × npts) batched Raman shape for
#                              many minutes on first use; production runs amortise that)
#     --primitives=0           skip the primitives micro-benchmark
#     --out=<csv path>         also append the summary rows to a CSV file
#
# Order of work: first the isolated RHS timings and the primitives for every case and array
# type (fast — the key numbers), then the propagations. Everything printed is flushed, so a
# job killed by the wall-time limit still leaves the tables in the .output file (Julia
# buffers stdout when it is redirected to a file).
#
# The results table is printed at the end (Markdown), so it can be pasted straight into
# the developer guide.
# =============================================================================
using Luna
import Luna: Interface, Maths, Utils, PhysData, Modes, Capillary, Nonlinear
import LinearAlgebra: mul!, norm
import FFTW
using Printf, Statistics, Dates

# ----------------------------------------------------------------------------- options
opts = Dict{String, String}()
for a in ARGS
    m = match(r"^--([a-z]+)=(.*)$", a)
    isnothing(m) && error("unrecognised argument $a")
    opts[m.captures[1]] = m.captures[2]
end
lengths = Dict("vuv" => 0.3, "ctc" => 0.02, "vector" => 0.1, "bignt" => 0.3)
# --scale (the old interface: a fraction of each case's full length) is still accepted for
# a job script submitted before --lengths existed; --lengths overrides it per case
if haskey(opts, "scale")
    sc = parse(Float64, opts["scale"])
    fulllengths = Dict("vuv" => 1.5, "ctc" => 1.5, "vector" => 0.5, "bignt" => 1.5)
    for (k, L) in fulllengths; lengths[k] = min(sc*L, lengths[k]); end
end
for kv in split(get(opts, "lengths", ""), ","; keepempty=false)
    k, v = split(kv, ":"); lengths[String(k)] = parse(Float64, v)
end
nrep = parse(Int, get(opts, "nrep", "20"))
doprim = get(opts, "primitives", "1") != "0"
csvpath = get(opts, "out", "")
fftwmode = get(opts, "fftw", "measure")
Luna.set_fftw_mode(Symbol(fftwmode))

cuda_ok = try
    Luna.resolve_arraytype(:cuda); true
catch e
    @info "CUDA not available: $(sprint(showerror, e))"; false
end
arraytypes = if haskey(opts, "arraytypes")
    Symbol.(split(opts["arraytypes"], ","))
else
    cuda_ok ? [:cpu, :cuda] : [:cpu]
end
casenames = haskey(opts, "cases") ? String.(split(opts["cases"], ",")) :
                                     ["vuv", "ctc", "vector", "bignt"]

println("=== Luna fixed-quadrature modal transform: CPU/GPU benchmark ===")
println("host: $(gethostname())  date: $(now())  julia $(VERSION)  threads: $(Threads.nthreads())")
println("FFTW planning: $fftwmode, FFTW threads: $(FFTW.get_num_threads())  arraytypes: $arraytypes  cases: $casenames")
println("propagation lengths [m]: ", join(["$k=$(lengths[k])" for k in casenames if haskey(lengths, k)], ", "))
flush(stdout)
if cuda_ok
    CUDA = Base.require(Base.PkgId(Luna.CUDA_UUID, "CUDA"))
    Base.invokelatest(getfield(CUDA, :versioninfo))
    dev = Base.invokelatest(getfield(CUDA, :device))
    println("GPU: ", Base.invokelatest(getfield(CUDA, :name), dev))
end
println()

# ----------------------------------------------------------------------------- cases
# `args`: positional prop_capillary arguments; `kw`: keywords; `flength` is the full length
# (the benchmark propagates the per-case `lengths`).
const CASES = Dict(
    "vuv" => (args=(125e-6, 1.5, :He, (0.8, 0)),
              kw=(λ0=800e-9, τfwhm=7.5e-15, energy=275e-6, modes=4, trange=400e-15,
                  λlims=(90e-9, 4e-6), shotnoise=false, saveN=11)),
    "ctc" => (args=(16e-6, 1.5, :H2, 40.0),
              kw=(λ0=515e-9, τfwhm=400e-15, energy=10e-6, modes=6, trange=4e-12,
                  λlims=(170e-9, 4e-6), loss=false, shotnoise=false, saveN=11)),
    "vector" => (args=(125e-6, 0.5, :Ar, 1.0),
                 kw=(λ0=800e-9, τfwhm=10e-15, energy=150e-6,
                     modes=(:HE11, :TE01, :TM01, :HE21), polarisation=:circular,
                     trange=300e-15, λlims=(150e-9, 3e-6), shotnoise=false, saveN=11,
                     nr=64, nθ=16)),
    "bignt" => (args=(125e-6, 1.5, :He, (0.8, 0)),
                kw=(λ0=800e-9, τfwhm=7.5e-15, energy=275e-6, modes=4, trange=1600e-15,
                    λlims=(90e-9, 4e-6), shotnoise=false, saveN=11)),
)

# ----------------------------------------------------------------------------- helpers
sync() = Luna.device_synchronize()
gib(x) = x/2^30
devfree() = (s = Luna.device_memory_status(); isnothing(s) ? NaN : s[1])

"Median wall time (ms) of `f()` over `n` synchronised repetitions after one warm-up."
function timeit(f; n=nrep)
    f(); sync()
    ts = map(1:n) do _
        t0 = time_ns(); f(); sync(); (time_ns() - t0)/1e6
    end
    median(ts)
end

function shortened(case, name)
    args = (case.args[1], lengths[name], case.args[3:end]...)
    (; args, kw=case.kw)
end

"Build a case (arraytype resolved), returning the Luna.run arguments and the transform."
function build(case, arraytype)
    A = Luna.resolve_arraytype(arraytype)
    Base.invokelatest(Interface.prop_capillary_args, case.args...; case.kw..., arraytype=A)
end

results = NamedTuple[]
prims = NamedTuple[]
built = Dict{Tuple{String, Symbol}, Any}()

# ----------------------------------------------------------- phase A: RHS and primitives
println("=" ^ 78); println("Phase A: isolated RHS and primitives"); flush(stdout)
for name in casenames
    haskey(CASES, name) || (println("unknown case $name, skipping"); continue)
    case = shortened(CASES[name], name)
    for arraytype in arraytypes
        Luna.device_reclaim()
        Eω, grid, linop, transform, FT, output = build(case, arraytype)
        nt, nto = length(grid.t), length(grid.to)
        nmodes = size(Eω, 2)
        npts = length(transform.quad); npol = transform.ts.npol
        resps = join(string.(nameof.(typeof.(transform.resp_eval))), ",")
        nl = similar(Eω)
        trhs = Base.invokelatest(timeit, () -> Base.invokelatest(transform, nl, Eω, 0.0))
        @printf("%-7s %-5s nt=%d nto=%d modes=%d nodes=%d npol=%d  RHS %.3f ms  (%.2e points/RHS; %s)\n",
                name, arraytype, nt, nto, nmodes, npts, npol, trhs, nto*npol*npts, resps)
        flush(stdout)
        built[(name, arraytype)] = (; Eω, grid, linop, transform, FT, output, nt, nto, nmodes, npts, npol, trhs)
        if doprim
            A = Luna.resolve_arraytype(arraytype)
            T = eltype(transform.Et)
            Emt = Luna.device_zeros(A, T, (nto, nmodes)) # time-domain modal field
            Emω = Luna.device_zeros(A, ComplexF64, (nto ÷ 2 + 1, nmodes))
            Et = Luna.device_zeros(A, T, (nto, npol*npts))
            S = Luna.device_zeros(A, T, (nmodes, npol*npts)); Wp = Luna.device_zeros(A, T, (npol*npts, nmodes))
            R = Luna.device_zeros(A, Float64, (nto, npts))
            tmp = Maths.scan_scratch(R)
            fill!(Emt, 0.1); fill!(S, 0.3); fill!(Wp, 0.2); fill!(Et, 1e10); fill!(R, 0.5)
            pl = Base.invokelatest(Utils.plan_rfft_backend, Emt, 1); ipl = inv(pl)
            t_fft = Base.invokelatest(timeit, () -> (mul!(Emω, pl, Emt); mul!(Emt, ipl, Emω)))
            t_gemm = Base.invokelatest(timeit, () -> (mul!(Et, Emt, S); mul!(Emt, Et, Wp)))
            Ep = reshape(Et, nto, npol, npts); E1 = view(Ep, :, 1, :)
            # an ionisation-rate-like elementwise pass (exp of a reciprocal), threaded in
            # chunks on the host as the pointwise responses are, one kernel on a device
            t_rate = Base.invokelatest(timeit, () -> (Utils.tchunks(R, E1) do R, E1
                                                          @. R = exp(-1e10/abs(E1))
                                                      end; nothing))
            R2 = similar(R)
            # the cumulative integrals: on the host the batched plasma runs the sequential
            # trapezoid rule per column (threaded over columns), on a device the scan
            scan3! = if Utils.isdevice(R)
                () -> (Maths.scan!(R2, R, tmp); Maths.scan!(R, R2, tmp); Maths.scan!(R2, R, tmp))
            else
                () -> for _ in 1:3
                    Utils.tforeach(npts; ntotal=length(R), minlen=4nto) do i
                        Maths.cumtrapz!(view(R2, :, i), view(R, :, i), 1e-17)
                    end
                end
            end
            t_scan = Base.invokelatest(timeit, scan3!)
            B = Luna.device_zeros(A, Float64, (2nto, npts)); Bω = Luna.device_zeros(A, ComplexF64, (nto+1, npts))
            fill!(B, 1.0)
            plB = Base.invokelatest(Utils.plan_rfft_backend, B, 1); iplB = inv(plB)
            t_raman = Base.invokelatest(timeit, () -> (mul!(Bω, plB, B); mul!(B, iplB, Bω)))
            hasraman = any(r -> r isa Nonlinear.RamanPolarFieldBatched, transform.resp_eval)
            total = t_fft + t_gemm + t_rate + t_scan + (hasraman ? t_raman : 0.0)
            @printf("        primitives: modal FFT pair %.3f, GEMM pair %.3f, rate pass %.3f, 3 scans %.3f, Raman FFT pair %.3f%s ms → sum %.3f vs RHS %.3f\n",
                    t_fft, t_gemm, t_rate, t_scan, t_raman, hasraman ? "" : " (not in RHS)", total, trhs)
            flush(stdout)
            push!(prims, (; case=name, arraytype, t_fft, t_gemm, t_rate, t_scan, t_raman, total, trhs))
            Emt = Emω = Et = S = Wp = R = R2 = B = Bω = tmp = nothing
            GC.gc(); Luna.device_reclaim()
        end
    end
end

# ---------------------------------------------------------------- phase B: propagations
println("=" ^ 78); println("Phase B: propagations"); flush(stdout)
for name in casenames
    haskey(CASES, name) || continue
    reffield = nothing
    for arraytype in arraytypes
        b = built[(name, arraytype)]
        Luna.device_reclaim()
        free0 = devfree()
        t0 = time()
        Base.invokelatest(Luna.run, b.Eω, b.grid, b.linop, b.transform, b.FT, b.output;
                          status_period=120, allow_device_stats=true)
        wall = time() - t0
        st = b.output["stats"]
        nsteps = length(st["z"])
        peakmem = isnan(free0) ? NaN : gib(free0 - devfree())
        Efinal = Array(b.output["Eω"][:, :, end])
        rel = isnothing(reffield) ? 0.0 : norm(Efinal .- reffield)/norm(reffield)
        arraytype == first(arraytypes) && (reffield = Efinal)
        @printf("%-7s %-5s propagation %.3f m: %.1f s, %d steps, %.2f ms/step; 7×RHS = %.2f ms (%.0f%% of step)",
                name, arraytype, lengths[name], wall, nsteps, 1e3*wall/nsteps, 7b.trhs, 100*7b.trhs/(1e3*wall/nsteps))
        isnan(peakmem) || @printf(", device memory %.2f GiB", peakmem)
        @printf("; final field vs %s: rel L2 %.2e; peak ρe %.2e m^-3\n", first(arraytypes), rel,
                maximum(st["electrondensity"]))
        flush(stdout)
        push!(results, (; case=name, arraytype, b.nt, b.nto, b.nmodes, b.npts, b.npol, trhs=b.trhs, wall,
                        nsteps, msstep=1e3*wall/nsteps, share=7b.trhs/(1e3*wall/nsteps), rel, peakmem))
        built[(name, arraytype)] = nothing
        GC.gc(); Luna.device_reclaim()
    end
end

# ----------------------------------------------------------------------------- summary
println("\n=== Summary (Markdown) ===")
println("| case | array | nt/nto | modes | nodes×pol | RHS ms | steps | ms/step | 7×RHS/step | rel L2 vs $(first(arraytypes)) | device GiB |")
println("|---|---|---|---|---|---|---|---|---|---|---|")
for r in results
    @printf("| %s | %s | %d/%d | %d | %d×%d | %.3f | %d | %.2f | %.0f%% | %.1e | %s |\n",
            r.case, r.arraytype, r.nt, r.nto, r.nmodes, r.npts, r.npol, r.trhs, r.nsteps,
            r.msstep, 100r.share, r.rel, isnan(r.peakmem) ? "-" : @sprintf("%.2f", r.peakmem))
end
if doprim
    println("\n| case | array | FFT pair | GEMM pair | rate pass | 3 scans | Raman FFTs | sum | RHS |")
    println("|---|---|---|---|---|---|---|---|---|")
    for p in prims
        @printf("| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |\n",
                p.case, p.arraytype, p.t_fft, p.t_gemm, p.t_rate, p.t_scan, p.t_raman, p.total, p.trhs)
    end
end
# speed-ups
if length(arraytypes) > 1
    println("\nGPU/CPU speed-up (RHS, ms/step):")
    for name in casenames
        rc = filter(r -> r.case == name && r.arraytype == :cpu, results)
        rd = filter(r -> r.case == name && r.arraytype != :cpu, results)
        (isempty(rc) || isempty(rd)) && continue
        @printf("  %-7s RHS %.2f×   step %.2f×\n", name, rc[1].trhs/rd[1].trhs, rc[1].msstep/rd[1].msstep)
    end
end
if !isempty(csvpath)
    open(csvpath, "a") do io
        for r in results
            println(io, join([gethostname(), string(now()), Threads.nthreads(), r.case, r.arraytype,
                              r.nt, r.nto, r.nmodes, r.npts, r.npol, r.trhs, r.wall, r.nsteps,
                              r.msstep, r.share, r.rel, r.peakmem], ","))
        end
    end
    println("appended $(length(results)) rows to $csvpath")
end
println("=== done ==="); flush(stdout)
