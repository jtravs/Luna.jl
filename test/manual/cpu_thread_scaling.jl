# CPU thread-scaling point for the fixed-quadrature modal transform. Run once per thread
# count in a fresh process (the thread count is fixed at start-up):
#
#     for n in 1 2 4 8 16 32; do
#         JULIA_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n julia --project test/manual/cpu_thread_scaling.jl
#     done
#
# Each invocation sets FFTW's thread count to the Julia thread count (so a point emulates
# an allocation of that many cores, whatever the machine has) and prints one flushed line:
#   the isolated RHS time of the RDW VUV case (nto 8192, 4 modes, PPT plasma) and of the
#   CtC-type H2 case (nto 65536, 6 modes, Kerr + ADK plasma + Raman), one call of the
#   z-dependent linear operator and of the default statistics of the VUV case, and the
#   ms/step of a short VUV propagation (--length, default 0.05 m ≈ 600 steps).
# The RHS lines show how the transform's own threading scales (plasma columns, batched
# FFTs, GEMMs); linop/stats/ms-per-step show where the rest of the step caps it.
# Optional --out=<csv> appends the line to a file.
using Luna
import Luna: Interface, Stats
import LinearAlgebra: mul!
using Printf, Statistics
import FFTW

opts = Dict{String, String}()
for a in ARGS
    m = match(r"^--([a-z]+)=(.*)$", a)
    isnothing(m) && error("unrecognised argument $a")
    opts[m.captures[1]] = m.captures[2]
end
len = parse(Float64, get(opts, "length", "0.05"))
csvpath = get(opts, "out", "")
Luna.set_fftw_mode(:measure)
Luna.set_fftw_threads(Threads.nthreads()) # emulate an allocation of nthreads cores
nthr = Threads.nthreads()

med(f; n=15) = (f(); median([(@elapsed f()) for _ in 1:n])*1e3)

# --- VUV: RHS, linear operator, statistics, short propagation ---------------------------
Eω, grid, linop, t, FT, out = Interface.prop_capillary_args(125e-6, len, :He, (0.8, 0);
        λ0=800e-9, τfwhm=7.5e-15, energy=275e-6, modes=4, trange=400e-15, λlims=(90e-9, 4e-6),
        shotnoise=false, saveN=3, nr=64)
nl = similar(Eω)
t_rhs_vuv = med(() -> t(nl, Eω, 0.01))
t_linop = med(() -> linop(nl, 0.01))
t_stats = med(() -> out.statsfun(Eω, 0.01, 1e-4))
t0 = time()
Luna.run(Eω, grid, linop, t, FT, out; status_period=300)
wall = time() - t0
nsteps = length(out["stats"]["z"])
msstep = 1e3*wall/nsteps

# --- CtC-type: RHS only ------------------------------------------------------------------
Eω2, grid2, linop2, t2, FT2, out2 = Interface.prop_capillary_args(16e-6, 0.01, :H2, 40.0;
        λ0=515e-9, τfwhm=400e-15, energy=10e-6, modes=6, trange=4e-12, λlims=(170e-9, 4e-6),
        loss=false, shotnoise=false, saveN=3)
nl2 = similar(Eω2)
t_rhs_ctc = med(() -> t2(nl2, Eω2, 0.0); n=7)

line = @sprintf("threads=%2d fftw=%2d blas=%s | RHS vuv %.3f ms | RHS ctc %.2f ms | linop %.3f ms | stats %.2f ms | vuv %.3f m: %d steps, %.2f ms/step (7×RHS+6×linop = %.1f%%)",
                nthr, FFTW.get_num_threads(), get(ENV, "OPENBLAS_NUM_THREADS", "?"),
                t_rhs_vuv, t_rhs_ctc, t_linop, t_stats, len, nsteps, msstep,
                100*(7t_rhs_vuv + 6t_linop)/msstep)
println(line); flush(stdout)
if !isempty(csvpath)
    open(csvpath, "a") do io
        println(io, join([gethostname(), nthr, FFTW.get_num_threads(), t_rhs_vuv, t_rhs_ctc,
                          t_linop, t_stats, len, nsteps, msstep], ","))
    end
end
