#!/usr/bin/env bash
# =============================================================================
# GPU-only Luna tests + benchmarks on a Runpod H100/H200 pod (paid time).
#
# Follows on from test/manual/runpodcoldstart.sh, which has already put on the network
# volume: Julia and the depot (/workspace/juliaup, /workspace/julia_depot), the Luna
# checkout (/workspace/code/Luna.jl, branch modal-fixed), the dev project
# (/workspace/code/dev, with CUDA), and /workspace/env.sh (JULIA_NUM_THREADS from the
# cgroup, capped at 8 — LUNA_THREADS= overrides; JULIA_CPU_TARGET=generic). This script
# only does what is needed to run the tests and benchmarks and stores everything under
# /workspace/runs/<timestamp>/ (the persistent volume): its own log, the benchmark CSV,
# the scan outputs and per-process logs.
#
# Nothing here runs CPU benchmarks or the CPU test suite — those are done on the HPC.
#
# STEPS (every line is flushed as printed, so a partial run is still useful):
#   pkgs        guard-add the packages the test/benchmark scripts import directly
#               (no-op after the first time; the cold start's first resolve adds them)
#   tests       Luna's hardware-gated CUDA tests (LUNA_TEST_CUDA=1)            ~4 min
#   bench       transform benchmark on CUDA only, cases vuv + ctc: phase A isolated RHS
#               + primitives, phase B short propagations                       ~5–8 min
#   production  full-length runs on CUDA: RDW VUV 1.5 m (nr 32/64, lean + default
#               statistics), CtC-type H2 CTC_LENGTH m (default 0.3; FULL=1 → 1.5 m,
#               ~50 steps/mm)                                                  ~5–10 min
#   scan        scan rehearsal: 2×2 energy × pressure RDW VUV scan through Luna's scan
#               machinery with lean settings, per-point timing, reduced results; then
#               the same with 4 processes sharing the GPU (--batch 4,i)        ~3–5 min
#   Total ≈ 20–30 min (the cold start has already paid the environment).
#
# USAGE (on the pod, after coldstart; env.sh is sourced here in case this is not a login
# shell):
#   bash /workspace/code/Luna.jl/test/manual/h200_gpu_suite.sh
#   STEPS=tests,bench bash /workspace/code/Luna.jl/test/manual/h200_gpu_suite.sh
#   FULL=1 bash /workspace/code/Luna.jl/test/manual/h200_gpu_suite.sh   # + 1.5 m CtC
#   LUNA_THREADS=16 bash ...                                            # more host threads
# Variables: STEPS (default all), FULL, CTC_LENGTH, LUNA_THREADS, RUNDIR (default
# /workspace/runs/h200-suite-<timestamp>).
#
# WHAT TO LOOK FOR
#   * tests: every testset passes; "CUDA modal transform vs host rel" ≲ 1e-11, "CUDA
#     modal propagation vs host" < 1e-8, "device norm" ratios 1 ± 1e-12, LAZY_CUDA_OK=true.
#   * bench phase A: "RHS x.xxx ms" for vuv and ctc on cuda — compare with the A40 job and
#     the CPU node; the primitives line shows the RHS against the sum of its kernels.
#   * production: ms/step and wall for the full-length runs; "default stats" minus "lean
#     stats" is the host statistics' share of a step.
#   * scan: seconds per point (setup vs propagation) → the 400-point estimate; the
#     4-process run shows the throughput gain from sharing the GPU.
# =============================================================================
set -uo pipefail

[ -r /workspace/env.sh ] && source /workspace/env.sh
LUNA="${LUNA:-/workspace/code/Luna.jl}"
DEV="${DEV:-/workspace/code/dev}"
STAMP=$(date +%Y%m%d-%H%M%S)
RUNDIR="${RUNDIR:-/workspace/runs/h200-suite-$STAMP}"
FULL="${FULL:-0}"
CTC_LENGTH="${CTC_LENGTH:-$([ "$FULL" = "1" ] && echo 1.5 || echo 0.3)}"
STEPS="${STEPS:-pkgs,tests,bench,production,scan}"

mkdir -p "$RUNDIR" /workspace/logs
LOG="$RUNDIR/suite.log"
exec > >(tee -a "$LOG") 2>&1     # everything below goes to the terminal and the log

export LUNA_TEST_CUDA=1
export LUNA_ARRAYTYPE=cuda
export JULIA_NUM_THREADS="${LUNA_THREADS:-${JULIA_NUM_THREADS:-8}}"
export OPENBLAS_NUM_THREADS="$JULIA_NUM_THREADS"

t0=$SECONDS
has() { case ",$STEPS," in *",$1,"*) return 0;; *) return 1;; esac; }
step() { printf '\n\033[1;36m==> %s\033[0m  \033[2m(%s, t+%ds)\033[0m\n' "$1" "$(date +%H:%M:%S)" "$((SECONDS-t0))"; }

step "H100/H200 suite → $RUNDIR"
echo "host $(hostname)  threads $JULIA_NUM_THREADS  cpu target ${JULIA_CPU_TARGET:-native}  Luna $(git -C "$LUNA" rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git -C "$LUNA" rev-parse --short HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,power.limit --format=csv || true
julia --version
ln -sfn "$RUNDIR" /workspace/runs/latest

if has pkgs; then
    step "matplotlib for PyPlot (Luna loads PyPlot; PyCall uses /usr/bin/python3)"
    python3 -c "import matplotlib" 2>/dev/null && echo "    present" || \
        python3 -m pip install --quiet matplotlib || echo "    WARN: matplotlib install failed; 'using Luna' will fail"
    step "packages the test/benchmark scripts import directly (idempotent)"
    julia --project="$DEV" -e '
        using Pkg
        need = ["JLArrays", "GPUArraysCore", "AbstractFFTs", "Adapt", "FFTW", "HDF5", "QuadGK",
                "LinearAlgebra", "Random", "Statistics", "Dates", "Printf", "Test", "Logging"]
        have = keys(Pkg.project().dependencies)
        missing = filter(p -> !(p in have), need)
        isempty(missing) ? println("    all present") : (println("    adding ", missing); Pkg.add(missing))
        Pkg.precompile()
        using CUDA
        println("    CUDA driver ", CUDA.driver_version(), ", runtime ", CUDA.runtime_version(),
                ", device ", CUDA.name(CUDA.device()))'
fi

if has tests; then
    step "CUDA hardware tests (test/test_cuda.jl)"
    time julia --project="$DEV" "$LUNA/test/runtests.jl" test_cuda
fi

if has bench; then
    step "transform benchmark on CUDA only: vuv, ctc (phase A: isolated RHS; phase B: short propagations)"
    time julia --project="$DEV" "$LUNA/test/manual/hpc_gpu_bench.jl" \
        --arraytypes=cuda --cases=vuv,ctc --out="$RUNDIR/bench.csv"
fi

if has production; then
    step "production-like full-length runs on CUDA (CTC_LENGTH=$CTC_LENGTH)"
    time CTC_LENGTH="$CTC_LENGTH" julia --project="$DEV" "$LUNA/test/manual/gpu_production_timing.jl"
fi

if has scan; then
    step "scan rehearsal: 2×2 RDW VUV, one process"
    time SCAN_N=2 SCAN_OUT="$RUNDIR/scan_1proc" \
        julia --project="$DEV" "$LUNA/test/manual/gpu_scan_rehearsal.jl"
    step "scan rehearsal: 2×2 RDW VUV, 4 processes sharing the GPU (--batch 4,i)"
    T0=$SECONDS
    for i in 1 2 3 4; do
        SCAN_N=2 SCAN_OUT="$RUNDIR/scan_4proc" \
            julia --project="$DEV" "$LUNA/test/manual/gpu_scan_rehearsal.jl" --batch 4,$i \
            > "$RUNDIR/scan_4proc_$i.log" 2>&1 &
    done
    wait
    echo "4 processes × 1 point each: wall $((SECONDS-T0)) s (each process paid its own compilation; compare the per-point propagation lines):"
    grep -h "point \|per point" "$RUNDIR"/scan_4proc_*.log || true
    echo "GPU memory now:"; nvidia-smi --query-gpu=memory.used --format=csv || true
fi

step "done in $((SECONDS-t0)) s — results in $RUNDIR (also /workspace/runs/latest)"
cp -f "$LOG" "/workspace/logs/h200-suite-$STAMP.log" 2>/dev/null || true
