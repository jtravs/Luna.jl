#!/bin/bash
# =============================================================================
# GPU-only test + benchmark suite for a rented H100/H200 machine (no Slurm; paid time).
#
# WHAT IT DOES (in this order; every step prints flushed lines, so a partial run is still
# useful; nothing here runs CPU benchmarks or CPU tests — those are done on the HPC):
#   0. environment: activate/create $ENVDIR, develop Luna from $LUNA, add the packages the
#      test and benchmark scripts import directly, precompile              (~10–20 min first
#      time: CUDA.jl artifacts download + precompilation dominate; seconds afterwards)
#   1. Luna's hardware-gated CUDA tests (LUNA_TEST_CUDA=1)                  (~4 min)
#   2. the transform benchmark on CUDA only, cases vuv + ctc: phase A isolated RHS +
#      primitives, phase B short propagations                                (~5–8 min)
#   3. production-like full-length runs on CUDA: RDW VUV 1.5 m (nr 32/64, lean and default
#      statistics), CtC-type H2 CTC_LENGTH m (default 0.3 m; ~50 steps/mm)  (~5–10 min;
#      FULL=1 makes the CtC run 1.5 m, ~15–30 min more)
#   4. scan rehearsal: a 2×2 energy × pressure scan of the RDW VUV example through Luna's
#      scan machinery with lean settings, per-point timing, reduced results extracted
#      from the outputs; then the same 2×2 with 4 processes sharing the GPU (--batch),
#      which is how a 400-point scan would be run                            (~3–5 min)
#   Total ≈ 30–45 min the first time (≈ 20–30 min if the environment already exists).
#
# USAGE (as the user, or an agent, on the GPU box; Julia ≥ 1.12 on PATH — juliaup):
#   git clone -b modal-fixed <luna-remote> ~/Luna     # or copy the checkout
#   bash ~/Luna/test/manual/h200_gpu_suite.sh 2>&1 | tee ~/h200-suite-$(date +%Y%m%d-%H%M).log
#   FULL=1 bash ~/Luna/test/manual/h200_gpu_suite.sh ...          # + full-length CtC
#   STEPS=tests,bench bash ~/Luna/test/manual/h200_gpu_suite.sh   # subset of steps
# Environment variables: LUNA (checkout, default ~/Luna), ENVDIR (Julia project, default
# ~/lunaenv), NTHREADS (host threads, default min(nproc,16)), FULL, CTC_LENGTH, STEPS.
#
# WHAT TO LOOK FOR
#   * tests: every testset passes; "CUDA modal transform vs host rel" ≲ 1e-11, "CUDA modal
#     propagation vs host" < 1e-8, "device norm" ratios 1 ± 1e-12, LAZY_CUDA_OK=true.
#   * bench phase A: "RHS x.xxx ms" for vuv and ctc on cuda — compare with the A40 job's
#     phase A and the CPU node's; the primitives line shows how far the RHS is from the sum
#     of its kernels (launch latency / syncs).
#   * production timing: ms/step and wall for the full-length runs; the "default stats"
#     row minus the "lean stats" row is the host statistics' share of a step.
#   * scan rehearsal: seconds per point (setup vs propagation) → the 400-point estimate;
#     the 4-process run shows the throughput gain from sharing the GPU.
# Do NOT run the CPU benchmark (--arraytypes=cpu) or the CPU test suite here.
# =============================================================================
set -uo pipefail

LUNA="${LUNA:-$HOME/Luna}"
ENVDIR="${ENVDIR:-$HOME/lunaenv}"
NPROC=$(nproc 2>/dev/null || echo 8)
NTHREADS="${NTHREADS:-$(( NPROC < 16 ? NPROC : 16 ))}"
FULL="${FULL:-0}"
CTC_LENGTH="${CTC_LENGTH:-$([ "$FULL" = "1" ] && echo 1.5 || echo 0.3)}"
STEPS="${STEPS:-env,tests,bench,production,scan}"

export JULIA_NUM_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS="$NTHREADS"
export LUNA_TEST_CUDA=1
export LUNA_ARRAYTYPE=cuda

has() { case ",$STEPS," in *",$1,"*) return 0;; *) return 1;; esac; }
stamp() { echo; echo "=== $1  ($(date +%H:%M:%S)) ==="; }

stamp "host: $(hostname)  threads: $NTHREADS  Luna: $(git -C "$LUNA" rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git -C "$LUNA" rev-parse --short HEAD 2>/dev/null)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv || true
julia --version

if has env; then
    stamp "environment: $ENVDIR (Luna from $LUNA)"
    time julia -e "using Pkg
        Pkg.activate(\"$ENVDIR\")
        Pkg.develop(path=\"$LUNA\")
        Pkg.add([\"CUDA\", \"JLArrays\", \"GPUArraysCore\", \"AbstractFFTs\", \"Adapt\", \"FFTW\",
                 \"HDF5\", \"QuadGK\", \"LinearAlgebra\", \"Random\", \"Statistics\", \"Dates\",
                 \"Printf\", \"Test\", \"Logging\"])
        Pkg.instantiate(); Pkg.precompile()
        using CUDA; CUDA.versioninfo()"
fi

if has tests; then
    stamp "CUDA hardware tests (Luna/test/test_cuda.jl)"
    time julia --project="$ENVDIR" "$LUNA/test/runtests.jl" test_cuda
fi

if has bench; then
    stamp "transform benchmark on CUDA only: vuv, ctc (phase A: isolated RHS; phase B: short propagations)"
    time julia --project="$ENVDIR" "$LUNA/test/manual/hpc_gpu_bench.jl" \
        --arraytypes=cuda --cases=vuv,ctc --out="$HOME/h200-bench-$(hostname).csv"
fi

if has production; then
    stamp "production-like full-length runs on CUDA (CTC_LENGTH=$CTC_LENGTH)"
    time CTC_LENGTH="$CTC_LENGTH" julia --project="$ENVDIR" "$LUNA/test/manual/gpu_production_timing.jl"
fi

if has scan; then
    stamp "scan rehearsal: 2×2 RDW VUV, one process"
    time SCAN_N=2 SCAN_OUT="$HOME/scan_rehearsal_1proc" \
        julia --project="$ENVDIR" "$LUNA/test/manual/gpu_scan_rehearsal.jl"
    stamp "scan rehearsal: 2×2 RDW VUV, 4 processes sharing the GPU (--batch 4,i)"
    T0=$(date +%s)
    for i in 1 2 3 4; do
        SCAN_N=2 SCAN_OUT="$HOME/scan_rehearsal_4proc" \
            julia --project="$ENVDIR" "$LUNA/test/manual/gpu_scan_rehearsal.jl" --batch 4,$i \
            > "$HOME/scan_rehearsal_4proc_$i.log" 2>&1 &
    done
    wait
    echo "4 processes × 1 point each: wall $(( $(date +%s) - T0 )) s (each process paid its own compilation; compare the per-point propagation lines):"
    grep -h "point \|per point" "$HOME"/scan_rehearsal_4proc_*.log || true
    echo "GPU memory now:"; nvidia-smi --query-gpu=memory.used --format=csv || true
fi

stamp "done"
