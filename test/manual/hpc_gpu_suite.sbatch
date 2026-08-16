#!/bin/bash -l
# =============================================================================
# Luna fixed-quadrature modal transform on a GPU node: tests + CPU/GPU benchmark.
#
# Runs, on one GPU compute node (A40), in this order and without stopping at a failure
# (every result is collected in one submission):
#   1. instantiate/precompile the environment (on the GPU node — see ONE-TIME SETUP)
#   2. Luna's test suite for the parts touched by this work, on the CPU:
#        modal quadrature, fixed transform, batched responses, device logic (JLArrays)
#   3. the hardware-gated CUDA tests (LUNA_TEST_CUDA=1): cuFFT r2c/c2r semantics, the
#      PPT spline / ADK / plasma / Raman device kernels, the native scan, the RK45 error
#      norm host-vs-device with cancellation, transform and end-to-end agreement with the
#      host, and the lazy `arraytype=:cuda` path in a fresh subprocess (world age)
#   4. the CPU-vs-GPU benchmark (test/manual/hpc_gpu_bench.jl) on production-like cases:
#        vuv    RDW VUV example (He gradient, 4 modes, PPT plasma, nt ≈ 8k)
#        ctc    CtC-type H2 supercontinuum (515 nm, 40 bar, 6 modes, Raman+ADK, nt ≈ 65k)
#        vector fully vectorial HE11(x,y)+TE01+TM01+HE21 with plasma, 2-D quadrature
#        bignt  the VUV case on a 4× longer time window
#      each case: isolated RHS time, a propagation over SCALE × the full length with
#      ms/step, transform share, GPU-vs-CPU agreement, resident device memory, and a
#      primitives micro-benchmark at the same shapes. Markdown summary at the end.
#   5. optionally (RDW=1) the full-length RDW VUV validation script on both backends
#      (test/manual/modal_fixed_rdw.jl vuv quick 1.5 {cpu,cuda}) — adds ~20 min.
#
# Expected duration with the defaults: ~1–1.5 h (dominated by the ctc case on the CPU);
# --time is set to 4 h for headroom (first-run precompilation and FFTW planning).
#
# Submit:
#   sbatch test/manual/hpc_gpu_suite.sbatch
#   SCALE=0.05 sbatch --export=ALL,SCALE=0.05 test/manual/hpc_gpu_suite.sbatch  (quicker)
#   sbatch --export=ALL,CASES=vuv,ctc,BENCH_ONLY=1 test/manual/hpc_gpu_suite.sbatch
#   sbatch --export=ALL,RDW=1 test/manual/hpc_gpu_suite.sbatch
#
# ONE-TIME SETUP (see the block at the bottom of this file).
# =============================================================================

#SBATCH --export=ALL
#SBATCH -o luna-gpu-suite-%j.output
#SBATCH -e luna-gpu-suite-%j.error
#SBATCH -J luna-gpu-suite
#SBATCH --time=04:00:00
## Host memory (TOTAL). The largest host footprint is the CPU side of the ctc case
## (nto ≈ 1.3e5 × 65 nodes × ~10 buffers ≈ 1 GB) plus the Julia + CUDA runtime.
#SBATCH --mem=32G
## CPUs: the CPU half of the benchmark is the comparison the GPU has to beat, so give it
## a realistic thread count. (A production CPU node run would use more; the per-step
## numbers scale roughly with threads for the plasma/Raman columns.)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres gpu:1
#SBATCH -p gpu

# NB: no `set -e` — a failing step must not stop the later ones.
set -uo pipefail

GPUROOT="${GPUROOT:-$HOME/sharedscratch/gpu}"
LUNA="$GPUROOT/Luna"
ENVDIR="$GPUROOT/env"
SCALE="${SCALE:-0.2}"
CASES="${CASES:-vuv,ctc,vector,bignt}"
BENCH_ONLY="${BENCH_ONLY:-0}"
RDW="${RDW:-0}"
NTHREADS="${SLURM_CPUS_PER_TASK:-16}"

export JULIA_NUM_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS="$NTHREADS"
export OMP_NUM_THREADS="$NTHREADS"
# Enables the hardware-gated tests in Luna/test/test_cuda.jl (skipped otherwise, and the
# job would pass vacuously)
export LUNA_TEST_CUDA=1
# Julia's own scratch (FFTW wisdom cache, PPT tables) lives under ~/.julia by default,
# which is shared across nodes here; leave it.

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-none}  threads: $NTHREADS ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv || true
echo "Luna: $(git -C "$LUNA" rev-parse --abbrev-ref HEAD) @ $(git -C "$LUNA" rev-parse --short HEAD)"
echo

# Precompilation happens here, on the GPU node: the login node faults (illegal
# instruction) while precompiling CUDA.jl's BFloat16s dependency. Cached after the first
# run. If a node ever faults loading CUDA.jl, add `-C x86-64-v3` to every julia call.
echo "=== instantiate/precompile ==="
time julia --project="$ENVDIR" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
echo

if [ "$BENCH_ONLY" != "1" ]; then
    echo "=== CPU tests: modal quadrature, fixed transform, batched responses, device logic ==="
    time julia --project="$ENVDIR" "$LUNA/test/runtests.jl" modal_quadrature modal_fixed device multimode vectorplasma
    echo

    echo "=== CUDA hardware tests ==="
    time julia --project="$ENVDIR" "$LUNA/test/runtests.jl" test_cuda
    echo
fi

echo "=== CPU vs GPU benchmark (scale=$SCALE cases=$CASES) ==="
time julia --project="$ENVDIR" "$LUNA/test/manual/hpc_gpu_bench.jl" \
    --cases="$CASES" --scale="$SCALE" --out="$GPUROOT/bench-$(hostname)-${SLURM_JOB_ID:-none}.csv"
echo

if [ "$RDW" = "1" ]; then
    echo "=== full-length RDW VUV validation, CPU ==="
    time julia --project="$ENVDIR" "$LUNA/test/manual/modal_fixed_rdw.jl" vuv quick 1.5 cpu
    echo
    echo "=== full-length RDW VUV validation, CUDA ==="
    time julia --project="$ENVDIR" "$LUNA/test/manual/modal_fixed_rdw.jl" vuv quick 1.5 cuda
    echo
fi

echo "=== done ==="

# =============================================================================
# ONE-TIME SETUP
#
# On the login node (installs only — the precompile failure there is expected and
# harmless, the job above redoes it on a GPU node):
#
#   mkdir -p ~/sharedscratch/gpu && cd ~/sharedscratch/gpu
#   git clone -b modal-fixed <luna-remote> Luna      # or copy the checkout
#   julia -e 'using Pkg; Pkg.activate(ENV["HOME"]*"/sharedscratch/gpu/env");
#             Pkg.develop(path=ENV["HOME"]*"/sharedscratch/gpu/Luna");
#             Pkg.add(["CUDA", "JLArrays", "GPUArraysCore", "AbstractFFTs", "Adapt",
#                      "FFTW", "HDF5", "LinearAlgebra", "Random", "Statistics", "Dates",
#                      "Printf", "Test", "Logging"])'
#
# The test files and the benchmark are run DIRECTLY (not through Pkg.test), so every
# package they `import` must be a direct dependency of this environment — including ones
# that are dependencies of Luna itself (FFTW, HDF5, AbstractFFTs, Adapt, GPUArraysCore),
# and the test-only extras (JLArrays). CUDA.jl is what turns Luna's LunaCUDAExt on.
#
# WHAT TO LOOK FOR IN THE OUTPUT
#   * test_cuda: every testset passes. "CUDA modal transform vs host rel" ≲ 1e-11 and
#     "CUDA modal propagation vs host" < 1e-8 are the acceptance checks; the "device
#     norm" ratios must be 1 to ~1e-12 at every delta; "LAZY_CUDA_OK=true" from the
#     subprocess proves the scan-script path.
#   * benchmark summary table: "RHS ms" and "ms/step" per case for cpu and cuda, the
#     "7×RHS/step" share (what fraction of a step is the transform — the rest is the
#     linear operator, RK45 update/error norm, statistics and output), "rel L2 vs cpu"
#     (expect ≲ 1e-8 over the shortened fibre), and "device GiB". The primitives table
#     shows how far each RHS is from the sum of its kernels (launch latency / syncs).
#   * "GPU/CPU speed-up" lines at the end. Expected (unmeasured before this job): ~1× or
#     worse for vuv (launch-bound), several × for ctc/vector/bignt.
# =============================================================================
