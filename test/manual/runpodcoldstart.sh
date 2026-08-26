#!/usr/bin/env bash
#
# coldstart.sh — bring a fresh Runpod pod to a working Luna.jl GPU dev state.
#
# Idempotent: run it on every new pod attached to the same network volume.
# First run does the full install (~10-20 min); later runs are ~2 min.
#
#   rsync -av coldstart.sh runpod:/workspace/
#   ssh runpod 'bash /workspace/coldstart.sh'
#
set -euo pipefail

VOL=/workspace
JULIA_CHANNEL=1.12
LUNA_URL=https://github.com/jtravs/Luna.jl.git
LUNA_BRANCH=modal-fixed
PNPS_URL=https://github.com/LupoLab/ModelPNPS.jl.git
PNPS_BRANCH=gpu

GIT_NAME="John Travers"
GIT_EMAIL="jtravs@gmail.com"

t0=$SECONDS
step() { printf '\n\033[1;36m==> %s\033[0m  \033[2m(t+%ds)\033[0m\n' "$1" "$((SECONDS-t0))"; }
warn() { printf '\033[1;33m    WARN: %s\033[0m\n' "$1"; }

# ---------------------------------------------------------------- guards ----
if [ ! -d "$VOL" ]; then
    echo "ERROR: $VOL does not exist. Attach the network volume at pod creation."
    exit 1
fi
if ! mountpoint -q "$VOL" 2>/dev/null; then
    warn "$VOL is not a mount point — this may be container disk, which is"
    warn "wiped on terminate. Check the pod was created with a volume attached."
fi

# (not juliaup: the installer refuses to install into an existing directory)
mkdir -p "$VOL"/{code,runs,claude,logs}

DEPOT_LOCAL=/root/julia_depot
DEPOT_TAR="$VOL/julia_depot.tar"

# `runpodcoldstart.sh save`: persist the container-disk depot to the volume as one
# tarball (uncompressed: sequential write at volume speed, ~30 s for 8 GB; gzip would
# take minutes of CPU for little gain). Run it at the end of a session in which packages
# were added or precompiled, before the pod is terminated.
if [ "${1:-}" = "save" ]; then
    [ -d "$DEPOT_LOCAL" ] || { echo "no depot at $DEPOT_LOCAL"; exit 1; }
    step "Saving depot $DEPOT_LOCAL → $DEPOT_TAR"
    # (GNU tar: --exclude is positional and must precede the paths it applies to)
    tar --exclude='*/logs/*' --exclude='*/lunacache/*.h5' \
        -cf "$DEPOT_TAR.tmp" -C "$(dirname "$DEPOT_LOCAL")" "$(basename "$DEPOT_LOCAL")" \
        && mv -f "$DEPOT_TAR.tmp" "$DEPOT_TAR"
    ls -lh "$DEPOT_TAR"; echo "  done in $((SECONDS-t0))s"
    exit 0
fi

# Runpod network volumes are quota-enforced on shared storage: df reports the backing
# cluster, not your allocation, so the only meaningful number is what you use. The depot
# tarball is ~6-8 GB; keep an eye on it against the volume size you rented.
_vol_used=$(du -sh "$VOL" 2>/dev/null | cut -f1 || echo '?')
echo "    volume in use: ${_vol_used} (df cannot see the quota of a network volume)"

# ------------------------------------------------------ system packages ----
# These live on the container disk and are lost on every new pod. Cheap — and only
# conveniences, so this step must never abort the bootstrap: it is skipped when the tools
# are already there, and an apt failure (a broken NVIDIA repo entry, the image's own
# apt lock, a legacy-keyring warning) is logged and ignored. Warnings such as
# "Key is stored in legacy trusted.gpg keyring" and "debconf: delaying package
# configuration" are normal on these images.
step "System packages"
if command -v git >/dev/null && command -v curl >/dev/null && \
   command -v tmux >/dev/null && command -v rsync >/dev/null; then
    echo "    already present"
else
    export DEBIAN_FRONTEND=noninteractive
    _aptlog="$VOL/logs/apt-$(date +%Y%m%d-%H%M%S).log"
    if apt-get update -qq >"$_aptlog" 2>&1 && \
       apt-get install -y -qq --no-install-recommends \
           curl ca-certificates git tmux rsync less vim htop unzip build-essential \
           >>"$_aptlog" 2>&1; then
        echo "    installed (log: $_aptlog)"
    else
        warn "apt-get failed — continuing without it (see $_aptlog)"
        for t in git curl; do
            command -v "$t" >/dev/null || { echo "ERROR: $t is required and missing"; exit 1; }
        done
    fi
fi

# Luna's Plotting module loads PyPlot at `using Luna`, and PyPlot needs matplotlib in the
# Python that PyCall is built against (/usr/bin/python3 on these images) — without it
# `using Luna` fails during PyPlot's __init__. Container disk, so every new pod needs it.
step "matplotlib for PyPlot"
if python3 -c "import matplotlib" 2>/dev/null; then
    echo "    already present"
else
    python3 -m pip install --quiet matplotlib >"$VOL/logs/pip-matplotlib.log" 2>&1 \
        || { export DEBIAN_FRONTEND=noninteractive; apt-get update -qq >/dev/null 2>&1; \
             apt-get install -y -qq --no-install-recommends python3-matplotlib >>"$VOL/logs/pip-matplotlib.log" 2>&1; } \
        || warn "could not install matplotlib — 'using Luna' will fail (see $VOL/logs/pip-matplotlib.log)"
    python3 -c "import matplotlib; print('    matplotlib', matplotlib.__version__)" 2>/dev/null || true
fi

# ------------------------------------------------------------ environment ---
step "Environment file"
cat > "$VOL/env.sh" <<'EOF'
# Sourced from ~/.bashrc on every pod. Edit here, not in .bashrc.

export JULIAUP_DEPOT_PATH=/workspace/juliaup
# The Julia depot (packages, artifacts, compile caches: ~100k small files, 6-8 GB) lives
# on the CONTAINER DISK, not the network volume: the volume does sequential I/O at
# ~350 MB/s but ~5 ms per small-file create, which turns Pkg.add/precompile into hours.
# It is persisted across pods as ONE tarball on the volume (/workspace/julia_depot.tar):
# the cold start restores it, `runpodcoldstart.sh save` refreshes it.
# The TRAILING COLON matters: it appends Julia's bundled depots (the shipped stdlib
# pkgimages under <julia>/share/julia). Without it the stdlibs are recompiled into our
# depot (Pkg alone ~75 s, then StyledStrings, Dates, ...), which is what a bare value did.
export JULIA_DEPOT_PATH=/root/julia_depot:
export PATH="/workspace/juliaup/bin:$HOME/.local/bin:$PATH"
# If package/artifact downloads crawl, try a region-specific Pkg server, e.g.
#   export JULIA_PKG_SERVER=https://us-east.pkg.julialang.org   (or eu-central, ...)

# The pod image ships its own CUDA toolkit and puts /usr/local/cuda/lib64 on
# LD_LIBRARY_PATH. CUDA.jl uses its own toolkit artifacts, and mixing the two gives
# loader errors such as
#   libcublas.so: undefined symbol: cublasLtDDDMatmulAlgoGetHeuristicForStream, version libcublasLt.so.13
# (the artifact's cublas resolved against the image's older cublasLt). Drop the toolkit
# entries; keep the driver ones (/usr/local/nvidia/...), which CUDA.jl needs.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH=$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/cuda' | paste -sd: - )
fi

# Containers LIE about resources: /proc/cpuinfo and free(1) report the whole
# HOST (possibly 192 cores / 2 TB), not your cgroup slice. JULIA_NUM_THREADS=auto
# would therefore start ~190 threads on a 24-vCPU pod. Read the cgroup instead.
_pod_cpus() {
    if [ -r /sys/fs/cgroup/cpu.max ]; then                    # cgroup v2
        read -r _q _p < /sys/fs/cgroup/cpu.max
        [ "$_q" != max ] && { echo $(( (_q + _p - 1) / _p )); return; }
    elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then     # cgroup v1
        _q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        _p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        [ "$_q" -gt 0 ] && { echo $(( (_q + _p - 1) / _p )); return; }
    fi
    nproc                                                      # cpuset fallback
}
_pod_mem_gb() {
    local m g
    m=$(cat /sys/fs/cgroup/memory.max 2>/dev/null \
        || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo 0)
    # "unlimited" appears either as the string 'max' or as a huge sentinel int.
    case "$m" in max|0|*[!0-9]*) echo 0; return ;; esac
    g=$(( m / 1073741824 ))
    [ "$g" -gt 100000 ] && g=0          # sentinel: no cgroup limit set
    echo "$g"
}
export POD_CPUS=$(_pod_cpus)
export POD_MEM_GB=$(_pod_mem_gb)

# For GPU-resident work, host-side threading buys little and adds GC pressure
# (Julia's GC thread count defaults to half of this). Capped at 8 deliberately —
# override with LUNA_THREADS=24 and compare before trusting the bigger number.
export JULIA_NUM_THREADS=${LUNA_THREADS:-$(( POD_CPUS < 8 ? POD_CPUS : 8 ))}
export OPENBLAS_NUM_THREADS=$JULIA_NUM_THREADS
# Luna's FFTW thread count: its default caps at Sys.CPU_THREADS, which inside a container
# is the HOST's core count (the cgroup quota is not in the affinity mask), so 8 Julia
# threads got 32 FFTW pthreads on an 8-vCPU slice. Pin it to the Julia thread count.
export LUNA_FFTW_THREADS=$JULIA_NUM_THREADS
# Same container lie, worse consequence: Pkg sizes its precompile worker pool from
# Sys.CPU_THREADS + 1, i.e. ~190 concurrent Julia processes on a 24-vCPU slice — the
# whole environment then precompiles at a crawl (an hour-long cold start). Cap it at the
# cgroup CPU count.
export JULIA_NUM_PRECOMPILE_TASKS=$POD_CPUS

# Package images are only loadable on a CPU that supports the features they were
# compiled for. Unset (= native) makes the depot tarball specific to the CPU of the pod
# that built it — a pod with a different CPU recompiles everything (~15 min); 'generic'
# is portable but slow host code. Julia's own x86-64 multi-target string gives both:
# a generic clone plus optimised ones, exactly what the shipped stdlib images use.
export JULIA_CPU_TARGET="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)"

# Claude Code: config + credentials on the volume, so you log in once ever.
export CLAUDE_CONFIG_DIR=/workspace/claude

export LUNA_DEV=/workspace/code/dev
alias dev='cd $LUNA_DEV'
EOF

grep -qF 'source /workspace/env.sh' ~/.bashrc || \
    echo 'source /workspace/env.sh' >> ~/.bashrc
# shellcheck source=/dev/null
source "$VOL/env.sh"

git config --global user.name  "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"
git config --global --add safe.directory '*'

# ------------------------------------------------------------ Julia depot ---
step "Julia depot on the container disk ($DEPOT_LOCAL)"
if [ -d "$DEPOT_LOCAL" ]; then
    echo "    present ($(du -sh "$DEPOT_LOCAL" 2>/dev/null | cut -f1))"
elif [ -f "$DEPOT_TAR" ]; then
    echo "    restoring from $DEPOT_TAR ($(du -h "$DEPOT_TAR" | cut -f1)) — sequential read, minutes at most"
    tar xf "$DEPOT_TAR" -C "$(dirname "$DEPOT_LOCAL")" && echo "    restored in $((SECONDS-t0))s"
elif [ -d "$VOL/julia_depot" ] && [ "${MIGRATE_DEPOT:-1}" = "1" ]; then
    # a depot from the earlier layout (files on the volume): stream it across once,
    # then it is only ever read again from the tarball. Slow-ish (per-file reads on the
    # volume) but faster than re-downloading on a slow network; MIGRATE_DEPOT=0 skips it.
    echo "    migrating the old on-volume depot $VOL/julia_depot (one-off; MIGRATE_DEPOT=0 to skip)"
    mkdir -p "$DEPOT_LOCAL"
    ( cd "$VOL/julia_depot" && tar cf - . ) | ( cd "$DEPOT_LOCAL" && tar xf - ) \
        && echo "    migrated in $((SECONDS-t0))s — remove $VOL/julia_depot when happy" \
        || warn "migration failed; a fresh depot will be built"
else
    echo "    fresh (packages will download; run '$0 save' at the end of the session)"
fi
mkdir -p "$DEPOT_LOCAL"
_root_free_gb=$(df -BG --output=avail /root 2>/dev/null | tail -1 | tr -dc '0-9')
[ -n "${_root_free_gb:-}" ] && [ "$_root_free_gb" -lt 12 ] && \
    warn "only ${_root_free_gb} GB free on the container disk — a full depot needs ~8 GB"

# ------------------------------------------------------------------ Julia ---
step "Julia $JULIA_CHANNEL"
if [ ! -x "$VOL/juliaup/bin/julia" ]; then
    # a leftover directory from a failed attempt (or the old mkdir above) makes the
    # installer refuse to run; it holds nothing worth keeping without bin/julia
    if [ -d "$VOL/juliaup" ]; then
        warn "removing incomplete $VOL/juliaup from an earlier attempt"
        rm -rf "$VOL/juliaup"
    fi
    # juliaup's installer: --add-to-path no (formerly --no-modify-path; PATH comes from
    # env.sh), --default-channel, --path (install location; JULIAUP_DEPOT_PATH is also
    # set by env.sh). Startup/background self-update checks off: nothing should phone home
    # or change versions under a benchmark.
    curl -fsSL https://install.julialang.org | sh -s -- \
        --yes --add-to-path no \
        --default-channel "$JULIA_CHANNEL" \
        --path "$VOL/juliaup" \
        --startup-selfupdate 0 --background-selfupdate 0
    hash -r
else
    echo "    already on volume"
fi
julia --version

# ------------------------------------------------------------ Claude Code ---
# Binary goes to ~/.local/bin (container disk, ~30s to reinstall each pod).
# Config and credentials go to $CLAUDE_CONFIG_DIR on the volume, so auth persists.
step "Claude Code"
if ! command -v claude >/dev/null 2>&1; then
    curl -fsSL https://claude.ai/install.sh | bash
fi
hash -r
if command -v claude >/dev/null 2>&1; then
    claude --version
    if [ -f "$CLAUDE_CONFIG_DIR/.credentials.json" ]; then
        echo "    credentials found on volume — already authenticated"
    else
        echo "    not yet authenticated: run 'claude' and follow the printed URL"
    fi
else
    warn "claude not on PATH; check ~/.local/bin"
fi

# ------------------------------------------------------------------ repos ---
sync_repo() {
    local url=$1 dir=$2 branch=$3 name
    name=$(basename "$dir")
    if [ -d "$dir/.git" ]; then
        git -C "$dir" fetch --quiet origin "$branch" || warn "$name: fetch failed"
        if [ -n "$(git -C "$dir" status --porcelain)" ]; then
            warn "$name: working tree dirty — not touching it"
        else
            git -C "$dir" checkout --quiet "$branch" 2>/dev/null || \
                warn "$name: could not check out $branch"
            git -C "$dir" merge --quiet --ff-only "origin/$branch" 2>/dev/null || \
                warn "$name: not fast-forwardable (local commits?) — left as is"
        fi
    else
        git clone --quiet --branch "$branch" "$url" "$dir"
    fi
    printf '    %-16s %-14s %s\n' "$name" "$branch" \
        "$(git -C "$dir" rev-parse --short HEAD)"
}

step "Repositories"
sync_repo "$LUNA_URL" "$VOL/code/Luna.jl"      "$LUNA_BRANCH"
sync_repo "$PNPS_URL" "$VOL/code/ModelPNPS.jl" "$PNPS_BRANCH"

# ------------------------------------------------------- Julia dev project --
# A shared environment that `dev`s both packages, so edits in either are live.
step "Julia environment"
DEV="$VOL/code/dev"
mkdir -p "$DEV"

# Do NOT pin the CUDA runtime. CUDA.jl picks the newest toolkit artifact it
# supports that the HOST DRIVER can run — which is the correct answer and
# differs per machine. A stale pin is how you manufacture
# "You are using CUDA 13.x but CUDA.jl was precompiled for 13.y" errors.
# (Pinning is only justified for a GPU-less Docker build.)
if grep -rqs 'version *=' "$DEV/LocalPreferences.toml" 2>/dev/null && \
   grep -qs 'CUDA_Runtime_jll' "$DEV/LocalPreferences.toml"; then
    warn "a pinned CUDA runtime is set in $DEV/LocalPreferences.toml"
    warn "remove it unless you set it deliberately: CUDA.reset_runtime_version!()"
fi

# Besides CUDA, the project needs every package that Luna's test files and manual
# benchmark scripts `import` directly (they are run as scripts, not through Pkg.test):
# the test-only extras (JLArrays), Luna dependencies that are not transitively
# importable (FFTW, HDF5, AbstractFFTs, Adapt, GPUArraysCore, QuadGK) and the stdlibs.
# test/manual/h200_gpu_suite.sh guard-adds the same list, so an older volume catches up.
if [ ! -f "$DEV/Manifest.toml" ]; then
    echo "    first-time resolve (this is the slow bit)"
    julia --project="$DEV" -e '
        using Pkg
        Pkg.develop(path="/workspace/code/Luna.jl")
        Pkg.develop(path="/workspace/code/ModelPNPS.jl")
        Pkg.add(["CUDA", "BenchmarkTools", "HDF5", "JLArrays", "GPUArraysCore",
                 "AbstractFFTs", "Adapt", "FFTW", "QuadGK", "LinearAlgebra", "Random",
                 "Statistics", "Dates", "Printf", "Test", "Logging"])
    '
else
    julia --project="$DEV" -e 'using Pkg; Pkg.instantiate()'
fi

# Non-fatal: a broken gpu branch shouldn't abort the whole bootstrap.
julia --project="$DEV" -e 'using Pkg; Pkg.precompile()' \
    || warn "precompile failed — drop into the REPL and debug interactively"

# ------------------------------------------------------------ GPU sanity ----
step "GPU"
nvidia-smi --query-gpu=name,memory.total,power.limit,power.max_limit \
           --format=csv,noheader || warn "nvidia-smi failed"

julia --project="$DEV" -e '
    using CUDA, Libdl
    if !CUDA.functional()
        println("CUDA NOT FUNCTIONAL"); CUDA.versioninfo(); exit(1)
    end
    d = CUDA.device()
    println("  device      : ", CUDA.name(d))
    println("  capability  : ", CUDA.capability(d), "   (9.0 = H100/H200)")
    println("  memory      : ", round(CUDA.total_memory()/2^30, digits=1), " GiB")
    # driver_version  = highest CUDA API level this host driver supports.
    #                   CUDA 13.x artifacts require an r580-series driver or newer.
    # runtime_version = the toolkit artifact CUDA.jl actually selected.
    # Record both alongside every benchmark; they can differ between pods.
    println("  driver CUDA : ", CUDA.driver_version())
    println("  runtime     : ", CUDA.runtime_version())
    println("  julia       : ", VERSION)
    # every library must come from the artifact depot, none from /usr/local/cuda: a mix
    # is what produces "undefined symbol ... libcublasLt.so.13"-type loader errors
    for lib in (CUDA.CUBLAS.libcublas, CUDA.CUFFT.libcufft)
        p = try Libdl.dlpath(lib) catch; "?" end
        println("  ", rpad(basename(string(lib)), 12), ": ", p)
    end
    VERSION >= v"1.13-" && println("  WARNING: CUDA.jl support for 1.13 is not settled; use 1.12")

    # Effective copy bandwidth. Compare against the spec sheet: a rented card
    # may be power-capped below nominal TDP, which shows up here first.
    n = 2^27                      # 1 GiB of Float64
    a = CUDA.zeros(Float64, n); b = similar(a)
    copyto!(b, a); CUDA.synchronize()          # warm up
    reps = 20
    t = CUDA.@elapsed begin
        for _ in 1:reps; copyto!(b, a); end
    end
    bw = 2 * sizeof(a) * reps / t / 1e9
    println("  copy BW     : ", round(bw, digits=0), " GB/s",
            "   (H100 ~3350, H200 ~4800 peak)")

    # What this card can actually hold, for the 3-D TG-FROG campaigns. The envelope rule
    # of thumb is 10 x the state array; the FIELD-RESOLVED path is not covered by it (its
    # state is twice as long in omega and its nonlinear grid twice as long again in time),
    # so quote the measured budgets at the 40 um / N=768 production shape.
    gib = CUDA.total_memory()/2^30
    println("  fits (3-D TG-FROG at N=768, 40 um):")
    for (what, need) in (("envelope", 24), ("field :thg or :nothg+ffac4", 65),
                         ("field :nothg (the campaign arm)", 92))
        println("      ", rpad(what, 32), need > 0.92gib ? "NO  " : "yes ",
                "(~", need, " GiB)")
    end
' || warn "GPU check failed"

# ----------------------------------------------------------------- summary --
step "Ready"
_depot_gb=$(du -sBG "$DEPOT_LOCAL" 2>/dev/null | tr -dc '0-9' || echo '?')
_depot_files=$(find "$DEPOT_LOCAL" 2>/dev/null | wc -l || echo '?')
_vol_used=$(du -sh "$VOL" 2>/dev/null | cut -f1 || echo '?')
if [ "${POD_MEM_GB:-0}" -gt 0 ]; then
    _mem_str="${POD_MEM_GB} GB"
else
    _mem_str="no cgroup limit visible — free(1) shows the HOST, not your slice"
fi

# HOST RAM is a real constraint for the 3-D free-space campaigns, and it is easy to miss
# on a pod chosen for its GPU. `build_setup` builds the beamlets on the host and holds the
# unmasked HE11 field, the three masked beamlets, their pre-summed gate pair and the
# window all at once — at the 40 µm production shape (N = 768) that is ~12 GiB in the
# envelope mode and ~25 GiB in FIELD mode, where every field is twice as long in ω.
_hostwarn=""
if [ "${POD_MEM_GB:-0}" -gt 0 ] && [ "${POD_MEM_GB}" -lt 32 ]; then
    _hostwarn="  NOTE: ${POD_MEM_GB} GB host RAM is tight for the N=768 campaigns
        (build_setup peaks at ~12 GiB envelope / ~25 GiB field mode, on the HOST).
        ModelPNPS's field-mode driver passes beamlets_on_host=true, which keeps two
        more fields resident there for the whole run. If a run dies during setup
        rather than on the card, this is why."
fi

cat <<EOF

  project   $DEV
  code      $VOL/code/{Luna.jl,ModelPNPS.jl}
  runs      $VOL/runs
  logs      $VOL/logs

${_hostwarn:+$_hostwarn
}  host slice   ${POD_CPUS} vCPU (cgroup), ${_mem_str} RAM
               JULIA_NUM_THREADS=${JULIA_NUM_THREADS}  (LUNA_THREADS= to override)
               JULIA_NUM_PRECOMPILE_TASKS=${JULIA_NUM_PRECOMPILE_TASKS}
  volume       ${_vol_used} in use (quota not visible to df)
  depot        ${_depot_gb} GB across ${_depot_files} files (container disk;
               persist with: bash $0 save   → $DEPOT_TAR)

  Total cold start: $((SECONDS-t0))s
  (If this is under ~5 min on a warm volume, a custom Docker image
   would not buy you much.)

  Next:
    tmux new -s dev
    cd \$LUNA_DEV && claude

  GPU tests + benchmarks (CUDA only; results under /workspace/runs/<timestamp>/):
    bash /workspace/code/Luna.jl/test/manual/h200_gpu_suite.sh          # modal transform
    bash /workspace/code/ModelPNPS.jl/examples/h200_modelpnps_suite.sh # 3-D TG-FROG

  Field-resolved (RealGrid) TG-FROG — the 1 fs envelope-vs-field question. Heavier than
  it looks: ~92 GiB on the card at the 40 µm production shape with the default response,
  so H200-only (:thg or FFAC=4 cut it to ~65 GiB and fit an 80 GB H100). Benchmark it
  before committing to a scan, and always dry-run the scan first:
    STEPS=fieldbench bash /workspace/code/ModelPNPS.jl/examples/h200_modelpnps_suite.sh
    PNPS_DRYRUN=1 bash /workspace/code/ModelPNPS.jl/examples/h200_field_mode.sh

  To get commits back to your laptop without any credentials on this pod:
    git remote add pod ssh://root@<IP>:<PORT>/workspace/code/Luna.jl
    git fetch pod && git log pod/$LUNA_BRANCH

EOF
