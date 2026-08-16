# =============================================================================
# CUDA-specific glue for Luna's device backend.
#
# Luna's device code paths are generic (broadcasts and reductions over anything
# following the GPUArrays interface), so this extension is small by design: it only
# provides the operations that genuinely cannot be expressed generically — selecting a
# device, returning pooled memory, and querying it.
#
# These are installed as function-valued hooks from `__init__` rather than defined as
# methods. Defining `Luna.device_reclaim()` here would *overwrite* the identically
# signatured stub in Device.jl, which Julia rejects during precompilation, leaving this
# extension uncompilable and recompiled in every process that loads it.
#
# Methods specialised on `CuArray` are a different matter — they *add* to a generic
# function rather than overwrite a stub, which is exactly what extensions are for — and
# are used below for the one hot-loop operation where a native CUDA implementation beats
# Luna's portable fallback: the prefix sum inside the plasma response.
#
# Loaded automatically when CUDA is available, including when CUDA is loaded indirectly
# by `Luna.resolve_arraytype(:cuda)`.
# =============================================================================
module LunaCUDAExt

import Luna
import Luna: Maths
import CUDA

# ---------------------------------------------------------------------------
# Native prefix sum. Luna's portable `Maths.scan!` is a Hillis–Steele doubling scan:
# 2*ceil(log2(nt)) broadcast kernels per cumulative integral, i.e. ~26 launches at
# nt = 8192, three times per plasma evaluation. CUDA's `cumsum!` is a single
# work-efficient scan (a handful of launches) with the same accuracy (rounding-level,
# different summation order), and it needs no scratch buffer — so `scan_scratch` returns
# `nothing` and PlasmaCumtrapzBatched allocates two field-sized buffers fewer.
# `AnyCuArray` covers reshapes and views of `CuArray`s (the plasma buffers are
# `reshape`d between (nt, npts) and (nt, 1, npts)).
# ---------------------------------------------------------------------------
Maths.scan!(out::CUDA.AnyCuArray, y::CUDA.AnyCuArray, tmp) = Base.cumsum!(out, y; dims=1)
Maths.scan_scratch(::CUDA.AnyCuArray) = nothing

# A GPU array library keeps a memory pool that is not returned by garbage collection
# alone; the incremental collection first makes freshly dead arrays eligible.
_reclaim() = (GC.gc(false); CUDA.reclaim())

_memory_status() = CUDA.functional() ? (CUDA.free_memory(), CUDA.total_memory()) : nothing

function _select(i::Integer)
    CUDA.functional() || error("cannot select device $i: CUDA is not functional")
    ndev = length(CUDA.devices())
    0 <= i < ndev || throw(ArgumentError(
        "device $i out of range: this process can see $ndev CUDA device(s). " *
        "Under Slurm, CUDA_VISIBLE_DEVICES restricts what is visible, so the index " *
        "is relative to that set."))
    CUDA.device!(i)
    return i
end

function __init__()
    Luna._DEVICE_RECLAIM[] = _reclaim
    Luna._DEVICE_SYNCHRONIZE[] = CUDA.synchronize
    Luna._DEVICE_MEMORY_STATUS[] = _memory_status
    Luna._DEVICE_SELECT[] = _select
    return nothing
end

end
