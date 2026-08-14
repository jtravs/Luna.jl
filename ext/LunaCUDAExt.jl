# =============================================================================
# CUDA-specific glue for Luna's device backend.
#
# Luna's device code paths are generic (broadcasts and reductions over anything
# following the GPUArrays interface), so this extension is small by design: it only
# implements the operations that genuinely cannot be expressed generically — selecting a
# device, returning pooled memory, and querying it.
#
# Loaded automatically when CUDA is available, including when CUDA is loaded indirectly
# by `Luna.resolve_arraytype(:cuda)`.
# =============================================================================
module LunaCUDAExt

import Luna
import CUDA

Luna.device_reclaim() = (GC.gc(false); CUDA.reclaim())

Luna.device_synchronize() = CUDA.synchronize()

function Luna.device_memory_status()
    CUDA.functional() || return nothing
    return (CUDA.free_memory(), CUDA.total_memory())
end

function Luna.select_device(i::Integer)
    CUDA.functional() || error("cannot select device $i: CUDA is not functional")
    ndev = length(CUDA.devices())
    0 <= i < ndev || throw(ArgumentError(
        "device $i out of range: this process can see $ndev CUDA device(s). " *
        "Under Slurm, CUDA_VISIBLE_DEVICES restricts what is visible, so the index " *
        "is relative to that set."))
    CUDA.device!(i)
    return i
end

end
