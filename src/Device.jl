# =============================================================================
# Device (GPU) support scaffolding.
#
# Luna has NO GPU dependency. Everything here is generic over the array type: the
# device kernels elsewhere in Luna are ordinary broadcasts and `mapreduce`s, so they
# compile for any array that follows the GPUArrays interface (`CuArray`, `ROCArray`,
# `JLArray`, …). The genuinely backend-specific operations — selecting a device,
# returning pooled memory, querying free memory — are declared here as no-ops and
# overridden by a package extension (see `ext/LunaCUDAExt.jl`) when the corresponding
# GPU package is loaded.
#
# These definitions live directly in the `Luna` module (this file is `include`d without
# a `module` wrapper) so that extensions can extend them as `Luna.device_reclaim` etc.
# =============================================================================

import Adapt
import .Utils: Backend, CPUBackend, DeviceBackend, backend, isdevice

#=================================================#
#=========  DEVICE MANAGEMENT (stubs)  ===========#
#=================================================#

# These are installed by a package extension (see `ext/LunaCUDAExt.jl`) from its
# `__init__`, rather than being defined as methods there. A method with an identical
# signature cannot be *overwritten* from an extension — Julia rejects that during
# precompilation ("Method overwriting is not permitted"), which leaves the extension
# uncompilable and silently recompiled in every process that loads it. Function-valued
# hooks sidestep the problem entirely.
#
# The hooks are called through `Base.invokelatest` because the GPU package may have been
# loaded *during* the call that is now using them (see `resolve_arraytype`), in which
# case its methods are "too new" for the calling world. These run once per simulation,
# never per step, so the dynamic dispatch costs nothing.
const _DEVICE_RECLAIM = Ref{Any}(nothing)
const _DEVICE_SYNCHRONIZE = Ref{Any}(nothing)
const _DEVICE_MEMORY_STATUS = Ref{Any}(nothing)
const _DEVICE_SELECT = Ref{Any}(nothing)

"""
    device_reclaim()

Return cached device memory to the driver. A no-op unless a GPU package is loaded (see
`ext/LunaCUDAExt.jl`). Call between independent simulations in a long-lived process:
GPU array libraries keep a memory pool that is not released on garbage collection alone.
"""
device_reclaim() = (f = _DEVICE_RECLAIM[]; isnothing(f) ? nothing : Base.invokelatest(f))

"""
    device_synchronize()

Block until all queued device work has completed. A no-op unless a GPU package is
loaded. Only needed for timing — Luna's own code paths are synchronised implicitly by
the data dependencies between kernels.
"""
device_synchronize() = (f = _DEVICE_SYNCHRONIZE[]; isnothing(f) ? nothing : Base.invokelatest(f))

"""
    device_memory_status() -> (free, total) in bytes, or `nothing`

Free and total device memory, or `nothing` when no GPU package is loaded. Useful for
logging the high-water mark of a propagation.
"""
device_memory_status() = (f = _DEVICE_MEMORY_STATUS[]; isnothing(f) ? nothing : Base.invokelatest(f))

"""
    select_device(i)

Select device number `i` (0-based, matching the GPU vendor's numbering) for this
process. A no-op unless a GPU package is loaded.

Under Slurm it is usually better to let the batch script set `CUDA_VISIBLE_DEVICES` per
process, so each process sees exactly one device and needs no explicit selection.
"""
select_device(i::Integer) = (f = _DEVICE_SELECT[]; isnothing(f) ? nothing : Base.invokelatest(f, i))

#=================================================#
#===========  ARRAY TYPE RESOLUTION  =============#
#=================================================#

const CUDA_UUID = Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba")

"""
    resolve_arraytype(x) -> Type

Resolve an array-type specification to a concrete array type. Accepts a type directly
(`Array`, `CuArray`, …), `nothing` (→ `Array`), or a `Symbol`: `:cpu` → `Array`,
`:cuda` → `CUDA.CuArray`.

The `Symbol` form exists so that a scan script can request a GPU **without loading the
GPU package at the top level**. That matters on clusters where the submitting host is a
login node with no GPU (and, in at least one observed case, a CPU that faults while
precompiling CUDA.jl): the script is parsed on the login node to generate and submit the
batch job, but `resolve_arraytype(:cuda)` is only ever called from inside the scan
closure, which runs on the compute node.

`Base.require` loads the package by UUID without any lexical reference to it, and still
triggers Luna's package extension for that backend.
"""
resolve_arraytype(A::Type) = A
resolve_arraytype(::Nothing) = Array

function resolve_arraytype(s::Symbol)
    s === :cpu && return Array
    s === :cuda || throw(ArgumentError(
        "unknown array type `:$s`; expected `:cpu`, `:cuda`, or an array type"))
    CUDA = try
        Base.require(Base.PkgId(CUDA_UUID, "CUDA"))
    catch e
        error("`arraytype=:cuda` requested but CUDA.jl could not be loaded on " *
              "$(gethostname()): $e")
    end
    Base.invokelatest(getproperty(CUDA, :functional)) || error(
        "`arraytype=:cuda` requested but CUDA is not functional on $(gethostname()). " *
        "Check that this process can see a GPU (CUDA_VISIBLE_DEVICES).")
    return Base.invokelatest(getproperty, CUDA, :CuArray)
end

#=================================================#
#==============  ALLOCATION HELPER  ==============#
#=================================================#

"""
    device_zeros(arraytype, T, dims)

Allocate a zero-filled `dims`-shaped array of element type `T` and array type
`arraytype`. Equivalent to `zeros(T, dims)` for `arraytype === Array`.

Field-sized buffers are allocated this way rather than built on the host and copied, so
that a device run never materialises a host array it does not need — at production sizes
that would be several GB per buffer.
"""
device_zeros(::Type{A}, ::Type{T}, dims) where {A<:AbstractArray, T} =
    fill!(A{T}(undef, dims), zero(T))

#=================================================#
#=============  GRID VECTOR MIRRORS  =============#
#=================================================#

"""
    GridVectors

The grid vectors that appear in elementwise kernels alongside the propagating field:
the angular frequency axis and the spectral/temporal apodisation windows.

`Grid.EnvGrid`/`Grid.RealGrid` hold plain host `Vector`s, and a host vector cannot be
broadcast against a device array. Rather than making the grid types themselves
device-aware — which would leak device arrays into the metadata written to output files
— the transform keeps a `GridVectors` mirror, adapted to its own array type at
construction. On the host this aliases the grid's own vectors: no copy, no extra memory,
and the CPU code path is unchanged.
"""
struct GridVectors{V}
    ω::V
    ωwin::V
    twin::V
    towin::V
end

"Build a [`GridVectors`](@ref) aliasing `grid`'s own vectors (no copy)."
gridvectors(grid) = GridVectors(grid.ω, grid.ωwin, grid.twin, grid.towin)

Adapt.adapt_structure(to, g::GridVectors) =
    GridVectors(Adapt.adapt(to, g.ω), Adapt.adapt(to, g.ωwin),
                Adapt.adapt(to, g.twin), Adapt.adapt(to, g.towin))

"""
    gridvectors(grid, arraytype)

[`GridVectors`](@ref) for `grid`, adapted to `arraytype`. For `Array` this returns
aliases of the grid's own vectors.
"""
gridvectors(grid, ::Type{Array}) = gridvectors(grid)
gridvectors(grid, arraytype::Type) = Adapt.adapt(arraytype, gridvectors(grid))

#=================================================#
#============  HOST/DEVICE OUTPUT  ===============#
#=================================================#

"""
    needs_host_y(output) -> Bool

Whether `output` inspects or stores the solution array `y` it is passed on every step
(as opposed to only the interpolated values it requests via `yfun`). Output handlers
that do must be given a host copy when propagating on a device — which costs a full
device-to-host transfer per step, so this defaults to `false`.
"""
needs_host_y(o) = false
needs_host_y(o::Output.HDF5Output) = o.cache

"""
    nostats_only(output) -> Bool

Whether `output` collects no per-step statistics. Statistics functions are called with
the solution array on every step, so on a device they would force a full device-to-host
copy per step rather than per save; [`Luna.run`](@ref) refuses that by default.
Conservatively `false` for output handlers whose statistics function cannot be
inspected.
"""
nostats_only(o) = false
nostats_only(o::Output.MemoryOutput) = o.statsfun === Output.nostats
nostats_only(o::Output.HDF5Output) = o.statsfun === Output.nostats

"""
    HostOutput(output, y)

Wrap an output handler so that it receives host arrays while the propagation runs on a
device. Saved fields are copied device-to-host into a reusable buffer, so a save costs
one transfer and no allocation.

Only the *interpolated* value is copied by default: that is what output handlers
actually store. The per-step solution `y` is passed through untouched unless
[`needs_host_y`](@ref) is true for the handler (e.g. an `HDF5Output` with `cache=true`,
which writes `y` itself), because copying it on every step — rather than on every save —
would dominate the runtime.

Constructed automatically by [`Luna.run`](@ref) when the propagating field is a device
array; `Output.jl` itself is unaware of devices.
"""
mutable struct HostOutput{O, A<:AbstractArray}
    o::O
    ibuf::A                    # buffer for the interpolated (saved) field
    ybuf::Union{Nothing, A}    # buffer for `y`, only when the handler needs it
end

function HostOutput(o, y)
    ibuf = Array{eltype(y)}(undef, size(y))
    ybuf = needs_host_y(o) ? Array{eltype(y)}(undef, size(y)) : nothing
    HostOutput(o, ibuf, ybuf)
end

_tohost!(buf, y) = isdevice(y) ? copyto!(buf, y) : y

function (h::HostOutput)(y, t, dt, yfun)
    yh = isnothing(h.ybuf) ? y : _tohost!(h.ybuf, y)
    h.o(yh, t, dt, ts -> _tohost!(h.ibuf, yfun(ts)))
end

# Metadata and any other calls (e.g. `output(dict; group="grid")`) pass straight through.
(h::HostOutput)(args...; kwargs...) = h.o(args...; kwargs...)

Base.getindex(h::HostOutput, args...) = getindex(h.o, args...)
Base.haskey(h::HostOutput, key) = haskey(h.o, key)
Output.check_cache(h::HostOutput, y, t, dt) = Output.check_cache(h.o, y, t, dt)
