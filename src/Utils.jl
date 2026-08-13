module Utils
import Dates
import FFTW
import Logging
import LibGit2
import FileWatching.Pidfile: mkpidlock
import HDF5
import Luna: settings
import Printf: @sprintf
import Scratch: get_scratch!, clear_scratchspaces!
import Luna

subzero = '\u2080'
subscript(digit::Char) = string(Char(codepoint(subzero)+parse(Int, digit)))
subscript(num::AbstractString) = prod([subscript(chi) for chi in num])
subscript(num::Int) = num >= 0 ? subscript(string(num)) : "₋"*subscript(string(abs(num)))

unsubscript(digit::Char) = string(codepoint(digit)-codepoint(subzero))
unsubscript(num::AbstractString) = prod([unsubscript(chi) for chi in num])

function git_commit()
    try
        repo = LibGit2.GitRepo(lunadir())
        commit = string(LibGit2.GitHash(LibGit2.head(repo)))
        LibGit2.isdirty(repo) && (commit *= " (dirty)")
        return commit
    catch
        "unavailable (Luna is not checkout out for development)"
    end
end

function git_branch()
    try
        repo = LibGit2.GitRepo(lunadir())
        n = string(LibGit2.name(LibGit2.head(repo)))
        branch = split(n, "/")[end]
        return branch
    catch
        "unavailable (Luna is not checkout out for development)"
    end
end

srcdir() = dirname(@__FILE__)

lunadir() = dirname(srcdir())

datadir() = joinpath(srcdir(), "data")

cachedir() = get_scratch!(Luna, "lunacache")

clear_cache() = clear_scratchspaces!(Luna)

function sourcecode()
    src = dirname(@__FILE__)
    luna = dirname(src)
    out = "#= Date: $(Dates.now())\n"
    out *= "git branch: $(git_branch())\n"
    out *= "git commit: $(git_commit())\n"
    out *= "hostname: $(gethostname())\n"
    out *= "=#"
    for folder in (src, luna)
        for obj in readdir(folder)
            if isfile(joinpath(folder, obj))
                if split(obj, ".")[end] in ("md", "jl", "txt", "toml") # avoid binary files
                    out *= "\n" * "#" * "="^8 * obj * "="^8 * "#" * "\n"^2
                    open(joinpath(folder, obj), "r") do file
                        out *= read(file, String)
                    end
                end
            end
        end
    end
    return out
end

function FFTWthreads()
    # An explicit `fftw_threads` setting always wins — including with a single
    # Julia thread. In that case FFTW.jl never installs its Julia-task
    # threading callback (providers.jl registers it only when
    # `Threads.nthreads() > 1`), so libfftw3 falls back to its own native
    # pthreads pool: the combination `JULIA_NUM_THREADS=1` +
    # `Luna.set_fftw_threads(n)` therefore gives threaded FFTs without the
    # partr callback, which segfaults inside `spawnloop`/`spawn_apply` on
    # Julia ≥ 1.12 (observed with FFTW.jl 1.10 on x64 Linux).
    nthr = settings["fftw_threads"]
    nthr > 0 && return nthr
    Threads.nthreads() == 1 ? 1 : 4*Threads.nthreads()
end

"""
    use_native_fftw_threads()

Hand FFT threading back to libfftw3's own pthread pool by deregistering FFTW.jl's
Julia-task (partr) threading callback, which segfaults on Julia ≥ 1.12. This makes
`JULIA_NUM_THREADS > 1` safe to combine with threaded FFTs. Bit-identical: the callback
only changes how FFTW's parallel jobs are executed, never the plans or their results.

Returns `true` if the callback was deregistered, `false` if not applicable (MKL provider,
or FFTW.jl internals changed).
"""
function use_native_fftw_threads()
    # Coupled to FFTW.jl internals (verified against FFTW.jl 1.10): feature-detect and
    # bail out gracefully rather than erroring on future versions.
    FFTW.fftw_provider == "fftw" || return false
    (isdefined(FFTW, :libfftw3) && isdefined(FFTW, :libfftw3f)) || return false
    try
        # Force both libraries to load first: on Julia ≥ 1.11, FFTW.jl installs its
        # callback lazily at first library load, so deregistering before both are loaded
        # would be silently undone. set_num_threads ccalls into both libraries.
        FFTW.set_num_threads(FFTWthreads())
        ccall((:fftw_threads_set_callback, FFTW.libfftw3), Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}), C_NULL, C_NULL)
        ccall((:fftwf_threads_set_callback, FFTW.libfftw3f), Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}), C_NULL, C_NULL)
        return true
    catch e
        Logging.@warn("Could not deregister the FFTW.jl threading callback: $e")
        return false
    end
end

# Global switches for the threaded elementwise kernels: `THREADING[]` turns them off
# entirely; arrays smaller than `THREADING_MINLEN[]` always run serially (threading
# overhead dominates there, e.g. for modal propagation).
const THREADING = Ref(true)
const THREADING_MINLEN = Ref(1<<20)

"Enable or disable Luna's threaded elementwise kernels (threaded and serial execution
are bit-identical; this only matters for benchmarking and testing)."
set_threading(on::Bool) = (THREADING[] = on)

_threading(n) = THREADING[] && n >= THREADING_MINLEN[] && Threads.nthreads() > 1

"""
    tforeach(f, n; ntotal=n)

Call `f(i)` for every `i in 1:n`, splitting the range into one contiguous chunk per
thread for large `n` (see `THREADING_MINLEN`). The work must be elementwise-independent;
threaded and serial execution are then bit-identical. When each `f(i)` covers more than
one element (e.g. one transverse column), pass the total element count as `ntotal` so the
threading threshold reflects the actual work.
"""
function tforeach(f, n::Integer; ntotal::Integer=n)
    if _threading(ntotal)
        nchunks = Threads.nthreads()
        Threads.@threads :static for c in 1:nchunks
            for i in (n*(c-1))÷nchunks + 1 : (n*c)÷nchunks
                f(i)
            end
        end
    else
        for i in 1:n
            f(i)
        end
    end
    return nothing
end

"""
    tchunks(f, arrs...)

Call `f` on matching contiguous linear-index views of `arrs`, one chunk per thread for
large arrays; `f(arrs...)` directly otherwise. Use to thread a fused elementwise
broadcast: `tchunks((d, s) -> (@. d = 2s), dest, src)`. The broadcast must be purely
elementwise; threaded and serial execution are then bit-identical.
"""
function tchunks(f, arrs...)
    n = length(first(arrs))
    if _threading(n)
        nchunks = Threads.nthreads()
        Threads.@threads :static for c in 1:nchunks
            rng = (n*(c-1))÷nchunks + 1 : (n*c)÷nchunks
            f(map(a -> view(a, rng), arrs)...)
        end
    else
        f(arrs...)
    end
    return nothing
end

function loadFFTwisdom()
    FFTW.set_num_threads(FFTWthreads())
    fpath = joinpath(cachedir(), "FFTWcache_$(FFTWthreads())threads")
    lockpath = joinpath(cachedir(), "FFTWlock")
    isdir(cachedir()) || mkpath(cachedir())
    if isfile(fpath)
        Logging.@info("Found FFTW wisdom at $fpath")
        mkpidlock(lockpath; stale_age=600) do
            FFTW.import_wisdom(fpath)
        end
    else
        Logging.@info("No FFTW wisdom found")
    end
end

function saveFFTwisdom()
    fpath = joinpath(cachedir(), "FFTWcache_$(FFTWthreads())threads")
    lockpath = joinpath(cachedir(), "FFTWlock")
    mkpidlock(lockpath; stale_age=600) do
        isfile(fpath) && rm(fpath)
        isdir(cachedir()) || mkpath(cachedir())
        FFTW.export_wisdom(fpath)
    end
    Logging.@info("FFTW wisdom saved to $fpath")
end

function save_dict_h5(fpath, d; force=false, rmold=false)
    if isfile(fpath) && rmold
        rm(fpath)
    end

    function dict2h5(k::AbstractString, v, parent)
        if HDF5.haskey(parent, k) && !force
            error("Dataset $k exists in $fpath. Set force=true to overwrite.")
        end
        parent[k] = v
    end

    function dict2h5(k::AbstractString, v::BitArray, parent)
        if HDF5.haskey(parent, k) && !force
            error("Dataset $k exists in $fpath. Set force=true to overwrite.")
        end
        parent[k] = Array{Bool, 1}(v)
    end

    function dict2h5(k::AbstractString, v::Nothing, parent)
        if HDF5.haskey(parent, k) && !force
            error("Dataset $k exists in $fpath. Set force=true to overwrite.")
        end
        parent[k] = Float64[]
    end

    function dict2h5(k::AbstractString, v::AbstractDict, parent)
        if !HDF5.haskey(parent, k)
            subparent = HDF5.create_group(parent, k)
        else
            subparent = parent[k]
        end
        for (kk, vv) in pairs(v)
            dict2h5(kk, vv, subparent)
        end
    end

    HDF5.h5open(fpath, "cw") do file
        for (k, v) in pairs(d)
            dict2h5(k, v, file)
        end
    end
end

function save_dict_h5(fpath, t::NamedTuple; kwargs...)
    d = Dict{String, Any}()
    for (k, v) in pairs(t)
        d[string(k)] = v
    end
    save_dict_h5(fpath, d; kwargs...)
end

function load_dict_h5(fpath)
    isfile(fpath) || error("Error loading file $fpath: file does not exist")

    function h52dict(x::HDF5.Dataset)
        return read(x)
    end

    function h52dict(x::Union{HDF5.Group, HDF5.File})
        dd = Dict{String, Any}()
        for n in keys(x)
            dd[n] = h52dict(x[n])
        end
        return dd
    end

    d = HDF5.h5open(fpath) do file
        h52dict(file)
    end
end

function format_elapsed(ms::Dates.Millisecond)
    stot = Dates.value(ms)/1000 # total seconds
    seconds = stot % 60
    stot -= seconds
    mtot = stot ÷ 60
    minutes = mtot % 60
    mtot -= minutes
    hours = mtot ÷ 60
    out = @sprintf("%.3f seconds", seconds)
    minstr = abs(minutes) == 1 ? "minute" : "minutes"
    hrstr = abs(hours) == 1 ? "hour" : "hours"
    if abs(hours) > 0
        out = @sprintf("%d %s, ", minutes, minstr) * out
        out = @sprintf("%d %s, ", hours, hrstr) * out
    elseif abs(minutes) > 0
        out = @sprintf("%d %s, ", minutes, minstr) * out
    end
    out
end

end
