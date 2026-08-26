module NonlinearRHS
import FFTW
import AbstractFFTs
import Hankel
import Cubature
import Base: show
import Adapt
import LinearAlgebra: mul!, ldiv!
import NumericalIntegration: integrate, SimpsonEven
import Luna
import Luna: PhysData, Modes, Maths, Grid, Nonlinear, Utils
import Luna.PhysData: wlfreq

"""
    to_time!(Ato, Aω, Aωo, IFTplan)

Transform ``A(ω)`` on normal grid to ``A(t)`` on oversampled time grid.
"""
function to_time!(Ato::AbstractArray{<:Real}, Aω, Aωo, IFTplan)
    N = size(Aω, 1)
    No = size(Aωo, 1)
    scale = (No-1)/(N-1) # Scale factor makes up for difference in FFT array length
    fill!(Aωo, 0)
    copy_scale!(Aωo, Aω, N, scale)
    mul!(Ato, IFTplan, Aωo)
end

function to_time!(Ato::AbstractArray{<:Complex}, Aω, Aωo, IFTplan)
    N = size(Aω, 1)
    No = size(Aωo, 1)
    scale = No/N # Scale factor makes up for difference in FFT array length
    fill!(Aωo, 0)
    copy_scale_both!(Aωo, Aω, N÷2, scale)
    mul!(Ato, IFTplan, Aωo)
end

"""
    to_freq!(Aω, Aωo, Ato, FTplan)

Transform oversampled A(t) to A(ω) on normal grid
"""
function to_freq!(Aω, Aωo, Ato::AbstractArray{<:Real}, FTplan)
    N = size(Aω, 1)
    No = size(Aωo, 1)
    scale = (N-1)/(No-1) # Scale factor makes up for difference in FFT array length
    mul!(Aωo, FTplan, Ato)
    copy_scale!(Aω, Aωo, N, scale)
end

function to_freq!(Aω, Aωo, Ato::AbstractArray{<:Complex}, FTplan)
    N = size(Aω, 1)
    No = size(Aωo, 1)
    scale = N/No # Scale factor makes up for difference in FFT array length
    mul!(Aωo, FTplan, Ato)
    copy_scale_both!(Aω, Aωo, N÷2, scale)
end

"""
    copy_scale!(dest, source, N, scale)

Copy first N elements from source to dest and simultaneously multiply by scale factor.
For multi-dimensional `dest` and `source`, work along first axis.
"""
function copy_scale!(dest::Vector, source::Vector, N, scale)
    for i = 1:N
        dest[i] = scale * source[i]
    end
end

"""
    copy_scale_both!(dest::Vector, source::Vector, N, scale)

Copy first and last N elements from source to first and last N elements in dest
and simultaneously multiply by scale factor.
For multi-dimensional `dest` and `source`, work along first axis.
"""
function copy_scale_both!(dest::Vector, source::Vector, N, scale)
    for i = 1:N
        dest[i] = scale * source[i]
    end
    for i = 1:N
        dest[end-i+1] = scale * source[end-i+1]
    end
end

function copy_scale!(dest, source, N, scale)
    (size(dest)[2:end] == size(source)[2:end]
     || error("dest and source must be same size except along first dimension"))
    _copy_scale!(Utils.backend(dest), dest, source, N, scale)
end

function _copy_scale!(::Utils.CPUBackend, dest, source, N, scale)
    idcs = CartesianIndices(size(dest)[2:end])
    _cpsc_core(dest, source, N, scale, idcs)
end

# On a device the scalar loop cannot run; the leading block is one broadcast over views.
function _copy_scale!(::Utils.DeviceBackend, dest, source, N, scale)
    view(dest, 1:N, ntuple(_ -> Colon(), ndims(dest)-1)...) .=
        scale .* view(source, 1:N, ntuple(_ -> Colon(), ndims(source)-1)...)
    nothing
end

# Threaded over the trailing indices, with the leading (contiguous) axis innermost, so each
# thread walks its own memory. Elementwise, so the result does not depend on the chunking.
function _cpsc_core(dest, source, N, scale, idcs)
    Utils.tforeach(length(idcs); ntotal=N*length(idcs)) do ii
        i = idcs[ii]
        @inbounds for j = 1:N
            dest[j, i] = scale * source[j, i]
        end
    end
end

function copy_scale_both!(dest, source, N, scale)
    (size(dest)[2:end] == size(source)[2:end]
     || error("dest and source must be same size except along first dimension"))
    _copy_scale_both!(Utils.backend(dest), dest, source, N, scale)
end

function _copy_scale_both!(::Utils.CPUBackend, dest, source, N, scale)
    idcs = CartesianIndices(size(dest)[2:end])
    _cpscb_core(dest, source, N, scale, idcs)
end

function _copy_scale_both!(::Utils.DeviceBackend, dest, source, N, scale)
    cd = ntuple(_ -> Colon(), ndims(dest)-1)
    cs = ntuple(_ -> Colon(), ndims(source)-1)
    view(dest, 1:N, cd...) .= scale .* view(source, 1:N, cs...)
    nd = size(dest, 1); ns = size(source, 1)
    view(dest, nd-N+1:nd, cd...) .= scale .* view(source, ns-N+1:ns, cs...)
    nothing
end

function _cpscb_core(dest, source, N, scale, idcs)
    for i in idcs
        for j = 1:N
            dest[j, i] = scale * source[j, i]
        end
        for j = 1:N
            dest[end-j+1, i] = scale * source[end-j+1, i]
        end
    end
end

# Note on noise and ionization/plasma: when the modified shot-noise model is active,
# Et_to_Pt! receives the combined field (field + noise). The noise amplitude is of order
# √(ħω·Δν) ≈ 5×10⁻⁴ √W per mode — roughly 10⁻¹⁴ of typical pulse peak power. This is
# completely negligible for the highly nonlinear ionization rate and plasma
# response, so including noise in the field passed to all response functions is physically
# reasonable. The noise meaningfully affects only Kerr and Raman processes, as intended.
"""
    Et_to_Pt!(Pt, Et, responses, density)

Accumulate responses induced by Et in Pt.
"""
function Et_to_Pt!(Pt, Et, responses, density::Number)
    fill!(Pt, 0)
    for resp! in responses
        resp!(Pt, Et, density)
    end
end

function Et_to_Pt!(Pt, Et, responses, density::AbstractVector)
    fill!(Pt, 0)
    for ii in eachindex(density)
        for resp! in responses[ii]
            resp!(Pt, Et, density[ii])
        end
    end
end

function Et_to_Pt!(Pt, Et, responses, density, idcs)
    for i in idcs
        Et_to_Pt!(view(Pt, :, i), view(Et, :, i), responses, density)
    end
end

"""
    Et_to_Pt_ordered!(Pt, Et, responses, density, idcs)

Like `Et_to_Pt!(Pt, Et, responses, density, idcs)` but applies pointwise responses
(see [`Nonlinear.pointwise`](@ref Luna.Nonlinear.pointwise)) as whole-array broadcasts.
Responses are applied in tuple order, so each element of `Pt` accumulates its
contributions in exactly the same order as the columnwise version — the results are
bit-identical.
"""
function Et_to_Pt_ordered!(Pt, Et, responses::Tuple, density::Number, idcs)
    fill!(Pt, 0)
    for resp! in responses
        if Nonlinear.pointwise(resp!)
            Utils.tchunks(Pt, Et) do Pt, Et
                Pt .+= Nonlinear.pointwise_P.(Ref(resp!), Et, density)
            end
        elseif Nonlinear.batched(resp!)
            resp!(Pt, Et, density) # array-level call with the full field
        else
            Utils.isdevice(Pt) && error(
                "response $(typeof(resp!)) is evaluated column by column, which cannot "*
                "run on a device. Use a response with a pointwise "*
                "(`Nonlinear.pointwise`) or batched (`Nonlinear.batched`) method — e.g. "*
                "`RamanPolarEnvBatched` instead of `RamanPolarEnv`.")
            for i in idcs
                resp!(view(Pt, :, i), view(Et, :, i), density)
            end
        end
    end
end

# fallback (e.g. gas mixtures, where density is a vector and responses is a tuple of
# tuples): use the columnwise version
Et_to_Pt_ordered!(Pt, Et, responses, density, idcs) = Et_to_Pt!(Pt, Et, responses, density, idcs)

"""
    pointwise_Pt!(Pt, Et, responses, density)

Compute the total nonlinear polarisation for a tuple of purely pointwise responses in a
single broadcast. `Pt` may alias `Et` (each output sample depends only on the
corresponding input sample). Reproduces `Et_to_Pt!`'s zero-fill + accumulate sequence
per element, so the result is bit-identical to the columnwise version.
"""
function pointwise_Pt!(Pt, Et, responses::Tuple, density::Number)
    Utils.tchunks(Pt, Et) do Pt, Et
        Pt .= _pointwise_P_total.(Ref(responses), Et, density)
    end
end

@inline _pointwise_P_total(responses, E, density) = _pw_accum(zero(E), responses, E, density)
@inline _pw_accum(p, ::Tuple{}, E, density) = p
@inline _pw_accum(p, resp::Tuple, E, density) =
    _pw_accum(p + Nonlinear.pointwise_P(first(resp), E, density), Base.tail(resp), E, density)

"""
    AbstractTransModal

Supertype of the multimode (modal) transforms [`TransModal`](@ref) (adaptive cubature) and
[`TransModalFixed`](@ref) (fixed quadrature). Both hold the mode collection in a
`Modes.ToSpace` under the field `ts` and the responses under `resp`, which `Stats`,
`Luna.save_modeinfo_maybe` and `Plotting.get_modes` rely on.
"""
abstract type AbstractTransModal end

"""
    TransModal

Transform E(ω) -> Pₙₗ(ω) for multimode propagation via spatial integration.

# Fields
- `Emω_noise`: modal noise field `(nω, nmodes)` for the modified shot-noise model, or
  `nothing`. When present, the noise is projected to real space at each integration point
  and combined with the field in a separate buffer (`Er_nl`) for nonlinear evaluation.
  The propagating field (`Er`) is never modified.
- `Er_noise`: preallocated buffer for the real-space time-domain noise, same shape as `Er`.
- `Er_nl`: preallocated buffer for the combined field + noise, passed to `Et_to_Pt!`.
"""
mutable struct TransModal{tsT, lT, TT, FTT, rT, gT, dT, ddT, nT, eT, enT, enlT} <: AbstractTransModal
    ts::tsT
    full::Bool
    dimlimits::lT
    Emω::Array{ComplexF64,2}
    Erω::Array{ComplexF64,2}
    Erωo::Array{ComplexF64,2}
    Er::Array{TT,2}
    Pr::Array{TT,2}
    Prω::Array{ComplexF64,2}
    Prωo::Array{ComplexF64,2}
    Prmω::Array{ComplexF64,2}
    FT::FTT
    resp::rT
    grid::gT
    densityfun::dT
    density::ddT
    norm!::nT
    ncalls::Int
    z::Float64
    rtol::Float64
    atol::Float64
    mfcn::Int
    err::Array{ComplexF64,2}
    Emω_noise::eT # modal noise field for modified shot-noise model, or nothing
    Er_noise::enT # buffer for real-space time-domain noise, or nothing
    Er_nl::enlT # buffer for field+noise passed to Et_to_Pt!, or nothing
end

function show(io::IO, t::TransModal)
    grid = "grid type: $(typeof(t.grid))"
    modes = "modes: $(t.ts.nmodes)\n"*" "^4*join([string(mi) for mi in t.ts.ms], "\n    ")
    p = t.ts.indices == 1:2 ? "x,y" : t.ts.indices == 1 ? "x" : "y"
    pol = "polarisation: $p"
    samples = "time grid size: $(length(t.grid.t)) / $(length(t.grid.to))"
    resp = "responses: "*join([string(typeof(ri)) for ri in t.resp], "\n    ")
    full = "full: $(t.full)"
    out = join(["TransModal", modes, pol, grid, samples, full, resp], "\n  ")
    print(io, out)
end

"""
    TransModal(grid, ts, FT, resp, densityfun, norm!; rtol=1e-3, atol=0.0, mfcn=300, full=false, noise_field=nothing)

Construct a `TransModal`, transform E(ω) -> Pₙₗ(ω) for modal fields.

# Arguments
- `grid::AbstractGrid` : the grid used in the simulation
- `ts::Modes.ToSpace` : pre-created `ToSpace` for conversion from modal fields to space
- `FT::FFTW.Plan` : the time-frequency Fourier transform for the oversampled time grid
- `resp` : `Tuple` of response functions
- `densityfun` : callable which returns the gas density as a function of `z`
- `norm!` : normalisation function as fctn of `z`, can be created via [`norm_modal`](@ref)
- `rtol::Float=1e-3` : relative tolerance on the `HCubature` integration
- `atol::Float=0.0` : absolute tolerance on the `HCubature` integration
- `mfcn::Int=512` : maximum number of function evaluations for one modal integration
- `full::Bool=false` : if `true`, use full 2-D mode integral, if `false`, only do radial integral
- `noise_field=nothing` : optional `(nω, nmodes)` noise field for the modified shot-noise
  model. Each mode column should contain independent noise with the one-photon-per-mode
  spectral density. Generate with [`Fields.generate_noise_field`](@ref Luna.Fields.generate_noise_field).
"""
function TransModal(tT, grid, ts::Modes.ToSpace, FT, resp, densityfun, norm!;
                    rtol=1e-3, atol=0.0, mfcn=512, full=false, noise_field=nothing)
    Emω = Array{ComplexF64,2}(undef, length(grid.ω), ts.nmodes)
    Erω = Array{ComplexF64,2}(undef, length(grid.ω), ts.npol)
    Erωo = Array{ComplexF64,2}(undef, length(grid.ωo), ts.npol)
    Er = Array{tT,2}(undef, length(grid.to), ts.npol)
    Pr = Array{tT,2}(undef, length(grid.to), ts.npol)
    Prω = Array{ComplexF64,2}(undef, length(grid.ω), ts.npol)
    Prωo = Array{ComplexF64,2}(undef, length(grid.ωo), ts.npol)
    Prmω = Array{ComplexF64,2}(undef, length(grid.ω), ts.nmodes)
    IFT = inv(FT)
    # For the modified shot-noise model, store the modal noise field and allocate a buffer
    # for the real-space time-domain noise. The noise is projected to space at each
    # integration point in Erω_to_Prω!, so we store it in the modal domain.
    if !isnothing(noise_field)
        Emω_noise = copy(noise_field)
        Er_noise = Array{tT,2}(undef, length(grid.to), ts.npol)
        Er_nl = Array{tT,2}(undef, length(grid.to), ts.npol)
    else
        Emω_noise = nothing
        Er_noise = nothing
        Er_nl = nothing
    end
    TransModal(ts, full, Modes.dimlimits(ts.ms[1]), Emω, Erω, Erωo, Er, Pr, Prω, Prωo, Prmω,
               FT, resp, grid, densityfun, densityfun(0.0), norm!, 0, 0.0, rtol, atol, mfcn,
               similar(Prmω), Emω_noise, Er_noise, Er_nl)
end

function TransModal(grid::Grid.RealGrid, args...; kwargs...)
    TransModal(Float64, grid, args...; kwargs...)
end

function TransModal(grid::Grid.EnvGrid, args...; kwargs...)
    TransModal(ComplexF64, grid, args...; kwargs...)
end

function reset!(t::TransModal, Emω::Array{ComplexF64,2}, z::Float64)
    t.Emω .= Emω
    t.ncalls = 0
    t.z = z
    t.dimlimits = Modes.dimlimits(t.ts.ms[1], z=z)
    t.density = t.densityfun(z)
end

function pointcalc!(fval, xs, t::TransModal)
    # TODO: parallelize this in Julia 1.3
    for i in 1:size(xs, 2)
        x1 = xs[1, i]
        # on or outside boundaries are zero
        if x1 <= t.dimlimits[2][1] || x1 >= t.dimlimits[3][1]
            fval[:, i] .= 0.0
            continue
        end
        if size(xs, 1) > 1 # full 2-D mode integral
            x2 = xs[2, i]
            if t.dimlimits[1] == :polar
                pre = x1
            else
                if x2 <= t.dimlimits[2][2] || x2 >= t.dimlimits[3][2]
                    fval[:, i] .= 0.0
                    continue
                end
                pre = 1.0
            end
        else
            if t.dimlimits[1] == :polar
                x2 = 0.0
                pre = 2π*x1
            else
                x2 = 0.0
                pre = 1.0
            end
        end
        x = (x1,x2)
        Erω_to_Prω!(t, x)
        t.ncalls += 1
        # now project back to each mode
        # matrix product (nω x npol) * (npol x nmodes) -> (nω x nmodes)
        mul!(t.Prmω, t.Prω, transpose(t.ts.Ems))
        fval[:, i] .= pre.*reshape(reinterpret(Float64, t.Prmω), length(t.Emω)*2)
    end
end

function Erω_to_Prω!(t, x)
    Modes.to_space!(t.Erω, t.Emω, x, t.ts, z=t.z)
    to_time!(t.Er, t.Erω, t.Erωo, inv(t.FT))
    # Modified shot-noise model: project noise modes to real space at this spatial point,
    # convert to oversampled time domain, and combine with field in a separate buffer (Er_nl)
    # so the propagating field (Er) is never contaminated.
    if !isnothing(t.Emω_noise)
        Modes.to_space!(t.Erω, t.Emω_noise, x, t.ts, z=t.z)
        to_time!(t.Er_noise, t.Erω, t.Erωo, inv(t.FT))
        @. t.Er_nl = t.Er + t.Er_noise
        Et_to_Pt!(t.Pr, t.Er_nl, t.resp, t.density)
    else
        Et_to_Pt!(t.Pr, t.Er, t.resp, t.density)
    end
    @. t.Pr *= t.grid.towin
    to_freq!(t.Prω, t.Prωo, t.Pr, t.FT)
    @. t.Prω *= t.grid.ωwin
    t.norm!(t.Prω)
end

function (t::TransModal)(nl, Eω, z)
    reset!(t, Eω, z)
    _, ll, ul = t.dimlimits
    if t.full
        val, err = Cubature.hcubature_v(
            length(Eω)*2,
            (x, fval) -> pointcalc!(fval, x, t),
            ll, ul, 
            reltol=t.rtol, abstol=t.atol, maxevals=t.mfcn, error_norm=Cubature.L2)
    else
        val, err = Cubature.pcubature_v(
            length(Eω)*2,
            (x, fval) -> pointcalc!(fval, x, t),
            (ll[1],), (ul[1],), 
            reltol=t.rtol, abstol=t.atol, maxevals=t.mfcn, error_norm=Cubature.L2)
    end
    t.err .= reshape(reinterpret(ComplexF64, err), size(nl))
    nl .= reshape(reinterpret(ComplexF64, val), size(nl))
end

"""
    norm_modal(grid; shock=true, arraytype=Array)

Normalisation function for modal propagation. If `shock` is `false`, the intrinsic frequency
dependence of the nonlinear response is ignored, which turns off optical shock formation/
self-steepening. `arraytype` is the array type of the field the function will be applied
to; for a device array type the frequency axis is mirrored there.
"""
function norm_modal(grid; shock=true, arraytype=Array)
    ω0 = PhysData.wlfreq(grid.referenceλ)
    ω = arraytype === Array ? grid.ω : Adapt.adapt(arraytype, grid.ω)
    withshock!(nl) = @. nl *= (-im * ω/4)
    withoutshock!(nl) = @. nl *= (-im * ω0/4)
    shock ? withshock! : withoutshock!
end

"""
    TransModalFixed

Transform E(ω) -> Pₙₗ(ω) for multimode propagation on a **fixed** transverse quadrature
rule (see [`Modes.TransverseQuadrature`](@ref)) — the array-generic replacement for the
adaptive-cubature [`TransModal`](@ref).

Per evaluation: the whole modal field `(nω × nmodes)` is transformed to the oversampled time
domain with one batched FFT, synthesised on the quadrature nodes with a GEMM
(`Et = Emt * S`), the nonlinear responses are applied to the whole `(nto, npol, npts)`
array, the result is projected back onto the modes with a second GEMM (`Pmt = Pt * Wp`,
quadrature weights folded into `Wp`) and transformed back with one batched FFT. All
operations are GEMMs, batched FFT plans and broadcasts, so the transform runs unchanged on
host arrays (threaded) and on device arrays (`arraytype`).

# Fields of note
- `ts::Modes.ToSpace`, `resp`: the mode collection and the responses as passed by the user
  (what `show`, `Stats` and `Luna.save_modeinfo_maybe` inspect); `resp_eval` are the
  responses actually evaluated (array-level variants where available, see
  [`Nonlinear.batched_response`](@ref)).
- `quad`: the quadrature rule; `S`, `Wp`, `Wc`: synthesis, projection and coarse
  (embedded-rule) projection matrices for the current `z`; `S0`, `Wp0`, `Wc0`: the same at
  the reference radius for scale-invariant tapers.
- `err`: the embedded error estimate `P_coarse - P_fine` of the last evaluation, filled by
  [`integral_error!`](@ref) on demand.
- `Et_scratch`: a `(nt × nmodes)` buffer of the field's element type which `Luna.run` uses
  as its window scratch (see [`scratch`](@ref)).
"""
mutable struct TransModalFixed{tsT, qT, ST, WT, EωT, EtmT, EtT, ErrT, FTT, IFTT, rT, reT,
                               gT, gvT, dT, ddT, nT, enT, enlT, esT} <: AbstractTransModal
    ts::tsT
    full::Bool
    quad::qT
    zconstant::Bool
    scale_invariant::Bool
    a0::Float64 # reference core radius for scale-invariant modes
    zmat::Float64 # z at which S/Wp/Wc were last built
    S::ST # (nmodes × npol⋅npts) synthesis matrix
    Wp::WT # (npol⋅npts × nmodes) projection matrix (fine rule)
    Wc::WT # (npol⋅npts × nmodes) projection matrix (embedded coarse rule)
    S0::ST
    Wp0::WT
    Wc0::WT
    Emωo::EωT # (nωo × nmodes) oversampled modal spectrum
    Emt::EtmT # (nto × nmodes) modal field in time
    Emt_nl::enlT # (nto × nmodes) field + noise, or nothing
    Emt_noise::enT # (nto × nmodes) modal noise in time, or nothing
    Et::EtT # (nto × npol⋅npts) real-space field
    Pt::EtT # (nto × npol⋅npts) real-space polarisation
    Pmt::EtmT # (nto × nmodes) modal polarisation in time
    Pmωo::EωT # (nωo × nmodes)
    err::ErrT # (nω × nmodes) embedded error estimate
    FT::FTT # forward plan on (nto × nmodes)
    IFT::IFTT # explicit inverse plan
    resp::rT
    resp_eval::reT
    grid::gT
    gv::gvT # grid vectors on the buffers' array type
    densityfun::dT
    density::ddT
    norm!::nT
    z::Float64
    Et_scratch::esT # (nt × nmodes) scratch for Luna.run
    ncalls::Int # number of quadrature points (kept for Stats compatibility)
end

function show(io::IO, t::TransModalFixed)
    grid = "grid type: $(typeof(t.grid))"
    modes = "modes: $(t.ts.nmodes)\n"*" "^4*join([string(mi) for mi in t.ts.ms], "\n    ")
    p = t.ts.indices == 1:2 ? "x,y" : t.ts.indices == 1 ? "x" : "y"
    pol = "polarisation: $p"
    samples = "time grid size: $(length(t.grid.t)) / $(length(t.grid.to))"
    resp = "responses: "*join([string(typeof(ri)) for ri in t.resp], "\n    ")
    full = "full: $(t.full)"
    q = t.quad
    quad = "quadrature: $(q.kind), nr=$(q.nr), nθ=$(q.nθ), kronrod=$(q.kronrod)"
    out = join(["TransModalFixed", modes, pol, grid, samples, full, quad, resp], "\n  ")
    print(io, out)
end

"""
    TransModalFixed(grid, ts, resp, densityfun, norm!; kwargs...)

Construct a [`TransModalFixed`](@ref).

# Arguments
- `grid::AbstractGrid`: the grid used in the simulation
- `ts::Modes.ToSpace`: the mode collection and polarisation components
- `resp`: `Tuple` of response functions (or tuple of tuples for gas mixtures)
- `densityfun`: callable which returns the gas density as a function of `z`
- `norm!`: normalisation function, see [`norm_modal`](@ref)

# Keyword arguments
- `full::Bool=false`: `true` for the 2-D (r,θ) rule, `false` for the radial rule with an
  azimuthally symmetric integrand (HE₁ₘ mode sets).
- `nr::Int=64`, `nθ::Int=16`: nodes along r (or x) and θ (or y).
- `kronrod::Bool=false`: use a Gauss–Kronrod rule in r (`nr` rounded up to odd) so that
  [`integral_error!`](@ref) has an embedded coarse rule to compare against.
- `noise_field=nothing`: optional `(nω, nmodes)` modal noise field for the modified
  shot-noise model (see [`TransModal`](@ref)).
- `zconstant=nothing`: whether the mode profiles are independent of `z`; `nothing` uses
  `Modes.zconstant` of the modes.
- `arraytype=Array`: array type of the buffers; pass a GPU array type to evaluate on a device.
"""
function TransModalFixed(tT, grid, ts::Modes.ToSpace, resp, densityfun, norm!;
                         full=false, nr=64, nθ=16, kronrod=false, noise_field=nothing,
                         zconstant=nothing, arraytype=Array)
    ms = ts.ms
    nmodes = ts.nmodes
    npol = ts.npol
    dl = Modes.dimlimits(ms[1], z=0.0)
    (dl[1] == :cartesian && !full) && error(
        "cartesian modes need the full 2-D quadrature rule (full=true)")
    quad = Modes.transverse_quadrature(dl, full; nr, nθ, kronrod)
    npts = length(quad)
    zc = isnothing(zconstant) ? all(Modes.zconstant, ms) : zconstant
    si = !zc && all(Modes.scale_invariant, ms) && dl[1] == :polar
    a0 = dl[1] == :polar ? dl[3][1] : NaN
    if full && quad.kind == :polar
        hs = [Modes.azimuthal_order(m) for m in ms]
        if all(!isnothing, hs)
            hmax = maximum(hs)
            nθ < 4hmax + 1 && @warn(
                "nθ=$nθ is below the exactness bound 4⋅$(hmax)+1 for cubic products of "*
                "modes with azimuthal order up to $hmax; use nθ >= $(4hmax+1).")
        end
    end
    # host copies of the matrices at z=0
    Sh, Wph, Wch = _mode_matrices(tT, ts, quad, 0.0)
    S = _to_arraytype(arraytype, Sh); Wp = _to_arraytype(arraytype, Wph)
    Wc = _to_arraytype(arraytype, Wch)
    S0 = si ? copy(S) : S; Wp0 = si ? copy(Wp) : Wp; Wc0 = si ? copy(Wc) : Wc
    nω = length(grid.ω); nωo = length(grid.ωo); nt = length(grid.t); nto = length(grid.to)
    Emωo = Luna.device_zeros(arraytype, ComplexF64, (nωo, nmodes))
    Pmωo = Luna.device_zeros(arraytype, ComplexF64, (nωo, nmodes))
    Emt = Luna.device_zeros(arraytype, tT, (nto, nmodes))
    Pmt = Luna.device_zeros(arraytype, tT, (nto, nmodes))
    Et = Luna.device_zeros(arraytype, tT, (nto, npol*npts))
    Pt = Luna.device_zeros(arraytype, tT, (nto, npol*npts))
    err = Luna.device_zeros(arraytype, ComplexF64, (nω, nmodes))
    Et_scratch = Luna.device_zeros(arraytype, tT, (nt, nmodes))
    FT = _plan_forward(Emt, 1)
    IFT = inv(FT)
    if !isnothing(noise_field)
        # the noise enters linearly, so it can be transformed to the time domain once
        # (a single-mode noise field may be a plain vector)
        noise_field = reshape(noise_field, nω, nmodes)
        Emt_noise_h = zeros(tT, (nto, nmodes))
        FTh = arraytype === Array ? FT : _plan_forward(Emt_noise_h, 1)
        to_time!(Emt_noise_h, noise_field, zeros(ComplexF64, (nωo, nmodes)), inv(FTh))
        Emt_noise = _to_arraytype(arraytype, Emt_noise_h)
        Emt_nl = similar(Emt)
    else
        Emt_noise = nothing
        Emt_nl = nothing
    end
    gv = Luna.gridvectors(grid, arraytype)
    resp_eval = Nonlinear.batched_responses(resp; arraytype)
    TransModalFixed(ts, full, quad, zc, si, a0, 0.0, S, Wp, Wc, S0, Wp0, Wc0,
                    Emωo, Emt, Emt_nl, Emt_noise, Et, Pt, Pmt, Pmωo, err, FT, IFT,
                    resp, resp_eval, grid, gv, densityfun, densityfun(0.0), norm!, 0.0,
                    Et_scratch, npts)
end

function TransModalFixed(grid::Grid.RealGrid, args...; kwargs...)
    TransModalFixed(Float64, grid, args...; kwargs...)
end

function TransModalFixed(grid::Grid.EnvGrid, args...; kwargs...)
    TransModalFixed(ComplexF64, grid, args...; kwargs...)
end

_to_arraytype(::Type{Array}, x) = x
_to_arraytype(arraytype, x) = Adapt.adapt(arraytype, x)

# forward FFT plan along `dims` for a real (r2c) or complex (c2c) prototype, on its backend
_plan_forward(x::AbstractArray{<:Real}, dims) = Utils.plan_rfft_backend(x, dims)
_plan_forward(x::AbstractArray{<:Complex}, dims) = Utils.plan_fft_backend(x, dims)

# Synthesis and projection matrices on the host for the mode collection at position z.
# Column p + (i-1)*npol of S (row of Wp) is polarisation component p at quadrature node i,
# matching a (nto, npol, npts) reshape of the real-space arrays.
function _mode_matrices(tT, ts::Modes.ToSpace, quad, z)
    dl = Modes.dimlimits(ts.ms[1], z=z)
    Ems = Modes.mode_matrix(ts.ms, ts.indices, Modes.quadrature_nodes(quad, dl); z)
    nmodes, npol, npts = size(Ems)
    w = Modes.quadrature_weights(quad, dl)
    wc = Modes.quadrature_weights(quad, dl; coarse=true)
    S = Matrix{tT}(reshape(Ems, nmodes, npol*npts))
    Wp = Matrix{tT}(transpose(reshape(Ems .* reshape(w, 1, 1, npts), nmodes, npol*npts)))
    Wc = Matrix{tT}(transpose(reshape(Ems .* reshape(wc, 1, 1, npts), nmodes, npol*npts)))
    S, Wp, Wc
end

"""
    update_matrices!(t::TransModalFixed, z)

Bring the synthesis/projection matrices of `t` to position `z`: nothing to do for
`z`-independent modes; a rescaling for scale-invariant (tapered Marcatili-type) modes;
otherwise a re-evaluation of the mode fields on the host and an upload.
"""
function update_matrices!(t::TransModalFixed, z)
    (t.zconstant || z == t.zmat) && return
    if t.scale_invariant
        a = Modes.dimlimits(t.ts.ms[1], z=z)[3][1]
        s = t.a0/a
        t.S .= t.S0 .* s
        t.Wp .= t.Wp0 ./ s
        t.Wc .= t.Wc0 ./ s
    else
        Sh, Wph, Wch = _mode_matrices(eltype(t.S), t.ts, t.quad, z)
        copyto!(t.S, Sh); copyto!(t.Wp, Wph); copyto!(t.Wc, Wch)
    end
    t.zmat = z
    nothing
end

function (t::TransModalFixed)(nl, Eω, z)
    t.z = z
    t.density = t.densityfun(z)
    update_matrices!(t, z)
    to_time!(t.Emt, Eω, t.Emωo, t.IFT) # (nω × nmodes) -> (nto × nmodes)
    Emt = t.Emt
    if !isnothing(t.Emt_noise)
        # Modified shot-noise model: field + noise in a separate buffer, the propagating
        # field is never contaminated.
        @. t.Emt_nl = t.Emt + t.Emt_noise
        Emt = t.Emt_nl
    end
    mul!(t.Et, Emt, t.S) # synthesise on the quadrature nodes
    apply_responses!(t.Pt, t.Et, t.resp_eval, t.density, t.ts.npol)
    mul!(t.Pmt, t.Pt, t.Wp) # project back onto the modes
    _finish_modal!(nl, t, t.Pmt)
end

# time window, transform to frequency, spectral window and normalisation of a modal
# time-domain polarisation
function _finish_modal!(nl, t::TransModalFixed, Pmt)
    Pmt .*= t.gv.towin # (nto × nmodes) .* (nto,)
    to_freq!(nl, t.Pmωo, Pmt, t.FT)
    nl .*= t.gv.ωwin
    t.norm!(nl)
    nl
end

"""
    integral_error!(t::TransModalFixed)

Fill `t.err` with the embedded error estimate `P_coarse(ω) - P_fine(ω)` of the **last**
evaluation of `t` (the real-space polarisation is still held in `t.Pt`), where the coarse
rule is the Gauss subset of the Kronrod rule in r (only if constructed with
`kronrod=true`) and every other node in θ. Returns `t.err`.
"""
function integral_error!(t::TransModalFixed)
    if !has_error_estimate(t)
        fill!(t.err, NaN)
        return t.err
    end
    mul!(t.Pmt, t.Pt, t.Wc)
    mul!(t.Pmt, t.Pt, t.Wp, -1.0, 1.0)
    _finish_modal!(t.err, t, t.Pmt)
end

"""
    has_error_estimate(t::TransModalFixed)

Whether the quadrature rule of `t` has an embedded coarse rule (Kronrod in r, or an even
number of θ nodes) so that [`integral_error!`](@ref) is meaningful.
"""
has_error_estimate(t::TransModalFixed) =
    t.quad.kronrod || (t.full && iseven(t.quad.nθ) && t.quad.nθ >= 4)

"""
    apply_responses!(Pt, Et, responses, density, npol)

Accumulate the nonlinear polarisation of the real-space field `Et` (shape
`(nto, npol⋅npts)`, polarisation component fastest) into `Pt` for the modal transforms.
Scalar fields (`npol == 1`) use [`Et_to_Pt_ordered!`](@ref) (pointwise/batched responses as
whole-array operations). Vector fields use array-level `(nto, npol, npts)` methods for
responses with `Nonlinear.batched(resp, npol)`, and otherwise the serial column-by-column
call with a contiguous `(nto, npol)` view per node — the legacy responses own internal
buffers and are not thread-safe, and this path cannot run on a device.
"""
function apply_responses!(Pt, Et, responses, density, npol)
    nto = size(Et, 1)
    npts = size(Et, 2) ÷ npol
    Et3 = reshape(Et, nto, npol, npts)
    Pt3 = reshape(Pt, nto, npol, npts)
    if npol == 1
        # (nto, 1, npts): a scalar field per column, as the batched free-space responses
        # expect a 3-D array and the columnwise ones a vector view per node
        Et_to_Pt_ordered!(Pt3, Et3, responses, density, CartesianIndices((1, npts)))
    else
        fill!(Pt3, 0)
        _apply_vector!(Pt3, Et3, responses, density)
    end
    Pt
end

function _apply_vector!(Pt3, Et3, responses::Tuple, density::Number)
    npol = size(Et3, 2)
    for resp! in responses
        if Nonlinear.batched(resp!, npol)
            resp!(Pt3, Et3, density)
        else
            Utils.isdevice(Pt3) && error(
                "response $(typeof(resp!)) has no array-level method for vector fields "*
                "and is evaluated column by column, which cannot run on a device.")
            for i in 1:size(Et3, 3)
                resp!(view(Pt3, :, :, i), view(Et3, :, :, i), density)
            end
        end
    end
end

# gas mixtures: responses is a tuple of tuples, density a vector
function _apply_vector!(Pt3, Et3, responses, density::AbstractVector)
    for ii in eachindex(density)
        _apply_vector!(Pt3, Et3, responses[ii], density[ii])
    end
end

"""
    TransModeAvg

Transform E(ω) -> Pₙₗ(ω) for mode-averaged single-mode propagation.

# Fields
- `Et_noise`: precomputed time-domain noise on the oversampled grid for the modified
  shot-noise model (Chen & Wise, arXiv:2410.20567), or `nothing` for the traditional model.
- `Et_nl`: preallocated buffer for the combined field + noise. When `Et_noise` is present,
  `Et_nl = Eto + Et_noise` is computed at each step and passed to `Et_to_Pt!`. The
  propagating field (`Eto`) is never modified; dispersion acts only on the physical field.
"""
struct TransModeAvg{TT, FTT, rT, gT, dT, nT, aT, eT, nlT}
    Pto::Vector{TT}
    Eto::Vector{TT}
    Eωo::Vector{ComplexF64}
    Pωo::Vector{ComplexF64}
    FT::FTT
    resp::rT
    grid::gT
    densityfun::dT
    norm!::nT
    aeff::aT # function which returns effective area
    Et_noise::eT # time-domain noise for modified shot-noise model, or nothing
    Et_nl::nlT # buffer for field+noise passed to Et_to_Pt!, or nothing
end

function show(io::IO, t::TransModeAvg)
    grid = "grid type: $(typeof(t.grid))"
    samples = "time grid size: $(length(t.grid.t)) / $(length(t.grid.to))"
    resp = "responses: "*join([string(typeof(ri)) for ri in t.resp], "\n    ")
    out = join(["TransModeAvg", grid, samples, resp], "\n  ")
    print(io, out)
end

"""
    TransModeAvg(TT, grid, FT, resp, densityfun, norm!, aeff; noise_field=nothing)

Construct a `TransModeAvg` transform for mode-averaged propagation.

# Keyword arguments
- `noise_field=nothing`: optional frequency-domain noise field (on the normal grid) for the
  modified shot-noise model. When provided, it is converted to the oversampled time grid and
  stored as `Et_noise` for injection into the nonlinear operator at every propagation step.
  Generate with [`Fields.generate_noise_field`](@ref Luna.Fields.generate_noise_field).
"""
function TransModeAvg(TT, grid, FT, resp, densityfun, norm!, aeff; noise_field=nothing)
    Eωo = zeros(ComplexF64, length(grid.ωo))
    Eto = zeros(TT, length(grid.to))
    Pto = similar(Eto)
    Pωo = similar(Eωo)
    # Precompute time-domain noise on the oversampled grid if noise_field is provided.
    # Uses the same ω→t conversion path as to_time!: copy_scale! into oversampled spectral
    # array, then inverse FFT. The result is constant throughout propagation.
    if !isnothing(noise_field)
        Eωo_noise = zeros(ComplexF64, length(grid.ωo))
        Et_noise = zeros(TT, length(grid.to))
        to_time!(Et_noise, noise_field, Eωo_noise, inv(FT))
        Et_nl = zeros(TT, length(grid.to))
    else
        Et_noise = nothing
        Et_nl = nothing
    end
    TransModeAvg(Pto, Eto, Eωo, Pωo, FT, resp, grid, densityfun, norm!, aeff, Et_noise, Et_nl)
end

function TransModeAvg(grid::Grid.RealGrid, FT, resp, densityfun, norm!, aeff; kwargs...)
    TransModeAvg(Float64, grid, FT, resp, densityfun, norm!, aeff; kwargs...)
end

function TransModeAvg(grid::Grid.EnvGrid, FT, resp, densityfun, norm!, aeff; kwargs...)
    TransModeAvg(ComplexF64, grid, FT, resp, densityfun, norm!, aeff; kwargs...)
end

const nlscale = sqrt(PhysData.ε_0*PhysData.c/2)

function (t::TransModeAvg)(nl, Eω, z)
    to_time!(t.Eto, Eω, t.Eωo, inv(t.FT))
    sc = nlscale*sqrt(t.aeff(z))
    @. t.Eto /= sc
    # Modified shot-noise model: compute field+noise in a separate buffer (Et_nl) so that
    # the propagating field (Eto) is never contaminated. The noise is scaled by the same
    # normalisation factor (nlscale × √Aeff) so it enters in physical units.
    if !isnothing(t.Et_noise)
        @. t.Et_nl = t.Eto + t.Et_noise / sc
        Et_to_Pt!(t.Pto, t.Et_nl, t.resp, t.densityfun(z))
    else
        Et_to_Pt!(t.Pto, t.Eto, t.resp, t.densityfun(z))
    end
    @. t.Pto *= t.grid.towin
    to_freq!(nl, t.Pωo, t.Pto, t.FT)
    t.norm!(nl, z)
    for i in eachindex(nl)
        !t.grid.sidx[i] && continue
        nl[i] *= t.grid.ωwin[i]
    end
end

function norm_mode_average(grid, βfun!, aeff; shock=true)
    β = zeros(Float64, length(grid.ω))
    shockterm = shock ? grid.ω.^2 : grid.ω .* PhysData.wlfreq(grid.referenceλ)
    pre = @. -im*shockterm/4 / nlscale / PhysData.c
    function norm!(nl, z)
        βfun!(β, z)
        sqrtaeff = sqrt(aeff(z))
        for i in eachindex(nl)
            !grid.sidx[i] && continue
            nl[i] *= pre[i]/β[i]*sqrtaeff
        end
    end
end

function norm_mode_average_gnlse(grid, aeff; shock=true)
    shockterm = shock ? grid.ω.^2 : grid.ω .* PhysData.wlfreq(grid.referenceλ)
    pre = @. -im*shockterm/(2*PhysData.c^(3/2)*sqrt(2*PhysData.ε_0))/(grid.ω/PhysData.c)
    function norm!(nl, z)
        sqrtaeff = sqrt(aeff(z))
        for i in eachindex(nl)
            !grid.sidx[i] && continue
            nl[i] *= pre[i]*sqrtaeff
        end
    end
end

"""
    TransRadial

Transform E(ω) -> Pₙₗ(ω) for radially symmetric free-space propagation.

# Fields
- `Et_noise`: precomputed time-domain noise on the oversampled real-space grid `(nto, nr)`
  for the modified shot-noise model, or `nothing`.
- `Et_nl`: preallocated buffer for the combined field + noise, passed to `Et_to_Pt!`. The
  propagating field (`Eto`) is never modified.
"""
struct TransRadial{TT, HTT, FTT, nT, rT, gT, dT, iT, eT, nlT}
    QDHT::HTT # Hankel transform (space to k-space)
    FT::FTT # Fourier transform (time to frequency)
    normfun::nT # Function which returns normalisation factor
    resp::rT # nonlinear responses (tuple of callables)
    grid::gT # time grid
    densityfun::dT # callable which returns density
    Pto::Array{TT,2} # Buffer array for NL polarisation on oversampled time grid
    Eto::Array{TT,2} # Buffer array for field on oversampled time grid
    Eωo::Array{ComplexF64,2} # Buffer array for field on oversampled frequency grid
    Pωo::Array{ComplexF64,2} # Buffer array for NL polarisation on oversampled frequency grid
    idcs::iT # CartesianIndices for Et_to_Pt! to iterate over
    Et_noise::eT # time-domain noise for modified shot-noise model, or nothing
    Et_nl::nlT # buffer for field+noise passed to Et_to_Pt!, or nothing
end

function show(io::IO, t::TransRadial)
    grid = "grid type: $(typeof(t.grid))"
    samples = "time grid size: $(length(t.grid.t)) / $(length(t.grid.to))"
    resp = "responses: "*join([string(typeof(ri)) for ri in t.resp], "\n    ")
    nr = "radial points: $(t.QDHT.N)"
    R = "aperture: $(t.QDHT.R)"
    out = join(["TransRadial", grid, samples, nr, R, resp], "\n  ")
    print(io, out)
end

"""
    TransRadial(TT, grid, HT, FT, responses, densityfun, normfun; noise_field=nothing)

Construct a `TransRadial` to calculate the reciprocal-domain nonlinear polarisation.

# Keyword arguments
- `noise_field=nothing`: optional `(nω, nk)` frequency/k-space noise field for the modified
  shot-noise model. When provided, it is converted to the real-space time domain `(nto, nr)`
  via inverse FFT and inverse Hankel transform, and stored as `Et_noise`.
  Generate with [`Fields.generate_noise_field`](@ref Luna.Fields.generate_noise_field).
"""
function TransRadial(TT, grid, HT, FT, responses, densityfun, normfun; noise_field=nothing)
    Eωo = zeros(ComplexF64, (length(grid.ωo), HT.N))
    Eto = zeros(TT, (length(grid.to), HT.N))
    Pto = similar(Eto)
    Pωo = similar(Eωo)
    idcs = CartesianIndices(size(Pto)[2:end])
    # Precompute time-domain noise in real space: ω→t via to_time!, then k→r via QDHT⁻¹
    if !isnothing(noise_field)
        Eωo_noise = zeros(ComplexF64, (length(grid.ωo), HT.N))
        Et_noise = zeros(TT, (length(grid.to), HT.N))
        to_time!(Et_noise, noise_field, Eωo_noise, inv(FT))
        ldiv!(Et_noise, HT, Et_noise)
        Et_nl = zeros(TT, (length(grid.to), HT.N))
    else
        Et_noise = nothing
        Et_nl = nothing
    end
    TransRadial(HT, FT, normfun, responses, grid, densityfun, Pto, Eto, Eωo, Pωo, idcs, Et_noise, Et_nl)
end

function TransRadial(grid::Grid.RealGrid, args...; kwargs...)
    TransRadial(Float64, grid, args...; kwargs...)
end

function TransRadial(grid::Grid.EnvGrid, args...; kwargs...)
    TransRadial(ComplexF64, grid, args...; kwargs...)
end

"""
    (t::TransRadial)(nl, Eω, z)

Calculate the reciprocal-domain (ω-k-space) nonlinear response due to the field `Eω` and
place the result in `nl`
"""
function (t::TransRadial)(nl, Eω, z)
    to_time!(t.Eto, Eω, t.Eωo, inv(t.FT)) # transform ω -> t
    ldiv!(t.Eto, t.QDHT, t.Eto) # transform k -> r
    # Modified shot-noise: compute field+noise in separate buffer (Et_nl) so the
    # propagating field (Eto) is never contaminated.
    if !isnothing(t.Et_noise)
        @. t.Et_nl = t.Eto + t.Et_noise
        Et_to_Pt!(t.Pto, t.Et_nl, t.resp, t.densityfun(z), t.idcs)
    else
        Et_to_Pt!(t.Pto, t.Eto, t.resp, t.densityfun(z), t.idcs)
    end
    @. t.Pto *= t.grid.towin # apodisation
    mul!(t.Pto, t.QDHT, t.Pto) # transform r -> k
    to_freq!(nl, t.Pωo, t.Pto, t.FT) # transform t -> ω
    nl .*= t.grid.ωwin .* (-im.*t.grid.ω)./(2 .* t.normfun(z))
end

"""
    const_norm_radial(ω, q, nfun)

Make function to return normalisation factor for radial symmetry without re-calculating at
every step. 
"""
function const_norm_radial(grid, q, nfun)
    nfunω = (ω; z) -> nfun(wlfreq(ω))
    normfun = norm_radial(grid, q, nfunω)
    out = copy(normfun(0.0))
    function norm(z)
        return out
    end
    return norm
end

"""
    norm_radial(ω, q, nfun)

Make function to return normalisation factor for radial symmetry. 

!!! note
    Here, `nfun(ω; z)` needs to take frequency `ω` and a keyword argument `z`.
"""
function norm_radial(grid, q, nfun)
    ω = grid.ω
    out = zeros(Float64, (length(ω), q.N))
    kr2 = q.k.^2
    k2 = zeros(Float64, length(ω))
    function norm(z)
        k2[grid.sidx] .= (nfun.(grid.ω[grid.sidx]; z=z).*grid.ω[grid.sidx]./PhysData.c).^2
        for ir = 1:q.N
            for iω in eachindex(ω)
                if ω[iω] == 0
                    out[iω, ir] = 1.0
                    continue
                end
                βsq = k2[iω] - kr2[ir]
                if βsq <= 0
                    out[iω, ir] = 1.0
                    continue
                end
                out[iω, ir] = sqrt(βsq)/(PhysData.μ_0*ω[iω])
            end
        end
        return out
    end
    return norm
end

"""
    TransFree

Transform E(ω) -> Pₙₗ(ω) for 3D free-space propagation.

# Fields
- `Et_noise`: precomputed time-domain noise on the oversampled real-space grid `(nto, ny, nx)`
  for the modified shot-noise model, or `nothing`.
- `Et_nl`: preallocated buffer for the combined field + noise, passed to `Et_to_Pt!`. The
  propagating field (`Eto`) is never modified.

# Fast path
When the grid is not oversampled (`length(grid.ωo) == length(grid.ω)`, `scale == 1` — the
usual case for `EnvGrid` with `thg=false`), the field is complex, and no noise field is
present, the copy through the oversampled buffers is an exact identity and is skipped:
`Eωo` and `Pωo` are `nothing` and the FFTs run directly between the solution array, `Eto`
and the output. If additionally all responses are pointwise
(see [`Nonlinear.pointwise`](@ref Luna.Nonlinear.pointwise)), the polarisation overwrites
`Eto` in place and `Pto` is `nothing` too. Both fast paths are bit-identical to the
general path; disable with the constructor keyword `fastpath=false`.

# General path
A real (`RealGrid`) field cannot take the fast path — the inverse of a real-to-complex plan
destroys its input, which on the fast path would be the solver's own state array — so it
always goes through the oversampled buffers. That path is fully backend-dispatched: it uses
the same threaded/device kernels as the fast path (`copy_scale!`, `Et_to_Pt_ordered!` or
`pointwise_Pt!`, `_apply_towin!`, `_scale_nl!`) and runs on a device provided every response
is pointwise or batched and there is no noise field. `Pωo` always aliases `Eωo`, and `Pto`
aliases `Eto` when every response is pointwise, so the general path costs two field-sized
buffers rather than four.
"""
mutable struct TransFree{FTT, iFTT, nT, rT, gT, gvT, xygT, dT, tT, pT, oT, iT, eT, nlT, wT}
    FT::FTT # 3D Fourier transform (space to k-space and time to frequency)
    IFT::iFTT # explicit inverse of FT on a device, `nothing` on the host (see below)
    normfun::nT # Function which returns normalisation factor
    resp::rT # nonlinear responses (tuple of callables)
    grid::gT # time grid
    gv::gvT # grid vectors (ω and the apodisation windows) on the buffers' array type
    xygrid::xygT
    densityfun::dT # callable which returns density
    Pto::pT # buffer for oversampled time-domain NL polarisation; `nothing` (fast path) or
            # `=== Eto` (general path) when every response is pointwise and it may alias
    Eto::tT # buffer for oversampled time-domain field
    Eωo::oT # buffer for oversampled frequency-domain field, or nothing (fast path)
    Pωo::oT # oversampled frequency-domain NL polarisation: `=== Eωo` on the general path
            # (the inverse transform consumes Eωo), or nothing (fast path)
    scale::Float64 # scale factor to be applied during oversampling
    idcs::iT # iterating over these slices Eto/Pto into Vectors, one at each position
    Et_noise::eT # time-domain noise for modified shot-noise model, or nothing
    Et_nl::nlT # buffer for field+noise passed to Et_to_Pt!, or nothing
    Et_win::wT # (nt, ny, nx) window-application scratch on the COARSE grid, or nothing
               # when the grid is not oversampled (Eto already has that shape)
end

function show(io::IO, t::TransFree)
    grid = "grid type: $(typeof(t.grid))"
    samples = "time grid size: $(length(t.grid.t)) / $(length(t.grid.to))"
    resp = "responses: "*join([string(typeof(ri)) for ri in t.resp], "\n    ")
    y = "y grid: $(minimum(t.xygrid.y)) to $(maximum(t.xygrid.y)), N=$(length(t.xygrid.y))"
    x = "x grid: $(minimum(t.xygrid.x)) to $(maximum(t.xygrid.x)), N=$(length(t.xygrid.x))"
    out = join(["TransFree", grid, samples, y, x, resp], "\n  ")
    print(io, out)
end

"""
    TransFree(TT, scale, grid, xygrid, FT, responses, densityfun, normfun;
              noise_field=nothing, fastpath=true, arraytype=Array)

Construct a `TransFree` to calculate the reciprocal-domain nonlinear polarisation for 3D
free-space propagation.

# Keyword arguments
- `noise_field=nothing`: optional `(nω, ny, nx)` frequency/k-space noise field for the
  modified shot-noise model. When provided, it is converted to the real-space oversampled
  time domain `(nto, ny, nx)` via `copy_scale!` and 3D inverse FFT, and stored as `Et_noise`.
  Generate with [`Fields.generate_noise_field`](@ref Luna.Fields.generate_noise_field).
- `fastpath=true`: allow the bit-identical fast path which skips the oversampled buffers
  when the grid is not oversampled (see [`TransFree`](@ref)). Set to `false` to force the
  general path (e.g. for testing).
- `arraytype=Array`: array type for the working buffers. Pass a GPU array type to run the
  transform on a device; the grid vectors are mirrored to the same type (see
  [`Luna.GridVectors`](@ref Luna.GridVectors)) and an explicit inverse FFT plan is stored, because
  `ldiv!` through a device plan's `ScaledPlan` would otherwise be built per call.
- `Eto=nothing`: adopt this `(nto, ny, nx)` array as the oversampled time-domain buffer
  instead of allocating one. `Luna.setup`'s real free-space method passes the very array it
  planned this `FT` against, so the plan prototype and the transform's buffer are one
  allocation rather than two — 9 GiB at field-resolved production shapes. The buffer's
  contents are irrelevant (the first act of every RHS evaluation is to overwrite it), which
  is what makes it safe to hand over an array an FFTW planner has scribbled on.
"""
function TransFree(TT, scale, grid, xygrid, FT, responses, densityfun, normfun;
                   noise_field=nothing, fastpath=true, arraytype=Array, Eto=nothing)
    Ny = length(xygrid.y)
    Nx = length(xygrid.x)
    # The fast path is only exact (and only safe: c2r ldiv! would destroy its input, and on
    # the fast path that input is the solver's own state) for complex (envelope) fields on a
    # non-oversampled grid without a noise field.
    fast = (fastpath && TT <: Complex && length(grid.ωo) == length(grid.ω)
            && scale == 1 && isnothing(noise_field))
    ondevice = arraytype !== Array
    # On a device a columnwise response cannot run at all, so substitute its batched
    # equivalent where one exists — the same thing TransModalFixed does. Deliberately
    # device-only: on the host the columnwise forms work, and swapping them silently would
    # move existing host results at the ~1e-15 level.
    responses = ondevice ? Nonlinear.batched_responses(responses; arraytype) : responses
    if ondevice
        isnothing(noise_field) || error(
            "device propagation with a noise field is not supported: the noise field is "*
            "generated and transformed on the host.")
        _device_responses_ok(responses) || error(
            "device propagation requires every response to be pointwise "*
            "(`Nonlinear.pointwise`) or batched (`Nonlinear.batched`); got "*
            "$(join(map(r -> string(typeof(r)), responses), ", ")). Columnwise responses "*
            "cannot run on a device.")
    end
    if isnothing(Eto)
        Eto = Luna.device_zeros(arraytype, TT, (length(grid.to), Ny, Nx))
    else
        (eltype(Eto) === TT && size(Eto) == (length(grid.to), Ny, Nx)) || error(
            "the supplied Eto buffer is $(eltype(Eto)) $(size(Eto)); this grid needs "*
            "$TT $((length(grid.to), Ny, Nx))")
    end
    allpointwise = all(Nonlinear.pointwise, responses)
    if fast
        Eωo = nothing
        Pωo = nothing
        Pto = allpointwise ? nothing : similar(Eto)
    else
        Eωo = Luna.device_zeros(arraytype, ComplexF64, (length(grid.ωo), Ny, Nx))
        # The polarisation's forward transform reuses the field's spectral buffer: the
        # inverse transform above consumes `Eωo` (a c2r transform destroys it outright) and
        # nothing reads it again, and the next call refills it from scratch. At 3-D
        # field-resolved production shapes that is 9 GiB not spent.
        Pωo = Eωo
        # ...and when every response is pointwise the polarisation may overwrite the field
        # buffer, exactly as the fast path does.
        Pto = allpointwise ? Eto : similar(Eto)
    end
    idcs = CartesianIndices((Ny, Nx))
    # Precompute time-domain noise in real space:
    # copy_scale! into oversampled spectral grid, then 3D IFFT: (ω,ky,kx) → (t,y,x)
    if !isnothing(noise_field)
        Eωo_noise = zeros(ComplexF64, (length(grid.ωo), Ny, Nx))
        N = length(grid.ω)
        copy_scale!(Eωo_noise, noise_field, N, scale)
        Et_noise = zeros(TT, (length(grid.to), Ny, Nx))
        ldiv!(Et_noise, FT, Eωo_noise)
        Et_nl = zeros(TT, (length(grid.to), Ny, Nx))
    else
        Et_noise = nothing
        Et_nl = nothing
    end
    # `Luna.run` needs a COARSE-grid time-domain buffer to apply the apodisation windows in.
    # When the grid is not oversampled that is `Eto` itself (see `scratch`); when it is, own
    # the buffer here rather than letting `Luna.run` allocate it with `FT \ Eω`. On a real
    # grid that expression applies a c2r transform to the solver's state, which FFTW happens
    # to survive (it copies) but which is not a property to rely on through a device FFT
    # library. It also gives `Luna.setup` a real prototype to plan the state rfft against.
    Et_win = length(grid.to) == length(grid.t) ? nothing :
             Luna.device_zeros(arraytype, TT, (length(grid.t), Ny, Nx))
    # The host path keeps using `ldiv!(dest, FT, src)` exactly as before; only the device
    # path needs the inverse plan materialised up front.
    IFT = ondevice ? inv(FT) : nothing
    gv = Luna.gridvectors(grid, arraytype)
    TransFree(FT, IFT, normfun, responses, grid, gv, xygrid, densityfun,
              Pto, Eto, Eωo, Pωo, scale, idcs, Et_noise, Et_nl, Et_win)
end

# A tuple of tuples (gas mixtures) never satisfies this — `pointwise`/`batched` fall back to
# `false` for a Tuple — which is the intended answer: those go through the columnwise
# fallback of `Et_to_Pt_ordered!`.
_device_responses_ok(responses) =
    all(r -> Nonlinear.pointwise(r) || Nonlinear.batched(r), responses)

function TransFree(grid::Grid.RealGrid, args...; kwargs...)
    N = length(grid.ω)
    No = length(grid.ωo)
    scale = (No-1)/(N-1)
    TransFree(Float64, scale, grid, args...; kwargs...)
end

function TransFree(grid::Grid.EnvGrid, args...; kwargs...)
    N = length(grid.ω)
    No = length(grid.ωo)
    scale = No/N
    TransFree(ComplexF64, scale, grid, args...; kwargs...)
end

"""
    scratch(transform)

Return a time-domain buffer owned by `transform` which is dead between RHS evaluations
and matches the shape of `FT \\ Eω` on the coarse grid, or `nothing` if no compatible
buffer exists. `Luna.run` reuses it as its window-application scratch instead of
allocating another field-sized array.
"""
scratch(t) = nothing
# `Et_win` exists exactly when the grid is oversampled; otherwise `Eto` already has the
# coarse shape and is dead between RHS evaluations.
scratch(t::TransFree) = isnothing(t.Et_win) ? t.Eto : t.Et_win
scratch(t::TransModalFixed) = t.Et_scratch

"""
    device_gridvectors(transform, grid)

The grid vectors (`ω` and the apodisation windows) on the transform's own array type, so
that `Luna.run`'s window kernels broadcast against a device state array correctly.
Transforms which keep no mirror fall back to the grid's own (host) vectors, which is
what every host propagation uses.
"""
device_gridvectors(t, grid) = Luna.gridvectors(grid)
device_gridvectors(t::TransFree, grid) = t.gv
device_gridvectors(t::TransModalFixed, grid) = t.gv

"""
    (t::TransFree)(nl, Eω, z)

Calculate the reciprocal-domain (ω-kx-ky-space) nonlinear response due to the field `Eω`
and place the result in `nl`.
"""
function (t::TransFree)(nl, Eωk, z)
    if t.Eωo === nothing # fast path (branch resolved at compile time)
        trans_free_fast!(nl, t, Eωk, z)
    else
        trans_free_general!(nl, t, Eωk, z)
    end
end

"""
General path: oversample into `Eωo`, transform, respond, window, transform back and crop.

Every stage is a backend-dispatched kernel shared with the fast path — `copy_scale!`,
`pointwise_Pt!`/`Et_to_Pt_ordered!`, `_apply_towin!`, `_scale_nl!` — so this is threaded on
the host and runs on a device. It performs the same floating-point operations in the same
order as the version that preceded it (which was one serial scalar loop per stage), and the
`fastpath=false` bit-identity test covers that.
"""
function trans_free_general!(nl, t::TransFree, Eωk, z)
    ρ = t.densityfun(z)
    N = length(t.grid.ω)
    fill!(t.Eωo, 0)
    copy_scale!(t.Eωo, Eωk, N, t.scale)
    _to_time!(t.Eto, t, t.Eωo) # transform (ω, ky, kx) -> (t, y, x)
    # Modified shot-noise: compute field+noise in a separate buffer (Et_nl) so the
    # propagating field (Eto) is never contaminated.
    Eresp = t.Eto
    if !isnothing(t.Et_noise)
        @. t.Et_nl = t.Eto + t.Et_noise
        Eresp = t.Et_nl
    end
    if t.Pto === t.Eto # all responses pointwise: the polarisation overwrites the field
        pointwise_Pt!(t.Pto, Eresp, t.resp, ρ)
    else
        Et_to_Pt_ordered!(t.Pto, Eresp, t.resp, ρ, t.idcs)
    end
    _apply_towin!(Utils.backend(t.Pto), t.Pto, t.gv.towin, t.idcs) # apodisation
    mul!(t.Pωo, t.FT, t.Pto) # transform (t, y, x) -> (ω, ky, kx)
    copy_scale!(nl, t.Pωo, N, 1/t.scale)
    _scale_nl!(Utils.backend(nl), nl, t, t.normfun(z))
end

# Fast path: no oversampling (scale == 1), complex field, no noise — the oversampled
# buffers would be exact copies, so transform directly between Eωk, Eto and nl.
# Bit-identical to trans_free_general! (see TransFree docstring).
function trans_free_fast!(nl, t::TransFree, Eωk, z)
    ρ = t.densityfun(z)
    Pto = _to_time_and_respond!(Utils.backend(Eωk), t, Eωk, ρ)
    mul!(nl, t.FT, Pto) # transform (t, y, x) -> (ω, ky, kx)
    _scale_nl!(Utils.backend(nl), nl, t, t.normfun(z))
end

"""
    _to_time_and_respond!(backend, t::TransFree, Eωk, ρ) -> Pto

`(ω, k_y, k_x)` → `(t, y, x)`, the nonlinear responses, and the temporal apodisation
window; returns the buffer holding the polarisation. Split by backend so the device can
fuse passes that the host gets for free from FFTW (see the device method).
"""
function _to_time_and_respond!(::Utils.CPUBackend, t::TransFree, Eωk, ρ)
    # `ldiv!` through the forward plan is FFTW's fused normalisation — no separate pass.
    # It preserves Eωk, which the caller relies on.
    _to_time!(t.Eto, t, Eωk)
    if t.Pto === nothing # all responses pointwise: polarisation overwrites Eto
        Pto = t.Eto
        pointwise_Pt!(Pto, t.Eto, t.resp, ρ)
    else
        Pto = t.Pto
        Et_to_Pt_ordered!(t.Pto, t.Eto, t.resp, ρ, t.idcs)
    end
    _apply_towin!(Utils.CPUBackend(), Pto, t.gv.towin, t.idcs)
    return Pto
end

"""
Device version. A device inverse plan is an `AbstractFFTs.ScaledPlan`, so applying it runs
the transform and then a **separate** full-array multiply by `1/N` — 4.5 GiB of traffic
per RHS at 3-D free-space production shapes, seven times per step.

When every response is pointwise, the next operation is itself an elementwise broadcast,
so the transform is applied unnormalised and the `1/N` is folded into that broadcast as a
prescale on read, together with the temporal window. Two whole-array passes disappear and
nothing is assumed about the responses: `f(s·E)` is what the normalised field would have
produced, for any `f`.

With a non-pointwise (batched or columnwise) response the polarisation is accumulated
across responses into a separate buffer, so there is no single broadcast to fold into and
the normalised plan is used as before.
"""
function _to_time_and_respond!(::Utils.DeviceBackend, t::TransFree, Eωk, ρ)
    if t.Pto === nothing
        raw, s = _ift_unscaled(t.IFT)
        mul!(t.Eto, raw, Eωk)
        Eto = t.Eto
        resp = t.resp
        towin = reshape(t.gv.towin, :, 1, 1)
        Eto .= towin .* _pointwise_P_total.(Ref(resp), s .* Eto, ρ)
        return Eto
    end
    _to_time!(t.Eto, t, Eωk)
    Et_to_Pt_ordered!(t.Pto, t.Eto, t.resp, ρ, t.idcs)
    _apply_towin!(Utils.DeviceBackend(), t.Pto, t.gv.towin, t.idcs)
    return t.Pto
end

"""
    _ift_unscaled(plan) -> (raw_plan, scale)

Split a stored inverse FFT plan into the raw transform and the normalisation factor it
would apply, so a caller that is about to broadcast anyway can fold the factor in rather
than pay a separate pass for it. `plan * x == scale * (raw_plan * x)`.
"""
_ift_unscaled(p::AbstractFFTs.ScaledPlan) = (p.p, p.scale)
_ift_unscaled(p) = (p, 1)

_to_time!(Eto, t::TransFree, Eωk) = _to_time!(Eto, t, Eωk, t.IFT)
_to_time!(Eto, t::TransFree, Eωk, ::Nothing) = ldiv!(Eto, t.FT, Eωk)
_to_time!(Eto, t::TransFree, Eωk, IFT) = mul!(Eto, IFT, Eωk)

# Time-domain apodisation, threaded over transverse points (host) or as one broadcast
# over the whole array (device). `towin` is a length-nt vector which expands along the
# leading dimension in both cases.
function _apply_towin!(::Utils.CPUBackend, Pto, towin, idcs)
    Utils.tforeach(length(idcs)) do ii
        Pcol = view(Pto, :, idcs[ii])
        @. Pcol *= towin
    end
end

_apply_towin!(::Utils.DeviceBackend, Pto, towin, idcs) = (Pto .*= towin; nothing)

# Spectral window, shock term and normalisation.
function _scale_nl!(::Utils.CPUBackend, nl, t::TransFree, norm)
    ωwin = t.gv.ωwin
    ω = t.gv.ω
    idcs = t.idcs
    Utils.tforeach(length(idcs)) do ii
        nlcol = view(nl, :, idcs[ii])
        normcol = view(norm, :, idcs[ii])
        @. nlcol *= ωwin * (-im*ω) / (2 * normcol)
    end
end

# The FreeNorm-specialised device method is defined below, once that type exists.
_scale_nl!(::Utils.DeviceBackend, nl, t::TransFree, norm) = error(
    "device propagation needs a factored normalisation (`const_norm_free(...; "*
    "factored=true)`); got a $(typeof(norm)).")

"""
    FreeNorm

Lazy normalisation-factor array for 3D free-space propagation: stores only `k²(ω)` and
`k⊥²(ky, kx)` and computes `√(k² - k⊥²)/(μ₀ω)` elements on demand, instead of caching the
full `(nω, nky, nkx)` `Float64` array. Elements are identical (bit-for-bit) to those
produced by [`norm_free`](@ref). Construct via `const_norm_free(...; factored=true)`.
"""
struct FreeNorm{Vt, Mt} <: AbstractArray{Float64, 3}
    ω::Vt
    k2::Vt
    kperp2::Mt
end

# Single-element normalisation factor, shared between `getindex` below and the device
# scaling kernel, so the lazy and broadcast forms can never drift apart (the same
# arrangement as `LinearOps._linop_xy_element`).
@inline function _freenorm_element(ω, k2ω, kperp2i)
    ω == 0 && return 1.0
    βsq = k2ω - kperp2i
    βsq <= 0 && return 1.0
    return sqrt(βsq)/(PhysData.μ_0*ω)
end

# See the note on `LinearOps.FactoredFreeLinop`: adapting moves the small factors to the
# device, where they are consumed by a broadcast kernel rather than by `getindex`.
Adapt.adapt_structure(to, n::FreeNorm) =
    FreeNorm(Adapt.adapt(to, n.ω), Adapt.adapt(to, n.k2), Adapt.adapt(to, n.kperp2))

Base.size(n::FreeNorm) = (length(n.ω), size(n.kperp2, 1), size(n.kperp2, 2))
Base.@propagate_inbounds Base.getindex(n::FreeNorm, iω::Int, iy::Int, ix::Int) =
    _freenorm_element(n.ω[iω], n.k2[iω], n.kperp2[iy, ix])

# Device scaling kernel: broadcast the normalisation's separable factors rather than the
# lazy struct itself (see `LinearOps._prop_factored!` for why the struct must stay out of
# a device broadcast). Same element function as `getindex` above, so the two agree.
function _scale_nl!(::Utils.DeviceBackend, nl, t::TransFree, norm::FreeNorm)
    ωwin = reshape(t.gv.ωwin, :, 1, 1)
    ω = reshape(t.gv.ω, :, 1, 1)
    nω = reshape(norm.ω, :, 1, 1)
    k2 = reshape(norm.k2, :, 1, 1)
    kperp2 = reshape(norm.kperp2, 1, size(norm.kperp2)...)
    @. nl *= ωwin * (-im*ω) / (2 * _freenorm_element(nω, k2, kperp2))
    return nothing
end

"""
    const_norm_free(grid, xygrid, nfun)

Make function to return normalisation factor for 3D propagation without re-calculating at
every step. With `factored=true` the returned function yields a lazy [`FreeNorm`](@ref)
(bit-identical elements, no field-sized cache).
"""
function const_norm_free(grid, xygrid, nfun; factored::Bool=false, arraytype=Array)
    if factored
        kperp2 = @. (xygrid.kx^2)' + xygrid.ky^2
        k2 = zero(grid.ω)
        # same k² as norm_free computes (z-independent here, so evaluated once)
        k2[grid.sidx] = (nfun.(wlfreq.(grid.ω[grid.sidx])).*grid.ω[grid.sidx]./PhysData.c).^2
        out = FreeNorm(copy(grid.ω), k2, kperp2)
        arraytype === Array || (out = Adapt.adapt(arraytype, out))
        return z -> out
    end
    arraytype === Array || error(
        "device propagation requires `factored=true`: a materialised normalisation "*
        "factor would occupy half a field-sized device array. Pass `factored=true` to "*
        "const_norm_free.")
    nfunω = (ω; z) -> nfun(wlfreq(ω))
    normfun = norm_free(grid, xygrid, nfunω)
    out = copy(normfun(0.0))
    function norm(z)
        return out
    end
    return norm
end

"""
    norm_free(grid, xygrid, nfun)

Make function to return normalisation factor for 3D propagation.

!!! note
    Here, `nfun(ω; z)` needs to take frequency `ω` and a keyword argument `z`.
"""
function norm_free(grid, xygrid, nfun)
    ω = grid.ω
    kperp2 = @. (xygrid.kx^2)' + xygrid.ky^2
    idcs = CartesianIndices((length(xygrid.ky), length(xygrid.kx)))
    k2 = zero(grid.ω)
    out = zeros(Float64, (length(grid.ω), length(xygrid.ky), length(xygrid.kx)))
    function norm(z)
        k2[grid.sidx] = (nfun.(grid.ω[grid.sidx]; z=z).*grid.ω[grid.sidx]./PhysData.c).^2
        for ii in idcs
            for iω in eachindex(ω)
                if ω[iω] == 0
                    out[iω, ii] = 1.0
                    continue
                end
                βsq = k2[iω] - kperp2[ii]
                if βsq <= 0
                    out[iω, ii] = 1.0
                    continue
                end
                out[iω, ii] = sqrt(βsq)/(PhysData.μ_0*ω[iω])
            end
        end
        return out
    end
end

end