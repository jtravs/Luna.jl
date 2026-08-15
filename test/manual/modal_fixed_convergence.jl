# Manual validation: convergence of the fixed-quadrature modal transform with the number of
# radial nodes, on frozen fields from a short RDW-type propagation, for the Kerr term alone
# and for the plasma term with the ADK rate, the literal channel-sum PPT rate and the
# default smooth (sum_integral) PPT rate; Gauss versus Kronrod rules; and a vector (x+y HE11) mode set.
# The reference is the same rule with 2048 nodes; the adaptive TransModal at its default
# tolerance is shown for comparison. Not part of the test suite (takes a few minutes).
#
#     julia --project -t 8 test/manual/modal_fixed_convergence.jl
using Luna
import Luna: Interface, NonlinearRHS, Modes, Nonlinear, Ionisation, PhysData, Fields, Grid
import Luna.NonlinearRHS: TransModal, TransModalFixed
using Printf, LinearAlgebra

const λ0 = 800e-9
relerr(x, y) = norm(x .- y)/norm(y)

# --- frozen fields: 0.4 m of the RDW VUV example (He gradient), 4 HE1m modes ------------
Eω, grid, linop, transform, FT, output = Interface.prop_capillary_args(
    125e-6, 0.4, :He, (0.8, 0); λ0, τfwhm=7.5e-15, energy=275e-6, modes=4,
    trange=400e-15, λlims=(90e-9, 4e-6), shotnoise=false, saveN=3,
    stats_kwargs=Dict(:mode_error => false), modal_integral=:fixed, nr=128)
Luna.run(Eω, grid, linop, transform, FT, output; status_period=30)
# use the middle of the fibre: the density (and hence the polarisation) vanishes at the exit
Ez = output["Eω"][:, :, 2]; z = output["z"][2]
ρ = transform.densityfun(z)
ts = transform.ts
Ip = PhysData.ionisation_potential(:He)
λ = PhysData.wlfreq.(grid.ω)
band = (100e-9 .<= λ .<= 140e-9)

function fixed(resp; nr, kronrod=false)
    TransModalFixed(grid, ts, resp, transform.densityfun, transform.norm!; nr, kronrod)
end
function adaptive(resp; rtol=1e-3, mfcn=512)
    xo = zeros(length(grid.to), ts.npol)
    TransModal(grid, ts, Luna.FFTW.plan_rfft(xo, 1), resp, transform.densityfun,
               transform.norm!; rtol, mfcn)
end

kerr = Nonlinear.Kerr_field(PhysData.γ3_gas(:He))
plasma(ir) = Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ir, Ip)
cases = [
    ("Kerr only", (kerr,)),
    ("plasma, ADK", (plasma(Ionisation.IonRateADK(:He)),)),
    ("plasma, PPT channel sum (sum_integral=false)", (plasma(Ionisation.IonRatePPTCached(:He, λ0; sum_integral=false)),)),
    ("plasma, PPT sum_integral (default)", (plasma(Ionisation.IonRatePPTCached(:He, λ0)),)),
]
nrs = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
println("Frozen field at z = $z m; errors are ‖ΔP‖/‖P‖ over all ω and modes, and in 100–140 nm")
for (label, resp) in cases
    println("\n=== $label ===")
    tref = fixed(resp; nr=2048); nlref = similar(Ez); tref(nlref, Ez, z)
    nl = similar(Ez)
    for kronrod in (false, true)
        println(kronrod ? "  Gauss–Kronrod (nr rounded up to odd):" : "  Gauss–Legendre:")
        for nr in nrs
            t = fixed(resp; nr, kronrod); t(nl, Ez, z)
            s = @sprintf("    nr=%4d  rel(all)=%.2e  rel(band)=%.2e", t.quad.nr,
                         relerr(nl, nlref), relerr(nl[band, :], nlref[band, :]))
            if kronrod
                e = NonlinearRHS.integral_error!(t)
                s *= @sprintf("  embedded estimate rel(all)=%.2e", norm(e)/norm(nl))
            end
            println(s)
        end
    end
    ta = adaptive(resp); ta(nl, Ez, z)
    @printf("  adaptive rtol=1e-3: ncalls=%d  rel(all)=%.2e  rel(band)=%.2e\n", ta.ncalls,
            relerr(nl, nlref), relerr(nl[band, :], nlref[band, :]))
end

# --- vector mode set: HE11 x + y with the same field in both, Kerr + ADK ----------------
println("\n=== HE11 x+y (vector), Kerr + ADK plasma ===")
modes = [Luna.Capillary.MarcatiliMode(125e-6, :He, 0.8; n=1, m=1, ϕ=0.0),
         Luna.Capillary.MarcatiliMode(125e-6, :He, 0.8; n=1, m=1, ϕ=π/2)]
tsv = Modes.ToSpace(modes, components=:xy)
Ev = hcat(Ez[:, 1], 0.7 .* Ez[:, 1])
respv = (kerr, Nonlinear.PlasmaCumtrapz(grid.to, zeros(length(grid.to), 2),
                                        Ionisation.IonRateADK(:He), Ip))
fixedv(nr) = TransModalFixed(grid, tsv, respv, transform.densityfun, transform.norm!; nr)
tref = fixedv(1024); nlref = similar(Ev); tref(nlref, Ev, z)
nl = similar(Ev)
for nr in (16, 24, 32, 48, 64)
    t = fixedv(nr); t(nl, Ev, z)
    @printf("    nr=%4d  rel(all)=%.2e\n", nr, relerr(nl, nlref))
end
