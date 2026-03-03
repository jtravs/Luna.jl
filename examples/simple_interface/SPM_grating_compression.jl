#= In this example we simulate single-stage SPM broadening of a Yb-laser-like pulse
(1030 nm, 300 fs, 600 µJ) in an argon-filled hollow capillary fibre, followed by
compression using a four-grating compressor.

The fibre parameters (125 µm core radius, 1 m length, 4 bar Ar) give significant SPM
broadening, and the resulting chirp is approximately linear—well suited for grating
compression.

The grating compressor uses 300 lines/mm gratings at Littrow angle (order m = -1). =#
using Luna

## -- Fibre parameters --
a = 125e-6      # core radius [m]
flength = 1.0   # fibre length [m]
gas = :Ar       # fill gas
pressure = 4.0  # gas pressure [bar]

## -- Pulse parameters --
λ0 = 1030e-9    # central wavelength [m]
τfwhm = 300e-15 # pulse duration FWHM [s]
energy = 600e-6 # pulse energy [J]

## -- Run the mode-averaged simulation --
output = prop_capillary(a, flength, gas, pressure;
    λ0, τfwhm, energy,
    λlims=(800e-9, 1300e-9), trange=2e-12)

## -- Grating compressor parameters --
Λ = 1/(300e3)             # grating period [m] (300 lines/mm)
m = -1                     # diffraction order
θi = Fields.littrow_angle(λ0, Λ; m)  # Littrow angle of incidence [rad]

## -- Define a propagation function that applies optimal grating compression --
function prop(grid, Eω)
    L_opt, Eωcomp = Fields.optcomp_gratings(Eω, grid, Λ, m, θi; λ0=λ0)
    @info "Optimal grating separation: $(round(L_opt*1000, digits=2)) mm"
    Eωcomp
end

## -- Plotting --
# Spectral evolution along the fibre
Plotting.prop_2D(output, :λ, λrange=(800e-9,1200e-9), trange=(-1000e-15, 1000e-15))

# Input and output spectra
Plotting.spec_1D(output, λrange=(800e-9,1200e-9))

# Compressed temporal profile
Plotting.time_1D(output; propagate=prop, trange=(-1000e-15, 1000e-15))
