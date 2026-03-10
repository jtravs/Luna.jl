# # Plotting examples
#
# This page demonstrates Luna's built-in plotting functions using the PythonPlot backend.
# The same functions work identically with CairoMakie, GLMakie, WGLMakie, and PyPlot.
#
# ## Setup
#
# First, we run a simple hollow capillary fibre simulation: dispersive wave emission
# in the deep UV from a few-cycle 800 nm pump pulse.

using Luna, PythonPlot

radius = 125e-6
flength = 1.5
gas = :Ar
pressure = 80e-3

λ0 = 800e-9
τfwhm = 10e-15
energy = 60e-6

output = prop_capillary(radius, flength, gas, pressure; λ0, τfwhm, energy,
                        modes=2, trange=200e-15, λlims=(150e-9, 4e-6))

# ## 2D propagation plots
#
# [`Plotting.prop_2D`](@ref) creates side-by-side spectral and temporal heatmaps.
# The first positional argument after `output` selects the spectral axis:
# `:λ` for wavelength, `:f` for frequency (default), or `:ω` for angular frequency.

Plotting.prop_2D(output, :λ; λrange=(150e-9, 1000e-9), trange=(-20e-15, 20e-15), dBmin=-30)

# For multimode simulations, use the `modes` keyword to select which modes to plot.
# `:sum` plots the coherent sum of all modes:

Plotting.prop_2D(output, :λ; modes=:sum,
                 λrange=(150e-9, 1000e-9), trange=(-20e-15, 20e-15), dBmin=-30)

# ## Time-domain line plots
#
# [`Plotting.time_1D`](@ref) plots time-domain slices at one or more propagation distances.
# The legend automatically includes the FWHM duration.

Plotting.time_1D(output; modes=:sum)

# Use `bandpass` to isolate a spectral window in the time domain:

Plotting.time_1D(output; modes=1, bandpass=(220e-9, 270e-9), trange=(-50e-15, 50e-15))

# ## Spectral line plots
#
# [`Plotting.spec_1D`](@ref) plots spectral slices. By default it uses a logarithmic y-axis.

Plotting.spec_1D(output; modes=:sum, λrange=(150e-9, 1000e-9))

# ## Spectrograms
#
# [`Plotting.spectrogram`](@ref) creates a time-frequency spectrogram using the Gabor transform.
# The `fw` parameter sets the gate function width.

Plotting.spectrogram(output, flength; trange=(-20e-15, 30e-15),
                     λrange=(150e-9, 1000e-9), N=256, fw=3e-15)

# !!! note
#     With the Makie backend, you can also create a 3D surface spectrogram using `surface3d=true`.

# ## Statistics
#
# [`Plotting.stats`](@ref) automatically plots all available propagation statistics
# (energy, peak power, FWHM, electron density, etc.).

Plotting.stats(output)

# ## Energy evolution
#
# [`Plotting.energy`](@ref) shows energy vs propagation distance with a secondary axis
# for conversion efficiency.

Plotting.energy(output)
