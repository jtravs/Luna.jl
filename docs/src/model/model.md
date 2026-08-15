# The numerical model
```@contents
Pages = ["model.md", "modal_decompositions.md", "nonlinear_responses.md"]
Depth = 2
```

`Luna` can solve several different variations of the unidirectional pulse propagation equation (UPPE). All of these variations have this basic form in common:
```math
\partial_z E(\omega, \mathbf{k}_\perp, z){\mathrm{d}z} = \mathcal{L}(\omega, \mathbf{k}_\perp, z)E(\omega, \mathbf{k}_\perp, z) + \frac{i\omega}{N_{\mathrm{nl}}} P_{\mathrm{nl}}(\omega, \mathbf{k}_\perp, z)\,,
```
where ``E(\omega, \mathbf{k}_\perp, z)`` is the electric field in "reciprocal space", i.e. frequency and transverse spatial frequency, ``\omega`` is angular frequency, ``\mathbf{k}_\perp`` is some generalised transverse spatial frequency, ``z`` is the propagation direction, ``\mathcal{L}(\omega, \mathbf{k}_\perp, z)`` is a linear operator describing dispersion, loss and diffraction, ``P_{\mathrm{nl}}(\omega, \mathbf{k}_\perp, z)`` is the nonlinear polarisation induced by the field ``E(\omega, \mathbf{k}_\perp, z)``, and ``N_{\mathrm{nl}}`` is a normalisation factor. Since calculating the nonlinear polarisation directly in the frequency domain is not feasible, this is done in the real-space-time domain instead, and ``P_{\mathrm{nl}}(\omega, \mathbf{k}_\perp, z)`` is obtained by transforming back:
```math
P_{\mathrm{nl}}(\omega, \mathbf{k}_\perp, z) = \int_{-\infty}^{\infty} \mathcal{T}_\perp\Big[\mathcal{P}_\mathrm{nl}(E(t, \mathbf{r}_\perp, z)\Big](t, \mathbf{k}_\perp, z)\mathrm{e}^{i\omega t}\mathrm{d}t\,,
```
where ``\mathcal{T}_\perp`` is a transform from (transverse) real space to reciprocal space (i.e. spatial frequency), ``\mathbf{r}_\perp`` is the transverse spatial coordinate, ``t`` is time, and  ``\mathcal{P}`` is an operator which calculates the nonlinear response of the medium given an electric field. Naturally, the real-space field ``E(t, \mathbf{r}_\perp, z)`` first has to be obtained from ``E(\omega, \mathbf{k}_\perp, z)``:
```math
E(t, \mathbf{r}_\perp, z)  = \int_{-\infty}^{\infty} \mathrm{d}\omega \mathcal{T}_\perp^{-1}\Big[E(\omega, \mathbf{k}_\perp, z)\Big]\mathrm{e}^{-i\omega t}\,,
```
where ``\mathcal{T}_\perp^{-1}`` is simply the inverse of ``\mathcal{T}_\perp`` so transforms from transverse reciprocal space to real space. The chief difference between variations of the UPPE implemented in `Luna` is the definition of ``\mathbf{k}_\perp`` and ``\mathcal{T}_\perp``, that is, the choice of [Modal decompositions](@ref) of the field.

## Absorbing boundaries

Both the frequency and the time axes are finite and periodic (they are sampled for FFTs), so energy which reaches the edge of either window does not leave the simulation — it wraps around, and the result is wrong. `Luna` therefore applies absorbing boundaries at the edges of both windows. These matter whenever the spectrum or the pulse genuinely reaches the edge of the grid: soliton self-compression, modulation instability, or a dispersive wave walking out of the time window.

The absorbers are defined by an absorption **rate per unit propagation distance** ``\alpha``, so that propagating a distance ``L`` attenuates the field by ``\mathrm{e}^{-\alpha L}`` no matter how the solver subdivides that distance. The rate is derived from the apodisation profiles ``W`` built by [`Grid`](@ref Grid) as
```math
\alpha(x) = -\frac{\log W(x)}{\ell}\,,
```
where the reference length ``\ell`` is the distance over which the historical profile is applied exactly once. By default ``\ell = z_\mathrm{max}/N`` with ``N`` set by `boundary_N`, i.e. the profile is applied ``N`` times over the whole propagation.

The spectral absorber is diagonal in ``\omega``, so it is simply an imaginary part of ``k(\omega)`` and is added to the linear operator ``\mathcal{L}``. The interaction-picture propagator then applies it *exactly*, over whatever sub-interval the adaptive stepper chooses, at no extra cost. The temporal absorber is diagonal in ``t`` and so cannot ride the propagator; it is applied as a factor ``\mathrm{e}^{-\alpha(t)\Delta z}`` after each accepted step. Because these factors telescope, the total absorption over a distance is again independent of the step layout.

!!! note "Change in behaviour"
    Before this scheme was introduced, the absorbing boundaries were applied by multiplying the solution by the fixed profile ``W`` once per accepted step. The cumulative absorption was then ``W^N`` with ``N`` the number of steps — a number the adaptive controller derives from `rtol`. The consequence was that the converged solution depended on the tolerance: tightening `rtol` did not approach a limit, it increased the absorption, and with typical step counts the graded taper acted as a brick wall which eroded genuine spectral wings. Results from that scheme can be reproduced with `boundary=:legacy`, and the absorbers can be switched off entirely with `boundary=:none`. See [`Luna.run`](@ref) and [`Boundaries`](@ref Boundaries.Boundaries).

Note that this is separate from the band-limiting of the nonlinear polarisation which happens inside [`NonlinearRHS`](@ref NonlinearRHS): that is part of the definition of the equation being solved, and is applied to ``P_\mathrm{nl}`` rather than to the field.

## A note on sign conventions
In optics, a plane wave is usually written as
```math
E(t, \mathbf{r}) = \mathrm{e}^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}
```
and hence a general field, the superposition of many plane waves, is
```math
E(t, \mathbf{r}) = \int_{-\infty}^\infty \tilde{E}(\omega, \mathbf{k})\mathrm{e}^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}\,\mathrm{d}\omega\mathrm{d}^3\mathbf{k}\,,
```
which means that for the *time-domain* Fourier transform, the sign convention is *opposite* to that used in mathematics, with the forward and inverse transforms given by
```math
\tilde{E}(\omega) = \mathcal{F}_t\left[E(t)\right] = \int_{-\infty}^\infty \!\! E(t)\mathrm{e}^{i\omega t}\,\mathrm{d} t 
\\
\\
E(t) = \mathcal{F}^{-1}_\omega\left[E(\omega)\right] = \frac{1}{2\pi}\int_{-\infty}^\infty\!\! E(\omega)\mathrm{e}^{-i\omega t}\,\mathrm{d} \omega \,.
```
In this convention with one sign in the exponent for space and the opposite for time, positive group-velocity dispersion (GVD) is indeed a positive parabola (``1/2\,\beta_2(\omega-\omega_0)^2`` with positive ``\beta_2``), waves with positive wave vectors move to larger ``\mathbf{r}`` for larger times ``t`` and so forth. However, fast Fourier transforms (FFTs) use the mathematics convention. For complex (envelope) fields, this could be circumvented by simply using `ifft` instead of `fft` and vice versa, but this is not possible for real-valued fields using real FFTs (rFFT). The sign conventions in `Luna` are:

1. All *physical* expressions and quantities (propagation constants, dispersion, nonlinear phases etc.) are given in the **optics convention**, i.e. as they would be found in a textbook.
2. The *fields* in the actual simulation are given in the **mathematics convention** as required for FFTs. This leads to the appearance in additional minus signs in the linear operator, see e.g. [`make_const_linop`](@ref LinearOps.make_const_linop). Similarly, to add e.g. some dispersion to a field used in or returned by a `Luna` simulation, the sign of that dispersion has to be flipped.
