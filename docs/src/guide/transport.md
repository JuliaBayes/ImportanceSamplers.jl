# [Fitted transport](@id transport-guide)

A transport maps a simple proposal into a curved distribution. Fit it from a
weighted pilot, freeze the fit, and draw a separate production sample. This
changes the proposal, not the base sampling algorithm.

## Pilot and production

The following target has a quadratic bend. If ``u,v`` are independent standard
normals, then ``x_1=u`` and ``x_2=v+1.25(u^2-1)`` follow this target.
Consequently, ``E[x_2^2]=1+2(1.25)^2=4.125``.

```@example transport
using ImportanceSamplers, Random, Statistics, LinearAlgebra

logtarget(x) = -(x[1]^2 + (x[2] - 1.25*(x[1]^2-1))^2)/2 - log(2pi)
initial = FactorStudentT(5.0, zeros(2), 2.0*Matrix{Float64}(I, 2, 2))
pilot = importance_sample(Xoshiro(17), logtarget,
    AMIS(initial; rounds=4, round_size=512))

fit = fit_transport(pilot)
production = prepare_sampler(Xoshiro(29), logtarget,
    ImportanceSampling(fit.proposal; nsamples=16_384); transform=fit.transform)
samples = importance_sample!(production)

(coefficient=fit.transform.coefficients,
 estimate=mean(x -> x[2]^2, samples), reference=4.125)
```

The fit contains an ordinary latent [`FactorStudentT`](@ref) and a
[`QuadraticCoupling`](@ref). Production uses fresh random draws. It does not
reweight the pilot under the newly fitted proposal or include pilot draws in
the production estimate. Self-normalized estimates retain finite-sample bias.

Each pair uses

```math
x_{2j-1}=z_{2j-1},\qquad
x_{2j}=z_{2j}+a_j(z_{2j-1}^2-1).
```

The inverse subtracts the shift. The Jacobian determinant is one, so
``\log q_x(x)=\log q_z(T^{-1}(x))``. The target receives physical coordinates,
the sampler operates in latent coordinates, and returned samples contain
physical coordinates. Do not apply another Jacobian correction.
An unpaired final coordinate is unchanged. A one-dimensional map is the identity.
Supply `floor(d/2)` coefficients. For example, `QuadraticCoupling([0.75])`
maps the first pair of a three-dimensional vector and leaves its third entry
unchanged. `QuadraticCoupling(Float64[])` leaves a one-dimensional vector unchanged.

This is an additive coupling, following
[Dinh et al., Section 3.2](https://arxiv.org/abs/1410.8516).
It is not a neural flow or a general geometry-learning algorithm.

## Fitting

Moment fitting regresses each even coordinate on the paired odd coordinate's
square, with an intercept and a linear nuisance term. Score fitting removes
the same linear trend. A translated Gaussian pair then has zero population
curvature. Finite pilots can still fit spurious curvature. Ordinary linear
dependence stays in the latent covariance. The latent Student-t fits the inverse-mapped
pilot's full weighted covariance, including correlations between pairs.

Optional cached scores fit a Gaussian conditional-score surrogate instead:

```@example transport
scores = similar(pilot.samples)
for j in axes(scores, 2)
    u, value = pilot.samples[:, j]
    residual = value - 1.25*(u^2-1)
    scores[1,j] = -u + 2.5u*residual
    scores[2,j] = -residual
end
score_fit = fit_transport(pilot; scores)
score_fit.transform.coefficients
```

`scores[:,j]` must be ``\nabla_x\log\pi(x_j)`` at the stored physical sample,
in the same coordinate order, precision, and device. It is not the proposal
score or a gradient pulled back to a previous sampler's latent coordinates.
Only the even-coordinate scores enter this fit. No target Hessian is needed.

Under the relevant integration-by-parts conditions, score and moment fitting
have the same population coefficient. Scores can improve a finite pilot, but
they add evaluation cost and do not guarantee a better proposal. Both fits
need finite moments of squared odd coordinates, including their fourth moments.

[`psis(pilot)`](@ref psis) can supply smoothed fitting weights:
`fit_transport(psis(pilot))`. This changes the pilot fit, not production weighting.
It does not repair missing support or certify finite moments.

`dof` defaults to 5 and must exceed 2. `ridge` defaults to `1e-6` and adds
`ridge * max(tr(covariance)/d, 1)` to the latent covariance diagonal.
The fit converts covariance to Student-t scale before constructing the proposal.
Undefined regressions, ill-conditioned score fits, nonnegative score slopes,
nonfinite moments, and failed factorizations raise errors without clipping or
an identity fallback.

## Adaptation and reuse

The same fit can initialize an adaptive sampler:

```julia
prepared = prepare_sampler(Xoshiro(31), logtarget,
    AMIS(fit.proposal; rounds=4, round_size=4096); transform=fit.transform)
samples = importance_sample!(prepared)
```

The map stays fixed while AMIS adapts the latent proposal. Population methods
likewise use latent proposal banks. Explicit target gradients use physical
coordinates and the sampler applies the map's gradient pullback.
Batch targets receive mapped sample matrices. [`retarget`](@ref) preserves the
map and committed latent proposal. See [Adaptation and reuse](@ref reuse-guide).

## Devices

Fitting works on CPU and CUDA in Float32 and Float64. Pilot arrays and score
arrays stay on their device. Only fitted coefficients, the latent mean and
covariance, and scalar checks return to CPU for proposal construction.
The fit returns CPU parameters for the usual prepare-then-transfer workflow:

```julia
import CUDA, MLDataDevices
CUDA.allowscalar(false)
device = MLDataDevices.gpu_device()
device_pilot = device(pilot)
fit = fit_transport(device_pilot)
prepared = device(prepare_sampler(Xoshiro(41), logtarget,
    ImportanceSampling(fit.proposal; nsamples=1_000_000); transform=fit.transform))
samples = importance_sample!(prepared)
```

Use `threaded=false` to disable CPU worker tasks during fitting or sampling.
The map adds no per-sample host transfer. Other backend support requires its own
validation. Use sampler `transform=` for native execution and latent adaptation,
rather than placing a transformed proposal inside a population bank.

## Limits

The family accepts nonempty flat, unconstrained vectors of any dimension.
It leaves odd coordinates unchanged and models only paired quadratic shifts.
It cannot remove arbitrary conditional scale changes, identify unseen modes,
or guarantee finite importance-weight variance. Named constraint-map composition
is not part of this interface.

Compare complete fitting and production cost against the same workflow without
the map. Use errors of the functions of interest, not weight ESS alone.
[`psis`](@ref) remains an optional, separate smoothing step on production weights.
