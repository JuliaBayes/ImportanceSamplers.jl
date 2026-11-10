# [Score control variates](@id control-variates-guide)

A control variate reduces error in an expectation by subtracting a function
whose target mean is known. Here that mean is zero. The correction uses target
gradients, called scores. It does not change samples, weights, or the proposal.

## A non-Gaussian example

Consider an equal mixture of unit-variance normals centered at -2 and 2.
Its second moment is exactly five. The following log target omits its irrelevant
normalizing constant. Its score is `-x + 2tanh(2x)`.

```@example controls
using ImportanceSamplers, Random, Statistics

function logtarget(x)
    a, b = -(x[1]-2)^2/2, -(x[1]+2)^2/2
    return max(a,b) + log1p(exp(-abs(a-b)))
end
score(x) = -x + 2tanh(2x)
f(x) = x[1]^2
proposal = SphericalStudentT(7.0, zeros(1), 2.0)
pilot_algorithm = ImportanceSampling(proposal; nsamples=4_096)
production_algorithm = ImportanceSampling(proposal; nsamples=16_384)

pilot = importance_sample(Xoshiro(11), logtarget, pilot_algorithm)
fit = fit_control_variate(f, pilot; scores=score.(pilot.samples))

production = importance_sample(Xoshiro(29), logtarget, production_algorithm)
raw = mean(f, production)
adjusted = control_variate_mean(f, production, fit;
    scores=score.(production.samples))
(raw, adjusted, reference=5.0)
```

Pilot and production use separate random draws. The coefficients stay fixed
during production. One comparison does not establish an improvement.
Repeated comparisons use independent complete workflows:

```@example controls
errors = map(1:64) do repetition
    pilot = importance_sample(Xoshiro(2repetition+100), logtarget, pilot_algorithm)
    fit = fit_control_variate(f, pilot; scores=score.(pilot.samples))
    production = importance_sample(Xoshiro(2repetition+101), logtarget, production_algorithm)
    raw = mean(f, production)
    adjusted = control_variate_mean(f, production, fit;
        scores=score.(production.samples))
    (raw-5, adjusted-5)
end
raw_errors, adjusted_errors = first.(errors), last.(errors)
gain = raw_errors.^2 .- adjusted_errors.^2
(raw_rmse=sqrt(mean(abs2,raw_errors)),
 adjusted_rmse=sqrt(mean(abs2,adjusted_errors)),
 raw_bias=mean(raw_errors), adjusted_bias=mean(adjusted_errors),
 paired_mse_gain=mean(gain), gain_mcse=std(gain)/sqrt(length(gain)))
```

The gain MCSE describes repeated-run uncertainty, not a posterior standard
deviation or an error bar for one estimate. Fitting and score evaluation add
cost. Compare complete workflows before claiming an efficiency gain.

## The correction

Let ``s(x)=\nabla\log p(x)`` and let ``c`` be the pilot's weighted center.
The controls are

```math
h_j(x)=s_j(x),\qquad h_{d+j}(x)=1+(x_j-c_j)s_j(x).
```

Integration by parts gives ``E_p[h]=0`` under the boundary conditions below.
For frozen coefficients ``\beta``, the adjusted estimator is

```math
\widehat\mu_{\mathrm{CV}}=
\sum_i\bar w_i\big[f(x_i)-\beta^\mathsf{T}h(x_i)\big].
```

The unknown target normalizing constant disappears from its log gradient.
Only target gradients are required, not target Hessians or derivatives of `f`.
An indicator, prediction, or parameter moment can therefore be the functional.
Fit separate coefficients for each functional. The score cache can be shared.

## Fitting and inputs

[`fit_control_variate`](@ref) uses a column-scaled ridge regression of
``w_i(f_i-\widehat\mu_{pilot})`` on ``w_i h(x_i)``. Thus its objective uses
**squared** importance weights. Control columns are scaled but never centered
at their empirical means. `ridge=0` requires full-rank controls.

For independent production draws from one fixed proposal, this objective
approximates the self-normalized estimator's asymptotic variance when the pilot
uses the same proposal. An adaptive or different-proposal pilot gives a
heuristic fit, not that optimization guarantee. No fit guarantees lower error.
Self-normalized estimates retain finite-sample ratio bias even with an
independent pilot.

| Input | Required meaning |
|:--|:--|
| `pilot`, `production` | Flat Float32/Float64 sample columns and aligned log weights |
| `f` | Scalar callable or cached vector aligned with that input's columns |
| `scores` | Exact target log-density gradients at those stored coordinates |
| `fit` | Frozen center and coefficients for the same target, coordinates, and functional |

Cached values permit external batched functional evaluation. Callable outputs
and cached values convert to the sample precision. Scores must already match
that precision. Zero-weight rows are ignored before functional or control
evaluation. Inputs remain unchanged.

The returned scalar is an expectation estimate. It is not a new weighted sample
set. [`normalized_weights`](@ref), [`lognormalizer`](@ref), resampling, and weight
ESS keep their original meanings. Adjusted estimates need not stay inside the
range of `f`, including `[0,1]` for a probability estimate.

## Boundary conditions

Differentiability alone is insufficient. The surface integrals of ``p e_j``
and ``p(x_j-c_j)e_j`` must vanish, with the required integrability. On the real
line these include vanishing ``p(x)`` and ``(x-c)p(x)`` at both ends.

A uniform density on `(0,1)` has interior score zero. Its second control is
one, not a zero-mean function. An exponential target on `(0,∞)` has constant
score -1, also not zero mean. Applying these controls directly is invalid.
The package cannot infer these conditions from finite samples.

Use valid unconstrained coordinates for constrained targets. The score must
then include the transformed density's existing log-Jacobian term, and all
controls must use those same coordinates. Approximate scores, missing modes,
or inadequate proposal support invalidate accuracy claims. A heavy weight tail
can prevent the finite-variance assumptions needed for regression and error
estimation. The correction does not repair that sampling problem.

## Devices

The fit and application use array operations on the input device. Only the
small regression system reaches CPU for factorization. The fit arrays return
to the input device. Production transfers only scalar checks and its final
estimate, not individual samples. Cached values and scores must share the
sample device. `threaded=false` disables CPU worker tasks.

Transfer the fit with the same `MLDataDevices` device as production. Callable
functionals on accelerators must satisfy the same restrictions as
[`mean(f, samples)`](@ref Statistics.mean). GPU score evaluation remains the
caller's responsibility. Native CPU and CUDA validation does not imply Metal
or Reactant support.

## References

[South et al. (2023)](https://arxiv.org/abs/1811.05073) describe polynomial
Stein control variates, regularization, boundary conditions, and split fitting.
The fixed-proposal squared-weight fitting objective here follows the
self-normalized IS influence function. It is not their unweighted regression
or the adaptive quadrature rule of
[Leluc et al. (2022)](https://arxiv.org/abs/2205.11890).

[`psis`](@ref) instead changes weights. [Fitted transport](@ref transport-guide)
instead changes the production proposal. These methods have different costs
and assumptions.
