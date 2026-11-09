# [Pareto smoothing](@id psis-guide)

Pareto-smoothed importance sampling (PSIS) replaces the largest importance
ratios with fitted generalized Pareto quantiles. It is optional postprocessing,
not a sampler or an adaptation rule.

## A light-tailed proposal

Estimate the mean of an exponential target with rate one using an exponential
proposal with rate three. Both distributions have support ``x\geq0``, but the
proposal undersamples the target's right tail. The target mean is exactly one.

```math
p(x)=e^{-x},\qquad q(x)=3e^{-3x},\qquad r(x)=\frac{p(x)}{q(x)}=\frac{e^{2x}}{3}.
```

Under the proposal, ``\Pr(r(X)>t)=(3t)^{-3/2}`` for ``t\geq1/3``.
The ratios therefore have exact Pareto shape ``k=2/3`` and infinite variance,
although the target has finite moments of every order. An occasional large
ratio can dominate the raw estimate.

```@example psis
using ImportanceSamplers, Random, Statistics

logtarget(x) = x[1] >= 0 ? -x[1] : -Inf
# Gamma shape 1 and scale 1/3 is an exponential distribution with rate 3.
algorithm = ImportanceSampling(IndependentGamma([1.0], [1 / 3]); nsamples=2_000)
samples = importance_sample(Xoshiro(20261110), logtarget, algorithm)
smoothed = psis(samples)

(
    raw_mean=mean(x -> x[1], samples),
    smoothed_mean=mean(x -> x[1], smoothed),
    reference=1.0,
    pareto_k=smoothed.diagnostics.pareto_k,
)
```

Both estimates use exactly the same samples. Only their weights differ.
A single run does not establish an improvement.

## Repeated comparison

Repeat the calculation 400 times with independent seeds. Compare each estimate
with one and pair the raw and smoothed squared errors within each run.

```@example psis
repetitions = 400
estimates = map(1:repetitions) do repetition
    raw = importance_sample(Xoshiro(20261109 + repetition), logtarget, algorithm)
    smooth = psis(raw; threaded=false)
    (mean(x -> x[1], raw), mean(x -> x[1], smooth))
end
raw_errors = first.(estimates) .- 1
smoothed_errors = last.(estimates) .- 1
gain = raw_errors.^2 .- smoothed_errors.^2

(
    raw_rmse=sqrt(mean(abs2, raw_errors)),
    smoothed_rmse=sqrt(mean(abs2, smoothed_errors)),
    raw_bias=mean(raw_errors),
    smoothed_bias=mean(smoothed_errors),
    mse_gain=mean(gain),
    mse_gain_mcse=std(gain) / sqrt(repetitions),
)
```

For these seeds, raw IS has RMSE 0.245 and PSIS has RMSE 0.182.
The mean paired squared-error reduction is 0.0270, with Monte Carlo standard
error 0.0111. This measures performance across repeated runs, not uncertainty
in the target distribution or an error bar for one estimate.

PSIS lowers RMSE here but increases negative bias: the mean errors are -0.0782
for raw IS and -0.147 for PSIS. The reduction in variance outweighs that bias.
PSIS does not guarantee lower error for every target, proposal, or functional.
A wider or adapted proposal can address the sampling problem itself.
PSIS instead reuses the draws already available.

## Weights and normalizers

`smoothed` shares the read-only samples and provenance but contains separate
log weights. The original result remains unchanged. Weighted statistics,
functionals, subsets, and resampling use the smoothed weights.

Keep `samples` for `lognormalizer(samples)`. Smoothing changes the normalizer
estimator, so `lognormalizer(smoothed)` is unavailable.
For raw vectors, `psis(logweights)` returns `(; logweights, diagnostics)`.
The output log weights remain unnormalized and preserve their original offset.

MIS and adaptive inputs must already contain the method's correct final weights.
PSIS does not recompute denominators or change adaptation.

## Diagnostics

The fit uses the largest `floor(min(n/5, 3sqrt(n/reff)))` ratios. The default
`reff=1` describes independent draws. For MCMC output, supply the relative MCMC
efficiency, not weight ESS divided by sample count. A single efficiency value
does not account for arbitrary adaptive dependence.

The estimated `pareto_k` describes the observed weight tail. The diagnostic
threshold is `min(1 - 1/log10(n), 0.7)`. `diagnostics.reliable` tests whether a
successful fit lies below this threshold. It does not certify mode coverage,
support, or Monte Carlo error for an arbitrary functional. A high estimate
warns of instability, not proof that a proper target has an infinite integral.

Too-short or degenerate tails and failed fits retain the raw weights. Their
shape diagnostic is `NaN` and `reliable=false`. An all-zero input throws
[`AllZeroWeightsError`](@ref).

Equal tail weights receive their block's average smoothed mass. Their order
therefore cannot change a weighted estimate. A tie across the cutoff, or an
exact zero ratio in the selected tail, retains all raw weights and reports
`status=:degenerate_tail`. These conservative rules extend the paper's
continuous-ratio algorithm and preserve known zero target mass.

## Devices and threads

Native CPU and CUDA execution have checks in `Float32` and `Float64`.
Arrays, sorting, and fitting stay on the selected device. Only scalar diagnostics
reach the host. `threaded=false` disables CPU worker tasks, not GPU execution.
Metal, AMDGPU, and Reactant PSIS are not validated.
See [Devices](@ref devices-guide) for transfer rules.

## Reference

[Vehtari et al. (2024)](https://jmlr.org/papers/v25/19-556.html), Algorithm 1
and Appendices F–H. The tail count follows the paper's floor rule, which differs
from implementations that round upward.

See [`psis`](@ref) and [`ParetoSmoothedSamples`](@ref) for the API.
