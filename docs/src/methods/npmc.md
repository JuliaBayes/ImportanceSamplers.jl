# [Nonlinear population Monte Carlo](@id npmc-method)

N-PMC fits one proposal from each round using clipped weights.
The returned samples keep their original importance weights.

## Example

```@example npmc
using ImportanceSamplers, Random, Statistics

logtarget(x) = -(abs2(x[1] - 1) + abs2((x[2] + 1 - 0.8(x[1] - 1)) / 0.6)) / 2
algorithm = NPMC(
    SphericalGaussian(zeros(2), 2.0);
    rounds=4, round_size=2_000,
)
prepared = prepare_sampler(Xoshiro(42), logtarget, algorithm)
samples = importance_sample!(prepared)

(mean=mean(samples), covariance=cov(samples))
```

The reference mean is `[1, -1]`.
Gaussian and Student-t vector proposals learn full covariance even when their
initial scale is spherical or diagonal.

## Estimation and adaptation weights

For each round:

1. Draw `n` samples from the current proposal.
2. Compute raw log weights `logtarget - logproposal`.
3. Clip fitting weights at the `isqrt(n)`-th largest weight.
4. Fit proposal moments from that round's clipped weights.
5. Retain the fitted proposal, with the [AMIS covariance ridge](@ref moment-ridge)
   for Gaussian/Student-t fits.

The estimator uses the original raw weights from every round.
There is no temporal-mixture denominator or resampling step.

Independent Gamma and Laplace use the marginal moment fits described for
[AMIS](@ref amis-method). They do not fit correlations or use a covariance ridge.
Degenerate or unrepresentable independent fits fail instead of clipping parameters.

`diagnostics.adaptation_ess` describes the clipped fitting weights.
`diagnostics.round_ess` describes each round's raw estimator weights.

## Scope and failures

This is an adaptation variant of N-PMC.
The original transformed-weight estimator is not the estimator returned here.

Student-t fits require `nu > 2` and retain `nu`.
A round with too few nonzero weights can give a zero clipping threshold and fail adaptation.
A failed fit preserves the proposal committed before the call.

For a fixed schedule, normalized proposals, support coverage, and integrability,
the underlying raw linear normalizer estimate is unbiased when the run completes almost surely.
This statement concerns the unconditional sampling law, not estimates selected for successful completion.
Keeping raw weights avoids substituting clipped ratios into the estimator.
It does not remove selection bias from discarding failed calls or retrying until success.
Its logarithm and self-normalized expectations generally have finite-sample bias.

See [Adaptation and reuse](@ref reuse-guide) and [Devices](@ref devices-guide).

## Sources

- Koblents and Míguez, [*A Population Monte Carlo Scheme with Transformed Weights and its Application to Stochastic Kinetic Models*](https://arxiv.org/abs/1208.5600).
- [Standalone example](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/examples/npmc.jl).
- [CUDA reproducer](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/validation/reproducers/cuda_npmc.jl).
