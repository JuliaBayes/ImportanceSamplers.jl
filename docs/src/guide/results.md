# [Weighted samples](@id results-guide)

Every sampler returns `WeightedSamples`.
The samples and their weights together define the estimate.

## Weighted moments

```@example results
using ImportanceSamplers, Random, Statistics

samples = importance_sample(
    Xoshiro(42), x -> -sum(abs2, x) / 2,
    ImportanceSampling(SphericalGaussian(zeros(3), 1.5); nsamples=5_000),
)

(mean=mean(samples), variance=var(samples), covariance=cov(samples))
```

These `Statistics` methods use normalized importance weights.
`var`, `std`, and `cov` describe the weighted target approximation.
They use `corrected=false`. The usual unweighted sample-size correction is not valid here.

Compute a scalar function of each sampled parameter vector directly:

```@example results
(
    expectation=mean(x -> sum(abs2, x), samples),
    variance=var(x -> sum(abs2, x), samples),
    probability=mean(x -> x[1] > 0, samples),
)
```

`std(f, samples)` follows the same convention.
The variance describes `f(X)` under the target, not the Monte Carlo error of its estimated mean.
The [logistic tutorial](@ref logistic-mc-error) estimates that error for independent
fixed-proposal production draws. Weight ESS is not a substitute for this calculation.

## Indexing and provenance

```@example results
samples[1]
```

Scalar indexing returns `(; sample, logweight, provenance)`.
Iteration returns the same aligned records.

| One draw | Storage in `samples.samples` |
|:--|:--|
| Scalar | Length-`n` vector |
| Length-`d` vector | `d × n` matrix |
| Named tuple | Named tuple of vector/matrix leaves |

`samples.logweights` holds the raw log weights.
`samples.provenance` holds aligned identifiers when the method has proposals or rounds.
`samples.diagnostics` contains run-level information.

Population methods expose `provenance.proposal_id`.
Adaptive methods expose `provenance.round`.
These are generating identifiers, not a request to recompute weights.

Interpret method diagnostics according to the samples and weights they summarize:

| Diagnostic | Interpretation |
|:--|:--|
| AMIS `round_ess` | Cumulative retained history after retrospective reweighting |
| N-PMC `round_ess`, `adaptation_ess` | Current-round raw estimator weights versus clipped fitting weights |
| CAIS `local_ess` | Proposal-local weights used for covariance fitting |
| LAIS `transition` | Initial, warmup, and production upper-chain work |

The [pilot example](@ref logistic-diagnostics) interprets an AMIS trace.
Compare concentration with retained counts and independent estimates, not an arbitrary ESS cutoff.

## Sample subsets

```@example results
subset = samples[1:100]
(length(subset), mean(subset))
```

Ranges, integer vectors, and Boolean masks return an aligned `WeightedSampleView`.
Its weighted summaries normalize within that subset.

A subset does not preserve the complete estimator.
`lognormalizer(subset)` is therefore unavailable.
A target-dependent selection also changes the population described by the summary.

Results contain separate arrays. Later sampler calls do not change an earlier result.
Treat result arrays as read-only to preserve alignment and weight semantics.

## Weighted quantiles

```@example results
(
    median=median(samples),
    interval=quantile(samples, [0.025, 0.975]),
)
```

Quantiles are component-wise weighted quantiles.
A vector of probabilities gives one output column per probability for vector samples.

Quantiles, medians, scalar indexing, and iteration require CPU storage.
Transfer a GPU result explicitly before using them.

## Resampling

```@example results
draws = resample(Xoshiro(7), samples)
(length(draws), mean(draws))
```

The default is multinomial resampling with replacement.
Omitting the count requests `length(samples)` draws.
Use `resample(rng, samples, n)` for another count.

The returned `UnweightedSamples` contains no importance weights.
Its summaries use ordinary unweighted definitions.
Resampling adds randomness and does not increase the information in the weighted result.
Keep the original result for weighted summaries and normalizer estimates.

## Effective sample size and normalizers

```@example results
weights = normalized_weights(samples)
(ess=inv(sum(abs2, weights)), lognormalizer=lognormalizer(samples))
```

Weight ESS is ``1/\sum_i\bar w_i^2``.
It measures concentration, not mode coverage, tail accuracy, or chain mixing.
Adaptive dependence also prevents treating it as a general independent-sample count.

For a complete result with `n` draws,

```math
\log\widehat Z=\operatorname{logsumexp}(\ell_1,\ldots,\ell_n)-\log n.
```

The linear estimate may be unbiased under a method's assumptions.
Taking its logarithm does not preserve unbiasedness.
Self-normalized expectations generally have finite-sample bias.
AMIS has additional [retrospective-weight caveats](@ref amis-method).

If every raw log weight is `-Inf`, the plain IS result remains inspectable and
`lognormalizer` returns `-Inf`. Normalized summaries throw `AllZeroWeightsError`.

## Pareto smoothing

Pareto-smoothed importance sampling (PSIS) replaces the largest importance
ratios with fitted generalized Pareto quantiles. It can stabilize an estimate
at the cost of finite-sample bias. It is optional postprocessing, not another
sampler and not an adaptation rule.

```@example results
smoothed = psis(samples)
(
    raw_mean=mean(samples),
    smoothed_mean=mean(smoothed),
    pareto_k=smoothed.diagnostics.pareto_k,
    status=smoothed.diagnostics.status,
    reliable=smoothed.diagnostics.reliable,
)
```

`smoothed` shares the read-only samples and provenance but contains separate
log weights. The original result remains unchanged. Weighted statistics,
functionals, subsets, and resampling work as before, using the smoothed weights.
Keep `samples` for `lognormalizer(samples)`: smoothing changes the normalizer
estimator, so `lognormalizer(smoothed)` is unavailable.

For raw log-weight vectors, `psis(logweights)` returns `(; logweights, diagnostics)`.
Log weights remain unnormalized and preserve their original offset.

The fit uses the largest `floor(min(n/5, 3sqrt(n/reff)))` ratios. The default
`reff=1` describes independent draws. For MCMC output, `reff` denotes a supplied
relative MCMC efficiency, not weight ESS divided by sample count. A single
efficiency value does not account for arbitrary adaptive dependence.

The shape estimate `pareto_k` describes the observed weight tail. The diagnostic
threshold is `min(1 - 1/log10(n), 0.7)`. A fit below the threshold is not proof
that the run found every mode or that an unbounded functional has a reliable
error estimate. A high estimate warns of instability, not proof that a proper
target has an infinite integral. MIS and adaptive inputs must already contain
the method's correct final weights. PSIS does not recompute their denominators.

Too-short or degenerate tails and failed fits retain the raw weights. Their
shape diagnostic is `NaN` and `reliable=false`. An all-zero result cannot be
normalized and throws `AllZeroWeightsError`.

Equal tail weights receive their block's average smoothed mass. Their order
therefore cannot change a weighted estimate. If a tie crosses the cutoff, or
an exact zero ratio enters the selected tail, PSIS retains all raw weights and
reports a degenerate tail. These conservative rules extend the paper's
continuous-ratio algorithm and preserve known zero target mass.

Arrays and fitting stay on the selected device. Only scalar diagnostics reach
the host. `threaded=false` disables CPU worker tasks.
See [Vehtari et al. (2024)](https://jmlr.org/papers/v25/19-556.html), Algorithm 1
and Appendices F–H. The tail count follows the paper's floor rule, which differs
from implementations that round upward.

## Device results

`normalized_weights`, moments, scalar functionals, slicing, and resampling have device paths.
Array outputs stay on the device. Scalar reductions can require synchronization.

For explicit CPU access:

```julia
import MLDataDevices
host_samples = MLDataDevices.CPUDevice()(samples)
```

See [Devices](@ref devices-guide) for backend limits and Reactant scalar results.
