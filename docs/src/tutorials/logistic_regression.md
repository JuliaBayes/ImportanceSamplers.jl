# [Logistic regression](@id logistic-tutorial)

This example infers an intercept and two slopes from binary observations.
It uses a short adaptive run to fit a proposal, then draws from that fixed proposal.

## Data

```@example logistic
using ImportanceSamplers, Random, Statistics, LinearAlgebra

rng = Xoshiro(12)
truth = [-0.4, 1.1, -0.8]
X = hcat(ones(80), randn(rng, 80, 2))
probability = 1 ./ (1 .+ exp.(-(X * truth)))
y = rand(rng, 80) .< probability
p = (; X, y, prior_scale=2.0)

(size(X), count(y))
```

Each row of `X` describes one observation. The first column supplies the intercept.
The context `p` holds data and constants. Only the coefficient vector `beta` is sampled.

## Posterior log density

Assume independent normal priors with standard deviation two for the coefficients.

```@example logistic
function logposterior(beta, p)
    value = -sum(abs2, beta) / (2p.prior_scale^2)
    for row in axes(p.X, 1)
        eta = zero(eltype(beta))
        for column in eachindex(beta)
            eta += p.X[row, column] * beta[column]
        end
        # Stable log(1 + exp(eta)).
        logpartition = max(eta, zero(eta)) + log1p(exp(-abs(eta)))
        value += p.y[row] * eta - logpartition
    end
    return value
end
nothing # hide
```

The log target adds the log prior and log likelihood.
It omits parameter-independent constants, so its normalizer is not the model evidence.

The two-argument signature separates the unknown `beta` from the fixed context `p`.
No wrapper is required.

## Proposal adaptation

```@example logistic
initial = SphericalStudentT(5.0, zeros(3), 2.0)
pilot_rounds, pilot_round_size = 4, 2_000
pilot = prepare_sampler(
    Xoshiro(31), logposterior, p,
    AMIS(initial; rounds=pilot_rounds, round_size=pilot_round_size),
)
pilot_samples = importance_sample!(pilot)
learned = current_proposal(pilot)

round.(mean(pilot_samples); digits=3)
```

AMIS adapts a location and full covariance. The Student-t degrees of freedom stay fixed.
`current_proposal` returns an independent snapshot in the original sampling coordinates.

The pilot starts broadly. It may still miss a remote mode.
A fitted proposal does not remove the need to inspect the final weights.

## [Proposal diagnostics](@id logistic-diagnostics)

Inspect the weight concentration during the pilot:

```@example logistic
history_draws = pilot_round_size .* (1:pilot_rounds)
history_ess = pilot_samples.diagnostics.round_ess
(
    draws=history_draws,
    weight_ess=history_ess,
    ess_fraction=history_ess ./ history_draws,
)
```

AMIS reports cumulative history under each round's retrospective weights.
These are not isolated round estimates, and the ESS fraction need not increase.
A low fraction suggests checking location, width and tail coverage before adding draws.
Even a high fraction cannot detect an unvisited mode.

## Fixed-proposal sampling

```@example logistic
samples = importance_sample(
    Xoshiro(32), logposterior, p,
    ImportanceSampling(learned; nsamples=20_000),
)

(
    mean=round.(mean(samples); digits=3),
    sd=round.(std(samples); digits=3),
    weight_ess=inv(sum(abs2, normalized_weights(samples))),
)
```

The final run uses a fresh random stream and a fixed proposal.
Its budget is independent of the pilot budget.
Only `samples` supplies the following estimates.
Check its weight ESS independently of the pilot's diagnostics.

## Posterior summaries and prediction

```@example logistic
intervals = quantile(samples, [0.025, 0.975])
new_predictors = [1.0, 0.5, -0.2]
prediction = mean(
    beta -> inv(1 + exp(-dot(new_predictors, beta))),
    samples,
)

(intervals=intervals, predicted_probability=prediction)
```

The intervals are marginal posterior intervals for each coefficient.
The prediction averages the success probability over posterior uncertainty.

More observations can make prior importance sampling inefficient.
Adapting the proposal helps only when the pilot reaches the posterior mass.

## [Monte Carlo error](@id logistic-mc-error)

The credible intervals above describe parameter uncertainty.
They do not measure the numerical error in the estimated prediction.
For these independent production draws, estimate that error with

```@example logistic
weights = normalized_weights(samples)
probabilities = [
    inv(1 + exp(-dot(new_predictors, beta))) for beta in eachcol(samples.samples)
]
mcse = sqrt(sum(abs2, weights .* (probabilities .- prediction)))
(predicted_probability=prediction, monte_carlo_se=mcse)
```

For normalized weights ``\bar w_i``, this computes
``\sqrt{\sum_i\bar w_i^2(h_i-\widehat\mu)^2}``.
It is an asymptotic plug-in standard error, conditional on the frozen proposal,
under the relevant finite-second-moment assumptions. It is not an exact error bound.
See [Owen, Section 9.2](https://artowen.su.domains/mc/Ch-var-is.pdf), equation (9.9).

Do not apply this independent-draw formula directly to adaptive or stratified MIS output.
It also cannot detect missed regions or remove self-normalization bias.
Compare independent production runs and larger budgets while keeping the proposal fixed.

## Related examples

- Use [batch targets](@ref batch-guide) to evaluate a design matrix against many coefficient vectors.
- Transfer the complete prepared sampler for [GPU execution](@ref devices-guide).
- See the standalone [logistic regression script](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/examples/logistic_regression.jl) for a prior-based pilot.
