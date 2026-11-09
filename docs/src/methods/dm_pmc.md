# [DM-PMC, GR-PMC, and LR-PMC](@id dmpmc-method)

DM-PMC moves proposal centres by resampling completed rounds.
It uses a current-population mixture denominator and retains configured widths.

## Example

```@example dmpmc
using ImportanceSamplers, Random, Statistics

function logtarget(x)
    a = log(0.3) - (abs2(x[1] + 3) + abs2(x[2])) / 2
    b = log(0.7) - (abs2(x[1] - 3) + abs2(x[2])) / 2
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m)) - log(2pi)
end
bank = ProposalBank([
    FactorGaussian([-2.0, 0.0], [1.2 0.0; 0.3 1.1]),
    FactorGaussian([ 2.0, 0.0], [1.2 0.0; 0.3 1.1]),
])
algorithm = DeterministicMixturePMC(
    bank; rounds=4, round_size=2_000,
    resampling=LocalResampling(),
)
prepared = prepare_sampler(Xoshiro(42), logtarget, algorithm)
samples = importance_sample!(prepared)

(
    mean=mean(samples),
    positive_probability=mean(x -> x[1] > 0, samples),
    count=length(samples),
)
```

The target mixes unit-covariance Gaussians centred at `[-3, 0]` and `[3, 0]`,
with masses 0.3 and 0.7. Its reference mean is `[1.2, 0]`.
The probability of a positive first coordinate is
``0.3\Phi(-3)+0.7\Phi(3)\approx0.6995``, where ``\Phi`` is the standard-normal CDF.
This functional checks region mass as well as the mean.
Neither check alone certifies complete tail coverage.

This equal-mass, equal-allocation configuration uses local resampling, the LR-PMC case.
Replace `LocalResampling()` with `GlobalResampling()` for the GR-PMC case.
Global resampling is the default.
Bank masses determine allocation, not the target's mixture masses.
Generating proposal IDs are not latent target-component assignments.

## Resampling policies

| Policy | Source of the next centre in each slot |
|:--|:--|
| `GlobalResampling()` | The complete weighted round |
| `LocalResampling()` | That proposal's own weighted sample group |

Both policies use the same complete-mixture weights.
Local resampling preserves one descendant per proposal.
Global resampling can select duplicate ancestors and reduce population diversity.

The selected sample becomes the next proposal mean or location.
Gaussian, Student-t and Laplace scales or factors remain unchanged.
Gamma instead preserves actual coordinate variance `v`, using
`scale = v/mean` and `shape = mean/scale`.
Resampling does not tune proposal widths.

## Allocation and weights

Equal masses and divisible round sizes give equal allocation.
For unequal masses, the package uses largest-remainder allocation with rotated ties.
Every active proposal needs at least one draw.

For realized counts ``n_{t,j}`` and total ``N_t``, the denominator is

```math
\psi_t(x)=\sum_j\frac{n_{t,j}}{N_t}q_{t,j}(x),
\qquad
\ell_{t,i}=\log\pi(x_{t,i})-\log\psi_t(x_{t,i}).
```

The coefficients are realized count fractions, not nominal masses.
Unequal allocation is a package extension to the equal-allocation paper cases.

Each sample keeps its own round's weight.
All rounds contribute to the returned result.

## Reuse and limitations

The last resampling step runs after the final returned round.
Its centres become the starting population for the next prepared call.

Gaussian, Student-t and Laplace proposals retain their original scales and Student-t degrees of freedom.
Gamma proposals retain coordinate variances, not their constructor scale parameters.
Adapted centres do not compensate for unsuitable scales.

An all-zero round fails.
Local resampling also fails when one proposal's sample group has no nonzero weight.
A failure preserves the pre-call population, but not the old RNG position.

The linear normalizer estimate is unbiased for a fixed schedule when each conditional
mixture covers the integrable target, adaptation uses only completed rounds, and the run completes almost surely.
This statement concerns the unconditional sampling law, not runs selected for successful completion.
Discarding failed runs or retrying until success can bias the estimate.
This does not make the logarithm or normalized expectations unbiased.

## Sources

- Elvira et al., [*Improving Population Monte Carlo: Alternative Weighting and Resampling Schemes*](https://victorelvira.github.io/assets/papers/elvira2017improving_pre.pdf).
- [Equation reproducer](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/validation/reproducers/dm_pmc_global.jl).
- [Standalone example](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/examples/dm_pmc.jl).

See [Adaptation and reuse](@ref reuse-guide) and [Devices](@ref devices-guide).
