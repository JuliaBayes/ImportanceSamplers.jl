# [Hamiltonian importance sampling](@id hais-method)

Native HMC moves proposal centres using target gradients and a Metropolis correction.
The returned observations remain importance samples drawn from normalized lower proposals.
HMC trajectory points are not additional output samples.

Centre exploration alone does not fit lower widths. When moments exist, mixture covariance
includes both component covariance and the spread of component centres. Wide lower
proposals can therefore become overdispersed as centres explore the target, especially
in many dimensions. Cooperation can also remove centres from a minor mode.
Check weight concentration and mode coverage rather than acceptance alone.

## Example

This three-dimensional target has quartic tails. Its coordinate means are zero.

```@example hais
using ImportanceSamplers, Random, Statistics

logtarget(z) = -sum(x -> x^4, z) / 4 - sum(abs2, z) / 2
score!(g, z) = (g .= .-z.^3 .- z; nothing)
target = LogTarget(logtarget; grad=score!)
bank = ProposalBank([
    SphericalStudentT(7.0, [-1.0, 0.0, 0.0], 1.0),
    SphericalStudentT(7.0, [ 1.0, 0.0, 0.0], 1.0),
])
upper = HamiltonianMonteCarlo(;
    step_size=0.15, leapfrog_steps=8, tuning=WarmupTuning(32),
)
algorithm = HAIS(bank; transition=upper, rounds=5, round_size=2_000)
prepared = prepare_sampler(Xoshiro(42), target, algorithm)
samples = importance_sample!(prepared)

(mean=mean(samples), draws=length(samples))
```

Explicit gradients, a selected DifferentiationInterface backend, and first-order
LogDensityProblems targets use the existing [gradient interface](@ref gradients-guide).
Lower sample weighting evaluates the log target without requesting a gradient.

With ForwardDiff installed, select it explicitly instead of supplying `grad=`:

```julia
using ADTypes, ForwardDiff
ad_target = LogTarget(logtarget, AutoForwardDiff())
ad_samples = importance_sample(Xoshiro(44), ad_target, algorithm)
```

For native CUDA, transfer a fresh complete prepared sampler before its first run:

```julia
using CUDA, PureRNGs
import MLDataDevices
CUDA.allowscalar(false)
device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)
gpu_sampler = device(prepare_sampler(Threefry4x32(45), target, algorithm))
gpu_samples = importance_sample!(gpu_sampler)
host_samples = MLDataDevices.CPUDevice()(gpu_samples)
```

The target and explicit score must support device execution. Ancillary arrays
passed through the target context move with the prepared sampler.
Metal and Reactant HMC execution are not enabled.

## HMC controls

`step_size` and `leapfrog_steps` are required. Trajectory length stays fixed.
Optional warmup tunes one shared step size from acceptance probabilities, then freezes it.
Warmup moves add cost and return no importance samples. Successful reuse does not repeat them.

`mass` is the momentum covariance, not the lower proposal covariance.
It accepts an isotropic scalar, `I`, `Diagonal`, an SPD matrix, or its Cholesky factorization.
Dense operations use factors and solves, not matrix inversion.
Mass does not adapt. Lower scales adapt only with `adapt_scales=true`.

Every positive vector dimension is supported. A one-dimensional problem uses `[x]`, not scalar-layout proposals.
Named constraint transforms retain their existing score pullback and Jacobian conventions.

## LAIS and HAIS

Use the same transition with [`LAIS`](@ref) to omit cooperation:

```@example hais
independent = LAIS(bank; transition=upper, rounds=5, round_size=2_000)
independent_samples = importance_sample(Xoshiro(43), target, independent)
mean(independent_samples)
```

| Method | Round order | Retained final bank |
|:--|:--|:--|
| HMC-driven LAIS | HMC, then lower draws and weights | Final centres and next-round widths |
| HAIS | Lower draws and weights, then HMC and cooperation | Adapted bank for the next round |

Both require equal positive masses and a round size divisible by the proposal count.
Gaussian, Student-t, and independent Laplace lower scales stay fixed by default.
All lower rounds enter the result with their original current-mixture weights.

## [Width adaptation](@id lower-width-adaptation)

Enable marginal-width fitting independently of the upper transition:

```@example hais
adaptive = HAIS(bank; transition=upper, rounds=5, round_size=2_000,
                adapt_scales=true)
fitted = prepare_sampler(Xoshiro(46), target, adaptive)
adaptive_samples = importance_sample!(fitted)
snapshot = current_proposal(fitted)
mean(adaptive_samples)
```

The same keyword works with any LAIS transition. This is a package extension,
not a covariance update specified by the HAIS paper. It changes neither HMC mass
nor the centre update.

For each generating group, normalize its current-round DM weights. Compute its
weighted coordinate means and centred variances, without a Bessel correction.
Each new marginal variance is half the previous variance plus half the fitted
variance. Gaussian scale squared equals variance. Student-t scale squared is
variance times `(nu - 2) / nu`, requiring `nu > 2`. Laplace scale squared is
half the variance. Every Student-t component retains its degrees of freedom.

These local moments estimate a density proportional to `π(x)qⱼ(x)/ψ(x)`,
not the entire target independently for each proposal. Diagonal proposals fit
each marginal width. Factor proposals scale their factor rows, preserving
correlations. New correlations are not fitted. Spherical inputs can become
diagonal proposals in `current_proposal`.

All-zero groups, computed local weight ESS below two, zero or nonfinite fitted variances,
and unrepresentable updated parameters retain that component's previous widths.
An invalid coordinate rejects the complete component update. No estimator
weights are clipped, tempered, or recomputed.

Widths change only after a lower round. HAIS first completes cooperation against
the unchanged generating bank. Each destination then uses its own generating
group's width fit, regardless of the selected ancestor. The final fitted widths
remain available for reuse, snapshots, and `retarget`. Failed calls preserve the
pre-call widths. Native CUDA fits remain on-device without additional numerical
host reads. Reactant width adaptation is not enabled.

Width adaptation can reduce overdispersion, but cannot guarantee adequate local
ESS or preserve a minor mode lost through centre cooperation. Inspect weight and
coverage diagnostics across independent runs. Narrow Gaussian proposals can
leave heavy target tails poorly covered despite plausible fitted variances.

## Cooperation and weights

For the lower bank in round `t`, define

```math
\psi_t(x)=\frac1M\sum_{j=1}^M q_{t,j}(x),
\qquad \log w=\log\pi(x)-\log\psi_t(x).
```

After lower weighting, HAIS moves each centre with HMC and selects centres independently
with replacement. Selection scores are proportional to `π(y)/ψₜ(y)`, evaluated at each
endpoint against the unchanged old bank. Destination slots retain their scales and IDs.

These are adaptation scores, not importance weights for the HMC endpoint law.
The selected centres need not follow the target. Estimation uses only lower draws with
their actual proposal mixture. Under support and integrability assumptions, the fixed-budget
unnormalized integral estimator is unbiased. Self-normalized means and log-normalizer estimates
are generally biased at finite sample size.

The implementation interprets the paper's inconsistent denominator indices pointwise.
The `1/M` mixture normalization matters for evidence, even though it cancels in normalized means.

For example, exact standard-normal HMC endpoints reweighted by a standard-normal
target divided by a `Normal(0, sqrt(2))` lower density converge to variance `2/3`, not one.
This illustrates why the centre-law claim differs from the valid lower estimator.

## Diagnostics and failures

`samples.diagnostics.transition` separates initial, warmup, and production target/gradient work.
It also records acceptance, support rejections, energy divergences, and the retained step size.
Include upper work and warmup when comparing complete-method performance.

A finite energy error above `divergence_threshold` records a divergence without changing MH acceptance.
An invalid numerical trajectory rejects. A malformed target or gradient fails the call.
Failure to represent a logical value or Jacobian under a configured transform
also fails the call, even when sampled coordinates remain finite.
Failures preserve pre-call proposal and tuning state, but do not rewind the RNG.
All-zero lower batches remain in the result. Their normalized summaries are undefined.

See [Reuse](@ref reuse-guide) for proposal snapshots and target replacement.

## Source

The [standalone curved-target example](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/examples/hamiltonian.jl)
includes analytic mean, variance, and normalizer references.

Mousavi, Monsefi, and Elvira,
[*Hamiltonian Adaptive Importance Sampling*](https://arxiv.org/abs/2209.13716v1).
