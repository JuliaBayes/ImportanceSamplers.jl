# [Troubleshooting](@id troubleshooting-guide)

Inspect the exception and its cause before increasing the sampling budget.
Many failures indicate a model, support, or execution mismatch.

## Weight concentration

Compute `inv(sum(abs2, normalized_weights(samples)))` and inspect the largest normalized weights.

Check proposal location, covariance, tails, and mode coverage.
Use an independent pilot or a broader family when appropriate.
A larger count does not repair missing support.

DM-PMC and APIS retain Gaussian, Student-t, and Laplace proposal scales.
Gamma proposals retain coordinate variances while their shape and scale change.
LAIS retains lower proposal scales. RAM tunes its upper MCMC increments only.

## All-zero weights

A valid `-Inf` target value means zero density.
For plain IS, an all-zero run retains its samples and reports `lognormalizer == -Inf`.
Normalized summaries throw `AllZeroWeightsError`.

Adaptive methods may fail earlier because a round or local fit has no usable mass.
Check constraints, proposal support, and target underflow.

## Nonfinite target values

`SamplerExecutionError` reports a phase and sample index.
A batch-wide failure uses index zero because no single sample is identified.
Its captured exception retains the original error and backtrace.

Use a stable log-density expression.
Do not replace invalid arithmetic with a finite number merely to continue sampling.

## Covariance adaptation

Inspect the method-specific round error, its `phase`, and its `cause`.
Check local sample counts, weight concentration, and repeated or degenerate samples.

CAIS uses exact covariance replacement and no ridge.
Its ESS threshold does not guarantee a positive-definite covariance.
AMIS, N-PMC, and first-order GRAMIS use their documented stabilization rules.

A failed adaptive call preserves the pre-call proposal state.
Its RNG still advances. Repeating the call is not an exact replay.

## Device transfer

`SamplerDeviceError.reason` identifies the rejected capability.
Common causes include:

- A generic proposal or product without a native representation on an accelerator.
- A host array captured by a closure rather than passed through `p`.
- An unspecified device precision policy.
- `threaded=false` for accelerator execution.
- An unsupported gradient or Reactant CPU kernel.

Consult [Devices](@ref devices-guide).
An unsupported contract has no silent CPU fallback.

## Scalar indexing

Transfer device results to CPU before scalar indexing, iteration, quantiles, or medians.
Array slicing and resident weighted moments use separate supported paths.

## Sampler migration and concurrent use

Transfer it before the first execution begins.
An accelerator-prepared sampler cannot itself migrate to another device.
Request an explicit result or proposal copy instead.

A prepared sampler is mutable and non-reentrant.
Use one sampler and RNG per concurrent task.

## Failure reports

Include a small target, proposal, seed, count schedule, and device selection.
Include `Project.toml`, `Manifest.toml`, the package commit, and the command that failed.
For a numerical error, include the expected mathematical quantity and observed result.
Do not include private data or credentials.
