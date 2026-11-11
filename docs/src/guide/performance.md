# [Performance](@id performance-guide)

Performance measurements should cover the complete application workflow,
with preparation, compilation, sampling, summaries, and explicit transfers reported separately.

## CPU threading

Launch Julia with a chosen thread budget:

```sh
julia --threads=8 --project
```

Sampling requests Julia's default thread pool unless you pass `threaded=false`.
With only one default thread, the request executes serially.
Prepared samplers keep their threading policy for later calls.

The coordinator consumes random numbers.
Native workers use prefilled random buffers.
Generic proposal draws occur before threaded target and density evaluation.

Targets and proposal densities must be thread-safe.
Explicit [batch callbacks](@ref batch-guide) control their parallelism.
Avoid oversubscribing the machine with both outer threads and multithreaded BLAS.

The largest thread count need not be fastest.
Compare several budgets on the same workload.

## Factor execution

```@example performance
using ImportanceSamplers, Random, LinearAlgebra

proposal = FactorGaussian(zeros(3), Matrix{Float64}(I, 3, 3))
algorithm = ImportanceSampling(proposal; nsamples=5_000)
prepared = prepare_sampler(
    Xoshiro(42), x -> -sum(abs2, x) / 2, algorithm;
    factor_execution=BatchedFactorExecution(),
)
samples = importance_sample!(prepared)
samples.diagnostics.factor_execution_policy
```

CPU defaults to fused sample kernels.
Supported accelerator factor paths default to batched multiplication and triangular solves.

[`FusedFactorExecution`](@ref) and [`BatchedFactorExecution`](@ref) request a policy explicitly.
There is no machine-specific sample-count threshold.
Unsupported batched cases use the fused path without changing the estimator.

Plain IS supports batching for [`FactorGaussian`](@ref) and [`FactorStudentT`](@ref)
with no sample transform or [`IdentityTransform`](@ref). Student-t draws use one radial multiplier per sample,
shared across coordinates. Packed bank/history paths support both families.
Static MIS batching requires a full-mixture denominator.
Mixed sample/log-weight precision can require an explicit batched request.

Independent uniform, Gamma, and Laplace proposals have no matrix factor.
They retain their coordinate-wise kernels; factor execution does not affect them.

Student-t batching uses one extra dimension-by-count workspace on the selected
device. It packs radially scaled normals into dense columns for BLAS, then reuses
that workspace for density solves. Fused execution avoids that extra workspace
and can be faster at modest dimensions. Compare both policies for the workload.

The diagnostic records the resolved policy.
It does not prove that every phase used a matrix operation.

## Gradient batches

HMC may have far fewer upper chains than retained lower samples. A scalar GPU
gradient assigns work to chains; serial loops inside the user callback do not
become parallel observation reductions. Larger lower budgets do not fix that
upper bottleneck.

Use [`LogTarget`](@ref)'s `batch_grad=` callback for matrix-batched values and
gradients. HMC and first-order GRAMIS share this derivative interface. Model
data, score buffers and retained samples stay on the device. Pair it with
`batch=` when lower value evaluation also benefits from batching.

See [Batched values and gradients](@ref batched-gradients) for the interface.
Measure complete runs, including configured warmup. Speed alone does not
establish adequate weight ESS or target coverage.

## Named product storage

Diagonal Gaussian products can use native CPU execution. Correlated products
keep blockwise generic CPU arithmetic where available, avoiding a dense solve
across independent blocks. Heterogeneous banks without a generic route use native packing.

Explicit accelerator transfer combines supported Gaussian products into one
native proposal. A factor block makes that joint factor dense: storage grows
as the square of the total sampling dimension. Factor batching still applies.

Native named layouts currently retain a flat sample batch while constructing
the named output arrays. Budget for both batches, alongside random buffers and
method workspaces. They stay on the selected device; this is not CPU readback.

## Warmed sampling

Use BenchmarkTools in a benchmark environment:

```julia
using BenchmarkTools, ImportanceSamplers, Random

algorithm = ImportanceSampling(SphericalGaussian(zeros(3), 1.5); nsamples=100_000)
prepared = prepare_sampler(Xoshiro(42), x -> -sum(abs2, x) / 2, algorithm)
importance_sample!(prepared) # Compile the path before timing.
@benchmark importance_sample!($prepared)
```

For CUDA, synchronize within the timed expression:

```julia
@benchmark CUDA.@sync importance_sample!($gpu_sampler)
```

A warmed adaptive benchmark retains its learned proposal across calls.
It is not a fresh-initialization accuracy experiment.
Use independent prepared samplers when comparing fresh-run accuracy.

## Memory and compilation

Every adaptive result retains all configured rounds.
AMIS additionally keeps proposal history and revisits denominators.
Data-heavy batch targets can allocate much more scratch than a scalar target.

Reported Julia allocations measure host allocation traffic, not peak device memory.
A cached GPU memory pool is not the same as live sample storage.

Reactant prepares built-in sampling phases once for the fixed schedule.
Result operations cache up to 64 executable signatures per process.
Further signatures and view operations may compile eagerly.
Functionals with captured state also compile eagerly so state changes remain visible.
Custom LAIS transitions do not receive the built-in compilation guarantees.

## Benchmark design

Interleave baseline and candidate timings on shared hardware.
Record thread counts, BLAS threads, device, precision, budgets, and host load.
Include pilots and upper-chain work in end-to-end comparisons.

Measure the error of the actual estimand as well as weight ESS.
See [Benchmarks](@ref benchmarks-guide) for reproducible model comparisons and accuracy reports.
