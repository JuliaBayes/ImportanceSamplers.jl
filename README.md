# ImportanceSamplers.jl

[![CI](https://github.com/JuliaBayes/ImportanceSamplers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaBayes/ImportanceSamplers.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/JuliaBayes/ImportanceSamplers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaBayes/ImportanceSamplers.jl)
[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliabayes.org/ImportanceSamplers.jl/stable/)
[![Development docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliabayes.org/ImportanceSamplers.jl/dev/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Plain, multiple, and adaptive importance sampling in Julia: AMIS, APIS, LAIS,
CAIS, N-PMC, DM-PMC, and first-order GRAMIS-CAIS. Supports threaded CPU execution,
native Gaussian and Student-t proposals, constrained parameters, and optional
CUDA, Metal, and Reactant paths with [documented support limits](https://juliabayes.org/ImportanceSamplers.jl/stable/guide/accelerators/).

Requires Julia 1.13 or later. Install the package in your project:

```julia
using Pkg
Pkg.add("ImportanceSamplers")
```

Sample a three-dimensional standard Gaussian using a broader Gaussian proposal:

```julia
using ImportanceSamplers, Random, Statistics

proposal = SphericalGaussian(zeros(3), 1.5)
logtarget(x) = -sum(abs2, x) / 2  # unnormalized log density, x is a 3-vector

samples = importance_sample(
    Xoshiro(42), logtarget, ImportanceSampling(proposal; nsamples=10_000),
)

mean(samples)           # coordinate means, approximately [0, 0, 0]
var(samples)            # coordinate variances, approximately [1, 1, 1]
samples.logweights      # raw log target/proposal ratios
lognormalizer(samples)  # estimate of log integral(exp(logtarget(x)))
```

The proposal must cover the target's support. Results retain samples and log
weights. Transfer a prepared sampler explicitly with `device(sampler)` for
accelerator execution.

## Benchmarks

[CPU and CUDA results](docs/src/guide/benchmarks.md)
compare six posterior models, with accuracy checks, timing details, and reproducible scripts.

[Documentation](https://juliabayes.org/ImportanceSamplers.jl/stable/) ·
[Logistic regression example](examples/logistic_regression.jl) ·
[Contributing](CONTRIBUTING.md)
