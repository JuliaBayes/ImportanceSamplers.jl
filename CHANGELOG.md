# Changelog

## Unreleased

## 0.0.1

Initial feature set. Requires Julia 1.13 or later.

### Added

- Plain importance sampling and static multiple importance sampling with selectable weighting schemes.
- AMIS, N-PMC, DM-PMC, GR-PMC, LR-PMC, APIS, CAIS, and LAIS.
- First-order GRAMIS-CAIS with gradients and local covariance adaptation, distinct from the Hessian-based GRAMIS algorithm.
- Spherical, diagonal, and factor Gaussian and Student-t proposals, uniform boxes, and independent Gamma and Laplace proposals.
- Proposal banks, product proposals, and optional Distributions.jl integration. Adaptive support depends on the method and proposal family.
- Scalar and batch log targets, target contexts, and DensityInterface and LogDensityProblems integration.
- Named parameters and identity, interval, positive, softplus, and simplex transforms.
- Explicit gradients and DifferentiationInterface-based CPU automatic differentiation.
- Reusable prepared samplers, adapted proposals, and target replacement through `retarget`.
- Weighted moments and functionals, log-normalizer estimates, weight ESS, resampling, and sample provenance.
- Serial and threaded CPU execution, with explicit device transfer through MLDataDevices.
- Optional CUDA, Metal, and Reactant execution. Supported methods, precision, and gradient routes vary by backend.
- Tutorials, method guides, equation reproducers, and CPU/CUDA model comparisons with recorded environments and accuracy checks.

The API is experimental. Backend and method restrictions are documented in the manual.
