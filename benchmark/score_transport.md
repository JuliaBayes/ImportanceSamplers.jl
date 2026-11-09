# Score-fitted quadratic transport experiment

This benchmark is a prototype, not a public proposal or fitting API.
It compares three workflows with a common AMIS pilot and independent production draws:

| Fit | Production proposal |
|:--|:--|
| `:none` | Student-t fitted to the pilot's weighted mean and covariance |
| `:moment` | Quadratic coupling fitted by weighted moments, with a latent Student-t fit |
| `:score` | The same coupling fitted by conditional score regression, with a latent Student-t fit |

Every Student-t proposal has five degrees of freedom. No sampler is added.
Each pair uses

```math
x_{2j-1}=z_{2j-1},\qquad
x_{2j}=z_{2j}+a_j(z_{2j-1}^2-1).
```

The map has an explicit inverse and determinant one. Production weights use
the composed target in latent coordinates and remain unchanged when samples
are mapped to physical coordinates. Fitting does not need a target Hessian.
The experiment reuses the package's private adoption helper for validated sampler
weights, avoiding another sample/weight copy. This is not a user construction API.

The score fit regresses the even-coordinate score on `x_odd^2-1` and `x_even`.
It is a Gaussian conditional surrogate, not full multivariate score matching.
Under regularity conditions both map fits have the same population coefficient.
Their difference is finite-pilot information and cost.
The map construction follows [NICE, Section 3.2](https://arxiv.org/html/1410.8516).
The conditional-score regression here is an experimental fit, not NICE's training procedure.

## CPU

Start Julia with the validation environment and a chosen thread count:

```sh
julia --project=validation --threads=8
```

```julia
using Pkg
Pkg.instantiate()
include("benchmark/score_transport.jl")
using .ScoreTransportExperiment: BentFunnel, Bimodal
using LinearAlgebra
BLAS.set_num_threads(1)

result = ScoreTransportExperiment.compare(; n=1_000_000, repetitions=80)
foreach(println, result.rows)
result.score_vs_moment
```

Use `threaded=false` for a serial CPU comparison. Each timed call includes
pilot preparation and sampling, fitting, production preparation and sampling,
mapped result storage, and summaries. Compilation is warmed first.
Four timing batches alternate method order. Median ranges describe those four
batch medians, not confidence intervals. Allocation fields use the last batch's
minimum. Repeated-run errors use paired pilot/production seeds across methods.

## CUDA

```julia
using CUDA
import MLDataDevices
CUDA.device!(0)  # Select a free device explicitly.
CUDA.allowscalar(false)
device = MLDataDevices.CUDADevice{typeof(CUDA.device()),Nothing}(CUDA.device())
gpu_result = ScoreTransportExperiment.compare(; device, n=1_000_000, repetitions=80)
foreach(println, gpu_result.rows)
```

Leave `threaded=true` on accelerators, as required by the package's transfer contract.
Samples, score fitting, and mapped results remain on-device. Only the fitted
mean/covariance and coupling coefficients return to CPU for proposal preparation.
`parameter_readback_bytes` reports those arrays, not every scalar synchronization
or the initial transfer of prepared workspaces. Fresh preparation costs remain
in the benchmark. Metal, AMDGPU, and Reactant are not checked by this experiment.

## Cases and interpretation

- `BentFunnel(1.25, 0.0)` is an exact positive control: its conditional score
  identifies the quadratic coefficient pointwise. It does not establish general
  score-learning performance.
- `BentFunnel(1.25, 0.45)` retains a varying conditional scale after unbending.
  It tests an imperfect map and a misspecified conditional score surrogate.
- `Bimodal(2.5, 0.7)` has modes in coordinates that the coupling leaves unchanged.
  It tests a case where this map cannot resolve the geometry.

Set `model=...`, `d=8`, `pilot=512`, or `seed=...` to change an experiment explicitly.
Even dimensions contain independent target pairs. Production counts do not change
with the pilot budget. Do not infer arbitrary high-dimensional performance from this family.

The measured functional is the first pair's second moment, with an analytic target
value. RMSE, bias, and paired squared-error differences use independent complete
workflows. Their reported Monte Carlo standard errors are empirical approximations.
`replicate_errors` retains the paired results. Weight ESS is descriptive, not the
accuracy criterion. Score fitting can add cost without reducing error.

## Algebra checks

```julia
include("validation/reproducers/score_transport.jl")
ScoreTransportValidation.analytic_checks()
ScoreTransportValidation.main()
ScoreTransportValidation.main(; T=Float32)
ScoreTransportValidation.main(; device)
ScoreTransportValidation.main(; device, T=Float32)
```

These check inverse mapping, the exact transformed-density identity, analytic
BentFunnel normalization and target gradients, and exact score-fit recovery on the positive control.
The CUDA checks use the same fixed inputs as CPU, with scalar indexing disabled.

`full_amis_workflow` supplies an additional ordinary AMIS reference that spends
the same total target budget across four adaptive rounds, retaining all draws:

```julia
ScoreTransportExperiment.full_amis_workflow(BentFunnel(1.25, 0.0); n=1_000_000)
```
