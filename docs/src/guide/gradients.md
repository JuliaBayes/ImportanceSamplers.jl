# [Gradients](@id gradients-guide)

`FirstOrderGRAMIS`, `HAIS`, and HMC-driven `LAIS` need target gradients.
Value-only operations still use the log target without requesting a gradient.

## Explicit gradients

```@example gradients
using ImportanceSamplers, Random, Statistics

logtarget(x, p) = -sum(abs2, x .- p.centre) / 2

function gradient!(g, x, p)
    g .= p.centre .- x
    return nothing
end

target = LogTarget(logtarget; grad=gradient!)
p = (; centre=[0.5, -0.5])
bank = ProposalBank([
    SphericalGaussian([-1.0, 0.0], 1.5),
    SphericalGaussian([ 1.0, 0.0], 1.5),
])
algorithm = FirstOrderGRAMIS(
    bank; rounds=3, round_size=2_000, repulsion_strength=0.0,
)
samples = importance_sample(Xoshiro(42), target, p, algorithm)

mean(samples)
```

The callback fills the gradient of the log target with respect to `x`.
The context `p` stays constant.

Without a context, use `gradient!(g, x)`.
On CPU, an out-of-place `gradient(x)` or `gradient(x, p)` is also accepted.
An applicable in-place form has priority.

A device gradient must support the supplied device arrays.
Do not retain its borrowed input or output buffers.

## Automatic differentiation

`LogTarget` accepts an ADTypes selector for an installed AD backend.
For example, with ForwardDiff installed:

```julia
using ADTypes, ForwardDiff

target = LogTarget(logtarget, AutoForwardDiff())
samples = importance_sample(Xoshiro(42), target, p, algorithm)
```

The package uses DifferentiationInterface on CPU.
It prepares derivative state for the chosen target and input.
It does not choose a backend automatically.

An explicit `grad=` takes priority over `adtype`.
A bare first-order LogDensityProblems target can supply its own value/gradient interface.
Failure in the selected source does not trigger a different source.

## Named-parameter gradients

For a target taking `theta = (; rate, offset)`, a logical gradient has matching fields:

```@example named_gradients
using ImportanceSamplers, Random, Statistics

layout = (
    rate=(1 => PositiveTransform()),
    offset=(2 => IdentityTransform()),
)
named_target(theta) = -theta.rate - theta.offset^2 / 2

function named_gradient!(g, theta)
    g.rate[] = -one(theta.rate)
    g.offset[] = -theta.offset
    return nothing
end

bank = ProposalBank([
    SphericalGaussian([-0.5, -0.5], 1.0),
    SphericalGaussian([ 0.5,  0.5], 1.0),
])
samples = importance_sample(
    Xoshiro(7), LogTarget(named_target; grad=named_gradient!),
    FirstOrderGRAMIS(bank; rounds=3, round_size=2_000, repulsion_strength=0.0);
    transform=layout,
)

mean(samples)
```

Scalar gradient fields are zero-dimensional writable views, so use `g.rate[]`.
Vector fields use ordinary indexed writes.
For a simplex of length `K`, supply `K` logical derivatives.

Differentiate only the logical log target.
The package adds the transform pullback and the log-Jacobian derivative.
CPU AD instead differentiates the composed flat-coordinate target.

## Accelerator gradients

| Execution | Gradient route |
|:--|:--|
| Native CUDA | Device-compatible scalar or batch callback, or supported reverse-mode Enzyme |
| Metal | Device-compatible scalar or batch callback |
| Reactant GPU | Supported compiled gradient, including `AutoEnzyme` |
| Ordinary CPU | Explicit callbacks, LogDensityProblems, or a supported DI backend |

A scalar target that runs on GPU need not be differentiable by the GPU AD backend.

For native CUDA with Enzyme, structured contexts can require explicit runtime activity:

```julia
using ADTypes, Enzyme

ad = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
target = LogTarget(logtarget, ad)
```

The supplied mode is preserved. The package batches scalar target gradients internally.
It does not copy each sample or gradient to CPU.

Targets with vector parameters must accept read-only `AbstractVector` inputs without
requiring a concrete view type or pointer layout. Direct reductions use sequential
accumulation. Other operations remain subject to the AD backend's device restrictions.

`batch=` controls value-only evaluation. Automatic differentiation still uses
the scalar target and may evaluate its primal.

## [Batched values and gradients](@id batched-gradients)

Use `batch_grad=` when matrix operations or device kernels can evaluate many
scores together. It applies to HMC-driven [`LAIS`](@ref), [`HAIS`](@ref), and
[`FirstOrderGRAMIS`](@ref). The scalar target remains required.

```@example batch_gradients
using ImportanceSamplers, Random, LinearAlgebra, Statistics

function regression_value(beta, p)
    value = -sum(abs2,beta)/8
    for i in eachindex(p.y)
        prediction = zero(eltype(beta))
        for j in eachindex(beta)
            prediction += p.X[i,j]*beta[j]
        end
        value -= (p.y[i]-prediction)^2/2
    end
    return value
end

function regression_gradient!(g, beta, p)
    mul!(g,transpose(p.X),p.y-p.X*beta)
    g .-= beta ./ 4
    return nothing
end

function regression_batch_gradient!(values, gradients, samples, p)
    residuals = p.y .- p.X * samples
    values .= .-vec(sum(abs2,residuals;dims=1)) ./ 2 .-
        vec(sum(abs2,samples;dims=1)) ./ 8
    mul!(gradients,transpose(p.X),residuals)
    gradients .-= samples ./ 4
    return nothing
end

rng = Xoshiro(21)
X = randn(rng,80,3)
p = (;X,y=X*[0.4,-0.7,1.0]+randn(rng,80))
target = LogTarget(regression_value;
    grad=regression_gradient!,batch_grad=regression_batch_gradient!)
algorithm = LAIS(ProposalBank([SphericalGaussian(zeros(3),0.3)]);
    transition=HamiltonianMonteCarlo(;step_size=0.02,leapfrog_steps=4),
    rounds=2,round_size=1000)
prepared = prepare_sampler(rng,target,p,algorithm)
samples = importance_sample!(prepared)
mean(samples)
```

On CPU, this example uses `grad=`. Transfer the complete prepared sampler
before its first execution to use `batch_grad=` on a supported accelerator.
Use the [device setup](@ref devices-guide) to preserve the chosen precision.
The matrix callback's operations must support that device.
An explicit `batch_grad=` takes priority over scalar gradients and AD in accelerator score phases.

```julia
using CUDA, MLDataDevices
device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(),nothing)
gpu_sampler = device(prepare_sampler(Xoshiro(21),target,p,algorithm))
gpu_samples = importance_sample!(gpu_sampler)
```

| Argument | Vector parameters | Named parameters |
|:--|:--|:--|
| `values` | One log density per column | One logical log density per draw |
| `gradients` | Dimension-by-count matrix | Named tuple of writable batched leaves |
| `samples` | Dimension-by-count matrix | Named tuple of logical batched leaves |
| `p` | Constant model context | Constant model context |

Without context, omit `p`. Fill every value and all gradient entries at finite
values. Each column must match its scalar target and score independently.
Return `-Inf` safely for out-of-support columns; their gradients are ignored.
An out-of-support column must not cause the whole callback to throw.
Accept array views and changing positive batch widths. Never retain borrowed buffers.

With transforms, provide logical scores only. The package adds the Jacobian
and its gradient, including simplex dimension changes and fitted transport.
Named mapping checks synchronize at batch boundaries, not once per sample.

On CPU, scalar `grad` or AD takes priority. If neither is supplied, the package
uses one-column batch calls in the default worker pool. Make that fallback
thread-safe or set `threaded=false`. A callback with shared mutable scratch
should supply a separate scalar CPU derivative.

The example allocates residuals for clarity. Reuse device-adaptable callable
scratch for repeated accelerator calls. Keep scratch valid for the actual width
and do not share it between concurrent runs. Use `batch=` separately to batch
value-only lower draws without computing unnecessary gradients.

See [Devices](@ref devices-guide) for setup and support limits.
The [DI documentation](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/#Compatibility)
describes backend-specific restrictions.
