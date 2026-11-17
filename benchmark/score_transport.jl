module ScoreTransportExperiment

using BenchmarkTools, ImportanceSamplers, LinearAlgebra, Random, Statistics
using PureRNGs
import MLDataDevices
import ImportanceSamplers: AcceleratedKernels

# Research prototype, not a new package API. Each pair uses an additive coupling
# x_odd=z_odd, x_even=z_even+a*(z_odd^2-1). Its determinant is exactly one.
struct CoupledPoint{T,V,C} <: AbstractVector{T}
    parent::V
    coefficients::C
end
CoupledPoint(x::AbstractVector{T}, a) where {T} = CoupledPoint{T,typeof(x),typeof(a)}(x, a)
Base.size(x::CoupledPoint) = size(x.parent)
Base.IndexStyle(::Type{<:CoupledPoint}) = IndexLinear()
@inline function Base.getindex(x::CoupledPoint, i::Int)
    value = x.parent[i]
    return iseven(i) ? value + x.coefficients[i÷2] * (abs2(x.parent[i-1]) - one(value)) :
           value
end

struct BentFunnel{T}
    bend::T
    logscale::T
end

struct Bimodal{T}
    separation::T
    scale::T
end

function logtarget(x, model::BentFunnel{T}) where {T}
    value = zero(T)
    for i = 1:2:length(x)
        u = x[i]
        residual = (x[i+1] - model.bend * (u^2 - one(T))) * exp(-model.logscale * u)
        value -= (u^2 + residual^2) / T(2) + model.logscale * u + log(T(2pi))
    end
    return value
end

function score!(g, x, model::BentFunnel{T}) where {T}
    for i = 1:2:length(x)
        u = x[i]
        residual = x[i+1] - model.bend * (u^2 - one(T))
        precision = exp(-T(2) * model.logscale * u)
        g[i+1] = -residual * precision
        g[i] =
            -u +
            T(2) * model.bend * u * residual * precision +
            model.logscale * (residual^2 * precision - one(T))
    end
    return g
end

function logtarget(x, model::Bimodal{T}) where {T}
    value = zero(T)
    variance = model.scale^2
    for i = 1:2:length(x)
        argument = model.separation * x[i] / variance
        logcosh = abs(argument) + log1p(exp(-T(2) * abs(argument))) - log(T(2))
        value +=
            logcosh - (x[i]^2 + model.separation^2 + x[i+1]^2) / (T(2) * variance) -
            log(T(2pi) * variance)
    end
    return value
end

function score!(g, x, model::Bimodal{T}) where {T}
    variance = model.scale^2
    for i = 1:2:length(x)
        g[i] =
            (-x[i] + model.separation * tanh(model.separation * x[i] / variance)) / variance
        g[i+1] = -x[i+1] / variance
    end
    return g
end

functional(x, ::BentFunnel) = abs2(x[2])
functional(x, ::Bimodal) = abs2(x[1])
reference(model::BentFunnel) = 2model.bend^2 + exp(2model.logscale^2)
reference(model::Bimodal) = model.separation^2 + model.scale^2
mapped_logtarget(z, p) = logtarget(CoupledPoint(z, p.coefficients), p.model)

function couple(samples, coefficients; inverse = false, threaded = true)
    output = similar(samples)
    d = size(samples, 1)
    tasks = threaded ? Threads.nthreads(:default) : 1
    direction = inverse ? -one(eltype(samples)) : one(eltype(samples))
    AcceleratedKernels.foreachindex(output; max_tasks = tasks, min_elems = 4096) do index
        i = mod1(index, d)
        value = samples[index]
        output[index] =
            iseven(i) ?
            value + direction * coefficients[i÷2] * (abs2(samples[index-1]) - one(value)) :
            value
    end
    return output
end

function fit_coefficients(samples, model, ::Val{method}; threaded = true) where {method}
    positions = samples.samples
    d, n = size(positions)
    T = eltype(positions)
    weights = normalized_weights(samples)
    coefficients = similar(weights, T, d ÷ 2)
    if method === :none
        fill!(coefficients, zero(T))
        return coefficients
    end
    method in (:moment, :score) || throw(ArgumentError("unknown coupling fit"))
    terms = similar(positions, T, 5(d ÷ 2), n)
    gradients = method === :score ? similar(positions) : positions
    tasks = threaded ? Threads.nthreads(:default) : 1
    AcceleratedKernels.foreachindex(weights; max_tasks = tasks, min_elems = 512) do j
        method === :score && score!(view(gradients, :, j), view(positions, :, j), model)
        for pair = 1:(d÷2)
            value = positions[2pair, j]
            phi = abs2(positions[2pair-1, j]) - one(value)
            score = method === :score ? gradients[2pair, j] : zero(value)
            offset = 5(pair - 1)
            terms[offset+1, j] = weights[j] * phi^2
            terms[offset+2, j] = weights[j] * phi * value
            terms[offset+3, j] = weights[j] * value^2
            terms[offset+4, j] = weights[j] * phi * score
            terms[offset+5, j] = weights[j] * value * score
        end
    end
    totals = sum(terms; dims = 2)
    AcceleratedKernels.foreachindex(coefficients; max_tasks = 1) do pair
        offset = 5(pair - 1)
        a, b, c, u, v = (totals[offset+i] for i = 1:5)
        # Regress s_even = beta_phi*phi + beta_x*x_even. A Gaussian
        # conditional surrogate has a=-beta_phi/beta_x, beta_x<0.
        denominator = u * b - v * a
        coefficients[pair] =
            method === :moment ? b / a :
            denominator > zero(a) && a * c > b^2 ? (u * c - v * b) / denominator :
            oftype(a, NaN)
    end
    return coefficients
end

function fit_base(samples, coefficients, ::Val{method}; threaded = true) where {method}
    fitted = if method === :none
        samples
    else
        ImportanceSamplers._adopt_validated_weighted_samples(
            couple(samples.samples, coefficients; inverse = true, threaded),
            samples.logweights,
            provenance = samples.provenance,
            diagnostics = samples.diagnostics,
        )
    end
    # Only d + d^2 moment entries and d/2 map coefficients leave the device.
    # CPU construction follows the package's prepare-then-transfer convention.
    location = Array(mean(fitted))
    covariance = Array(cov(fitted))
    T = eltype(location)
    ridge = T(1e-6) * max(tr(covariance) / length(location), one(T))
    factor = cholesky(Symmetric(covariance + ridge * I))
    return FactorStudentT(T(5), location, sqrt(T(3) / T(5)) * Matrix(factor.L))
end

function workflow(
    model,
    method::Symbol;
    n = 100_000,
    pilot = 4096,
    d = 2,
    seed = 2718,
    device = MLDataDevices.cpu_device(),
    threaded = true,
)
    return _workflow(model, Val(method), n, pilot, d, seed, device, threaded)
end

function _workflow(model, method::Val{M}, n, pilot, d, seed, device, threaded) where {M}
    iseven(d) && d >= 2 ||
        throw(ArgumentError("the experiment requires an even dimension >= 2"))
    pilot % 4 == 0 || throw(ArgumentError("pilot must divide into four rounds"))
    T = typeof(reference(model))
    initial = FactorStudentT(T(5), zeros(T, d), T(2) * Matrix{T}(I, d, d))
    prepared = device(
        prepare_sampler(
            device isa MLDataDevices.AbstractAcceleratorDevice ? Threefry4x32(seed) : Xoshiro(seed),
            logtarget,
            model,
            AMIS(initial; rounds = 4, round_size = pilot ÷ 4);
            threaded,
        ),
    )
    training = importance_sample!(prepared)
    coefficients = fit_coefficients(training, model, method; threaded)
    host_coefficients = Array(coefficients)
    all(isfinite, host_coefficients) ||
        error("score fit has no valid Gaussian conditional surrogate")
    proposal = fit_base(training, coefficients, method; threaded)
    target = M === :none ? logtarget : mapped_logtarget
    context = M === :none ? model : (; model, coefficients = host_coefficients)
    production = device(
        prepare_sampler(
            device isa MLDataDevices.AbstractAcceleratorDevice ?
                Threefry4x32(seed + 1_000_000) : Xoshiro(seed + 1_000_000),
            target,
            context,
            ImportanceSampling(proposal; nsamples = n);
            threaded,
        ),
    )
    result = importance_sample!(production)
    if M !== :none
        physical = couple(result.samples, device(host_coefficients); threaded)
        result = ImportanceSamplers._adopt_validated_weighted_samples(
            physical,
            result.logweights;
            provenance = result.provenance,
            diagnostics = result.diagnostics,
        )
    end
    estimate = mean(x -> functional(x, model), result)
    return (;
        estimate,
        error = estimate - reference(model),
        ess = inv(sum(abs2, normalized_weights(result))),
        lognormalizer = lognormalizer(result),
        coefficients = host_coefficients,
        target_evaluations = pilot + n,
        score_evaluations = M === :score ? pilot : 0,
        parameter_readback_bytes = device isa MLDataDevices.AbstractCPUDevice ? 0 :
                                   sizeof(T) * (d + d^2 + d ÷ 2),
    )
end

"""Run ordinary four-round AMIS with the same total target-evaluation budget."""
function full_amis_workflow(
    model;
    n = 100_000,
    pilot = 4096,
    d = 2,
    seed = 2718,
    device = MLDataDevices.cpu_device(),
    threaded = true,
)
    T = typeof(reference(model))
    initial = FactorStudentT(T(5), zeros(T, d), T(2) * Matrix{T}(I, d, d))
    total = n + pilot
    schedule = [total ÷ 4 + (r <= total % 4) for r = 1:4]
    prepared = device(
        prepare_sampler(
            device isa MLDataDevices.AbstractAcceleratorDevice ? Threefry4x32(seed) : Xoshiro(seed),
            logtarget,
            model,
            AMIS(initial; rounds = 4, round_size = schedule);
            threaded,
        ),
    )
    result = importance_sample!(prepared)
    estimate = mean(x -> functional(x, model), result)
    return (;
        estimate,
        error = estimate - reference(model),
        ess = inv(sum(abs2, normalized_weights(result))),
        target_evaluations = total,
    )
end

"""
    compare(; model=BentFunnel(1.25, 0.0), n=100_000, pilot=4096, d=2,
        repetitions=40, seed=2718, device=MLDataDevices.cpu_device(), threaded=true)

Compare a frozen AMIS-fitted Student-t (five degrees of freedom) with moment-fitted and score-fitted
quadratic couplings. Each timed call includes the same four-round pilot,
fitting, production, mapped sample storage, and summaries. Production uses
an independent stream. Warm compilation before alternating timing order.
Report repeated-run error, paired squared-error differences, and time/allocations.
"""
function compare(;
    model = BentFunnel(1.25, 0.0),
    n = 100_000,
    pilot = 4096,
    d = 2,
    repetitions = 40,
    seed = 2718,
    device = MLDataDevices.cpu_device(),
    threaded = true,
)
    methods = (:none, :moment, :score)
    run =
        (method, run_seed) ->
            workflow(model, method; n, pilot, d, seed = run_seed, device, threaded)
    foreach(method -> run(method, seed), methods)
    estimates = [run(method, seed + r) for r = 1:repetitions, method in methods]
    timing = [Float64[] for _ in methods]
    bytes = zeros(Int, 3)
    allocations = zeros(Int, 3)
    for round = 1:4
        for j in (isodd(round) ? (1:3) : (3:-1:1))
            method = methods[j]
            trial = @benchmark $run($method, $seed) samples = 5 evals = 1 seconds = 10
            push!(timing[j], median(trial).time / 1e6)
            bytes[j] = minimum(trial).memory
            allocations[j] = minimum(trial).allocs
        end
    end
    rows = map(eachindex(methods)) do j
        errors = [estimates[r, j].error for r = 1:repetitions]
        squared = errors .^ 2
        rmse = sqrt(mean(squared))
        gain = [estimates[r, 1].error^2 - squared[r] for r = 1:repetitions]
        (;
            method = methods[j],
            rmse,
            rmse_mcse = std(squared) / (2rmse * sqrt(repetitions)),
            bias = mean(errors),
            bias_mcse = std(errors) / sqrt(repetitions),
            mse_gain = mean(gain),
            mse_gain_mcse = std(gain) / sqrt(repetitions),
            median_ms = median(timing[j]),
            median_range_ms = extrema(timing[j]),
            host_bytes = bytes[j],
            allocations = allocations[j],
            mean_ess = mean(estimates[r, j].ess for r = 1:repetitions),
            target_evaluations = estimates[1, j].target_evaluations,
            score_evaluations = estimates[1, j].score_evaluations,
            parameter_readback_bytes = estimates[1, j].parameter_readback_bytes,
        )
    end
    score_gain = [estimates[r, 2].error^2 - estimates[r, 3].error^2 for r = 1:repetitions]
    return (;
        model,
        n,
        pilot,
        d,
        repetitions,
        seed,
        precision = typeof(reference(model)),
        julia = VERSION,
        threads = Threads.nthreads(:default),
        blas_threads = BLAS.get_num_threads(),
        device = string(device),
        versions = (;
            ImportanceSamplers = Base.pkgversion(ImportanceSamplers),
            BenchmarkTools = Base.pkgversion(BenchmarkTools),
            AcceleratedKernels = Base.pkgversion(AcceleratedKernels),
            MLDataDevices = Base.pkgversion(MLDataDevices),
        ),
        rows,
        score_vs_moment = (;
            mse_gain = mean(score_gain),
            mcse = std(score_gain) / sqrt(repetitions),
        ),
        replicate_errors = map(result -> result.error, estimates),
    )
end

end
