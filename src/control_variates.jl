"""
    SteinControlVariate

A frozen scalar-functional correction returned by [`fit_control_variate`](@ref).
Its center and coefficients stay on the pilot device and must remain read-only.
Transfer it with the same `MLDataDevices` device used for production samples.
Apply it with [`control_variate_mean`](@ref), not by changing sample weights.
"""
struct SteinControlVariate{C,B}
    center::C
    coefficients::B
end

function Adapt.adapt_structure(to, fit::SteinControlVariate)
    SteinControlVariate(Adapt.adapt(to, fit.center), Adapt.adapt(to, fit.coefficients))
end

"""
    fit_control_variate(f, pilot; scores, ridge=1e-6, threaded=true)

Fit a [`SteinControlVariate`](@ref) for a scalar expectation. Supply a callable
`f(x)` or an aligned cached vector. The pilot must contain nonempty flat
Float32/Float64 vectors. `scores[:,i]` is the exact physical-coordinate target
score `∇logtarget(x_i)`, with matching shape, precision, and device.

The controls are `scores[j,i]` and `1 + (x[j,i]-center[j])*scores[j,i]`.
Their target expectations must be zero: the target must be differentiable and
the corresponding boundary flux must vanish. These conditions are not inferred
from samples. Hard constraints require valid unconstrained coordinates or
boundary-compatible controls. No Hessian or derivative of `f` is needed.

Fit `w*(f-mean(f,pilot))` against `w*controls`, with column scaling and ridge.
The **squared**-weight objective targets fixed-proposal self-normalized IS
asymptotic variance when pilot and production use the same proposal. Other
pilots give a heuristic fit. Improvement is not guaranteed. `ridge=0` requires
full-rank pilot controls; singular systems fail without a fallback.

Freeze the fit and use independent production draws. Self-normalization still
has finite-sample ratio bias. Inputs remain unchanged, zero-weight rows are
ignored, and scalar values use the sample precision. Only the small regression
system returns to CPU for a solve. The fit stays on the pilot device.
`threaded=false` disables CPU worker tasks, not accelerator execution.
See [Score control variates](@ref control-variates-guide).
"""
function fit_control_variate(f, pilot::_AbstractWeightedSamples;
                             scores, ridge=1e-6, threaded::Bool=true)
    return _with_backend_device(_storage_device(pilot.logweights)) do
        samples, weights, values = _stein_data(f, pilot, scores, threaded)
        T = eltype(samples)
        regularization = T(ridge)
        isfinite(regularization) && regularization >= zero(T) ||
            throw(ArgumentError("ridge must be finite and nonnegative"))
        d, n = size(samples)
        z, unit = zero(T), one(T)
        features = similar(samples, 2d, n)
        tasks = threaded ? Threads.nthreads(:default) : 1
        # Reuse the feature buffer for the pilot center. Ignore zero-mass rows
        # before reading their coordinates or scores.
        AcceleratedKernels.foreachindex(samples; max_tasks=tasks, min_elems=4096) do i
            j, k = mod1(i, d), (i - 1) ÷ d + 1
            features[j, k] = iszero(weights[k]) ? z : weights[k] * samples[j, k]
        end
        center = vec(sum(view(features, 1:d, :); dims=2))
        all(isfinite, center) || throw(ArgumentError("nonfinite pilot center"))
        mean_value = LinearAlgebra.dot(weights, values)
        weights ./= maximum(weights)
        AcceleratedKernels.foreachindex(features; max_tasks=tasks, min_elems=4096) do i
            row, k = mod1(i, 2d), (i - 1) ÷ (2d) + 1
            j = mod1(row, d)
            features[i] = if iszero(weights[k])
                z
            else
                score = scores[j, k]
                control = row <= d ? score : unit + (samples[j, k] - center[j]) * score
                weights[k] * control
            end
        end
        all(isfinite, features) || throw(ArgumentError("nonfinite weighted controls"))
        # Max scaling prevents overflow in squared norms. Never subtract empirical
        # control means: the control expectation is known to be zero, not fitted.
        maxima = maximum(abs, features; dims=2)
        maxima .= ifelse.(iszero.(maxima), one(T), maxima)
        features ./= maxima
        response = _scaled_deviation.(values, mean_value, weights)
        gram = Array(features * adjoint(features))
        rhs = Array(features * response)
        norms = sqrt.(LinearAlgebra.diag(gram))
        norms .= ifelse.(iszero.(norms), one(T), norms)
        gram ./= norms * transpose(norms)
        rhs ./= norms
        gram[LinearAlgebra.diagind(gram)] .+= regularization
        coefficients = (LinearAlgebra.cholesky(LinearAlgebra.Symmetric(gram)) \ rhs) ./
                       norms ./ vec(Array(maxima))
        all(isfinite, coefficients) || throw(ArgumentError("nonfinite control fit"))
        device_coefficients = similar(weights, 2d)
        copyto!(device_coefficients, coefficients)
        return SteinControlVariate(center, device_coefficients)
    end
end

"""
    control_variate_mean(f, samples, fit::SteinControlVariate; scores, threaded=true)

Return `sum(normalized_weights(samples) .* (f - controls' * fit.coefficients))`.
Supply a scalar callable or aligned cached values for the same functional used
in fitting. Scores use the same target and coordinates as the fit. Samples,
scores, fit arrays, and cached values must share a device. Scores and fit arrays
must match the sample precision and dimension.

The fit must be independent of these production draws. Start with independent
fixed-proposal production IS. This function does not certify adaptive-history
uncertainty, boundary conditions, or proposal support. It does not change the
samples, weights, weight ESS, resampling, or normalizer. The result is an
expectation estimate, not a weighted sample container or a Monte Carlo error.
Scores and sample-sized work stay on-device. Only scalar checks and the final
estimate reach the host. See [`fit_control_variate`](@ref).
"""
function control_variate_mean(f, result::_AbstractWeightedSamples,
                              fit::SteinControlVariate; scores, threaded::Bool=true)
    return _with_backend_device(_storage_device(result.logweights)) do
        samples, weights, values = _stein_data(f, result, scores, threaded)
        d = size(samples, 1)
        length(fit.center) == d && length(fit.coefficients) == 2d ||
            throw(DimensionMismatch("control fit must match the sample dimension"))
        for array in (fit.center, fit.coefficients)
            eltype(array) === eltype(samples) &&
                _storage_device(array) == _storage_device(samples) ||
                throw(ArgumentError("control fit must match sample precision and device"))
        end
        tasks = threaded ? Threads.nthreads(:default) : 1
        AcceleratedKernels.foreachindex(values; max_tasks=tasks, min_elems=4096) do k
            if iszero(weights[k])
                values[k] = zero(eltype(values))
                return
            end
            correction = zero(eltype(values))
            for j in 1:d
                score = scores[j, k]
                control = one(score) + (samples[j, k] - fit.center[j]) * score
                correction += fit.coefficients[j] * score +
                              fit.coefficients[d + j] * control
            end
            values[k] -= correction
        end
        all(isfinite, values) || throw(ArgumentError("nonfinite corrected values"))
        return LinearAlgebra.dot(weights, values)
    end
end

function _stein_data(f, result, scores, threaded)
    samples = result.samples
    samples isa AbstractMatrix{<:_TransformFloat} && size(samples, 1) > 0 ||
        throw(ArgumentError("control variates require a flat Float32/Float64 sample matrix"))
    Base.require_one_based_indexing(samples, scores)
    size(scores) == size(samples) || throw(DimensionMismatch("scores must match samples"))
    eltype(scores) === eltype(samples) &&
        _storage_device(scores) == _storage_device(samples) ||
        throw(ArgumentError("scores must match sample precision and device"))
    weights = normalized_weights(result)
    weights = eltype(weights) === eltype(samples) ? weights : eltype(samples).(weights)
    values = _stein_values(f, result, weights, threaded)
    all(isfinite, values) || throw(ArgumentError("nonfinite functional values"))
    return samples, weights, values
end

function _stein_values(values::AbstractVector{<:Real}, result, weights, threaded)
    Base.require_one_based_indexing(values)
    length(values) == length(result) ||
        throw(DimensionMismatch("cached values must match the sample count"))
    _storage_device(values) == _storage_device(weights) ||
        throw(ArgumentError("cached values must match the sample device"))
    T = eltype(weights)
    return ifelse.(iszero.(weights), zero(T), T.(values))
end

function _stein_values(f, result, weights, threaded)
    value_type = Base.promote_op(f, _functional_sample_type(result))
    value_type <: Real && isconcretetype(value_type) ||
        throw(ArgumentError("functional must return one concrete real value per sample"))
    values = similar(weights)
    samples = result.samples
    tasks = threaded ? Threads.nthreads(:default) : 1
    AcceleratedKernels.foreachindex(values; max_tasks=tasks, min_elems=4096) do k
        values[k] = iszero(weights[k]) ? zero(eltype(values)) : f(_sample_at(samples, k))
    end
    return values
end
