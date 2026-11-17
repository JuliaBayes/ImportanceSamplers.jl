"""
    ParetoSmoothedSamples

A weighted-sample result returned by [`psis`](@ref). Samples, saved log targets, and provenance are
shared with the input and must remain read-only. `logweights` contains a separate
array of unnormalized Pareto-smoothed log weights. Existing weighted statistics,
indexing, and resampling use these weights.

`diagnostics` reports `pareto_k`, `tail_length`, `threshold`, `relative_efficiency`,
`status`, `reliable`, and transfer counts for PSIS and later result operations,
not the generation of the input. `reliable` concerns the fitted weight
tail, not support coverage, arbitrary functionals, or adaptive dependence.
[`lognormalizer`](@ref) is deliberately unavailable: smoothing changes that
estimator. Keep the original result for its raw normalizer.
"""
struct ParetoSmoothedSamples{R,S,W,P,D,L,C} <: _AbstractWeightedSamples{R}
    samples::S
    logweights::W
    provenance::P
    diagnostics::D
    logtargets::L
    retained::C
end

function _new_pareto_smoothed_samples(samples, logweights, provenance, diagnostics, logtargets, retained=nothing)
    R = _result_record_type(samples, logweights, provenance)
    return ParetoSmoothedSamples{
        R,
        typeof(samples),
        typeof(logweights),
        typeof(provenance),
        typeof(diagnostics),
        typeof(logtargets),
        typeof(retained),
    }(
        samples,
        logweights,
        provenance,
        diagnostics,
        logtargets,
        retained,
    )
end

Base.iterate(result::ParetoSmoothedSamples) = (result[1], 2)
_result_transfers(result::ParetoSmoothedSamples) = result.diagnostics.transfers

function (device::MLDataDevices.AbstractDevice)(result::ParetoSmoothedSamples)
    return _new_pareto_smoothed_samples(
        _transfer_result_storage(device, result.samples),
        _transfer_result_storage(device, result.logweights),
        _transfer_result_storage(device, result.provenance),
        _transfer_result_storage(device, result.diagnostics),
        _transfer_result_storage(device, result.logtargets),
        _transfer_result_storage(device, result.retained),
    )
end

function lognormalizer(::ParetoSmoothedSamples)
    throw(
        ArgumentError("PSIS changes the normalizer estimator; use the original raw result"),
    )
end

"""
    psis(result; reff=1, threaded=true)
    psis(logweights::AbstractVector; reff=1, threaded=true)

Apply Pareto-smoothed importance sampling to final, unnormalized log weights.
The input remains unchanged. A sample result gives [`ParetoSmoothedSamples`](@ref),
sharing its samples and provenance. A log-weight vector gives
`(; logweights, diagnostics)`.

Fit the largest `floor(min(n/5, 3sqrt(n/reff)))` ratios using Zhang–Stephens
estimation and shape regularization, replace them by fitted Pareto quantiles,
and cap them at the largest raw ratio. This follows Algorithm 1 and Appendices
F–H of [Vehtari et al. (2024)](https://jmlr.org/papers/v25/19-556.html).

`reff` is a supplied relative MCMC efficiency in `(0, 1]`, not weight ESS divided
by `n`. The default describes independent draws. It does not correct arbitrary
adaptive dependence. The diagnostic threshold is `min(1 - 1/log10(n), 0.7)`.
A large `pareto_k` flags unstable estimates, not proof that the target is improper.

`status == :smoothed` means a tail fit succeeded. Too-short or degenerate tails
and numerical fit failures retain the raw weights, with an unavailable (`NaN`)
`pareto_k` and `reliable=false`. Tied tail values receive their block's mean
smoothed mass, so their order cannot change an estimate. A tie across the cutoff,
or a selected zero ratio, retains all raw weights with `status=:degenerate_tail`.
These conservative tie rules extend the paper's continuous-ratio algorithm.
All-zero weights throw [`AllZeroWeightsError`](@ref).
NaN and positive-infinite log weights are invalid.

Arrays retain their device and precision. Only scalar diagnostics reach the host.
`threaded=false` disables CPU worker tasks, not GPU execution. GPU execution needs
device sorting and floating-point support from the selected backend. Smoothing
does not discover missed modes, repair support, or change any sampler's adaptation.
"""
function psis(
    logweights::AbstractVector{T};
    reff::Real = 1,
    threaded::Bool = true,
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(logweights)
    isempty(logweights) && throw(ArgumentError("PSIS requires at least one log weight"))
    efficiency = T(reff)
    isfinite(efficiency) && zero(T) < efficiency <= one(T) ||
        throw(ArgumentError("reff must lie in (0, 1]"))
    return _with_backend_device(_storage_device(logweights)) do
        transfers = _ResultTransferCounter(0, 0)
        _validate_logweights(logweights, transfers, Val(false))
        maximum_logweight = maximum(logweights)
        _record_device_scalar_transfer!(transfers, logweights, T)
        maximum_logweight == -Inf && throw(AllZeroWeightsError())
        n = length(logweights)
        m = floor(Int, min(n / 5, 3sqrt(n / efficiency)))
        output = copy(logweights)
        threshold = n > 1 ? min(T(0.7), one(T) - inv(log10(T(n)))) : T(NaN)
        k, status = if m < 5
            (T(NaN), :insufficient_tail)
        else
            _psis_smooth!(output, logweights, maximum_logweight, m, threaded, transfers)
        end
        diagnostics = (;
            pareto_k = k,
            tail_length = m,
            threshold,
            relative_efficiency = efficiency,
            status,
            reliable = status === :smoothed && k < threshold,
            transfers,
        )
        return (; logweights = output, diagnostics)
    end
end

function psis(result::_AbstractWeightedSamples; kwargs...)
    smoothed = psis(result.logweights; kwargs...)
    return _new_pareto_smoothed_samples(
        result.samples,
        smoothed.logweights,
        result.provenance,
        smoothed.diagnostics,
        result.logtargets,
        result.retained,
    )
end

_psis_tail_order(::MLDataDevices.AbstractCPUDevice, logweights, m) =
    partialsortperm(logweights, (length(logweights)-m):length(logweights))
function _psis_tail_order(device, logweights, m)
    order = AcceleratedKernels.sortperm(logweights)
    return view(order, (length(order)-m):length(order))
end

# The fitted tail uses shifted ratios in [0, 1]. expm1 retains small excesses.
function _psis_smooth!(
    output,
    logweights::AbstractVector{T},
    maximum_logweight,
    m,
    threaded,
    transfers,
) where {T}
    order = _psis_tail_order(_storage_device(logweights), logweights, m)
    tail = similar(logweights, T, m + 1)
    profiles = similar(logweights, T, 2, 30 + isqrt(m))
    parameters = similar(logweights, T, 3)
    tasks = threaded ? Threads.nthreads() : 1
    AcceleratedKernels.foreachindex(tail; max_tasks = tasks, min_elems = 512) do i
        cutoff = logweights[order[1]]
        value = logweights[order[i]]
        # A selected exact zero cannot receive positive fitted mass. Mark that
        # tail unavailable explicitly, rather than producing 0 * NaN by accident.
        tail[i] =
            i == 1 ? exp(cutoff - maximum_logweight) :
            value == -Inf ? T(NaN) : exp(value - maximum_logweight) * -expm1(cutoff - value)
    end
    _psis_fit_tail!(_storage_device(logweights), parameters, profiles, tail, tasks)
    AcceleratedKernels.foreachindex(
        view(order, 2:length(order));
        max_tasks = tasks,
        min_elems = 512,
    ) do j
        k, scale = parameters[1], parameters[2]
        # One worker handles each equal-value block. This preserves its fitted
        # total mass in O(m) work without rank-dependent weights for tied draws.
        if isfinite(k) && (j == 1 || logweights[order[j+1]] != logweights[order[j]])
            last = j
            while last < m && logweights[order[last+2]] == logweights[order[j+1]]
                last += 1
            end
            mass = zero(T)
            for rank = j:last
                mass += _psis_tail_quantile(k, scale, tail[1], rank, m)
            end
            logweight = maximum_logweight + log(mass / T(last - j + 1))
            for rank = j:last
                output[order[rank+1]] = logweight
            end
        end
    end
    host = _is_host_storage(parameters) ? parameters : Array(parameters)
    _record_device_scalar_transfer!(transfers, parameters, NTuple{3,T})
    status =
        host[3] == zero(T) ? :smoothed : host[3] == one(T) ? :degenerate_tail : :failed_fit
    return host[1], status
end

function _psis_fit_tail!(
    ::MLDataDevices.AbstractCPUDevice,
    parameters,
    profiles,
    tail,
    tasks,
)
    # Each profile reads m ratios. Amortize task scheduling over log evaluations.
    AcceleratedKernels.foreachindex(
        view(profiles, 1, :);
        max_tasks = tasks,
        min_elems = max(1, cld(4096, length(tail) - 1)),
    ) do j
        profiles[1, j], profiles[2, j] = _psis_profile(tail, j, size(profiles, 2))
    end
    return _psis_fit!(parameters, profiles, tail)
end

function _psis_fit_tail!(device, parameters, profiles, tail, tasks)
    backend = KernelAbstractions.get_backend(tail)
    width = _LOCAL_REDUCTION_WORKGROUP_SIZE
    _psis_profiles_kernel!(backend, width)(
        profiles,
        tail;
        ndrange = width * size(profiles, 2),
    )
    _psis_parameters_kernel!(backend, width)(parameters, profiles, tail; ndrange = width)
    return nothing
end

_psis_theta(tail::AbstractVector{T}, j, count) where {T} =
    inv(tail[end]) +
    (one(T) - sqrt(T(count) / (T(j) - T(0.5)))) / (T(3) * tail[1+(length(tail)+1)÷4])

@kernel function _psis_profiles_kernel!(profiles, tail)
    lane = @index(Local, Linear)
    candidate = @index(Group, Linear)
    @uniform width = @groupsize()[1]
    @uniform T = eltype(tail)
    scratch = @localmem eltype(tail) (_LOCAL_REDUCTION_WORKGROUP_SIZE,)
    theta = _psis_theta(tail, candidate, size(profiles, 2))
    valid = _psis_valid_tail(tail)
    total = zero(T)
    if valid
        for i = (lane+1):width:length(tail)
            total += iszero(theta) ? tail[i] : log1p(-theta * tail[i])
        end
    end
    scratch[lane] = total
    @synchronize
    stride = width ÷ 2
    while stride > 0
        if lane <= stride
            scratch[lane] += scratch[lane+stride]
        end
        @synchronize
        stride ÷= 2
    end
    if lane == 1
        average = scratch[1] / T(length(tail) - 1)
        shape = iszero(theta) ? zero(T) : average
        scale = iszero(theta) ? average : -shape / theta
        profiles[1, candidate] = valid ? theta : T(NaN)
        profiles[2, candidate] =
            valid ? -T(length(tail) - 1) * (log(scale) + shape + one(T)) : T(-Inf)
    end
end

@kernel function _psis_parameters_kernel!(parameters, profiles, tail)
    lane = @index(Local, Linear)
    @uniform width = @groupsize()[1]
    @uniform T = eltype(tail)
    sums = @localmem eltype(tail) (_LOCAL_REDUCTION_WORKGROUP_SIZE,)
    products = @localmem eltype(tail) (_LOCAL_REDUCTION_WORKGROUP_SIZE,)
    maximum_likelihood = T(-Inf)
    for j = lane:width:size(profiles, 2)
        maximum_likelihood = max(maximum_likelihood, profiles[2, j])
    end
    sums[lane] = maximum_likelihood
    @synchronize
    stride = width ÷ 2
    while stride > 0
        if lane <= stride
            sums[lane] = max(sums[lane], sums[lane+stride])
        end
        @synchronize
        stride ÷= 2
    end
    maximum_likelihood = sums[1]
    @synchronize
    mass = zero(T)
    weighted_theta = zero(T)
    if isfinite(maximum_likelihood)
        for j = lane:width:size(profiles, 2)
            weight = exp(profiles[2, j] - maximum_likelihood)
            mass += weight
            weighted_theta += weight * profiles[1, j]
        end
    end
    sums[lane] = mass
    products[lane] = weighted_theta
    @synchronize
    stride = width ÷ 2
    while stride > 0
        if lane <= stride
            sums[lane] += sums[lane+stride]
            products[lane] += products[lane+stride]
        end
        @synchronize
        stride ÷= 2
    end
    theta = products[1] / sums[1]
    @synchronize
    total = zero(T)
    if isfinite(theta)
        for i = (lane+1):width:length(tail)
            total += iszero(theta) ? tail[i] : log1p(-theta * tail[i])
        end
    end
    sums[lane] = total
    @synchronize
    stride = width ÷ 2
    while stride > 0
        if lane <= stride
            sums[lane] += sums[lane+stride]
        end
        @synchronize
        stride ÷= 2
    end
    if lane == 1
        average = sums[1] / T(length(tail) - 1)
        shape = iszero(theta) ? zero(T) : average
        scale = iszero(theta) ? average : -shape / theta
        _psis_store_fit!(parameters, shape, scale, tail)
    end
end

function _psis_tail_quantile(k::T, scale, cutoff, rank, count) where {T}
    log_survival = log1p(-(T(rank) - T(0.5)) / T(count))
    excess = iszero(k) ? -scale * log_survival : scale * expm1(-k * log_survival) / k
    return min(one(T), cutoff + excess)
end

function _psis_valid_tail(tail)
    m = length(tail) - 1
    minimum = tail[2]
    return isfinite(minimum) && minimum > zero(minimum) && tail[1+(m+2)÷4] > minimum
end

function _psis_tail_mean(f, tail)
    total = zero(eltype(tail))
    for i = 2:length(tail)
        total += f(tail[i])
    end
    return total / (length(tail) - 1)
end

_psis_tail_mean_log(theta, tail) = _psis_tail_mean(x -> log1p(-theta * x), tail)

function _psis_profile(tail::AbstractVector{T}, j, count) where {T}
    m = length(tail) - 1
    _psis_valid_tail(tail) || return (T(NaN), T(-Inf))
    theta = _psis_theta(tail, j, count)
    shape = _psis_tail_mean_log(theta, tail)
    scale = iszero(theta) ? _psis_tail_mean(identity, tail) : -shape / theta
    likelihood = -T(m) * (log(scale) + shape + one(T))
    return theta, likelihood
end

function _psis_fit!(parameters::AbstractVector{T}, profiles, tail) where {T}
    parameters[1] = T(NaN)
    parameters[2] = T(NaN)
    parameters[3] = _psis_valid_tail(tail) ? T(2) : one(T)
    maximum_likelihood = T(-Inf)
    for j in axes(profiles, 2)
        maximum_likelihood = max(maximum_likelihood, profiles[2, j])
    end
    isfinite(maximum_likelihood) || return nothing
    total = zero(T)
    weighted_theta = zero(T)
    for j in axes(profiles, 2)
        weight = exp(profiles[2, j] - maximum_likelihood)
        total += weight
        weighted_theta += weight * profiles[1, j]
    end
    theta = weighted_theta / total
    shape = _psis_tail_mean_log(theta, tail)
    scale = iszero(theta) ? _psis_tail_mean(identity, tail) : -shape / theta
    return _psis_store_fit!(parameters, shape, scale, tail)
end

function _psis_store_fit!(parameters::AbstractVector{T}, shape, scale, tail) where {T}
    m = length(tail) - 1
    k = (T(m) * shape + T(5)) / T(m + 10)
    parameters[1] = T(NaN)
    parameters[2] = T(NaN)
    parameters[3] = _psis_valid_tail(tail) ? T(2) : one(T)
    if isfinite(k) && isfinite(scale) && scale > zero(T)
        parameters[1] = k
        parameters[2] = scale
        parameters[3] = zero(T)
    end
    return nothing
end
