"""
    QuadraticCoupling(coefficients)

Map pairs by `x[2j-1] = z[2j-1]` and
`x[2j] = z[2j] + coefficients[j] * (z[2j-1]^2 - 1)`. The inverse subtracts
the shift. The log absolute Jacobian determinant is zero. Coefficients are
copied, must remain read-only, and must match the proposal precision.
An unpaired final coordinate stays unchanged. Empty coefficients give the 1D identity.
Supply `floor(d/2)` coefficients for an input vector of dimension `d`.

Pass `transform=` to [`prepare_sampler`](@ref) or [`importance_sample`](@ref).
Native samplers adapt in latent coordinates and return physical coordinates.
Targets and explicit gradients use physical coordinates, including batch targets.
CPU and accelerator execution need no per-sample host transfer.

See [`fit_transport`](@ref) and [Fitted transport](@ref transport-guide).
This uses the additive map in [Dinh et al., Section 3.2](https://arxiv.org/abs/1410.8516),
not their neural-network training procedure or a general geometry fit.
"""
struct QuadraticCoupling{V} <: AbstractSampleTransform
    coefficients::V

    QuadraticCoupling(coefficients::V, ::_PreparedProposalToken) where {V} =
        new{V}(coefficients)
end

function QuadraticCoupling(coefficients::AbstractVector{T}) where {T<:_TransformFloat}
    Base.require_one_based_indexing(coefficients)
    all(isfinite, coefficients) ||
        throw(ArgumentError("coupling coefficients must be finite"))
    return QuadraticCoupling(copy(coefficients), _PreparedProposalToken())
end

Adapt.adapt_structure(to, transform::QuadraticCoupling) =
    QuadraticCoupling(Adapt.adapt(to, transform.coefficients), _PreparedProposalToken())

struct _CoupledPoint{T,V,C} <: AbstractVector{T}
    parent::V
    coefficients::C
end

_CoupledPoint(x::AbstractVector{T}, coefficients) where {T} =
    _CoupledPoint{T,typeof(x),typeof(coefficients)}(x, coefficients)
Base.size(point::_CoupledPoint) = size(point.parent)
Base.IndexStyle(::Type{<:_CoupledPoint}) = IndexLinear()
@inline function Base.getindex(point::_CoupledPoint, index::Int)
    value = point.parent[index]
    isodd(index) && return value
    coefficient = point.coefficients[index÷2]
    iszero(coefficient) && return value
    return value + coefficient * (abs2(point.parent[index-1]) - one(value))
end

function _validate_known_transform_input(transform::QuadraticCoupling, input)
    input === nothing && return nothing
    input isa AbstractVector &&
        !isempty(input) &&
        length(input) ÷ 2 == length(transform.coefficients) || throw(
        DimensionMismatch("coupling requires one coefficient per complete coordinate pair"),
    )
    eltype(input) === eltype(transform.coefficients) ||
        throw(ArgumentError("coupling and coordinate precision must match"))
    return nothing
end

function _named_target_layout(proposal, transform::QuadraticCoupling)
    dimension = _proposal_dimension(proposal)
    dimension isa Int && dimension > 0 && dimension ÷ 2 == length(transform.coefficients) ||
        throw(
            DimensionMismatch(
                "coupling requires one coefficient per complete coordinate pair",
            ),
        )
    _coupling_precision_matches(transform, proposal) ||
        throw(ArgumentError("coupling and proposal precision must match"))
    return transform
end

function _resolve_prepared_target(
    target::_PreparedLogTarget,
    proposal,
    transform::QuadraticCoupling,
)
    resolved = _resolve_prepared_target(target, proposal)
    _named_target_layout(proposal, transform)
    isempty(transform.coefficients) && return resolved
    return _prepare_named_target(resolved, proposal, transform)
end

_coupling_precision_matches(transform, proposal) =
    _native_fused_float_type(proposal) === eltype(transform.coefficients)
_coupling_precision_matches(transform, bank::ProposalBank) =
    all(proposal -> _coupling_precision_matches(transform, proposal), bank.proposals)

_logical_dimension(::QuadraticCoupling, sampling_dimension) = sampling_dimension
_logical_binding_sample(transform::QuadraticCoupling, coordinate) =
    _CoupledPoint(coordinate, transform.coefficients)
_logical_gradient(::QuadraticCoupling, scratch) = scratch
_allocate_flat_samples(prototype, ::Type{T}, ::QuadraticCoupling, count) where {T} =
    similar(prototype, T, size(prototype, 1), count)
_native_failure_location(transform::QuadraticCoupling, block) =
    block > length(transform.coefficients) ? 2block - 1 : (2block-1):2block
_throw_named_transform_failure(transform::QuadraticCoupling, reason, block) = throw(
    InvalidTransformError(
        _native_transform_failure_reason(reason),
        _native_failure_location(transform, block),
    ),
)

@inline function _coordinate_to_logical(transform::QuadraticCoupling, coordinate)
    logical = _CoupledPoint(coordinate, transform.coefficients)
    for index in eachindex(coordinate)
        reason =
            !isfinite(coordinate[index]) ? _NATIVE_TRANSFORM_NONFINITE_INPUT :
            !isfinite(logical[index]) ? _NATIVE_TRANSFORM_NONFINITE_OUTPUT : UInt16(0)
        iszero(reason) || return logical, zero(eltype(coordinate)), reason, (index + 1) ÷ 2
    end
    return logical, zero(eltype(coordinate)), UInt16(0), 0
end

function _transform_with_logjac(transform::QuadraticCoupling, coordinate::AbstractVector)
    _validate_known_transform_input(transform, coordinate)
    return _ad_coordinate_to_logical(transform, coordinate, Val(true))
end

function _inverse_with_logjac(transform::QuadraticCoupling, physical::AbstractVector)
    inverse = QuadraticCoupling(-transform.coefficients, _PreparedProposalToken())
    return _transform_with_logjac(inverse, physical)
end

function _ad_coordinate_to_logical(transform::QuadraticCoupling, coordinate, ::Val{true})
    logical, logjac, reason, block = _coordinate_to_logical(transform, coordinate)
    iszero(reason) || _throw_named_transform_failure(transform, reason, block)
    return collect(logical), logjac
end

function _pullback_layout!(
    destination,
    transform::QuadraticCoupling,
    sample,
    logical,
    gradient,
)
    for pair in eachindex(transform.coefficients)
        odd = 2pair - 1
        even = 2pair
        coefficient = transform.coefficients[pair]
        destination[odd] =
            iszero(coefficient) ? gradient[odd] :
            gradient[odd] + coefficient * (2sample[odd] * gradient[even])
        destination[even] = gradient[even]
    end
    isodd(length(sample)) && (destination[end] = gradient[end])
    return destination
end

function _couple_samples(samples, coefficients, weights, threaded)
    output = similar(samples)
    dimension = size(samples, 1)
    tasks = threaded ? Threads.nthreads(:default) : 1
    AcceleratedKernels.foreachindex(output; max_tasks = tasks, min_elems = 4096) do index
        coordinate = mod1(index, dimension)
        value = samples[index]
        if iszero(weights[(index-1)÷dimension+1])
            output[index] = zero(value)
            return
        end
        coefficient = iseven(coordinate) ? coefficients[coordinate÷2] : zero(value)
        output[index] =
            iszero(coefficient) ? value :
            value + coefficient * (abs2(samples[index-1]) - one(value))
    end
    return output
end

"""
    fit_transport(samples; scores=nothing, dof=5, ridge=1e-6, threaded=true)

Return a fitted `(; proposal, transform)` containing a latent [`FactorStudentT`](@ref)
and a [`QuadraticCoupling`](@ref), with CPU parameters. The weighted pilot must
contain nonempty flat Float32/Float64 vectors. Inputs remain unchanged.
Complete coordinate pairs are fitted and an unpaired final coordinate stays unchanged.
Use a separate RNG and production sampler:

```julia
fit = fit_transport(pilot)
production = importance_sample(rng, logtarget,
    ImportanceSampling(fit.proposal; nsamples=100_000); transform=fit.transform)
```

The default fits weighted slopes of `x_even` on `x_odd^2-1`, with an intercept.
Optional `scores` fits a Gaussian conditional-score surrogate. Supply finite
`∇logtarget(x)` values at the stored physical samples, matching their matrix
shape, device, and precision. Scores need no Hessian and do not guarantee improvement.

Both fits match inverse-mapped weighted moments with fixed `dof > 2`.
Add `ridge * max(tr(covariance)/d, 1)` to the covariance diagonal, then convert
to Student-t scale. Undefined or ill-conditioned fits, nonnegative score slopes,
and nonfinite parameters fail without clipping or an identity fallback.

Only fitted coefficients and moments return to CPU. Transfer the prepared sampler
to the chosen device. `threaded=false` disables CPU workers, not accelerators.
Production counts and methods stay independent of the pilot. Self-normalized
estimates retain ratio bias. See [Fitted transport](@ref transport-guide).
"""
function fit_transport(
    result::_AbstractWeightedSamples;
    scores = nothing,
    dof = 5,
    ridge = 1e-6,
    threaded::Bool = true,
)
    samples = result.samples
    samples isa AbstractMatrix{<:_TransformFloat} || throw(
        ArgumentError("transport fitting requires a flat Float32 or Float64 sample matrix"),
    )
    d, n = size(samples)
    d > 0 || throw(ArgumentError("transport fitting requires a positive dimension"))
    T = eltype(samples)
    nu, regularization = T(dof), T(ridge)
    isfinite(nu) && nu > T(2) || throw(ArgumentError("dof must be finite and > 2"))
    isfinite(regularization) && regularization >= zero(T) ||
        throw(ArgumentError("ridge must be finite and nonnegative"))
    if scores !== nothing
        size(scores) == size(samples) ||
            throw(DimensionMismatch("scores must match the pilot shape"))
        eltype(scores) === T ||
            throw(ArgumentError("scores must match the pilot precision"))
        _storage_device(scores) == _storage_device(samples) ||
            throw(ArgumentError("scores must match the pilot device"))
    end
    return _with_backend_device(_storage_device(samples)) do
        scores === nothing ||
            all(isfinite, scores) ||
            throw(ArgumentError("scores must be finite"))
        weights = normalized_weights(result)
        weights = eltype(weights) === T ? weights : T.(weights)
        coefficients = _fit_coupling_coefficients(samples, weights, scores, threaded)
        host_coefficients = Array(coefficients)
        all(isfinite, host_coefficients) || throw(ArgumentError("degenerate coupling fit"))
        latent = _couple_samples(samples, -coefficients, weights, threaded)
        center = _weighted_sum(latent, weights)
        covariance = Array(_weighted_covariance(latent, weights, center))
        location = Array(center)
        all(isfinite, location) && all(isfinite, covariance) ||
            throw(ArgumentError("nonfinite latent moments"))
        diagonal_ridge =
            regularization * max(sum(x -> x / d, LinearAlgebra.diag(covariance)), one(T))
        for index = 1:d
            covariance[index, index] += diagonal_ridge
        end
        factor = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(covariance))
        proposal = FactorStudentT(nu, location, sqrt((nu - T(2)) / nu) * Matrix(factor.L))
        return (; proposal, transform = QuadraticCoupling(host_coefficients))
    end
end

function _coupling_statistics!(terms, samples, weights, scores, centers, tasks)
    z = zero(eltype(samples))
    AcceleratedKernels.foreachindex(weights; max_tasks = tasks, min_elems = 512) do j
        for pair = 1:(size(samples, 1)÷2)
            offset = 5(pair - 1)
            w = weights[j]
            phi = iszero(w) ? z : abs2(samples[2pair-1, j]) - one(z)
            y = iszero(w) ? z : samples[2pair, j]
            s = scores === nothing || iszero(w) ? z : scores[2pair, j]
            if centers !== nothing && !iszero(w)
                phi -= centers[offset+1]
                y -= centers[offset+2]
                s -= centers[offset+3]
            end
            statistics =
                centers === nothing ? (w * phi, w * y, w * s, z, z) :
                (w * phi^2, w * phi * y, w * y^2, w * phi * s, w * y * s)
            terms[offset+1, j] = statistics[1]
            terms[offset+2, j] = statistics[2]
            terms[offset+3, j] = statistics[3]
            terms[offset+4, j] = statistics[4]
            terms[offset+5, j] = statistics[5]
        end
    end
    return sum(terms; dims = 2)
end

function _fit_coupling_coefficients(samples, weights, scores, threaded)
    pairs, n = size(samples, 1) ÷ 2, size(samples, 2)
    iszero(pairs) && return similar(weights, eltype(samples), 0)
    terms = similar(samples, 5pairs, n)
    tasks = threaded ? Threads.nthreads(:default) : 1
    centers = _coupling_statistics!(terms, samples, weights, scores, nothing, tasks)
    totals = _coupling_statistics!(terms, samples, weights, scores, centers, tasks)
    coefficients = similar(weights, eltype(samples), pairs)
    AcceleratedKernels.foreachindex(coefficients; max_tasks = 1) do pair
        offset = 5(pair - 1)
        a, b, c, u, v = (totals[offset+i] for i = 1:5)
        T = eltype(totals)
        coefficient = oftype(a, NaN)
        # Resolve predictor variation above the rounding error in its weighted center.
        if isfinite(a) && sqrt(a) > 8eps(T) * max(abs(centers[offset+1]), one(T))
            if scores === nothing
                coefficient = b / a
            elseif isfinite(c) && c > zero(T)
                sa, sc = sqrt(a), sqrt(c)
                correlation = (b / sa) / sc
                determinant = (one(T) - correlation) * (one(T) + correlation)
                us, vs = u / sa, v / sc
                slope = vs - correlation * us
                if determinant > sqrt(eps(T)) &&
                   slope < -sqrt(eps(T)) * max(abs(us), abs(vs))
                    coefficient = ((correlation * vs - us) / slope) * (sc / sa)
                end
            end
        end
        coefficients[pair] = coefficient
    end
    return coefficients
end
