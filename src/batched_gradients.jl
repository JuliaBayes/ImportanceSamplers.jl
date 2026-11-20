_has_batch_gradient(target) = false
_has_batch_gradient(target::_PreparedLogTarget) = target.gradient isa _BatchedGradient
_has_batch_gradient(target::_NamedPreparedTarget) = _has_batch_gradient(target.target)
_batch_only_gradient(target) = false
_batch_only_gradient(target::_PreparedLogTarget) =
    _has_batch_gradient(target) &&
    isnothing(target.gradient.scalar) &&
    target.adtype isa ADTypes.NoAutoDiff
_batch_only_gradient(target::_NamedPreparedTarget) = _batch_only_gradient(target.target)

struct _BoundExplicitBatchGradient{T} <: _BoundBatchGradient
    target::T
end

struct _BoundSingleBatchGradient{P} <: _BoundGradient
    preparations::P
end

struct _BoundNamedBatchGradient{T,B,M,G,J,F,C} <: _BoundBatchGradient
    target::T
    inner::B
    mapped::M
    gradients::G
    jacobians::J
    failures::F
    transfers::C
end

function _prepare_bound_gradient(
    target::_PreparedLogTarget{F,P,A,<:_BatchedGradient},
    sample,
    worker_count,
) where {F,P,A}
    return _prepare_bound_gradient(_scalar_target(target), sample, worker_count)
end

function _prepare_bound_gradient(target, sample, worker_count, ::Type{D}) where {D}
    _batch_only_gradient(target) &&
        return _prepare_single_batch_gradient(target, sample, worker_count, D)
    return _prepare_bound_gradient(target, sample, worker_count)
end

function _prepare_single_batch_gradient(target, sample, worker_count, ::Type{D}) where {D}
    _validate_gradient_worker_count(worker_count)
    preparations = _prepare_gradient_pool(worker_count) do
        values = similar(sample, D, 1)
        bound = _prepare_explicit_batch_gradient(target, reshape(sample, :, 1), values)
        (; bound, values)
    end
    return _BoundSingleBatchGradient(preparations)
end

function _value_and_gradient!(
    destination,
    bound::_BoundSingleBatchGradient,
    sample,
    slot = 1,
)
    preparation = bound.preparations.preparations[slot]
    _batch_value_and_gradient!(
        preparation.values,
        reshape(destination, :, 1),
        preparation.bound,
        reshape(sample, :, 1),
    )
    return only(preparation.values)
end

function _gradient!(destination, bound::_BoundSingleBatchGradient, sample, slot = 1)
    value = _value_and_gradient!(destination, bound, sample, slot)
    isfinite(value) ||
        throw(DomainError(value, "batch_grad must write a finite value at a score point"))
    return destination
end

_hmc_gradient_requested(::_BoundSingleBatchGradient, value) = 1

function _prepare_accelerator_gradient(
    target::_PreparedLogTarget{F,P,A,<:_BatchedGradient},
    locations,
    values,
) where {F,P,A}
    return _prepare_explicit_batch_gradient(target, locations, values)
end

function _prepare_explicit_batch_gradient(
    target::_PreparedLogTarget,
    locations,
    values,
    gradients = locations,
)
    target = _without_retention(target)
    callback = target.gradient.batch
    arguments =
        target.context isa _NoTargetContext ? (values, gradients, locations) :
        (values, gradients, locations, target.context)
    applicable(callback, arguments...) || throw(
        ArgumentError(
            "batch_grad must accept (values, gradients, samples) with optional context p",
        ),
    )
    return _BoundExplicitBatchGradient(target)
end

_call_batch_gradient!(callback, values, gradients, samples, context) =
    callback(values, gradients, samples, context)
_call_batch_gradient!(callback, values, gradients, samples, ::_NoTargetContext) =
    callback(values, gradients, samples)

function _batch_value_and_gradient!(
    values,
    gradients,
    bound::_BoundExplicitBatchGradient,
    locations,
)
    fill!(values, eltype(values)(NaN))
    _fill_batch_gradient!(gradients)
    try
        _call_batch_gradient!(
            bound.target.gradient.batch,
            values,
            gradients,
            locations,
            bound.target.context,
        )
    catch
        KernelAbstractions.synchronize(KernelAbstractions.get_backend(values))
        rethrow()
    end
    return nothing
end

_fill_batch_gradient!(gradients::AbstractArray) = fill!(gradients, eltype(gradients)(NaN))
_fill_batch_gradient!(gradients::NamedTuple) =
    foreach(_fill_batch_gradient!, values(gradients))

function _prepare_explicit_batch_gradient(target::_NamedPreparedTarget, locations, values)
    target = _without_retention(target)
    mapped =
        _allocate_flat_samples(locations, eltype(locations), target.layout, length(values))
    gradients = similar(
        locations,
        _logical_dimension(target.layout, size(locations, 1)),
        length(values),
    )
    logical_gradients = _logical_gradient(target.layout, gradients)
    inner =
        _prepare_explicit_batch_gradient(target.target, mapped, values, logical_gradients)
    failures = _allocate_failure_storage(
        KernelAbstractions.get_backend(values),
        values,
        length(values),
    )
    return _BoundNamedBatchGradient(
        target,
        inner,
        mapped,
        gradients,
        similar(values),
        failures,
        _ResultTransferCounter(0, 0),
    )
end

_logical_gradient_view(scratch::AbstractMatrix, block::_LocatedTransform{Int}, offset) =
    view(scratch, offset, :)
_logical_gradient_view(
    scratch::AbstractMatrix,
    block::_LocatedTransform{<:UnitRange},
    offset,
) = view(scratch, offset:(offset+_logical_block_length(block)-1), :)

@kernel function _pullback_batch_gradients_kernel!(
    values,
    gradients,
    logical_gradients,
    mapped,
    jacobians,
    locations,
    layout,
)
    column = @index(Global, Linear)
    logical = _native_sample_at(mapped, column)
    gradient = _logical_gradient(layout, view(logical_gradients, :, column))
    _pullback_layout!(
        view(gradients, :, column),
        layout,
        view(locations, :, column),
        logical,
        gradient,
    )
    values[column] += jacobians[column]
end

function _batch_value_and_gradient!(
    values,
    gradients,
    bound::_BoundNamedBatchGradient,
    locations,
)
    backend = KernelAbstractions.get_backend(values)
    fill!(bound.failures, zero(eltype(bound.failures)))
    _map_named_batch_kernel!(backend)(
        _result_blocks(bound.mapped),
        bound.jacobians,
        locations,
        bound.target.layout,
        bound.failures;
        ndrange = length(values),
        workgroupsize = 64,
    )
    KernelAbstractions.synchronize(backend)
    snapshot = _device_failure_snapshot(_DeviceFailureRecord(bound.failures))
    _record_reported_transfer!(
        bound.transfers,
        snapshot.transfers.count,
        snapshot.transfers.bytes,
        Val(:failure_snapshot),
    )
    _throw_batch_mapping_failure(bound.target.layout, snapshot.failure, false)
    logical_gradients = _logical_gradient(bound.target.layout, bound.gradients)
    _batch_value_and_gradient!(values, logical_gradients, bound.inner, bound.mapped)
    _pullback_batch_gradients_kernel!(backend)(
        values,
        gradients,
        bound.gradients,
        bound.mapped,
        bound.jacobians,
        locations,
        bound.target.layout;
        ndrange = length(values),
        workgroupsize = 64,
    )
    return nothing
end

function _record_gradient_transfers!(transfers, bound::_BoundNamedBatchGradient)
    _record_reported_transfer!(
        transfers,
        bound.transfers.count,
        bound.transfers.bytes,
        Val(:failure_snapshot),
    )
    bound.transfers.count = bound.transfers.bytes = 0
    bound.transfers.reasons.failure_snapshot = _ReportedTransfer(0, 0)
    return nothing
end
