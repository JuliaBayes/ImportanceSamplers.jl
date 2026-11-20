"""
    RetainCoordinates(evaluate, sample; batch=nothing)

Retain mapped coordinates during importance draws. Pass this configuration as
`LogTarget(f; retain=...)`. `sample` specifies the mapped sample shape and names.
Scalar types are inferred from `evaluate` for the actual input and context,
without evaluating the target.
Only the shape is stored; prototype arrays are not copied to accelerators.
Mapped numeric leaves and the Jacobian must have inferable concrete types.

`evaluate(z)` (or `evaluate(z, p)` with target context) returns
`(logtarget, x, logabsdet)`. Its log target must equal `f(z)`. `logabsdet` is
`log|dx/dz|`, relative to the logical coordinates supplied to the callback.
An engine `transform=` adds its Jacobian to both the saved log target and
saved map Jacobian. Derivatives, MCMC and adaptation trials use `f`, not this
callback. Target values are retained automatically.

The optional batch callback is
`batch!(values, coordinates, logjacobians, samples[, p])`. Fill every output
column. Inputs are read-only and all buffers are borrowed. Keep device data
resident, as for `LogTarget`'s ordinary batch callback.

Results expose `retained = (; coordinates, logjacobians)`. Their restored log
densities are `result.logtargets - result.retained.logjacobians`. These arrays
are independent of input views, callback scratch and later sampler runs.
"""
struct RetainCoordinates{F,S,B}
    evaluate::F
    shape::S
    batch::B
end

RetainCoordinates(evaluate, sample; batch = nothing) =
    RetainCoordinates(evaluate, _retained_shape(sample), batch)

_retained_shape(sample::Number) = ()
_retained_shape(sample::AbstractVector) = (length(sample),)
_retained_shape(sample::NamedTuple) = map(_retained_shape, sample)

Adapt.@adapt_structure RetainCoordinates

_target_callable_has_opaque_host_closure(policy::RetainCoordinates, device) =
    _target_callable_has_opaque_host_closure(policy.evaluate, device) ||
    _target_callable_has_opaque_host_closure(policy.batch, device)
_field_transfer_rewrites_opaque_closure(policy::RetainCoordinates) =
    _field_transfer_rewrites_opaque_closure(policy.evaluate) ||
    _field_transfer_rewrites_opaque_closure(policy.batch)

_without_retention(target::_PreparedLogTarget) =
    isnothing(target.retain) ? target :
    _PreparedLogTarget(
        target.logdensity,
        target.context,
        target.adtype,
        target.gradient,
        target.batch,
    )

function _without_retention(target::_NamedPreparedTarget)
    inner = _without_retention(target.target)
    return inner === target.target ? target : _NamedPreparedTarget(inner, target.layout)
end

_retention_plan(target) = nothing
_retention_plan(target::_PreparedLogTarget) =
    isnothing(target.retain) ? nothing :
    (policy = target.retain, context = target.context, layout = nothing)
function _retention_plan(target::_NamedPreparedTarget)
    plan = _retention_plan(target.target)
    return isnothing(plan) ? nothing : (; plan..., layout = target.layout)
end

function _preflight_retained_target(bound, target, prototype, sample)
    plan = _retention_plan(target)
    return _bind_retained_target(
        bound,
        plan,
        _allocate_retained(plan, prototype, 1, sample),
    )
end

_allocate_retained(::Nothing, prototype, count, sample) = nothing
function _allocate_retained(plan, prototype, count, sample)
    prototype = _retained_prototype(prototype)
    input, outer_jacobian = _retained_input_types(plan.layout, typeof(sample))
    returned = Base.promote_op(
        _call_retained,
        typeof(plan.policy.evaluate),
        input,
        typeof(plan.context),
    )
    returned !== Union{} && returned <: Tuple{Any,Any,Any} || throw(
        ArgumentError(
            "retained evaluation must infer (logtarget, coordinates, logjacobian)",
        ),
    )
    mapped, jacobian = fieldtype(returned, 2), fieldtype(returned, 3)
    isconcretetype(mapped) && isconcretetype(jacobian) && jacobian <: Real || throw(
        ArgumentError(
            "retained coordinates and Jacobian must infer concrete numeric types",
        ),
    )
    J = isnothing(outer_jacobian) ? jacobian : promote_type(jacobian, outer_jacobian)
    _retained_numeric_type(J)
    return (
        coordinates = _allocate_retained_batch(plan.policy.shape, prototype, count, mapped),
        logjacobians = similar(prototype, J, count),
    )
end
_retained_prototype(storage::AbstractArray) = storage
_retained_prototype(storage::NamedTuple) = _retained_prototype(first(storage))

_retained_input_types(::Nothing, S) = (S, nothing)
function _retained_input_types(layout, S)
    mapped = Base.promote_op(_coordinate_to_logical, typeof(layout), S)
    return fieldtype(mapped, 1), fieldtype(mapped, 2)
end

_retained_numeric_type(T) =
    isconcretetype(T) && T <: Number ? T :
    throw(ArgumentError("retained coordinate leaves must infer concrete numeric types"))

_allocate_retained_batch(::Tuple{}, prototype, count, ::Type{T}) where {T<:Number} =
    similar(prototype, _retained_numeric_type(T), count)
_allocate_retained_batch(
    shape::Tuple{Int},
    prototype,
    count,
    ::Type{<:AbstractVector{T}},
) where {T} = similar(prototype, _retained_numeric_type(T), first(shape), count)
function _allocate_retained_batch(
    sample::NamedTuple,
    prototype,
    count,
    ::Type{T},
) where {T<:NamedTuple}
    keys(sample) == fieldnames(T) ||
        throw(ArgumentError("retained coordinate names differ from the shape prototype"))
    leaves = map(keys(sample), values(sample)) do name, leaf
        _allocate_retained_batch(leaf, prototype, count, fieldtype(T, name))
    end
    return NamedTuple{keys(sample)}(leaves)
end

_retained_view(::Nothing, indices) = nothing
_retained_view(retained, indices) = (
    coordinates = _sample_view(retained.coordinates, indices),
    logjacobians = view(retained.logjacobians, indices),
)

struct _RetainingTarget{T,P,O} <: _BoundTarget
    target::T
    plan::P
    output::O
end
Adapt.@adapt_structure _RetainingTarget

# Keep type inference and non-retained callers on the ordinary scalar target.
@inline (target::_RetainingTarget)(sample) = target.target(sample)
@inline _target_value(target, sample, slot) = target(sample)
@inline _target_value(target::_BoundBatchTarget, sample, slot) =
    _target_value(target.scalar, sample, slot)
@inline function _target_value(target::_RetainingTarget, sample, slot)
    value, coordinates, jacobian =
        _call_retained(target.plan.policy.evaluate, sample, target.plan.context)
    _store_retained!(target.output.coordinates, coordinates, slot)
    target.output.logjacobians[slot] = jacobian
    return value
end

@inline _call_retained(f, sample, context) = f(sample, context)
@inline _call_retained(f, sample, ::_NoTargetContext) = f(sample)

@inline function _store_retained!(output::AbstractVector, sample, slot)
    output[slot] = sample
    return nothing
end
@inline function _store_retained!(output::AbstractMatrix, sample, slot)
    length(sample) == size(output, 1) ||
        throw(DimensionMismatch("retained coordinate length changed"))
    for i in axes(output, 1)
        output[i, slot] = sample[i]
    end
    return nothing
end
@inline function _store_retained!(output::NamedTuple, sample::NamedTuple, slot)
    keys(output) == keys(sample) ||
        throw(ArgumentError("retained coordinate names changed"))
    foreach((a, x) -> _store_retained!(a, x, slot), values(output), values(sample))
    return nothing
end

_bind_retained_target(target, ::Nothing, ::Nothing) = target
function _bind_retained_target(target, plan, output)
    scalar = _RetainingTarget(_scalar_target(target), plan, output)
    isnothing(plan.policy.batch) && return scalar
    return _BoundBatchTarget(
        scalar,
        _RetainingBatch(plan.policy.batch, output),
        plan.context,
        _batch_transfers_or_new(target),
    )
end
_bind_retained_target(target::_BoundNamedTarget, plan, output) =
    _BoundNamedTarget(_bind_retained_target(target.target, plan, output), target.layout)
_bind_retained_target(target::_BoundNamedTarget, ::Nothing, ::Nothing) = target

_batch_transfers_or_new(target) = _ResultTransferCounter(0, 0)
_batch_transfers_or_new(target::_BoundBatchTarget) = target.transfers

struct _RetainingBatch{F,O}
    callback::F
    output::O
end
Adapt.@adapt_structure _RetainingBatch
(batch::_RetainingBatch)(logs, samples, context) = batch.callback(
    logs,
    batch.output.coordinates,
    batch.output.logjacobians,
    samples,
    context,
)
(batch::_RetainingBatch)(logs, samples) =
    batch.callback(logs, batch.output.coordinates, batch.output.logjacobians, samples)

@inline function _target_value(target::_BoundNamedTarget, coordinate, slot)
    logical, jacobian, reason, block = _coordinate_to_logical(target.layout, coordinate)
    iszero(reason) || _throw_named_transform_failure(target.layout, reason, block)
    value = _target_value(target.target, logical, slot)
    _validate_target_logdensity(value)
    _add_retained_jacobian!(target.target, slot, jacobian)
    return value + jacobian
end

@inline _add_retained_jacobian!(target, slot, jacobian) = nothing
@inline _add_retained_jacobian!(target::_RetainingTarget, slot, jacobian) =
    (target.output.logjacobians[slot] += jacobian)

_retained_jacobians(target) = nothing
_retained_jacobians(target::_RetainingTarget) = target.output.logjacobians
_retained_jacobians(target::_BoundBatchTarget) = _retained_jacobians(target.scalar)
_add_retained_batch_jacobians!(::Nothing, jacobians) = nothing
_add_retained_batch_jacobians!(output, jacobians) = (output .+= jacobians)

_retained_output(target) = nothing
_retained_output(target::_RetainingTarget) = target.output
_retained_output(
    target::Union{
        _NativeCPUTarget,
        _NativeDeviceTarget,
        _NativeBatchTarget,
        _BoundNamedTarget,
        _SavingBatchTarget,
    },
) = _retained_output(target.target)
_retained_output(target::_BoundBatchTarget) = _retained_output(target.scalar)
_retained_output(target::_SavingTargetEvaluator) = _retained_output(target.evaluator)

_save_retained_target(target, logs, ::Nothing, ::Nothing, backend, failures) =
    _save_target_evaluator(target, logs)
function _save_retained_target(
    target::Union{_NativeCPUTarget{L},_NativeDeviceTarget{L},_NativeBatchTarget{L}},
    logs,
    plan::NamedTuple,
    output::NamedTuple,
    backend,
    failures,
) where {L}
    bound = _bind_retained_target(target.target, plan, output)
    evaluator, _ = _native_target_evaluator(backend, bound, L, failures)
    return _save_target_evaluator(evaluator, logs)
end

_validate_retained(::Nothing, count, device) = nothing
function _validate_retained(retained, count, device)
    _sample_count(retained.coordinates) == count &&
    length(retained.logjacobians) == count ||
        throw(ArgumentError("retained coordinates and samples must align"))
    _validate_numeric_storage(retained.coordinates, "retained coordinate") == device &&
    _storage_device(retained.logjacobians) == device ||
        throw(ArgumentError("retained coordinates and samples must use the same device"))
    return nothing
end

@inline (target::_NativeDeviceTarget{L,<:_RetainingTarget})(sample, slot) where {L} =
    _native_device_target_result(target.target, sample, L, slot)
@inline (target::_NativeDeviceTarget{L,<:_BoundNamedTarget{<:_RetainingTarget}})(
    sample,
    slot,
) where {L} = _native_device_target_result(target.target, sample, L, slot)

@inline function _native_device_target_result(
    target::_RetainingTarget,
    sample,
    ::Type{L},
    slot,
) where {L}
    value = convert(L, _target_value(target, sample, slot))
    reason = _native_target_reason(value)
    return value, reason, !iszero(reason)
end

@inline function _native_device_target_result(
    target::_BoundNamedTarget,
    coordinate,
    ::Type{L},
    slot,
) where {L}
    isnothing(_retained_jacobians(target.target)) &&
        return _native_device_target_result(target, coordinate, L)
    logical, jacobian, reason, _ = _coordinate_to_logical(target.layout, coordinate)
    iszero(reason) || return zero(L), _NATIVE_TARGET_NAN, true
    value = convert(L, _target_value(target.target, logical, slot) + jacobian)
    _add_retained_jacobian!(target.target, slot, jacobian)
    reason = _native_target_reason(value)
    return value, reason, !iszero(reason)
end
