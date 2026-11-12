module ImportanceSamplersReactantExt

import ImportanceSamplers as IS
import Reactant
import Random
import PureRNGs
import MLDataDevices
import ADTypes
import DifferentiationInterface as DI
import KernelAbstractions as KA
import LinearAlgebra as LA

const _ReactantStorage = Union{Reactant.AnyConcreteRArray,
    SubArray{T,N,P} where {T,N,P<:Reactant.AnyConcreteRArray}}

# Retained phases guard callbacks on-device. Host failure snapshots belong
# after the executable, not inside its trace.
IS._check_batch_failure!(target, storage::AbstractArray{<:Reactant.TracedRNumber},
    transform=IS._NoSampleTransform()) = nothing
IS._check_batch_mapping!(target, storage::AbstractArray{<:Reactant.TracedRNumber}, capture) = nothing

_batch_trace_samples(samples::AbstractArray) = Reactant.TracedUtils.materialize_traced_array(samples)
_batch_trace_samples(samples::NamedTuple) = map(_batch_trace_samples, samples)

# Keep target reductions separate from proposal-recovery closures. Combining
# them can leave unreachable recursive reductions in Reactant's kernel IR.
_separate_target(target) = target
_separate_target(target::IS._NativeDeviceTarget{L}) where {L} =
    IS._NativeBatchTarget{L,typeof(target)}(target)

KA.@kernel function _scalar_target_values_kernel!(logs, samples, target, failures, ready)
    index = KA.@index(Global, Linear)
    value = eltype(logs)(-Inf)
    if ready === nothing || ready[1]
        value, reason, failed = target(IS._native_sample_at(samples, index), index)
        if failed
            value = oftype(value, -Inf)
            failures === nothing || IS._record_native_failure!(failures, index, 0, reason)
        end
    end
    logs[index] = value
end

function IS._invoke_batch_target!(target::IS._NativeDeviceTarget,
    logs::AbstractArray{<:Reactant.TracedRNumber}, samples, failures, execution;
    transform=IS._NoSampleTransform())
    isempty(logs) && return nothing
    # Snapshot before this kernel records target failures, so its threads agree.
    ready = failures === nothing ? nothing : iszero.(view(failures, 1:1))
    _scalar_target_values_kernel!(KA.get_backend(logs))(logs, samples, target, failures, ready;
        ndrange=length(logs))
    return nothing
end

function IS._invoke_batch_target!(target::IS._BoundBatchTarget,
    logs::AbstractArray{<:Reactant.TracedRNumber}, samples, failures, execution;
    transform=IS._NoSampleTransform())
    isempty(logs) && return nothing
    fill!(logs, eltype(logs)(NaN))
    batch, context = target.batch, target.context
    values = Reactant.TracedUtils.materialize_traced_array(logs)
    samples = _batch_trace_samples(samples)
    if failures === nothing
        IS._call_batch_target!(batch, values, samples, context)
    else
        Reactant.@trace if iszero(sum(view(failures, 1:1)))
            IS._call_batch_target!(batch, values, samples, context)
        end
    end
    copyto!(logs, values)
    if failures !== nothing
        IS._validate_batch_values_kernel!(KA.get_backend(logs))(logs, failures;
            ndrange=length(logs))
    end
    return nothing
end

IS._validate_transfer_rng(::MLDataDevices.ReactantDevice, ::PureRNGs.AbstractPureRNG) = nothing
function IS._validate_transfer_rng(device::MLDataDevices.ReactantDevice, rng)
    MLDataDevices.get_device_type(rng) === MLDataDevices.ReactantDevice &&
    applicable(PureRNGs.rand_next, rng, Float32) ||
        throw(IS.SamplerDeviceError(device, :accelerator_rng_unavailable))
    actual = MLDataDevices.get_device(rng)
    (ismissing(device.client) || device.client == actual.client) &&
    (ismissing(device.device) || device.device == actual.device) ||
        throw(IS.SamplerDeviceError(device, :device_residency_mismatch))
    if !ismissing(device.sharding)
        expected = _reactant_rng_sharding(device.sharding)
        all(sharding -> _same_rng_sharding(sharding, expected), values(actual.sharding)) ||
            throw(IS.SamplerDeviceError(device, :device_residency_mismatch))
    end
    return nothing
end

_reactant_rng_sharding(sharding) = sharding
_same_rng_sharding(actual, expected) = actual == expected
_same_rng_sharding(actual::Reactant.Sharding.NamedSharding, expected::Reactant.Sharding.Replicated) =
    actual.mesh == expected.mesh && all(partition -> all(isnothing, partition), actual.partition_spec)
function _reactant_rng_sharding(sharding::IdDict)
    all(x -> x isa Reactant.Sharding.NoSharding, values(sharding)) &&
        return Reactant.Sharding.NoSharding()
    meshes = unique(x.mesh for x in values(sharding))
    return Reactant.Sharding.Replicated(only(meshes))
end

function IS._transfer_rng(device::MLDataDevices.ReactantDevice, rng::PureRNGs.AbstractPureRNG)
    kwargs = (;)
    ismissing(device.client) || (kwargs = (; kwargs..., client=device.client))
    ismissing(device.device) || (kwargs = (; kwargs..., device=device.device))
    ismissing(device.sharding) ||
        (kwargs = (; kwargs..., sharding=_reactant_rng_sharding(device.sharding)))
    return Reactant.to_rarray(rng; kwargs...)
end
function IS._transfer_rng(device::MLDataDevices.ReactantDevice, rng)
    IS._validate_transfer_rng(device, rng)
    return deepcopy(rng)
end
IS._clone_rng(::MLDataDevices.ReactantDevice, rng::PureRNGs.AbstractPureRNG) = rng
IS._clone_rng(::MLDataDevices.ReactantDevice, rng::Random.AbstractRNG) = copy(rng)
function IS._clone_rng(device::MLDataDevices.ReactantDevice, rng)
    IS._validate_transfer_rng(device, rng)
    return deepcopy(rng)
end

# `@jit` compiles on every call. Standalone result operations retain one
# executable per function, argument types, shapes and device. Views keep `@jit`:
# Reactant traces their indices as constants, so equal shapes can differ.
# The limit bounds executable memory when sample counts vary across calls.
const _RETAINED_CALLS = Dict{Any,Any}()
const _RETAINED_CALLS_LOCK = ReentrantLock()
const _RETAINED_CALL_LIMIT = 64

_retained_shape(x::Reactant.ConcreteRArray) = (size(x), Reactant.XLA.device(x))
_retained_shape(x::Reactant.ConcreteRNumber) = ((), Reactant.XLA.device(x))
_retained_shape(x::Bool) = x
function _retained_shape(x::NamedTuple)
    shapes = map(_retained_shape, x)
    return any(isnothing, shapes) ? nothing : shapes
end
# Carriers have a generator-specific fixed state shape. Numerical arguments
# supply the device key; key and cursor values must stay runtime inputs.
_retained_shape(x) = Base.issingletontype(typeof(x)) || applicable(PureRNGs.rand_next, x, Float32) ? () : nothing

# Select compilation from resident data, never from the process-wide default.
_call_client() = nothing
_call_client(x::Union{Reactant.ConcreteRArray,Reactant.ConcreteRNumber}, rest...) = Reactant.XLA.client(x)
function _call_client(x::AbstractArray, rest...)
    storage = parent(x)
    return storage === x ? _call_client(rest...) : _call_client(storage, rest...)
end
_call_client(x, rest...) = _call_client(rest...)

function _retained_call(f, arguments...)
    client = _call_client(arguments...)
    shapes = map(_retained_shape, arguments)
    any(isnothing, shapes) && return Reactant.@jit client=client f(arguments...)
    key = (f, client, map(typeof, arguments), shapes)
    compiled = @lock _RETAINED_CALLS_LOCK begin
        stored = get(_RETAINED_CALLS, key, nothing)
        if stored === nothing && length(_RETAINED_CALLS) < _RETAINED_CALL_LIMIT
            stored = _RETAINED_CALLS[key] = Reactant.@compile client=client f(arguments...)
        end
        stored
    end
    compiled === nothing && return Reactant.@jit client=client f(arguments...)
    return compiled(arguments...)
end

function IS._rand_uniform!(rng, values::Reactant.ConcreteRArray; threaded = false)
    isempty(values) && return rng
    _, rng = _retained_call(PureRNGs.rand_next!, rng, values)
    return rng
end

function IS._rand_normal!(rng, values::Reactant.ConcreteRArray; threaded = false)
    isempty(values) && return rng
    _, rng = _retained_call(PureRNGs.randn_next!, rng, values)
    return rng
end

function IS._rand_uniform!(
    rng,
    values::SubArray{<:Reactant.TracedRNumber};
    threaded = false
)
    isempty(values) && return rng
    traced = Reactant.TracedUtils.materialize_traced_array(values)
    _, rng = PureRNGs.rand_next!(rng, traced; threaded)
    copyto!(values, traced)
    return rng
end

function IS._rand_normal!(rng, values::SubArray{<:Reactant.TracedRNumber}; threaded = false)
    isempty(values) && return rng
    traced = Reactant.TracedUtils.materialize_traced_array(values)
    _, rng = PureRNGs.randn_next!(rng, traced; threaded)
    copyto!(values, traced)
    return rng
end

IS._rand_uniform!(
    rng::Random.AbstractRNG,
    values::Reactant.ConcreteRArray;
    threaded = false
) = invoke(IS._rand_uniform!, Tuple{Random.AbstractRNG,Any}, rng, values; threaded)
IS._rand_normal!(
    rng::Random.AbstractRNG,
    values::Reactant.ConcreteRArray;
    threaded = false
) = invoke(IS._rand_normal!, Tuple{Random.AbstractRNG,Any}, rng, values; threaded)
IS._rand_uniform!(
    rng::Random.AbstractRNG,
    values::SubArray{<:Reactant.TracedRNumber};
    threaded = false
) = invoke(IS._rand_uniform!, Tuple{Random.AbstractRNG,Any}, rng, values; threaded)
IS._rand_normal!(
    rng::Random.AbstractRNG,
    values::SubArray{<:Reactant.TracedRNumber};
    threaded = false
) = invoke(IS._rand_normal!, Tuple{Random.AbstractRNG,Any}, rng, values; threaded)

function IS._resample_next_on_device(
    device::MLDataDevices.ReactantDevice,
    rng::PureRNGs.AbstractPureRNG,
    result,
    count,
    method
)
    throw(
        ArgumentError(
            "Reactant resampling requires a compiled RNG carrier; use current_rng(sampler) or Reactant.to_rarray(rng)"
        )
    )
end

function IS._resample_next_on_device(
    device::MLDataDevices.ReactantDevice,
    rng,
    result,
    count,
    method
)
    IS._validate_transfer_rng(device, rng)
    return IS._resample_next(rng, result, count, method)
end

struct _NativePhase{D,F,E}
    device::D
    factor_execution::F
    execution::E
end

_copy_native_samples(samples::AbstractArray) = copy(samples)
_copy_native_samples(samples::NamedTuple) = map(_copy_native_samples, samples)

function (phase::_NativePhase)(rng, buffers, samples, logweights, target, proposal)
    rng = IS._fill_random_buffers!(rng, buffers)
    record = buffers.failure_scratch.record
    fill!(record.storage, zero(UInt64))
    base, transform = IS._native_fused_components(proposal)
    IS._launch_native_batch!(samples, logweights, record, buffers, target, base,
        transform, phase.execution, phase.device, phase.factor_execution)
    return (_copy_native_samples(samples), copy(logweights)), rng
end

function _map_samples!(samples, layout, record)
    count = IS._sample_count(samples)
    mapped = IS._allocate_flat_samples(samples, eltype(samples), layout, count)
    fill!(record.storage, zero(UInt64))
    IS._map_named_result_kernel!(KA.get_backend(samples))(
        IS._result_blocks(mapped), samples, layout, record.storage; ndrange=count)
    return mapped
end

_prepare_result_map(target, samples, record) = nothing
_prepare_result_map(target::IS._NamedPreparedTarget, samples, record) =
    Reactant.@compile client=Reactant.XLA.client(record.storage) _map_samples!(samples, target.layout, record)

struct _CompiledNativeExecution{F,A,M}
    compiled::F
    arguments::A
    map_result::M
end

function (phase::_NativePhase)(rng, buffers, samples, logweights, proposal_ids, target,
    state::IS._PreparedStaticMIS)
    rng = IS._fill_random_buffers!(rng, buffers)
    fill!(buffers.failure_scratch.record.storage, zero(UInt64))
    IS._launch_packed_static_mis!(samples, logweights, proposal_ids, buffers, target,
        state, phase.execution, phase.device, phase.factor_execution)
    return (copy(samples), copy(logweights), copy(proposal_ids)), rng
end

function _prepare_execution(sampler, state::IS._PreparedStaticMIS)
    buffers, count = sampler.random_buffers, sampler.algorithm.nsamples
    samples = IS._allocate_packed_static_mis_samples(buffers.normal, state.bank, count)
    binding = IS._native_binding_sample(samples)
    bound = IS._bind_resolved_target(sampler.target, binding)
    L = IS._resolve_packed_static_mis_logweight_type(bound, state.bank, typeof(binding))
    logweights = similar(buffers.normal, L, count)
    proposal_ids = similar(buffers.assignments, Int, count)
    target, _ = IS._native_target_evaluator(KA.get_backend(buffers.normal), bound, L,
        buffers.failure_scratch.target_failures)
    phase = _NativePhase(sampler.device, sampler.factor_execution, IS._ThreadedCPUExecution())
    arguments = (buffers, samples, logweights, proposal_ids, target, state)
    rng = sampler.rng
    compiled = Reactant.@compile client=Reactant.XLA.client(buffers.normal) phase(rng, arguments...)
    mapper = _prepare_result_map(sampler.target, samples, buffers.failure_scratch.record)
    return _CompiledNativeExecution(compiled, arguments, mapper)
end

struct _MomentDrawPhase{A,D,F,E}
    algorithm::A
    device::D
    factor_execution::F
    execution::E
    schedule::Vector{Int}
    offsets::Vector{Int}
    round::Int
end

function (phase::_MomentDrawPhase)(rng, workspace, history, logcounts, buffers, target, round_ids)
    rng = IS._fill_moment_random_buffers!(rng, buffers, history)
    state = (; workspace, history, logcounts, phase.schedule, phase.offsets)
    IS._launch_moment_round!(phase.algorithm, state, buffers, _separate_target(target), round_ids,
        phase.round, phase.execution, phase.device, phase.factor_execution)
    return rng
end

struct _MomentFitPhase
    round::Int
    count::Int
end

function (phase::_MomentFitPhase)(workspace, history::IS._FactorProposalHistory, failure_storage)
    count = phase.count
    moments = _normalize_weights!(workspace.normalized_weights, workspace.logweights, count)
    IS._weighted_moments!(workspace.candidate_mean, workspace.covariance,
        view(workspace.centered_scaled, :, 1:count), view(workspace.samples, :, 1:count),
        view(workspace.normalized_weights, 1:count))
    backend = KA.get_backend(workspace.covariance)
    IS._add_gaussian_factor_ridge_kernel!(backend)(workspace.covariance, history.factors,
        history.family, phase.round; ndrange=1)
    copyto!(workspace.candidate_scale, workspace.covariance)
    success = _factor!(workspace.candidate_scale)
    IS._scale_covariance_factor!(workspace.candidate_scale, history.family)
    IS._finish_gaussian_factor_candidate_kernel!(backend)(workspace.candidate_mean,
        workspace.candidate_scale, workspace.candidate_lognormalizer, failure_storage,
        count + 1, history.family, workspace.candidate_exponent_bound;
        ndrange=length(workspace.candidate_scale))
    return moments, success
end

function (phase::_MomentFitPhase)(workspace, history::IS._ScalarProposalHistory, failure_storage)
    count = phase.count
    moments = _normalize_weights!(workspace.normalized_weights, workspace.logweights, count)
    samples = view(workspace.samples, 1:count)
    weights = view(workspace.normalized_weights, 1:count)
    workspace.candidate_mean .= sum(samples .* weights; dims=1)
    # Avoid a reshape of a view: Julia's broadcast alias check requests host pointers.
    centered = similar(samples)
    centered .= (samples .- workspace.candidate_mean) .* sqrt.(weights)
    workspace.covariance .= sum(abs2, centered; dims=1)
    backend = KA.get_backend(workspace.covariance)
    IS._add_gaussian_scalar_ridge_kernel!(backend)(workspace.covariance, history.scales,
        history.family, phase.round; ndrange=1)
    IS._finish_gaussian_scalar_candidate_kernel!(backend)(workspace.candidate_scale,
        workspace.candidate_lognormalizer, workspace.covariance, failure_storage,
        count + 1, history.family; ndrange=1)
    return moments, true
end

function (phase::_MomentFitPhase)(workspace, history::IS._IndependentProposalHistory, failure_storage)
    moments = _normalize_weights!(workspace.normalized_weights, workspace.logweights, phase.count)
    IS._fit_independent_parameters!(workspace, history, phase.count, failure_storage)
    return moments, true
end

# Array-level lgamma avoids Reactant's scalar CUDA-intrinsic lowering in KA kernels.
function IS._independent_logconstants!(constants::AbstractArray{<:Reactant.TracedRNumber},
    shapes, scales, ::IS._GammaFamily)
    constants .= ifelse.(shapes .< 32,
        -IS.SpecialFunctions.loggamma.(shapes) .- shapes .* log.(scales),
        IS._gamma_center_logconstant.(shapes, scales))
    return constants
end

struct _PublishMomentPhase
    slot::Int
end

struct _NPMCFitPhase{A,D}
    algorithm::A
    device::D
    offsets::Vector{Int}
    fit::_MomentFitPhase
end

function (phase::_NPMCFitPhase)(workspace, history, failure_storage)
    state = (; workspace, phase.offsets)
    fit_workspace = IS._moment_adaptation_workspace!(phase.algorithm, phase.device,
        state, phase.fit.round)
    moments, success = phase.fit(fit_workspace, history, failure_storage)
    indices = IS._moment_summary_indices(phase.algorithm, state, phase.fit.round)
    raw = _logweight_moments(view(workspace.logweights, indices))
    return moments, success, raw
end

function IS._sort_clipping_weights!(::MLDataDevices.ReactantDevice,
    scratch::AbstractArray{<:Reactant.TracedRNumber}, threshold_index)
    copyto!(scratch, sort(Reactant.TracedUtils.materialize_traced_array(scratch)))
    return nothing
end
(phase::_PublishMomentPhase)(history, workspace) =
    _store_moment_candidate!(history, phase.slot, workspace)

# Reactant cannot trace `fill!` on the one-element view of a two-round history,
# nor an empty slice of a one-round history.
function _reset_moment_history!(history::IS._FactorProposalHistory, record)
    if length(history.lognormalizers) > 1
        history.means[:, 2:end] = zero.(history.means[:, 2:end])
        history.factors[:, :, 2:end] = zero.(history.factors[:, :, 2:end])
        history.lognormalizers[2:end] = zero.(history.lognormalizers[2:end])
        isnothing(history.exponent_bounds) ||
            (history.exponent_bounds[2:end] = zero.(history.exponent_bounds[2:end]))
    end
    fill!(record.storage, zero(UInt64))
    return nothing
end

function _reset_moment_history!(history::IS._ScalarProposalHistory, record)
    history.means[2:end] = zero.(history.means[2:end])
    history.scales[2:end] = zero.(history.scales[2:end])
    history.lognormalizers[2:end] = zero.(history.lognormalizers[2:end])
    fill!(record.storage, zero(UInt64))
    return nothing
end

function _reset_moment_history!(history::IS._IndependentProposalHistory, record)
    if length(history.lognormalizers) > 1
        history.first_parameters[:, 2:end] = zero.(history.first_parameters[:, 2:end])
        history.second_parameters[:, 2:end] = zero.(history.second_parameters[:, 2:end])
        history.logconstants[:, 2:end] = zero.(history.logconstants[:, 2:end])
        history.lognormalizers[2:end] = zero.(history.lognormalizers[2:end])
    end
    fill!(record.storage, zero(UInt64))
    return nothing
end

struct _CompiledMomentExecution{D,F,P,R,M}
    draws::D
    fits::F
    publish::P
    reset::R
    map_result::M
end

function _prepare_execution(sampler, state::IS._PreparedMomentSampler)
    sampler.algorithm isa Union{IS.AMIS,IS.NPMC} || return nothing
    workspace, history, buffers = state.workspace, state.history, sampler.random_buffers
    binding = IS._native_binding_sample(workspace.samples)
    bound = IS._bind_resolved_target(sampler.target, binding)
    target, _ = IS._native_target_evaluator(KA.get_backend(workspace.samples), bound,
        eltype(workspace.logweights), buffers.failure_scratch.target_failures)
    round_ids = similar(workspace.logweights, Int, length(workspace.logweights))
    rng = sampler.rng
    schedule, offsets = collect(state.schedule), collect(state.offsets)
    client = Reactant.XLA.client(workspace.samples)
    draws = map(eachindex(state.schedule)) do round
        phase = _MomentDrawPhase(sampler.algorithm, sampler.device,
            sampler.factor_execution, IS._ThreadedCPUExecution(),
            schedule, offsets, round)
        logcounts = state.logcounts
        Reactant.@compile client=client phase(rng, workspace, history, logcounts, buffers, target, round_ids)
    end
    fits = map(eachindex(state.schedule)) do round
        phase = _MomentFitPhase(round, IS._moment_adaptation_count(sampler.algorithm, state, round))
        if sampler.algorithm isa IS.NPMC
            phase = _NPMCFitPhase(sampler.algorithm, sampler.device, offsets, phase)
        end
        storage = buffers.failure_scratch.record.storage
        Reactant.@compile client=client phase(workspace, history, storage)
    end
    publish = map(eachindex(state.schedule)) do slot
        phase = _PublishMomentPhase(slot)
        Reactant.@compile client=client phase(history, workspace)
    end
    record = buffers.failure_scratch.record
    reset = Reactant.@compile client=client _reset_moment_history!(history, record)
    mapper = _prepare_result_map(sampler.target, workspace.samples, record)
    return _CompiledMomentExecution(draws, fits, publish, reset, mapper)
end

IS._execute_prepared_sampler!(sampler, ::_CompiledMomentExecution, threaded) =
    IS._importance_sample_cpu!(sampler, threaded)
IS._reset_prepared_moment_history!(plan::_CompiledMomentExecution, history, rounds, record) =
    plan.reset(history, record)
IS._store_prepared_moment_candidate!(plan::_CompiledMomentExecution, history, slot, workspace) =
    plan.publish[slot](history, workspace)
IS._prepared_moment_adaptation_workspace!(::_CompiledMomentExecution, algorithm, device, state, round) =
    state.workspace
IS._prepared_moment_summary(::_CompiledMomentExecution, algorithm, state, round, fit, transfers) =
    fit.raw_summary
IS._draw_moment_round!(sampler, plan::_CompiledMomentExecution, state, round_ids, target, round, execution) = (
    sampler.rng = plan.draws[round](sampler.rng, state.workspace, state.history, state.logcounts,
        sampler.random_buffers, target, round_ids
    )
)

function IS._fit_prepared_moment_proposal!(plan::_CompiledMomentExecution, device,
    workspace, history, round, count, transfers, failure_storage)
    phase = :moment
    try
        fit = plan.fits[round](workspace, history, failure_storage)
        values = Array(fit[1])
        IS._record_reported_transfer!(transfers, 1, sizeof(values), Val(:logweight_moments))
        isfinite(values[2]) && values[2] > 0 || throw(IS.AllZeroWeightsError())
        phase = IS._moment_fit_phase(history)
        Bool(fit[2]) || throw(LA.PosDefException(0))
        summary = IS._logweight_summary(values..., count)
        if length(fit) == 3
            raw = Array(fit[3])
            IS._record_reported_transfer!(transfers, 1, sizeof(raw), Val(:logweight_moments))
            return (; summary..., raw_summary=IS._logweight_summary(raw..., count))
        end
        return summary
    catch cause
        IS._throw_moment_stage(phase, cause)
    end
end

function IS._moment_result_samples(plan::_CompiledMomentExecution, sampler, workspace, transfers)
    isnothing(plan.map_result) && return copy(workspace.samples)
    return _map_compiled_result(plan.map_result, sampler, workspace.samples, transfers)
end

function IS._prepare_backend_execution(sampler::IS._PreparedImportanceSampler{R,B,T,A,M,D}
) where {R,B,T,A,M,D<:MLDataDevices.ReactantDevice}
    plan = _prepare_execution(sampler, sampler.method_state)
    return IS._PreparedImportanceSampler(sampler.rng, sampler.random_buffers,
        sampler.target, sampler.algorithm, sampler.method_state, sampler.device,
        sampler.factor_execution, sampler.threaded, sampler.running, sampler.executed, plan)
end

_prepare_execution(sampler, state) = nothing

function _prepare_execution(sampler, ::IS._SingleProposalMethodState)
    buffers = sampler.random_buffers
    proposal = sampler.algorithm.proposal
    samples = IS._allocate_native_samples(buffers.normal, proposal, sampler.algorithm.nsamples)
    binding = IS._native_binding_sample(samples)
    bound = IS._bind_resolved_target(sampler.target, binding)
    base, _ = IS._native_fused_components(proposal)
    L = IS._resolve_native_logweight_type(bound, base, typeof(binding))
    logweights = similar(buffers.normal, L, sampler.algorithm.nsamples)
    target, _ = IS._native_target_evaluator(KA.get_backend(buffers.normal), bound, L,
        buffers.failure_scratch.target_failures)
    execution = IS._sampling_execution(proposal, true)
    phase = _NativePhase(sampler.device, sampler.factor_execution, execution)
    arguments = (buffers, samples, logweights, target, proposal)
    rng = sampler.rng
    compiled = Reactant.@compile client=Reactant.XLA.client(samples) phase(rng, arguments...)
    mapper = _prepare_result_map(sampler.target, samples, buffers.failure_scratch.record)
    return _CompiledNativeExecution(compiled, arguments, mapper)
end

_map_compiled_result(::Nothing, sampler, samples, transfers) = samples
function _map_compiled_result(compiled, sampler, samples, transfers)
    record = sampler.random_buffers.failure_scratch.record
    mapped = compiled(samples, sampler.target.layout, record)
    snapshot = IS._device_failure_snapshot(record)
    failure = snapshot.failure
    iszero(failure.count) || IS._throw_named_result_failure(sampler.target.layout,
        failure.reason_bits, failure.first_logical_index, failure.first_block)
    IS._record_reported_transfer!(transfers, snapshot.transfers.count,
        snapshot.transfers.bytes, Val(:failure_snapshot))
    return mapped
end

function IS._execute_prepared_sampler!(sampler, plan::_CompiledNativeExecution, threaded)
    output, sampler.rng = plan.compiled(sampler.rng, plan.arguments...)
    return _compiled_native_result(sampler, plan, sampler.method_state, output, threaded)
end

function _compiled_native_transfers(sampler, transform)
    scratch = sampler.random_buffers.failure_scratch
    snapshot = IS._device_failure_snapshot(scratch.record)
    IS._throw_native_failures(snapshot.failure, snapshot.draw_failure,
        scratch.target_failures, transform)
    return IS._ResultTransferCounter(snapshot.transfers.count, snapshot.transfers.bytes)
end

function _compiled_native_result(sampler, plan, ::IS._SingleProposalMethodState, output, threaded)
    samples, logweights = output
    _, transform = IS._native_fused_components(sampler.algorithm.proposal)
    transfers = _compiled_native_transfers(sampler, transform)
    samples = _map_compiled_result(plan.map_result, sampler, samples, transfers)
    execution = IS._sampling_execution(sampler.algorithm.proposal, threaded)
    return IS._single_proposal_result(sampler, execution, samples, logweights, transfers)
end

function _compiled_native_result(sampler, plan, ::IS._PreparedStaticMIS, output, threaded)
    samples, logweights, proposal_ids = output
    transfers = _compiled_native_transfers(sampler, IS._NoSampleTransform())
    samples = _map_compiled_result(plan.map_result, sampler, samples, transfers)
    execution = threaded ? IS._ThreadedCPUExecution() : IS._SerialCPUExecution()
    return IS._packed_static_mis_result(sampler, samples, logweights, proposal_ids,
        execution, transfers)
end

struct _CompiledPopulationExecution{N,D,A,R,C,P,M}
    normals::N
    draws::D
    adaptation::A
    reset::R
    commit::C
    publish::P
    map_result::M
end

# Borrow the fixed host tuples. Reactant cannot trace the unsized Vararg field
# in _HostIntSequence. All numerical arrays, including swapped banks, stay live.
_population_trace_plan(plan) = IS._DeterministicAllocationPlan(plan.schedule.values,
    plan.counts, plan.assignments, plan.logcoefficients, plan.offsets.values)
_population_trace_state(state::IS._PreparedAPIS) = IS._PreparedAPIS(state.bank,
    state.run_bank, _population_trace_plan(state.plan), state.workspace)
_population_trace_state(state::IS._PreparedDMPMC) = IS._PreparedDMPMC(state.bank,
    state.run_bank, _population_trace_plan(state.plan), state.workspace)
_population_trace_state(state::IS._PreparedCAIS) = IS._PreparedCAIS(state.bank,
    state.run_bank, state.candidate_bank, _population_trace_plan(state.plan),
    state.covariance_ess_threshold, state.tempering_tolerance,
    state.tempering_max_iterations, state.workspace)

_reset_population_arrays!(state) = IS._reset_population_run!(state, state.run_bank, state.bank)
_population_draw_state(state) = state

_prepare_execution(sampler, state::Union{IS._PreparedAPIS,IS._PreparedDMPMC,IS._PreparedCAIS}) =
    _prepare_population_execution(sampler, state)

function _prepare_population_execution(sampler, state)
    state = _population_trace_state(state)
    buffers = sampler.random_buffers
    rng = sampler.rng
    bound = IS._bind_resolved_target(sampler.target, IS._population_binding_sample(state.bank))
    target, _ = IS._native_target_evaluator(KA.get_backend(buffers.normals), bound,
        eltype(state.workspace.round_logweights), buffers.failure_scratch.target_failures)
    execution = IS._population_execution(sampler, true, state)
    device, factor_execution = sampler.device, sampler.factor_execution
    client = Reactant.XLA.client(buffers.normals)
    normals = Reactant.@compile client=client IS._fill_population_normals!(rng, buffers)
    draws = map(eachindex(state.plan.schedule)) do round
        phase = (state, buffers, target) -> IS._launch_population_round!(
            _population_draw_state(state), buffers, target, round, execution, device, factor_execution)
        Reactant.@compile client=client phase(state, buffers, target)
    end
    adaptation = _prepare_population_adaptation(sampler, state)
    record = buffers.failure_scratch.record
    reset_phase = function (state, record)
        _reset_population_arrays!(state)
        fill!(record.storage, zero(UInt64))
        return nothing
    end
    reset = Reactant.@compile client=client reset_phase(state, record)
    commit_phase = state -> IS._commit_population_round!(state, state.run_bank)
    commit = Reactant.@compile client=client commit_phase(state)
    count = last(state.plan.offsets) - 1
    samples = IS._allocate_packed_static_mis_samples(buffers.normals, state.bank, count)
    logweights = similar(state.workspace.round_logweights, count)
    round_ids = similar(state.workspace.round_proposal_ids, count)
    proposal_ids = similar(round_ids)
    publish = map(eachindex(state.plan.schedule)) do round
        phase = (state, samples, logweights, round_ids, proposal_ids) ->
            IS._publish_population_round!(nothing, state, round, samples, logweights, round_ids, proposal_ids)
        Reactant.@compile client=client phase(state, samples, logweights, round_ids, proposal_ids)
    end
    mapper = _prepare_result_map(sampler.target, samples, record)
    return _CompiledPopulationExecution(normals, draws, adaptation, reset, commit, publish, mapper)
end

function _prepare_population_adaptation(sampler, state::IS._PreparedAPIS)
    client = Reactant.XLA.client(sampler.random_buffers.normals)
    return map(eachindex(state.plan.schedule)) do round
        phase = function (state)
            workspace = state.workspace
            views = IS._population_round_views(state, round)
            IS._launch_local_weighted_means!(workspace.candidate_locations,
                workspace.proposal_maxima, views.samples, views.scaled_local_weights,
                views.logtargets, views.generating_logdensities, views.assignments,
                state.plan.counts, round, IS._ThreadedCPUExecution())
            return _allfinite(workspace.proposal_maxima), _logweight_moments(views.logweights)
        end
        Reactant.@compile client=client phase(state)
    end
end

function _prepare_population_adaptation(sampler, state::IS._PreparedDMPMC)
    rng, buffers = sampler.rng, sampler.random_buffers
    client = Reactant.XLA.client(buffers.normals)
    return map(eachindex(state.plan.schedule)) do round
        if sampler.algorithm.resampling isa IS.LocalResampling
            phase = function (state, buffers)
                workspace = state.workspace
                views = IS._population_round_views(state, round)
                IS._launch_local_resample!(
                    views.cdf, buffers.resampling_uniforms,
                    workspace.ancestors, views.samples, workspace.candidate_locations,
                    views.logweights, views.assignments, workspace.resampling_maxima,
                    state.plan.counts, round, IS._ThreadedCPUExecution())
                return minimum(workspace.ancestors), _logweight_moments(views.logweights)
            end
            return (; local_phase=Reactant.@compile client=client phase(state, buffers))
        end
        cdf_phase = function (state)
            views = IS._population_round_views(state, round)
            moments = _normalize_weights!(views.cdf, views.logweights, views.round_size;
                normalize=false)
            views.cdf .= cumsum(views.cdf)
            views.cdf ./= views.cdf[end:end]
            return moments
        end
        select_phase = function (rng, state, buffers)
            views = IS._population_round_views(state, round)
            rng = IS._rand_uniform!(rng, buffers.resampling_uniforms)
            IS._resample_and_gather!(views.cdf, views.logweights, buffers.resampling_uniforms,
                state.workspace.ancestors, views.samples, state.workspace.candidate_locations,
                IS._ThreadedCPUExecution())
            return rng
        end
        cdf = Reactant.@compile client=client cdf_phase(state)
        select = Reactant.@compile client=client select_phase(rng, state, buffers)
        return (; cdf, select)
    end
end

function _prepare_population_adaptation(sampler, state::IS._PreparedCAIS)
    execution = IS._population_execution(sampler, true, state)
    client = Reactant.XLA.client(sampler.random_buffers.normals)
    return map(eachindex(state.plan.schedule)) do round
        weight_phase = state -> IS._cais_weights!(state, IS._population_round_views(state, round), round, execution)
        fit_phase = state -> IS._cais_fit!(state, IS._population_round_views(state, round), round, execution)
        install_phase = function (state)
            IS._cais_install!(state, round, execution)
            return _logweight_moments(IS._population_round_views(state, round).logweights)
        end
        weights = Reactant.@compile client=client weight_phase(state)
        fit = Reactant.@compile client=client fit_phase(state)
        install = Reactant.@compile client=client install_phase(state)
        return (; weights, fit, install)
    end
end

IS._execute_prepared_sampler!(sampler, ::_CompiledPopulationExecution, threaded) =
    IS._importance_sample_cpu!(sampler, threaded)
IS._reset_prepared_population!(plan::_CompiledPopulationExecution, state, record) =
    plan.reset(_population_trace_state(state), record)

IS._fill_population_normals!(plan::_CompiledPopulationExecution, sampler::IS._PreparedImportanceSampler) =
    (sampler.rng = plan.normals(sampler.rng, sampler.random_buffers))
IS._draw_population_round!(plan::_CompiledPopulationExecution, sampler, state, target, round, execution) =
    plan.draws[round](_population_trace_state(state), sampler.random_buffers, target)
IS._publish_population_round!(plan::_CompiledPopulationExecution, state, round, samples, weights, rounds, proposals) =
    plan.publish[round](_population_trace_state(state), samples, weights, rounds, proposals)
IS._commit_prepared_population!(plan::_CompiledPopulationExecution, state) =
    plan.commit(_population_trace_state(state))
IS._population_result_samples(plan::_CompiledPopulationExecution, sampler, samples, transfers) =
    _map_compiled_result(plan.map_result, sampler, samples, transfers)

function IS._advance_prepared_population!(plan::_CompiledPopulationExecution, sampler,
    state::IS._PreparedAPIS, views, round, execution, transfers)
    valid, moments = plan.adaptation[round](_population_trace_state(state))
    success = Bool(valid)
    IS._record_device_scalar_transfer!(transfers, state.workspace.proposal_maxima,
        Bool, Val(:local_mean_validity))
    success || throw(IS.AllZeroWeightsError())
    values = Array(moments)
    IS._record_reported_transfer!(transfers, 1, sizeof(values), Val(:logweight_moments))
    return IS._logweight_summary(values..., views.round_size)
end

function IS._advance_prepared_population!(plan::_CompiledPopulationExecution, sampler,
    state::IS._PreparedDMPMC, views, round, execution, transfers)
    state = _population_trace_state(state)
    phase = plan.adaptation[round]
    if sampler.algorithm.resampling isa IS.LocalResampling
        sampler.rng =
            IS._rand_uniform!(sampler.rng, sampler.random_buffers.resampling_uniforms)
        ancestor, moments = phase.local_phase(state, sampler.random_buffers)
        valid = Int(ancestor) > 0
        IS._record_device_scalar_transfer!(transfers, state.workspace.ancestors,
            Int, Val(:local_resampling_validity))
        valid || throw(IS.AllZeroWeightsError())
        values = Array(moments)
    else
        values = Array(phase.cdf(state))
        IS._record_reported_transfer!(transfers, 1, sizeof(values), Val(:cdf_sum))
        isfinite(values[2]) && values[2] > 0 || throw(IS.AllZeroWeightsError())
        sampler.rng = phase.select(sampler.rng, state, sampler.random_buffers)
        return IS._logweight_summary(values..., views.round_size)
    end
    IS._record_reported_transfer!(transfers, 1, sizeof(values), Val(:logweight_moments))
    return IS._logweight_summary(values..., views.round_size)
end

function IS._advance_prepared_population!(plan::_CompiledPopulationExecution, sampler,
    state::IS._PreparedCAIS, views, round, execution, transfers)
    state = _population_trace_state(state)
    phase = plan.adaptation[round]
    phase.weights(state)
    IS._throw_cais_weight_failure(sampler.device, state.workspace.factor_status, transfers)
    phase.fit(state)
    IS._throw_cais_factor_failure(sampler.device, state.workspace.factor_info, transfers)
    values = Array(phase.install(state))
    IS._record_reported_transfer!(transfers, 1, sizeof(values), Val(:logweight_moments))
    return IS._logweight_summary(values..., views.round_size)
end

include("reactant_lais.jl")
include("reactant_gramis.jl")

# Reactant arguments remain managed arrays, not isbits kernel pointers. Its live
# preflight compiles the target and gradients before sampling consumes the RNG.
function IS._preflight_kernel_argument(device::MLDataDevices.ReactantDevice, kernel, argument)
    try
        KA.argconvert(kernel, argument)
    catch
        throw(IS.SamplerDeviceError(device, :kernel_argument_unsupported))
    end
    return nothing
end

function IS._preflight_accelerator_method(
    device::MLDataDevices.ReactantDevice, target, algorithm::IS.FirstOrderGRAMIS,
    state::IS._PreparedFirstOrderGRAMIS, buffers, factor_execution,
)
    # Unraised cooperative kernels retain NVVM barriers on Reactant's CPU
    # backend. Reject before its compiler aborts the Julia process.
    Reactant.XLA.device_kind(Reactant.XLA.device(state.committed.locations)) == "cpu" &&
        throw(IS.SamplerDeviceError(device, :reactant_cpu_cooperative_kernels))
    return invoke(IS._preflight_accelerator_method,
        Tuple{Any,Any,IS.FirstOrderGRAMIS,IS._PreparedFirstOrderGRAMIS,Any,Any},
        device, target, algorithm, state, buffers, factor_execution)
end

# APIS, CAIS and locally resampled DM-PMC also launch workgroup kernels whose
# barriers abort Reactant's CPU compiler. Globally resampled DM-PMC compiles.
function _reject_reactant_cpu(device, buffers)
    Reactant.XLA.device_kind(Reactant.XLA.device(buffers.normals)) == "cpu" &&
        throw(IS.SamplerDeviceError(device, :reactant_cpu_cooperative_kernels))
    return nothing
end

function IS._preflight_accelerator_method(
    device::MLDataDevices.ReactantDevice, target, algorithm::IS.APIS,
    state::IS._PreparedAPIS, buffers::IS._PopulationNormalBuffers, factor_execution,
)
    _reject_reactant_cpu(device, buffers)
    return invoke(IS._preflight_accelerator_method,
        Tuple{Any,Any,IS.APIS,IS._PreparedAPIS,IS._PopulationNormalBuffers,Any},
        device, target, algorithm, state, buffers, factor_execution)
end

function IS._preflight_accelerator_method(
    device::MLDataDevices.ReactantDevice, target, algorithm::IS.CAIS,
    state::IS._PreparedCAIS, buffers::IS._PopulationNormalBuffers, factor_execution,
)
    _reject_reactant_cpu(device, buffers)
    return invoke(IS._preflight_accelerator_method,
        Tuple{Any,Any,IS.CAIS,IS._PreparedCAIS,IS._PopulationNormalBuffers,Any},
        device, target, algorithm, state, buffers, factor_execution)
end

function IS._preflight_accelerator_method(
    device::MLDataDevices.ReactantDevice, target,
    algorithm::IS.DeterministicMixturePMC{<:IS.ProposalBank,<:Any,IS.LocalResampling},
    state::IS._PreparedDMPMC, buffers::IS._DMPMCRandomBuffers, factor_execution,
)
    _reject_reactant_cpu(device, buffers)
    return invoke(IS._preflight_accelerator_method,
        Tuple{Any,Any,IS.DeterministicMixturePMC,IS._PreparedDMPMC,IS._DMPMCRandomBuffers,Any},
        device, target, algorithm, state, buffers, factor_execution)
end

# Preserve the existing solve workspace. Reactant cannot trace the coalesced
# PermutedDimsArray/reshape view used by the native CUDA kernel.
IS._fused_mis_solve_scratch(scratch::Reactant.AnyConcreteRArray, backend) = scratch
IS._fused_mis_solve_scratch(scratch::AbstractArray{<:Reactant.TracedRNumber}, backend) =
    Reactant.TracedUtils.materialize_traced_array(scratch)

function IS._factor_batch_solve!(
    scratch::AbstractMatrix{<:Reactant.TracedRNumber}, samples, source, slot,
)
    factor = Reactant.TracedUtils.materialize_traced_array(IS._factor_batch_factor(source, slot))
    centered = samples .- reshape(IS._factor_batch_location(source, slot), :, 1)
    scratch .= Reactant.Ops.triangular_solve(factor, centered;
        left_side=true, lower=true, unit_diagonal=false, transpose_a='N')
    # Pass a traced tensor, not a Julia reshape wrapper, to the following kernel.
    return Reactant.TracedUtils.materialize_traced_array(scratch)
end

function _pooled_covariance!(covariance, factors)
    packed = reshape(factors, size(factors, 1), :)
    covariance .= (packed * transpose(packed)) / eltype(factors)(size(factors, 3))
    return nothing
end

function IS._pooled_covariance!(covariance::Reactant.AnyConcreteRArray, factors, ::IS._KernelExecution)
    Reactant.@jit client=_call_client(covariance) _pooled_covariance!(covariance, factors)
    return nothing
end

IS._pooled_covariance!(covariance::AbstractArray{<:Reactant.TracedRNumber}, factors, ::IS._KernelExecution) =
    _pooled_covariance!(covariance, factors)

function _factor!(factor)
    decomposition = LA.cholesky(LA.Hermitian(factor, :L); check=false)
    factor .= transpose(decomposition.factors)
    return decomposition.info
end

function IS._factor_pooled_covariance!(factor::Reactant.AnyConcreteRArray)
    success = Reactant.@jit client=_call_client(factor) _factor!(factor)
    Bool(success) || throw(IS._FirstOrderGRAMISRepulsionError(
        0, :pooled_factorization_failed, :nonfinite_result))
    return nothing
end

function IS._gaussian_potrf!(::MLDataDevices.ReactantDevice, factor::Reactant.AnyConcreteRArray)
    success = Reactant.@jit client=_call_client(factor) _factor!(factor)
    Bool(success) || throw(LA.PosDefException(0))
    return factor
end

function IS._preflight_gaussian_factorization!(
    device::MLDataDevices.ReactantDevice, state::IS._PreparedMomentSampler{S,O,L,H},
) where {S,O,L,H<:IS._FactorProposalHistory}
    # Moment-sampler support stays GPU-only while cooperative CPU lowering
    # remains unvalidated. Check before launching any adaptive kernels.
    Reactant.XLA.device_kind(Reactant.XLA.device(state.workspace.covariance)) == "cpu" &&
        throw(IS.SamplerDeviceError(device, :reactant_cpu_cooperative_kernels))
    IS._gaussian_potrf!(device, state.workspace.covariance)
    return nothing
end

function IS._weighted_moments!(mean::Reactant.AnyConcreteRArray, covariance, centered, samples, weights)
    Reactant.@jit client=_call_client(mean) IS._weighted_moments!(mean, covariance, centered, samples, weights)
    return nothing
end

function _store_moment_candidate!(history::IS._FactorProposalHistory, slot, workspace)
    history.means[:, slot] = workspace.candidate_mean
    history.factors[:, :, slot] = workspace.candidate_scale
    history.lognormalizers[slot:slot] = workspace.candidate_lognormalizer
    isnothing(history.exponent_bounds) ||
        (history.exponent_bounds[slot:slot] = workspace.candidate_exponent_bound)
    return nothing
end

function _store_moment_candidate!(history::IS._ScalarProposalHistory, slot, workspace)
    history.means[slot:slot] = workspace.candidate_mean
    history.scales[slot:slot] = workspace.candidate_scale
    history.lognormalizers[slot:slot] = workspace.candidate_lognormalizer
    return nothing
end

function _store_moment_candidate!(history::IS._IndependentProposalHistory, slot, workspace)
    history.first_parameters[:, slot] = workspace.candidate_first
    history.second_parameters[:, slot] = workspace.candidate_second
    history.logconstants[:, slot] = workspace.candidate_logconstants
    history.lognormalizers[slot:slot] = workspace.candidate_lognormalizer
    return nothing
end

function IS._store_moment_candidate!(
    history::IS._FactorProposalHistory{<:Reactant.AnyConcreteRArray}, slot, workspace::IS._MomentWorkspace,
)
    Reactant.@jit client=Reactant.XLA.client(workspace.candidate_mean) _store_moment_candidate!(history, slot, workspace)
    return nothing
end

function IS._clip_logweights!(clipped::_ReactantStorage, raw, threshold)
    Reactant.@jit client=_call_client(clipped) IS._clip_logweights!(clipped, raw, threshold)
    return clipped
end

function _normalize_weights!(weights, logweights, count; normalize=true)
    active = view(weights, 1:count)
    values = view(logweights, 1:count)
    maximum_logweight = maximum(values; dims=1)
    active .= exp.(values .- ifelse.(isfinite.(maximum_logweight), maximum_logweight, zero(eltype(values))))
    total = sum(active; dims=1)
    squared_sum = sum(abs2, active; dims=1)
    normalize && (active ./= ifelse.(iszero.(total), one(eltype(active)), total))
    return vcat(maximum_logweight, total, squared_sum)
end

function IS._normalize_moment_weights!(
    weights::_ReactantStorage, logweights, count, transfers::IS._ResultTransferCounter,
)
    moments = Array(Reactant.@jit client=_call_client(weights) _normalize_weights!(weights, logweights, count))
    IS._record_reported_transfer!(transfers, 1, sizeof(moments), Val(:logweight_moments))
    isfinite(moments[2]) && moments[2] > 0 || throw(IS.AllZeroWeightsError())
    return IS._logweight_summary(moments..., count)
end

function _whiten_means!(output, factor, means)
    output .= Reactant.Ops.triangular_solve(factor, means;
        left_side=true, lower=true, unit_diagonal=false, transpose_a='N')
    return nothing
end

function IS._whiten_means!(output::Reactant.AnyConcreteRArray, factor, means)
    Reactant.@jit client=_call_client(output) _whiten_means!(output, factor, means)
    return nothing
end

IS._whiten_means!(output::AbstractArray{<:Reactant.TracedRNumber}, factor, means) =
    _whiten_means!(output, factor, means)

function _logweight_moments(values)
    maximum_logweight = maximum(values; dims=1)
    shifted = exp.(values .- ifelse.(isfinite.(maximum_logweight), maximum_logweight, zero(eltype(values))))
    return vcat(maximum_logweight, sum(shifted; dims=1), sum(abs2, shifted; dims=1))
end

function IS._logweight_moments(values::_ReactantStorage)
    return Tuple(Array(Reactant.@jit client=_call_client(values) _logweight_moments(values)))
end

function _logsumexp(values)
    A = promote_type(eltype(values), Float32)
    maximum_logweight = A.(maximum(values; dims=1))
    shifted = exp.(A.(values) .- ifelse.(isfinite.(maximum_logweight), maximum_logweight, zero(A)))
    return vcat(maximum_logweight, sum(shifted; dims=1))
end

function IS._logsumexp_accumulator(values::_ReactantStorage)
    # Reactant tensors cannot store the struct-valued reduction used by CUDA.
    return IS._LogSumExpAccumulator(Array(_retained_call(_logsumexp, values))...)
end

IS._normalized_weights(values::_ReactantStorage, maximum_logweight, total) =
    Reactant.@jit client=_call_client(values) IS._normalized_weights(values, maximum_logweight, total)
# Both scalars must remain live inputs to the retained executable.
IS._normalized_weights(values::Reactant.ConcreteRArray, maximum_logweight, total) =
    _retained_call(IS._normalized_weights, values,
        _transition_number(values, maximum_logweight), _transition_number(values, total))

function _materialized_statistic(f, arguments...)
    value = f(arguments...)
    # Keep vec/reduction wrappers inside the trace, not in eager result operations.
    return value isa AbstractArray ? Reactant.materialize_traced_array(value) : value
end

# XLA otherwise factors out the scale, recreating an overflowing difference or square.
IS._statistic_rounding_barrier(value::Reactant.TracedRNumber) =
    only(Reactant.Ops.optimization_barrier(value))

for N in (1, 2)
    @eval begin
        IS._weighted_sum(samples::Reactant.AnyConcreteRArray{T,$N}, weights) where {T} =
            _retained_call(IS._weighted_sum, samples, weights)
        IS._weighted_variance(samples::Reactant.AnyConcreteRArray{T,$N}, weights, center) where {T} =
            _retained_call(_materialized_statistic, IS._weighted_variance, samples, weights, center)
        IS._unweighted_mean(samples::Reactant.AnyConcreteRArray{T,$N}) where {T} =
            _retained_call(_materialized_statistic, IS._unweighted_mean, samples)
        IS._unweighted_variance(samples::Reactant.AnyConcreteRArray{T,$N}, corrected) where {T} =
            _retained_call(_materialized_statistic, IS._unweighted_variance, samples, corrected)
    end
end

IS._weighted_covariance(samples::Reactant.AnyConcreteRArray{T,2}, weights, center) where {T} =
    _retained_call(IS._weighted_covariance, samples, weights, center)
IS._unweighted_covariance(samples::Reactant.AnyConcreteRArray{T,2}, corrected) where {T} =
    _retained_call(IS._unweighted_covariance, samples, corrected)
IS._sqrt_summary(value::Union{Reactant.AnyConcreteRArray,Reactant.ConcreteRNumber}) =
    _retained_call(IS._sqrt_summary, value)
IS._evaluate_functional_values!(values::Reactant.AnyConcreteRArray, samples, f) =
    _retained_call(IS._evaluate_functional_values!, values, samples, f)

function _resampling_cdf!(cdf, logweights)
    moments = _normalize_weights!(cdf, logweights, length(cdf); normalize=false)
    cdf .= cumsum(cdf)
    cdf ./= cdf[end:end]
    return moments[1:2]
end

function IS._resampling_cdf!(
    cdf::_ReactantStorage, logweights, transfers::IS._ResultTransferCounter=IS._ResultTransferCounter(0, 0),
)
    summary = Array(_retained_call(_resampling_cdf!, cdf, logweights))
    # The CDF maximum and sum share one transfer, attributed to cdf_sum.
    IS._record_reported_transfer!(transfers, 1, sizeof(summary), Val(:cdf_sum))
    isfinite(summary[2]) && summary[2] > 0 || throw(IS.AllZeroWeightsError())
    return cdf
end

IS._resample_and_gather!(cdf::Reactant.ConcreteRArray, logweights, uniforms, ancestors, source, destination, execution) =
    _retained_call(IS._resample_and_gather!, cdf, logweights, uniforms, ancestors, source, destination, execution)

_allfinite(values) = all(isfinite.(values))
IS._local_means_valid(values::Reactant.AnyConcreteRArray) = Bool(Reactant.@jit client=_call_client(values) _allfinite(values))

function _gramis_diagnostics(status, steps, trials)
    return vcat(sum(Int.(status .== IS._POPULATION_ALL_ZERO_LOCAL); dims=1),
        sum(Int.(status .== IS._POPULATION_TEMPERING_FAILED); dims=1),
        sum(Int.(iszero.(steps)); dims=1), sum(trials; dims=1))
end

function IS._first_order_gramis_diagnostic_summary(
    ::MLDataDevices.ReactantDevice, status, steps, trials, transfers, ::IS._KernelExecution,
)
    values = Array(Reactant.@jit client=Reactant.XLA.client(status) _gramis_diagnostics(status, steps, trials))
    transfers.count += 1
    transfers.bytes += sizeof(values)
    return NamedTuple{(:all_zero, :tempering, :backtracking, :target_trials)}(Tuple(values))
end

struct _BoundReactantGradient{F,T,S,E} <: IS._BoundBatchGradient
    compiled::F
    target::T
    seeds::S
    status::E
end

IS._record_gradient_transfers!(transfers, ::_BoundReactantGradient) =
    IS._record_scalar_transfer!(transfers, Int32)

@inline _target_value(f, p, x) =
    (IS._batch_target_value(f, x, p), UInt16(0), 0)

@inline function _target_value(f::IS._NamedADLogDensity, p, x)
    logical, logjac, reason, block = IS._coordinate_to_logical(f.layout, x)
    iszero(reason) || return (oftype(logjac, NaN), reason, block)
    return (IS._batch_target_value(f.logdensity, logical, p) + logjac, reason, block)
end

KA.@kernel function _target_kernel!(values, locations, f, context, status)
    column = KA.@index(Global, Linear)
    value, reason, block = _target_value(f, context[], view(locations, :, column))
    values[column] = value
    status[1, column] = reason
    status[2, column] = block
end

function _batch_target!(values, locations, f, context, status)
    _target_kernel!(KA.get_backend(locations), 64)(
        values, locations, f, context, status; ndrange=size(locations, 2))
    return nothing
end

function _gradient!(values, gradients, locations, target, seeds, status)
    # Prepare inside the trace. DI preparations contain the concrete input types.
    DI.pullback!(_batch_target!, values, (gradients,), target.adtype, locations,
        (seeds,), DI.Constant(target.logdensity), DI.Constant(Ref(target.context)),
        DI.Constant(status))
    # Exceptions stay on the host. Return one scalar, not per-sample host reads.
    return minimum(ifelse.(iszero.(view(status, 1, :)),
        typemax(Int32), Int32.(1:size(locations, 2))))
end

function IS._prepare_accelerator_gradient(
    target::IS._PreparedLogTarget{F,P,A,Nothing},
    locations::Reactant.AnyConcreteRArray, values,
) where {F,P,A<:ADTypes.AutoEnzyme}
    ADTypes.mode(target.adtype) isa Union{ADTypes.ReverseMode,ADTypes.ForwardOrReverseMode} ||
        IS._reject_cpu_gradient_on_accelerator(view(locations, :, 1))
    seeds = similar(values)
    fill!(seeds, one(eltype(seeds)))
    status = similar(locations, Int32, 2, size(locations, 2))
    gradients = similar(locations)
    # Pre-AD tensor optimization can lose accumulated slice tangents. Keep
    # optimization after AD, where the numerical regression preserves them.
    options = Reactant.CompileOptions(; optimization_passes=:after_enzyme,
        raise=true, raise_first=true, excluded_passes=["transpose_is_reshape", "cse_reshape"])
    compiled = Reactant.@compile client=Reactant.XLA.client(locations) compile_options=options _gradient!(
        values, gradients, locations, target, seeds, status)
    return _BoundReactantGradient(compiled, target, seeds, status)
end

function IS._batch_value_and_gradient!(
    values, gradients, bound::_BoundReactantGradient, locations,
)
    first_failure = Int(bound.compiled(
        values, gradients, locations, bound.target, bound.seeds, bound.status))
    if first_failure != typemax(Int32)
        reason, block = Array(view(bound.status, :, first_failure))
        IS._throw_named_transform_failure(bound.target.logdensity.layout, UInt16(reason), Int(block))
    end
    return nothing
end

end
