struct _IndependentProposalHistory{F,P,N,L}
    family::F
    first_parameters::P
    second_parameters::P
    logconstants::P
    lognormalizers::N
    layout::L
end

struct _IndependentMomentWorkspace{S,T,N,W,P,C,V,M,F,L}
    samples::S
    logtargets::T
    lognumerators::N
    logweights::W
    normalized_weights::P
    centered_scaled::C
    variances::V
    candidate_mean::M
    candidate_first::F
    candidate_second::F
    candidate_logconstants::F
    candidate_lognormalizer::L
end

Adapt.@adapt_structure _IndependentProposalHistory
Adapt.@adapt_structure _IndependentMomentWorkspace

const _IndependentMomentProposal = Union{_GammaProposal,_LaplaceProposal}

function _validate_moment_proposal(proposal::_IndependentMomentProposal)
    first, second = _independent_parameters(proposal)
    supported = first isa _NativeGaussianFloat || first isa AbstractVector{<:_NativeGaussianFloat}
    supported && typeof(first) === typeof(second) || throw(
        ArgumentError(
            "adaptive independent proposals require matching native Float32 or Float64 parameters",
        ),
    )
    return nothing
end

function _moment_storage_prototype(proposal::_IndependentMomentProposal)
    first, _ = _independent_parameters(proposal)
    return first isa _NativeGaussianFloat ? Vector{typeof(first)}(undef, 0) : first
end

function _moment_binding_sample(proposal::_IndependentMomentProposal)
    first, _ = _independent_parameters(proposal)
    return first isa _NativeGaussianFloat ?
           _independent_binding_scalar(_independent_family(proposal), first) : view(first, :)
end

_independent_binding_scalar(::_GammaFamily, value) = one(value)
_independent_binding_scalar(::_LaplaceFamily, value) = value

function _allocate_moment_history(prototype, proposal::_IndependentMomentProposal, rounds)
    dimension = _proposal_dimension(proposal)
    T = _native_fused_float_type(proposal)
    first = similar(prototype, T, dimension, rounds)
    second = similar(first)
    constants = similar(first)
    normalizers = similar(prototype, T, rounds)
    a, b = _independent_parameters(proposal)
    if _native_scalar_layout(proposal)
        fill!(view(first, :, 1), a)
        fill!(view(second, :, 1), b)
    else
        copyto!(view(first, :, 1), a)
        copyto!(view(second, :, 1), b)
    end
    family = _independent_family(proposal)
    _independent_initial_constants!(view(constants, :, 1), proposal)
    fill!(view(normalizers, 1:1), proposal.lognormalizer)
    layout = _native_scalar_layout(proposal) ? _ScalarGaussianLayout() : _VectorGaussianLayout()
    return _IndependentProposalHistory(family, first, second, constants, normalizers, layout)
end

function _independent_initial_constants!(constants, proposal::_GammaProposal)
    if _native_scalar_layout(proposal)
        fill!(constants, proposal.logconstants)
    else
        copyto!(constants, proposal.logconstants)
    end
    return nothing
end

function _independent_initial_constants!(constants, proposal::_LaplaceProposal)
    T = eltype(constants)
    constants .= -log(T(2)) .- log.(proposal.scale)
    return nothing
end

function _allocate_moment_workspace(
    prototype,
    proposal::_IndependentMomentProposal,
    capacity,
    ::Type{L},
) where {L<:_NativeGaussianFloat}
    T = _native_fused_float_type(proposal)
    dimension = _proposal_dimension(proposal)
    samples = _allocate_native_samples(prototype, proposal, capacity)
    return _IndependentMomentWorkspace(
        samples,
        similar(prototype, L, capacity),
        similar(prototype, L, capacity),
        similar(prototype, L, capacity),
        similar(prototype, T, capacity),
        similar(samples),
        similar(prototype, T, dimension),
        similar(prototype, T, dimension),
        similar(prototype, T, dimension),
        similar(prototype, T, dimension),
        similar(prototype, T, dimension),
        similar(prototype, T, 1),
    )
end

_initialize_moment_workspace!(::_IndependentMomentWorkspace, ::_IndependentProposalHistory) =
    nothing
_moment_history_parameters(history::_IndependentProposalHistory) = history.first_parameters
_mis_dimension(history::_IndependentProposalHistory) = size(history.first_parameters, 1)
_moment_workspace_candidate(::_IndependentProposalHistory, accelerator) = true
_moment_fit_diagnostics(workspace::_IndependentMomentWorkspace) = workspace.variances
_moment_fit_phase(::_IndependentProposalHistory) = :moment
_moment_invalid_fit(::_IndependentProposalHistory) = throw(
    DomainError(
        nothing,
        "independent moment fit requires finite, representable parameters and positive variances",
    ),
)

function _moment_history_binding_sample(history::_IndependentProposalHistory)
    parameters = history.first_parameters
    return history.layout isa _ScalarGaussianLayout ?
           _independent_binding_scalar(history.family, zero(eltype(parameters))) :
           view(parameters, :, 1)
end

function _copy_moment_history(device, history::_IndependentProposalHistory)
    return _IndependentProposalHistory(
        history.family,
        _copy_to_device(device, history.first_parameters),
        _copy_to_device(device, history.second_parameters),
        _copy_to_device(device, history.logconstants),
        _copy_to_device(device, history.lognormalizers),
        history.layout,
    )
end

function _copy_moment_workspace(device, workspace::_IndependentMomentWorkspace)
    return _IndependentMomentWorkspace(
        _copy_to_device(device, workspace.samples),
        _copy_to_device(device, workspace.logtargets),
        _copy_to_device(device, workspace.lognumerators),
        _copy_to_device(device, workspace.logweights),
        _copy_to_device(device, workspace.normalized_weights),
        _copy_to_device(device, workspace.centered_scaled),
        _copy_to_device(device, workspace.variances),
        _copy_to_device(device, workspace.candidate_mean),
        _copy_to_device(device, workspace.candidate_first),
        _copy_to_device(device, workspace.candidate_second),
        _copy_to_device(device, workspace.candidate_logconstants),
        _copy_to_device(device, workspace.candidate_lognormalizer),
    )
end

function _moment_workspace_view(workspace::_IndependentMomentWorkspace, indices, clipped, scratch)
    return _IndependentMomentWorkspace(
        _sample_view(workspace.samples, indices),
        clipped,
        scratch,
        clipped,
        view(workspace.normalized_weights, indices),
        _sample_view(workspace.centered_scaled, indices),
        workspace.variances,
        workspace.candidate_mean,
        workspace.candidate_first,
        workspace.candidate_second,
        workspace.candidate_logconstants,
        workspace.candidate_lognormalizer,
    )
end

_preflight_gaussian_factorization!(
    device,
    state::_PreparedMomentSampler{S,O,L,H,W},
) where {S,O,L,H<:_IndependentProposalHistory,W} = nothing

function _moment_snapshot_parameters(
    state::_PreparedMomentSampler{S,O,L,H,W},
) where {S,O,L,H<:_IndependentProposalHistory,W}
    if state.committed_in_workspace
        return state.workspace.candidate_first, state.workspace.candidate_second
    end
    return view(state.history.first_parameters, :, 1), view(state.history.second_parameters, :, 1)
end

function _moment_proposal_snapshot(history::_IndependentProposalHistory, first, second)
    a, b = history.layout isa _ScalarGaussianLayout ? (first[1], second[1]) : (first, second)
    return _independent_proposal_snapshot(history.family, a, b)
end

_independent_proposal_snapshot(::_GammaFamily, first, second) = IndependentGamma(first, second)
_independent_proposal_snapshot(::_LaplaceFamily, first, second) = IndependentLaplace(first, second)

function _reset_moment_history!(history::_IndependentProposalHistory, rounds)
    slots = 2:rounds
    isempty(slots) && return nothing
    for parameters in (history.first_parameters, history.second_parameters, history.logconstants)
        fill!(view(parameters, :, slots), zero(eltype(parameters)))
    end
    fill!(view(history.lognormalizers, slots), zero(eltype(history.lognormalizers)))
    return nothing
end

function _store_moment_candidate!(
    history::_IndependentProposalHistory,
    slot,
    workspace::_IndependentMomentWorkspace,
)
    copyto!(view(history.first_parameters, :, slot), workspace.candidate_first)
    copyto!(view(history.second_parameters, :, slot), workspace.candidate_second)
    copyto!(view(history.logconstants, :, slot), workspace.candidate_logconstants)
    copyto!(view(history.lognormalizers, slot:slot), workspace.candidate_lognormalizer)
    return nothing
end

# This view is used only by the existing independent coordinate/density primitives.
@inline function _independent_density_bank(history::_IndependentProposalHistory)
    constants = _independent_density_constants(history.family, history)
    return _PackedIndependentBank(
        history.family,
        history.first_parameters,
        history.second_parameters,
        constants,
        nothing,
        nothing,
        nothing,
        history.layout,
    )
end

@inline _independent_density_constants(::_GammaFamily, history) = history.logconstants
@inline _independent_density_constants(::_LaplaceFamily, history) = history.lognormalizers

@inline _native_proposal_coordinate(
    history::_IndependentProposalHistory,
    inputs,
    offset,
    coordinate,
    slot,
) = _native_proposal_coordinate(
    _independent_density_bank(history),
    inputs,
    offset,
    coordinate,
    slot,
)

@inline _mis_proposal_logdensity(
    ::Type{T},
    history::_IndependentProposalHistory,
    sample,
    slot,
    solve_scratch,
    sample_index,
) where {T} = _mis_proposal_logdensity(T, _independent_density_bank(history), sample, slot)

function _allocate_random_buffers(
    ::MLDataDevices.AbstractDevice,
    proposal::_IndependentMomentProposal,
    state::_PreparedMomentSampler,
    sample_budget,
)
    prototype = state.workspace.samples
    history = state.history
    T = eltype(history.first_parameters)
    count = maximum(state.schedule)
    values = similar(prototype, T, _mis_dimension(history) * count)
    draws = _allocate_static_draw_buffers(prototype, _independent_density_bank(history), count)
    failures = _allocate_native_failure_scratch(values, count; capacity = sample_budget)
    return _RandomBuffers(similar(prototype, T, 0), values, failures, draws, nothing)
end

function _fill_moment_random_buffers!(
    rng,
    buffers,
    history::_IndependentProposalHistory{_GammaFamily};
    threaded = false,
)
    return _fill_radial_buffers!(rng, buffers.radial; threaded)
end

function _fill_moment_random_buffers!(
    rng,
    buffers,
    history::_IndependentProposalHistory{_LaplaceFamily};
    threaded = false,
)
    return _rand_uniform!(rng, buffers.normal; threaded)
end

@inline function _independent_moment_parameters(::_GammaFamily, mean::T, variance::T) where {T}
    scale = variance / mean
    shape = mean / scale
    valid =
        isfinite(mean) &&
        mean > zero(T) &&
        isfinite(variance) &&
        variance > zero(T) &&
        isfinite(shape) &&
        shape > zero(T) &&
        isfinite(scale) &&
        scale > zero(T)
    return shape, scale, valid
end

@inline function _independent_moment_parameters(::_LaplaceFamily, mean::T, variance::T) where {T}
    scale = sqrt(variance) / sqrt(T(2))
    valid =
        isfinite(mean) &&
        isfinite(variance) &&
        variance > zero(T) &&
        isfinite(scale) &&
        scale > zero(T)
    return mean, scale, valid
end

@kernel function _finish_independent_moment_parameters!(
    first,
    second,
    means,
    variances,
    family,
    failures,
    failure_index,
)
    coordinate = @index(Global, Linear)
    a, b, valid = _independent_moment_parameters(family, means[coordinate], variances[coordinate])
    first[coordinate] = valid ? a : oftype(a, NaN)
    second[coordinate] = valid ? b : oftype(b, NaN)
    valid || _record_native_failure!(failures, failure_index, 0, _MOMENT_FIT_INVALID)
end

_independent_logconstants!(constants, first, second, ::_GammaFamily) =
    (constants .= _gamma_logconstant.(first, second))
_independent_logconstants!(constants, first, second, ::_LaplaceFamily) =
    (constants .= -log(eltype(constants)(2)) .- log.(second))

@kernel function _validate_independent_normalizer!(normalizer, failures, failure_index)
    isfinite(normalizer[1]) ||
        _record_native_failure!(failures, failure_index, 0, _MOMENT_FIT_INVALID)
end

function _fit_independent_parameters!(
    workspace::_IndependentMomentWorkspace,
    history::_IndependentProposalHistory,
    count,
    failure_storage,
)
    dimension = _mis_dimension(history)
    samples = reshape(_sample_view(workspace.samples, 1:count), dimension, count)
    centered = reshape(_sample_view(workspace.centered_scaled, 1:count), dimension, count)
    weights = view(workspace.normalized_weights, 1:count)
    LinearAlgebra.mul!(workspace.candidate_mean, samples, weights)
    centered .= _weighted_squared_deviation.(
        samples,
        reshape(workspace.candidate_mean, :, 1),
        reshape(weights, 1, :),
    )
    sum!(reshape(workspace.variances, :, 1), centered)
    backend = KernelAbstractions.get_backend(workspace.samples)
    _finish_independent_moment_parameters!(backend)(
        workspace.candidate_first,
        workspace.candidate_second,
        workspace.candidate_mean,
        workspace.variances,
        history.family,
        failure_storage,
        count + 1;
        ndrange = dimension,
    )
    KernelAbstractions.synchronize(backend)
    _independent_logconstants!(
        workspace.candidate_logconstants,
        workspace.candidate_first,
        workspace.candidate_second,
        history.family,
    )
    sum!(workspace.candidate_lognormalizer, workspace.candidate_logconstants)
    _validate_independent_normalizer!(backend)(
        workspace.candidate_lognormalizer,
        failure_storage,
        count + 1;
        ndrange = 1,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _fit_moment_proposal!(
    workspace::_IndependentMomentWorkspace,
    history::_IndependentProposalHistory,
    previous_slot,
    count,
)
    try
        _normalize_moment_weights!(workspace.normalized_weights, workspace.logweights, count)
        failures = zeros(UInt64, 3)
        _fit_independent_parameters!(workspace, history, count, failures)
        iszero(failures[1]) || _moment_invalid_fit(history)
        return nothing
    catch cause
        _throw_moment_stage(:moment, cause)
    end
end

function _fit_moment_proposal!(
    device::MLDataDevices.AbstractAcceleratorDevice,
    workspace::_IndependentMomentWorkspace,
    history::_IndependentProposalHistory,
    previous_slot,
    count,
    transfers,
    failure_storage,
)
    try
        summary = _normalize_moment_weights!(
            workspace.normalized_weights,
            workspace.logweights,
            count,
            transfers,
        )
        _fit_independent_parameters!(workspace, history, count, failure_storage)
        return summary
    catch cause
        _throw_moment_stage(:moment, cause)
    end
end

function _preflight_moment_fit_kernels(
    device,
    state::_PreparedMomentSampler{S,O,L,H,W},
    buffers,
) where {S,O,L,H<:_IndependentProposalHistory,W}
    workspace = state.workspace
    backend = KernelAbstractions.get_backend(workspace.samples)
    finish = _finish_independent_moment_parameters!(backend)
    failure_index = state.offsets[end]
    for argument in (
        workspace.candidate_first,
        workspace.candidate_second,
        workspace.candidate_mean,
        workspace.variances,
        state.history.family,
        buffers.failure_scratch.record.storage,
        failure_index,
    )
        _preflight_kernel_argument(device, finish, argument)
    end
    validate = _validate_independent_normalizer!(backend)
    for argument in
        (workspace.candidate_lognormalizer, buffers.failure_scratch.record.storage, failure_index)
        _preflight_kernel_argument(device, validate, argument)
    end
    return nothing
end
