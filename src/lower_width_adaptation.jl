# Lower adaptation changes only future proposal densities. It never changes
# upper centres, upper dynamics, or already computed importance weights.
function _validate_lower_widths(bank, enabled)
    enabled || return nothing
    for proposal in bank.proposals
        if proposal isa _NativeRadialProposal
            _validate_moment_family(proposal.family)
        elseif !(proposal isa _LaplaceProposal)
            throw(
                ArgumentError(
                    "lower width adaptation requires Gaussian, Student-t or Laplace proposals",
                ),
            )
        end
    end
    return nothing
end

function _allocate_lower_widths(enabled, bank, plan, ::Type{L}) where {L}
    enabled || return nothing
    prototype = bank.locations
    T = promote_type(eltype(prototype), L)
    count = size(prototype, 2)
    return (
        weights=similar(prototype, T, maximum(plan.schedule)),
        ess=similar(prototype, T, count),
        powers=similar(prototype, T, count),
        status=similar(prototype, UInt8, count),
        ratios=similar(prototype, T, size(prototype)),
        starts=_population_group_starts(plan.counts),
        thresholds=zeros(T, size(plan.counts)),
    )
end

_lower_width_bank(bank, ::Nothing) = bank
_lower_width_bank(bank, workspace) = _population_state_bank(bank)
_lower_width_storage(bank::_PackedDiagonalBank) = bank.scales
_lower_width_storage(bank::_PackedFactorBank) = bank.factors

_reset_lower_widths!(run, committed, ::Nothing) = nothing
function _reset_lower_widths!(run, committed, workspace)
    copyto!(_lower_width_storage(run), _lower_width_storage(committed))
    copyto!(run.lognormalizers, committed.lognormalizers)
    _copy_exponent_bounds!(_exponent_bounds(run), _exponent_bounds(committed))
    return nothing
end

_lower_width_multiplier(family, slot, ::Type{T}) where {T} =
    _covariance_multiplier(_radial_family_at(family, slot), T)
_lower_width_multiplier(::_LaplaceFamily, slot, ::Type{T}) where {T} = T(2)

function _lower_width_sd(bank::_PackedDiagonalBank, row, slot)
    T = eltype(bank.scales)
    return bank.scales[row, slot] * sqrt(_lower_width_multiplier(bank.family, slot, T))
end
function _lower_width_sd(bank::_PackedFactorBank, row, slot)
    T = eltype(bank.factors)
    scale = zero(T)
    for column = 1:row
        scale = hypot(scale, bank.factors[row, column, slot])
    end
    return scale * sqrt(_lower_width_multiplier(bank.family, slot, T))
end

function _lower_width_ratio(workspace, bank, samples, first, last, origin_index, row, slot)
    T = eltype(workspace.ratios)
    workspace.status[slot] == _POPULATION_COVARIANCE_READY && workspace.ess[slot] >= T(2) ||
        return T(NaN)
    origin = T(_population_sample_coordinate(samples, row, origin_index))
    sd = T(_lower_width_sd(bank, row, slot))
    mean = zero(T)
    for sample = first:last
        iszero(workspace.weights[sample]) && continue
        delta = (T(_population_sample_coordinate(samples, row, sample)) - origin) / sd
        mean += workspace.weights[sample] * delta
    end
    variance = zero(T)
    for sample = first:last
        iszero(workspace.weights[sample]) && continue
        delta = (T(_population_sample_coordinate(samples, row, sample)) - origin) / sd - mean
        variance += abs2(sqrt(workspace.weights[sample]) * delta)
    end
    # Average variances, not standard deviations. The old half keeps widths positive.
    return isfinite(variance) && variance > zero(T) ? sqrt(T(0.5) + variance / T(2)) : T(NaN)
end

@kernel function _lower_width_ratios_kernel!(
    ratios,
    weights,
    ess,
    status,
    samples,
    bank,
    starts,
    counts,
    round,
)
    group = @index(Group, Linear)
    lane = @index(Local, Linear)
    @uniform dimension = size(ratios, 1)
    @uniform lanes = @groupsize()[1]
    partials = @localmem eltype(ratios) (_POPULATION_REDUCTION_WORKGROUP_SIZE,)
    indices = @localmem Int (_POPULATION_REDUCTION_WORKGROUP_SIZE,)
    mean = @localmem eltype(ratios) (1,)
    row = (group - 1) % dimension + 1
    slot = (group - 1) ÷ dimension + 1
    T = eltype(ratios)
    first = starts[slot, round]
    last = first + counts[slot, round] - 1
    active = status[slot] == _POPULATION_COVARIANCE_READY && ess[slot] >= T(2)
    # A zero-weight extreme draw must not set the origin of the central moments.
    largest = T(-Inf)
    origin_index = first
    if active
        for sample = (first+lane-1):lanes:last
            if weights[sample] > largest
                largest = weights[sample]
                origin_index = sample
            end
        end
    end
    partials[lane] = largest
    indices[lane] = origin_index
    @synchronize()
    for offset in _POPULATION_REDUCTION_OFFSETS
        if lane <= offset && partials[lane+offset] > partials[lane]
            partials[lane] = partials[lane+offset]
            indices[lane] = indices[lane+offset]
        end
        @synchronize()
    end
    origin = T(_population_sample_coordinate(samples, row, indices[1]))
    sd = T(_lower_width_sd(bank, row, slot))
    total = zero(T)
    if active
        for sample = (first+lane-1):lanes:last
            iszero(weights[sample]) && continue
            delta = (T(_population_sample_coordinate(samples, row, sample)) - origin) / sd
            total += weights[sample] * delta
        end
    end
    partials[lane] = total
    @synchronize()
    for offset in _POPULATION_REDUCTION_OFFSETS
        if lane <= offset
            partials[lane] += partials[lane+offset]
        end
        @synchronize()
    end
    lane == 1 && (mean[1] = partials[1])
    @synchronize()
    total = zero(T)
    if active
        for sample = (first+lane-1):lanes:last
            iszero(weights[sample]) && continue
            delta = (T(_population_sample_coordinate(samples, row, sample)) - origin) / sd - mean[1]
            total += abs2(sqrt(weights[sample]) * delta)
        end
    end
    partials[lane] = total
    @synchronize()
    for offset in _POPULATION_REDUCTION_OFFSETS
        if lane <= offset
            partials[lane] += partials[lane+offset]
        end
        @synchronize()
    end
    if lane == 1
        variance = partials[1]
        ratios[row, slot] =
            active && isfinite(variance) && variance > zero(T) ? sqrt(T(0.5) + variance / T(2)) :
            T(NaN)
    end
end

function _lower_width_ratios!(
    workspace,
    state,
    views,
    round,
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution},
)
    apply = function (slot)
        first, count = _population_local_group(workspace.starts, state.plan.counts, slot, round)
        origin_index = first
        if workspace.status[slot] == _POPULATION_COVARIANCE_READY
            origin_index += argmax(view(workspace.weights, first:(first+count-1))) - 1
        end
        for row in axes(workspace.ratios, 1)
            workspace.ratios[row, slot] = _lower_width_ratio(
                workspace,
                state.run_bank,
                views.samples,
                first,
                first + count - 1,
                origin_index,
                row,
                slot,
            )
        end
    end
    if execution isa _ThreadedCPUExecution
        _threaded_foreach(apply, axes(workspace.ratios, 2))
    else
        foreach(apply, axes(workspace.ratios, 2))
    end
    return nothing
end

function _lower_width_ratios!(workspace, state, views, round, ::_KernelExecution)
    backend = KernelAbstractions.get_backend(workspace.ratios)
    _lower_width_ratios_kernel!(backend, _POPULATION_REDUCTION_WORKGROUP_SIZE)(
        workspace.ratios,
        workspace.weights,
        workspace.ess,
        workspace.status,
        views.samples,
        state.run_bank,
        workspace.starts,
        state.plan.counts,
        round;
        ndrange=length(workspace.ratios) * _POPULATION_REDUCTION_WORKGROUP_SIZE,
    )
    return nothing
end

_lower_width_row(bank::_PackedDiagonalBank, row, slot) = view(bank.scales, row:row, slot)
_lower_width_row(bank::_PackedFactorBank, row, slot) = view(bank.factors, row, 1:row, slot)
_lower_width_exponents!(bank::_PackedDiagonalBank, slot) = nothing
_lower_width_exponents!(bank::_PackedFactorBank, slot) =
    _set_factor_exponent_bound!(bank.exponent_bounds, slot, view(bank.factors, :, :, slot))

_lower_width_lognormalizer(bank, logabsdet, slot) = _radial_lognormalizer(
    _radial_family_at(bank.family, slot),
    eltype(bank.lognormalizers),
    size(bank.locations, 1),
    logabsdet,
)
_lower_width_lognormalizer(bank::_PackedLaplacePopulation, logabsdet, slot) =
    -eltype(bank.lognormalizers)(size(bank.locations, 1)) * log(eltype(bank.lognormalizers)(2)) -
    logabsdet

@inline function _install_lower_width!(bank, ratios, slot)
    T = eltype(bank.lognormalizers)
    logabsdet = zero(T)
    for row in axes(ratios, 1)
        ratio = ratios[row, slot]
        isfinite(ratio) && ratio > zero(ratio) || return nothing
        for entry in _lower_width_row(bank, row, slot)
            isfinite(T(entry * ratio)) || return nothing
        end
        diagonal = T(last(_lower_width_row(bank, row, slot)) * ratio)
        diagonal > zero(T) || return nothing
        logabsdet += log(diagonal)
    end
    # Recompute from rounded storage, rather than accumulating ratio-rounding drift.
    normalizer = _lower_width_lognormalizer(bank, logabsdet, slot)
    isfinite(normalizer) || return nothing
    for row in axes(ratios, 1)
        storage = _lower_width_row(bank, row, slot)
        for column in eachindex(storage)
            storage[column] *= ratios[row, slot]
        end
    end
    bank.lognormalizers[slot] = normalizer
    _lower_width_exponents!(bank, slot)
    return nothing
end

@kernel function _install_lower_widths_kernel!(bank, ratios)
    _install_lower_width!(bank, ratios, @index(Global, Linear))
end

_adapt_lower_widths!(::Nothing, state, views, round, execution) = nothing
function _adapt_lower_widths!(workspace, state, views, round, execution)
    weights = view(workspace.weights, eachindex(views.logweights))
    backend = KernelAbstractions.get_backend(weights)
    if !(backend isa KernelAbstractions.CPU) && !(execution isa _KernelExecution)
        execution = _KernelExecution(execution)
    end
    _population_local_weights!(
        weights,
        workspace.ess,
        workspace.powers,
        workspace.status,
        views.logweights,
        workspace.starts,
        state.plan.counts,
        workspace.thresholds,
        round,
        zero(eltype(weights)),
        0,
        execution,
    )
    # All three phases use the same stream. The outer round supplies its barrier.
    _lower_width_ratios!(workspace, state, views, round, execution)
    if execution isa _KernelExecution
        _install_lower_widths_kernel!(backend)(
            state.run_bank,
            workspace.ratios;
            ndrange=size(workspace.ratios, 2),
        )
    else
        foreach(
            slot -> _install_lower_width!(state.run_bank, workspace.ratios, slot),
            axes(workspace.ratios, 2),
        )
    end
    return nothing
end

function _population_snapshot_proposals(
    locations,
    proposal_ids,
    sampler::_PreparedImportanceSampler{R,B,T,A},
) where {R,B,T,A<:Union{LAIS,HAIS}}
    sampler.algorithm.adapt_scales ||
        return _fixed_population_snapshot_proposals(locations, proposal_ids, sampler)
    bank = sampler.method_state.bank
    parameters = _with_backend_device(sampler.device) do
        Array(_lower_width_storage(bank))
    end
    family = _with_backend_device(sampler.device) do
        _copy_to_device(MLDataDevices.cpu_device(), bank.family)
    end
    return [
        _lower_width_snapshot(bank, family, locations, parameters, slot) for
        slot in sortperm(proposal_ids)
    ]
end

function _lower_width_snapshot(bank::_PackedDiagonalBank, family, locations, scales, slot)
    location = _population_snapshot_location(locations, slot, bank)
    scale = bank.layout isa _ScalarGaussianLayout ? scales[1, slot] : copy(scales[:, slot])
    family isa _LaplaceFamily && return IndependentLaplace(location, scale)
    radial = _radial_family_at(family, slot)
    return _lower_width_diagonal_snapshot(radial, location, scale)
end
_lower_width_diagonal_snapshot(::GaussianFamily, location::Number, scale) =
    SphericalGaussian(location, scale)
_lower_width_diagonal_snapshot(family::StudentTFamily, location::Number, scale) =
    SphericalStudentT(family.dof, location, scale)
_lower_width_diagonal_snapshot(::GaussianFamily, location, scales) =
    DiagonalGaussian(location, scales)
_lower_width_diagonal_snapshot(family::StudentTFamily, location, scales) =
    DiagonalStudentT(family.dof, location, scales)
function _lower_width_snapshot(bank::_PackedFactorBank, family, locations, factors, slot)
    return _moment_proposal_snapshot(
        _radial_family_at(family, slot),
        copy(locations[:, slot]),
        copy(factors[:, :, slot]),
    )
end
