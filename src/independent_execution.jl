_supports_native_fused_cpu(::_BoxUniform) = true
_accelerator_proposal_limit(::_BoxUniform) = nothing
_native_fused_components(proposal::_NativeIndependentProposal) = (proposal, _NoSampleTransform())
_native_scalar_layout(proposal::_BoxUniform) = proposal.lower isa _NativeGaussianFloat
_native_normal_stride(proposal::_BoxUniform) = _proposal_dimension(proposal)
_native_uniform_stride(proposal::_BoxUniform) = _proposal_dimension(proposal)
# Untransformed box densities do not use inverse-coordinate scratch.
_native_normal_count(::_BoxUniform, count) = 0

# A precision change must rebuild numeric caches before copying prepared arrays.
_independent_source_precision(proposal::_NativeIndependentProposal, state) =
    typeof(proposal.lognormalizer)
_independent_source_precision(
    bank::ProposalBank{<:AbstractVector{<:_NativeIndependentProposal}},
    state,
) = eltype(_static_bank_parameters(state.bank))
_independent_source_precision(proposal, state) = nothing

function _accelerator_precision_source(device, sampler::_PreparedImportanceSampler)
    T = applicable(eltype, device) ? eltype(device) : Nothing
    (T === Float32 || T === Float64) || return nothing
    source_precision =
        _independent_source_precision(_algorithm_proposal(sampler.algorithm), sampler.method_state)
    (isnothing(source_precision) || source_precision === T) && return nothing
    cpu = MLDataDevices.with_eltype(MLDataDevices.CPUDevice(), T)
    return _transfer_prepared_sampler(cpu, sampler, Val(false))
end

_copy_dm_pmc_active_proposal(device, proposal::_NativeIndependentProposal) =
    _copy_to_device(device, proposal)

_native_buffer_strategy(::_BoxUniform) = _BoxFamily()

function _fill_random_buffers!(rng, buffers::_RandomBuffers{U,N,F,_BoxFamily};
    threaded = false
) where {U,N,F}
    return _rand_uniform!(rng, buffers.uniform; threaded)
end

function _allocate_native_samples(prototype, base::_NativeIndependentProposal, count)
    T = _native_fused_float_type(base)
    return _native_scalar_layout(base) ? similar(prototype, T, count) :
           similar(prototype, T, _proposal_dimension(base), count)
end

@inline _native_draw_inputs(::_BoxUniform, normals, uniforms, offset, slot) = (uniforms, UInt16(0))
@inline _native_sample_coordinate(base::_BoxUniform, uniforms, offset, coordinate) =
    _box_coordinate(base, uniforms[offset+coordinate-1], coordinate)
@inline _native_proposal_coordinate(base::_BoxUniform, inputs, offset, coordinate, slot) =
    _native_sample_coordinate(base, inputs, offset, coordinate)

@inline function _native_generated_logdensity(
    base::_NativeIndependentProposal,
    ::_NoSampleTransform,
    sample,
    coordinates,
    offset,
    exponents,
)
    return DensityInterface.logdensityof(base, sample), UInt16(0)
end

_static_bank_parameters(bank) = bank.locations
_static_bank_parameters(bank::_PackedIndependentBank) = bank.first_parameters
_mis_dimension(bank::_PackedIndependentBank) = size(bank.first_parameters, 1)
_active_proposal_count(bank::_PackedIndependentBank) = size(bank.first_parameters, 2)
_allocate_mis_solve_scratch(prototype, bank::_PackedIndependentBank, count) = _NoMISSolveScratch()

function _copy_packed_bank(device, bank::_PackedIndependentBank)
    return _PackedIndependentBank(
        bank.family,
        _copy_to_device(device, bank.first_parameters),
        _copy_to_device(device, bank.second_parameters),
        _copy_to_device(device, bank.lognormalizers),
        _copy_to_device(device, bank.logmasses),
        _copy_to_device(device, bank.cdf),
        _copy_to_device(device, bank.proposal_ids),
        bank.layout,
    )
end

function _prepare_static_mis_bank(
    bank::ProposalBank{<:AbstractVector{<:_NativeIndependentProposal}},
)
    ids, logmasses, cdf = _prepare_active_proposal_metadata(bank)
    proposals = view(bank.proposals, ids)
    first_proposal = first(proposals)
    T = typeof(first_proposal.lognormalizer)
    dimension = _proposal_dimension(first_proposal)
    scalar = _native_scalar_layout(first_proposal)
    family = _independent_family(first_proposal)
    all(
        p ->
            typeof(p.lognormalizer) === T &&
            _proposal_dimension(p) == dimension &&
            _native_scalar_layout(p) == scalar &&
            typeof(_independent_family(p)) === typeof(family),
        proposals,
    ) || throw(
        ArgumentError(
            "native independent proposals must share family, dimension, precision and layout",
        ),
    )
    lower = Matrix{T}(undef, dimension, length(ids))
    upper = similar(lower)
    for (slot, proposal) in pairs(proposals), coordinate = 1:dimension
        a, b = _independent_parameters(proposal)
        lower[coordinate, slot] = _packed_sample_coordinate(a, coordinate)
        upper[coordinate, slot] = _packed_sample_coordinate(b, coordinate)
    end
    layout = scalar ? _ScalarGaussianLayout() : _VectorGaussianLayout()
    return _PackedIndependentBank(
        family,
        lower,
        upper,
        _pack_independent_constants(family, proposals),
        logmasses,
        cdf,
        ids,
        layout,
    )
end

_pack_independent_constants(::_BoxFamily, proposals) = [p.lognormalizer for p in proposals]
_pack_independent_constants(::_GammaFamily, proposals) = [
    _packed_sample_coordinate(p.logconstants, i) for
    i = 1:_proposal_dimension(first(proposals)), p in proposals
]

_accelerator_proposal_limit(::ProposalBank{<:AbstractVector{<:_NativeIndependentProposal}}) =
    nothing
_allocate_static_draw_buffers(prototype, bank, count) =
    _allocate_radial_buffers(prototype, bank.family, count)
_allocate_radial_buffers(prototype, family::_BoxFamily, count) = family
_prepare_mis_normals!(normals, ::_BoxFamily, bank, assignments, failures, execution) = nothing

function _fill_random_buffers!(
    rng,
    buffers::_PackedStaticMISRandomBuffers{U,N,A,S,F,_BoxFamily};
    threaded = false,
) where {U,N,A,S,F}
    rng = _rand_uniform!(rng, buffers.uniform; threaded)
    return _rand_uniform!(rng, buffers.normal; threaded)
end

_supports_native_fused_cpu(::_GammaProposal) = true
_accelerator_proposal_limit(::_GammaProposal) = nothing
_native_scalar_layout(proposal::_GammaProposal) = proposal.shape isa _NativeGaussianFloat
_native_buffer_strategy(::_GammaProposal) = _GammaFamily()
_native_normal_stride(proposal::_GammaProposal) =
    _GAMMA_DRAW_ATTEMPTS * _proposal_dimension(proposal)
_native_uniform_stride(proposal::_GammaProposal) =
    (_GAMMA_DRAW_ATTEMPTS + 1) * _proposal_dimension(proposal)
_fill_radial_buffers!(rng, ::_GammaFamily; threaded = false) = rng

@inline function _native_draw_inputs(base::_GammaProposal, normals, uniforms, offset, slot)
    dimension = _proposal_dimension(base)
    for i = 1:dimension
        value, reason = _buffered_unit_gamma(
            _packed_sample_coordinate(base.shape, i),
            normals,
            uniforms,
            offset - 1 + (i - 1) * _GAMMA_DRAW_ATTEMPTS,
            ((slot - 1) * dimension + i - 1) * (_GAMMA_DRAW_ATTEMPTS + 1),
        )
        iszero(reason) || return normals, reason
        value *= _packed_sample_coordinate(base.scale, i)
        isfinite(value) && value > zero(value) || return normals, _NATIVE_PROPOSAL_DRAW_EXHAUSTED
        normals[offset+i-1] = value
    end
    return normals, UInt16(0)
end

@inline _native_sample_coordinate(base::_GammaProposal, inputs, offset, coordinate) =
    inputs[offset+coordinate-1]
@inline _native_proposal_coordinate(base::_GammaProposal, inputs, offset, coordinate, slot) =
    _native_sample_coordinate(base, inputs, offset, coordinate)

struct _GammaDrawBuffers{A}
    normal::A
    uniform::A
end
Adapt.@adapt_structure _GammaDrawBuffers

function _fill_radial_buffers!(rng, buffers::_GammaDrawBuffers; threaded = false)
    rng = _rand_normal!(rng, buffers.normal; threaded)
    return _rand_uniform!(rng, buffers.uniform; threaded)
end

function _allocate_static_draw_buffers(prototype, bank::_PackedIndependentBank{_GammaFamily}, count)
    return _allocate_gamma_draw_buffers(prototype, _mis_dimension(bank), count)
end

function _allocate_gamma_draw_buffers(prototype, dimension, count)
    total = dimension * count
    T = eltype(prototype)
    return _GammaDrawBuffers(
        similar(prototype, T, _GAMMA_DRAW_ATTEMPTS * total),
        similar(prototype, T, (_GAMMA_DRAW_ATTEMPTS + 1) * total),
    )
end

function _fill_random_buffers!(
    rng,
    buffers::_PackedStaticMISRandomBuffers{U,N,A,S,F,<:_GammaDrawBuffers};
    threaded = false,
) where {U,N,A,S,F}
    rng = _rand_uniform!(rng, buffers.uniform; threaded)
    return _fill_radial_buffers!(rng, buffers.radial; threaded)
end

@kernel function _independent_gamma_draws!(values, normals, uniforms, bank, assignments, failures)
    sample = @index(Global, Linear)
    slot = assignments[sample]
    dimension = _mis_dimension(bank)
    for i = 1:dimension
        index = (sample - 1) * dimension + i
        value, reason = _buffered_unit_gamma(
            bank.first_parameters[i, slot],
            normals,
            uniforms,
            (index - 1) * _GAMMA_DRAW_ATTEMPTS,
            (index - 1) * (_GAMMA_DRAW_ATTEMPTS + 1),
        )
        value *= bank.second_parameters[i, slot]
        isfinite(value) && value > zero(value) || (reason = _NATIVE_PROPOSAL_DRAW_EXHAUSTED)
        if !iszero(reason)
            _record_native_failure!(failures, sample, 0, reason)
            value = oftype(value, NaN)
        end
        values[index] = value
    end
end

_gamma_draw_workgroupsize(backend, execution, count) = _native_workgroupsize(execution, count)
# Bound register demand per block, including Reactant's Float64 vector lowering.
_gamma_draw_workgroupsize(::KernelAbstractions.GPU, execution, count) = 256

function _prepare_mis_normals!(
    values,
    buffers::_GammaDrawBuffers,
    bank,
    assignments,
    failures,
    execution,
)
    backend = KernelAbstractions.get_backend(values)
    _independent_gamma_draws!(backend)(
        values,
        buffers.normal,
        buffers.uniform,
        bank,
        assignments,
        failures;
        ndrange = length(assignments),
        workgroupsize = _gamma_draw_workgroupsize(backend, execution, length(assignments)),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

@inline _native_proposal_coordinate(
    bank::_PackedIndependentBank{_GammaFamily},
    values,
    offset,
    coordinate,
    slot,
) = values[offset+coordinate-1]

@inline function _mis_proposal_logdensity(
    ::Type{T},
    bank::_PackedIndependentBank{_GammaFamily},
    sample,
    slot,
) where {T}
    value = zero(eltype(bank.first_parameters))
    for i = 1:_mis_dimension(bank)
        value += _gamma_logdensity(
            bank.first_parameters[i, slot],
            bank.second_parameters[i, slot],
            bank.lognormalizers[i, slot],
            _packed_sample_coordinate(sample, i),
        )
    end
    return T(value)
end

function _allocate_packed_static_mis_samples(prototype, bank::_PackedIndependentBank, count)
    T = eltype(bank.first_parameters)
    return bank.layout isa _ScalarGaussianLayout ? similar(prototype, T, count) :
           similar(prototype, T, _mis_dimension(bank), count)
end

@inline function _native_proposal_coordinate(
    bank::_PackedIndependentBank{_BoxFamily},
    uniforms,
    offset,
    coordinate,
    slot,
)
    return _bounded_interval_value(
        bank.first_parameters[coordinate, slot],
        bank.second_parameters[coordinate, slot],
        uniforms[offset+coordinate-1],
    )
end

@inline function _mis_proposal_logdensity(
    ::Type{T},
    bank::_PackedIndependentBank{_BoxFamily},
    sample,
    slot,
) where {T}
    for coordinate = 1:_mis_dimension(bank)
        x = _packed_sample_coordinate(sample, coordinate)
        bank.first_parameters[coordinate, slot] <= x <= bank.second_parameters[coordinate, slot] ||
            return T(-Inf)
    end
    return T(bank.lognormalizers[slot])
end

@inline function _native_generated_logdensity(
    base::_NativeIndependentProposal,
    transform,
    sample,
    coordinates,
    offset,
    exponents,
)
    logabsjac, reason = _native_inverse_sample!(coordinates, offset, sample, transform)
    iszero(reason) || return base.lognormalizer, _NATIVE_PROPOSAL_INVALID
    values = view(coordinates, offset:(offset+_proposal_dimension(base)-1))
    return DensityInterface.logdensityof(base, values) - logabsjac, UInt16(0)
end
