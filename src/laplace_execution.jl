_supports_native_fused_cpu(::_LaplaceProposal) = true
_accelerator_proposal_limit(::_LaplaceProposal) = nothing
_native_scalar_layout(proposal::_LaplaceProposal) = proposal.location isa _NativeGaussianFloat
_native_normal_stride(proposal::_LaplaceProposal) = _proposal_dimension(proposal)
_native_uniform_stride(proposal::_LaplaceProposal) = _proposal_dimension(proposal)
_native_normal_count(::_LaplaceProposal, count) = 0

# The box strategy fills only uniforms; Laplace keeps its own proposal family.
_native_buffer_strategy(::_LaplaceProposal) = _BoxFamily()
@inline _native_draw_inputs(::_LaplaceProposal, normals, uniforms, offset, slot) =
    (uniforms, UInt16(0))

@inline function _native_sample_coordinate(base::_LaplaceProposal, uniforms, offset, coordinate)
    return _laplace_coordinate(
        _packed_sample_coordinate(base.location, coordinate),
        _packed_sample_coordinate(base.scale, coordinate),
        uniforms[offset+coordinate-1],
    )
end

@inline _native_proposal_coordinate(base::_LaplaceProposal, inputs, offset, coordinate, slot) =
    _native_sample_coordinate(base, inputs, offset, coordinate)

_pack_independent_constants(::_LaplaceFamily, proposals) = [p.lognormalizer for p in proposals]
_allocate_static_draw_buffers(prototype, bank::_PackedIndependentBank{_LaplaceFamily}, count) =
    _BoxFamily()

@inline function _native_proposal_coordinate(
    bank::_PackedIndependentBank{_LaplaceFamily},
    uniforms,
    offset,
    coordinate,
    slot,
)
    return _laplace_coordinate(
        bank.first_parameters[coordinate, slot],
        bank.second_parameters[coordinate, slot],
        uniforms[offset+coordinate-1],
    )
end

@inline function _mis_proposal_logdensity(
    ::Type{T},
    bank::_PackedIndependentBank{_LaplaceFamily},
    sample,
    slot,
) where {T}
    value = bank.lognormalizers[slot]
    for i = 1:_mis_dimension(bank)
        value -= abs(
            _standardized_difference(
                _packed_sample_coordinate(sample, i),
                bank.first_parameters[i, slot],
                bank.second_parameters[i, slot],
            ),
        )
    end
    return T(value)
end
