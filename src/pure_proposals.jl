function _rand_proposal(
    rng::PureRNGs.AbstractPureRNG,
    proposal::_GaussianProposal{F,T},
) where {F,T<:_NativeGaussianFloat}
    normal, rng = PureRNGs.randn_next(rng, T)
    return _gaussian_affine_coordinate(proposal.location, proposal.scale.scale, normal), rng
end

function _rand_proposal(
    rng::PureRNGs.AbstractPureRNG,
    proposal::_GaussianProposal{F,Vector{T}},
) where {F,T}
    normals, rng = PureRNGs.randn_next(rng, T, length(proposal.location))
    return _gaussian_from_normal(proposal.location, proposal.scale, normals), rng
end

function _rand_unit_gamma_next(rng, shape::T) where {T<:_NativeGaussianFloat}
    adjusted_shape = shape < one(T) ? shape + one(T) : shape
    offset, coefficient = _gamma_parameters(adjusted_shape)
    while true
        normal, rng = PureRNGs.randn_next(rng, T)
        uniform, rng = PureRNGs.rand_next(rng, T)
        gamma, accepted = _gamma_candidate(offset, coefficient, normal, uniform)
        accepted || continue
        shape >= one(T) && return gamma, rng
        correction, rng = PureRNGs.rand_next(rng, T)
        correction > zero(T) && return gamma * correction^inv(shape), rng
    end
end

function _student_t_multiplier_next(rng, dof::T) where {T<:_NativeGaussianFloat}
    if dof == one(T)
        normal, rng = PureRNGs.randn_next(rng, T)
        return inv(abs(normal)), rng
    end
    gamma, rng = _rand_unit_gamma_next(rng, dof / T(2))
    return sqrt(dof / (T(2) * gamma)), rng
end

function _rand_proposal(
    rng::PureRNGs.AbstractPureRNG,
    proposal::_StudentTProposal{F,T},
) where {F,T<:_NativeGaussianFloat}
    multiplier, rng = _student_t_multiplier_next(rng, proposal.family.dof)
    normal, rng = PureRNGs.randn_next(rng, T)
    sample = _gaussian_affine_coordinate(
        proposal.location,
        proposal.scale.scale,
        normal * multiplier,
    )
    return sample, rng
end

function _rand_proposal(
    rng::PureRNGs.AbstractPureRNG,
    proposal::_StudentTProposal{F,Vector{T}},
) where {F,T}
    normals, rng = PureRNGs.randn_next(rng, T, length(proposal.location))
    multiplier, rng = _student_t_multiplier_next(rng, proposal.family.dof)
    normals .*= multiplier
    return _gaussian_from_normal(proposal.location, proposal.scale, normals), rng
end

function _rand_independent_coordinate_next(rng, proposal::_GammaProposal, i)
    shape = _packed_sample_coordinate(proposal.shape, i)
    gamma, rng = _rand_unit_gamma_next(rng, shape)
    value = gamma * _packed_sample_coordinate(proposal.scale, i)
    isfinite(value) && value > zero(value) || throw(
        DomainError(value, "Gamma draw is not finite and positive in the chosen precision"),
    )
    return value, rng
end

function _rand_independent_coordinate_next(rng, proposal::_BoxUniform, i)
    uniform, rng = PureRNGs.rand_next(rng, typeof(proposal.lognormalizer))
    return _box_coordinate(proposal, uniform, i), rng
end

function _rand_independent_coordinate_next(rng, proposal::_LaplaceProposal, i)
    uniform, rng = PureRNGs.rand_next(rng, typeof(proposal.lognormalizer))
    value = _laplace_coordinate(
        _packed_sample_coordinate(proposal.location, i),
        _packed_sample_coordinate(proposal.scale, i),
        uniform,
    )
    isfinite(value) ||
        throw(DomainError(value, "Laplace draw is not finite in the chosen precision"))
    return value, rng
end

function _rand_proposal(rng::PureRNGs.AbstractPureRNG, proposal::_NativeIndependentProposal)
    _native_scalar_layout(proposal) &&
        return _rand_independent_coordinate_next(rng, proposal, 1)
    first_parameter, _ = _independent_parameters(proposal)
    sample = similar(first_parameter)
    for i in eachindex(sample)
        sample[i], rng = _rand_independent_coordinate_next(rng, proposal, i)
    end
    return sample, rng
end

function _draw_proposal!(
    sampler::_PreparedImportanceSampler{<:PureRNGs.AbstractPureRNG},
    proposal::ProductProposal,
)
    return map(block -> _draw_proposal!(sampler, block), proposal.blocks)
end

function _draw_proposal!(
    sampler::_PreparedImportanceSampler{<:PureRNGs.AbstractPureRNG},
    proposal::TransformedProposal,
)
    coordinates = _draw_proposal!(sampler, proposal.base)
    logical_value, _ = _transform_with_logjac(proposal.transform, coordinates)
    return logical_value
end
