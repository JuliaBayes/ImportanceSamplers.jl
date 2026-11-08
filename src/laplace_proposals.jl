struct _LaplaceFamily <: AbstractProposalFamily end

struct _LaplaceProposal{P,T} <: _NativeIndependentProposal
    location::P
    scale::P
    lognormalizer::T
end

_independent_parameters(proposal::_LaplaceProposal) = (proposal.location, proposal.scale)
_independent_family(::_LaplaceProposal) = _LaplaceFamily()

"""
    IndependentLaplace(location, scale)

Normalized scalar Laplace proposal, or a product of independent Laplace
coordinates. Parameters must be finite, matching `Float32` or `Float64`
scalars or vectors, with strictly positive scales. Vector inputs are copied.
The mean is `location` and coordinate variances are `2 .* scale.^2`.

Plain IS and homogeneous static MIS banks use fixed parameters. [`AMIS`](@ref)
and [`NPMC`](@ref) fit the weighted marginal mean and `sqrt(variance / 2)`.
This is moment matching, not the weighted-median/absolute-deviation maximum
likelihood fit. Correlations are not fitted. Invalid fits fail without replacing
the committed proposal. Each coordinate uses one uniform
draw. A nonfinite rounded affine draw fails the call without redrawing.
[`APIS`](@ref), [`DeterministicMixturePMC`](@ref), and [`LAIS`](@ref) update
locations while retaining each coordinate scale.
"""
function IndependentLaplace(location, scale)
    stored_location, stored_scale = _independent_parameters(location, scale)
    all(>(0), stored_scale) || throw(ArgumentError("Laplace scales must be positive"))
    T = _gaussian_float_type(stored_location)
    lognormalizer = -sum(s -> log(T(2)) + log(s), stored_scale)
    return _LaplaceProposal(stored_location, stored_scale, lognormalizer)
end

Adapt.@adapt_structure _LaplaceProposal
_independent_constructor(::_LaplaceProposal) = IndependentLaplace
_independent_fields(proposal::_LaplaceProposal) =
    (proposal.location, proposal.scale, proposal.lognormalizer)
function Adapt.adapt_structure(device::MLDataDevices.AbstractDevice, proposal::_LaplaceProposal)
    return _LaplaceProposal(_adapt_independent_fields(device, proposal)...)
end
DensityInterface.DensityKind(::_LaplaceProposal) = DensityInterface.HasDensity()
_proposal_dimension(proposal::_LaplaceProposal) = _gaussian_dimension(proposal.location)
_default_mass_type(proposal::_LaplaceProposal) = typeof(proposal.lognormalizer)

@inline function _laplace_coordinate(location, scale, uniform::T) where {T}
    # GPU RNGs can return 1. Identify it with 0 to accept either half-open interval.
    uniform == one(T) && return location
    # Both half-intervals start at zero, avoiding log(0) at the other endpoints.
    standardized = uniform < T(0.5) ? log1p(-T(2) * uniform) : -log1p(-(T(2) * uniform - one(T)))
    return location + scale * standardized
end

function _laplace_draw(rng, location::T, scale::T) where {T}
    value = _laplace_coordinate(location, scale, Random.rand(rng, T))
    isfinite(value) ||
        throw(DomainError(value, "Laplace draw is not finite in the chosen precision"))
    return value
end

Random.rand(
    rng::Random.AbstractRNG,
    proposal::_LaplaceProposal{T},
) where {T<:_NativeGaussianFloat} = _laplace_draw(rng, proposal.location, proposal.scale)

function Random.rand(
    rng::Random.AbstractRNG,
    proposal::_LaplaceProposal{<:AbstractVector{T}},
) where {T}
    return [
        _laplace_draw(rng, proposal.location[i], proposal.scale[i]) for
        i in eachindex(proposal.location)
    ]
end

function DensityInterface.logdensityof(proposal::_LaplaceProposal, sample)
    dimension = _proposal_dimension(proposal)
    length(sample) == dimension ||
        throw(DimensionMismatch("sample dimension differs from Laplace proposal"))
    sample_type = sample isa Number ? typeof(sample) : eltype(sample)
    result =
        convert(promote_type(typeof(proposal.lognormalizer), sample_type), proposal.lognormalizer)
    for i = 1:dimension
        result -= abs(
            _standardized_difference(
                _packed_sample_coordinate(sample, i),
                _packed_sample_coordinate(proposal.location, i),
                _packed_sample_coordinate(proposal.scale, i),
            ),
        )
    end
    return result
end
