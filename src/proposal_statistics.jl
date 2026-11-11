"""
    proposal_scale_factor(proposal)

Return the stored scale factor of a native Gaussian or Student-t proposal:
a scalar for spherical proposals, a `Diagonal` wrapper for diagonal proposals,
or the stored lower-triangular matrix for factor proposals. For a Student-t law this
is an elliptical scale factor, not a covariance factor.

The result borrows the proposal's storage. Treat it as read-only, and copy it
explicitly if independent mutable storage is needed. No parameters are copied,
converted in precision, or transferred between devices. An independent
[`current_proposal`](@ref) snapshot owns its parameters independently of the
sampler and remains valid across subsequent sampling calls.
"""
proposal_scale_factor(proposal::_NativeRadialProposal) =
    _proposal_scale_factor(proposal.scale)

_proposal_scale_factor(scale::_SphericalGaussianScale) = scale.scale
_proposal_scale_factor(scale::_DiagonalGaussianScale) = LinearAlgebra.Diagonal(scale.scales)
_proposal_scale_factor(scale::_FactorGaussianScale) = scale.factor

Statistics.mean(proposal::_GaussianProposal) = proposal.location
Statistics.mean(proposal::_StudentTProposal) =
    proposal.family.dof > 1 ? proposal.location : oftype.(proposal.location, NaN)
Statistics.mean(proposal::_GammaProposal) = proposal.shape .* proposal.scale
Statistics.mean(proposal::_LaplaceProposal) = proposal.location
Statistics.mean(proposal::_BoxUniform) = _bounded_interval_value.(
    proposal.lower,
    proposal.upper,
    oftype(proposal.lognormalizer, 0.5),
)

_radial_variance(location::Number, scale::_SphericalGaussianScale) = abs2(scale.scale)
_radial_variance(location::AbstractVector, scale::_SphericalGaussianScale) =
    zero.(location) .+ abs2(scale.scale)
_radial_variance(location, scale::_DiagonalGaussianScale) = abs2.(scale.scales)
_radial_variance(location, scale::_FactorGaussianScale) =
    vec(sum(abs2, scale.factor; dims = 2))

Statistics.var(proposal::_GaussianProposal) =
    _radial_variance(proposal.location, proposal.scale)

function Statistics.var(proposal::_StudentTProposal)
    dof = proposal.family.dof
    dof > 2 && return abs2.(Statistics.std(proposal))
    # Match TDist for scalars and MvTDist for vectors, preserving shape/precision.
    value = proposal.location isa Number && dof > 1 ? Inf : NaN
    return oftype.(proposal.location, value)
end

Statistics.var(proposal::_GammaProposal) =
    (proposal.shape .* proposal.scale) .* proposal.scale
Statistics.var(proposal::_LaplaceProposal) = (2 .* proposal.scale) .* proposal.scale
Statistics.var(proposal::_BoxUniform) = abs2.(Statistics.std(proposal))

_radial_std(location::Number, scale::_SphericalGaussianScale) = scale.scale
_radial_std(location::AbstractVector, scale::_SphericalGaussianScale) =
    zero.(location) .+ scale.scale
_radial_std(location, scale::_DiagonalGaussianScale) = scale.scales
function _radial_std(location, scale::_FactorGaussianScale)
    row_scale = maximum(abs, scale.factor; dims = 2)
    return vec(row_scale .* sqrt.(sum(abs2, scale.factor ./ row_scale; dims = 2)))
end

Statistics.std(proposal::_GaussianProposal) = _radial_std(proposal.location, proposal.scale)
function Statistics.std(proposal::_StudentTProposal)
    dof = proposal.family.dof
    dof > 2 || return sqrt.(Statistics.var(proposal))
    return sqrt(_covariance_multiplier(proposal.family, typeof(dof))) .*
           _radial_std(proposal.location, proposal.scale)
end
Statistics.std(proposal::_GammaProposal) = sqrt.(proposal.shape) .* proposal.scale
Statistics.std(proposal::_LaplaceProposal) =
    sqrt(oftype(proposal.lognormalizer, 2)) .* proposal.scale
function Statistics.std(proposal::_BoxUniform)
    divisor = sqrt(oftype(proposal.lognormalizer, 12))
    return _box_std.(proposal.lower, proposal.upper, divisor)
end

function _box_std(lower, upper, divisor)
    width = upper - lower
    return isfinite(width) ? width / divisor : upper / divisor - lower / divisor
end

_proposal_covariance(variance::Number) = variance
_proposal_covariance(variance::AbstractVector) = LinearAlgebra.Diagonal(variance)

Statistics.cov(proposal::_GaussianProposal) = _radial_covariance(proposal, proposal.scale)
_radial_covariance(proposal, scale) = _proposal_covariance(Statistics.var(proposal))
_radial_covariance(proposal, scale::_FactorGaussianScale) = scale.factor * scale.factor'

function Statistics.cov(proposal::_StudentTProposal)
    dof = proposal.family.dof
    if proposal.location isa Number
        return Statistics.var(proposal)
    elseif dof <= 2
        covariance =
            similar(proposal.location, length(proposal.location), length(proposal.location))
        return fill!(covariance, oftype(dof, NaN))
    elseif proposal.scale isa _FactorGaussianScale
        factor =
            sqrt(_covariance_multiplier(proposal.family, typeof(dof))) .*
            proposal.scale.factor
        return factor * factor'
    end
    return _proposal_covariance(Statistics.var(proposal))
end

Statistics.cov(proposal::_NativeIndependentProposal) =
    _proposal_covariance(Statistics.var(proposal))
