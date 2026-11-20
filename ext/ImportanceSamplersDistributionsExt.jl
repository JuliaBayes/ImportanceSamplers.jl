module ImportanceSamplersDistributionsExt

import Distributions
import ImportanceSamplers
import LinearAlgebra
import Statistics

function _cpu_gaussian_location(proposal)
    location = Statistics.mean(proposal)
    device = ImportanceSamplers.MLDataDevices.get_device(location)
    device === nothing || device isa ImportanceSamplers.MLDataDevices.AbstractCPUDevice ||
        throw(ArgumentError("Distributions Gaussian conversion requires a CPU proposal; transfer it explicitly first"))
    return location
end

function _representable_gaussian_variance(variance)
    all(value -> isfinite(value) && value > zero(value), variance) || throw(
        ArgumentError("MvNormal conversion requires finite positive covariance in the proposal precision"),
    )
    return variance
end

function Distributions.Normal(proposal::ImportanceSamplers._GaussianProposal{F,L}) where {F,L<:Number}
    return Distributions.Normal(_cpu_gaussian_location(proposal), Statistics.std(proposal))
end

function Distributions.MvNormal(
    proposal::ImportanceSamplers._GaussianProposal{F,L,S},
) where {F,L<:AbstractVector,S<:ImportanceSamplers._SphericalGaussianScale}
    location = _cpu_gaussian_location(proposal)
    variance = _representable_gaussian_variance(abs2(ImportanceSamplers.proposal_scale_factor(proposal)))
    return Distributions.MvNormal(location, variance * LinearAlgebra.I)
end

function Distributions.MvNormal(
    proposal::ImportanceSamplers._GaussianProposal{F,L,S},
) where {F,L<:AbstractVector,S<:ImportanceSamplers._DiagonalGaussianScale}
    location = _cpu_gaussian_location(proposal)
    variance = _representable_gaussian_variance(Statistics.var(proposal))
    return Distributions.MvNormal(location, LinearAlgebra.Diagonal(variance))
end

function Distributions.MvNormal(
    proposal::ImportanceSamplers._GaussianProposal{F,L,S},
) where {F,L<:AbstractVector,S<:ImportanceSamplers._FactorGaussianScale}
    location = _cpu_gaussian_location(proposal)
    # PDMat retains this existing factor; it materializes covariance but does not refactor it.
    factor = LinearAlgebra.Cholesky(copy(ImportanceSamplers.proposal_scale_factor(proposal)), 'L', 0)
    return Distributions.MvNormal(location, Distributions.PDMat(factor))
end

ImportanceSamplers._default_mass_type(proposal::Distributions.Distribution) =
    float(eltype(proposal))

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.Uniform)
    return ImportanceSamplers.BoxUniform(Distributions.params(proposal)...)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.Gamma)
    return ImportanceSamplers.IndependentGamma(Distributions.params(proposal)...)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.Laplace)
    return ImportanceSamplers.IndependentLaplace(Distributions.params(proposal)...)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.Normal)
    location, scale = Distributions.params(proposal)
    return ImportanceSamplers.SphericalGaussian(location, scale)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.IsoNormal)
    location = collect(Statistics.mean(proposal))
    scale = first(Statistics.std(proposal))
    return ImportanceSamplers.SphericalGaussian(location, scale)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.DiagNormal)
    location = collect(Statistics.mean(proposal))
    scales = collect(Statistics.std(proposal))
    return ImportanceSamplers.DiagonalGaussian(location, scales)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.FullNormal)
    location = collect(Statistics.mean(proposal))
    _, covariance = Distributions.params(proposal)
    factor = LinearAlgebra.cholesky(covariance)
    return ImportanceSamplers.FactorGaussian(location, factor)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.TDist)
    dof = only(Distributions.params(proposal))
    T = typeof(dof)
    return ImportanceSamplers.SphericalStudentT(dof, zero(T), one(T))
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.Cauchy)
    location, scale = Distributions.params(proposal)
    return ImportanceSamplers.SphericalStudentT(one(location), location, scale)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.IsoTDist)
    dof, location, _ = Distributions.params(proposal)
    scale_matrix = Distributions.scale(proposal)
    scale = sqrt(first(LinearAlgebra.diag(scale_matrix)))
    return ImportanceSamplers.SphericalStudentT(dof, collect(location), scale)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.DiagTDist)
    dof, location, _ = Distributions.params(proposal)
    scales = sqrt.(LinearAlgebra.diag(Distributions.scale(proposal)))
    return ImportanceSamplers.DiagonalStudentT(dof, collect(location), scales)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.GenericMvTDist)
    dof, location, _ = Distributions.params(proposal)
    factor = LinearAlgebra.cholesky(
        LinearAlgebra.Symmetric(Distributions.scale(proposal)),
    )
    return ImportanceSamplers.FactorStudentT(dof, collect(location), factor)
end

function ImportanceSamplers._prepare_proposal_inputs(
    proposals::AbstractVector{<:Union{
        Distributions.Normal,
        Distributions.Uniform,
        Distributions.Gamma,
        Distributions.Laplace,
        Distributions.MvNormal,
        Distributions.TDist,
        Distributions.Cauchy,
        Distributions.IsoTDist,
        Distributions.DiagTDist,
        Distributions.GenericMvTDist,
    }},
)
    return map(ImportanceSamplers._prepare_proposal_input, proposals)
end

end
