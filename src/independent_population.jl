# Diagonal population storage also represents independent location/moment families.
struct _GammaPopulationFamily{P}
    shapes::P
    variances::P
end
Adapt.@adapt_structure _GammaPopulationFamily

const _PackedLaplacePopulation =
    _PackedDiagonalBank{L,S,N,M,C,I,R,_LaplaceFamily} where {L,S,N,M,C,I,R}
const _PackedGammaPopulation =
    _PackedDiagonalBank{L,S,N,M,C,I,R,_GammaPopulationFamily{P}} where {L,S,N,M,C,I,R,P}
const _PackedIndependentPopulation = Union{_PackedLaplacePopulation,_PackedGammaPopulation}

function _prepare_population_bank(bank::ProposalBank{<:AbstractVector{<:_LaplaceProposal}})
    packed = _prepare_static_mis_bank(bank)
    return _PackedDiagonalBank(
        packed.first_parameters,
        packed.second_parameters,
        packed.lognormalizers,
        packed.logmasses,
        packed.cdf,
        packed.proposal_ids,
        packed.layout,
        packed.family,
    )
end

function _prepare_population_bank(bank::ProposalBank{<:AbstractVector{<:_GammaProposal}})
    packed = _prepare_static_mis_bank(bank)
    means = packed.first_parameters .* packed.second_parameters
    variances = means .* packed.second_parameters
    all(isfinite, means) && all(>(0), means) && all(isfinite, variances) && all(>(0), variances) ||
        throw(ArgumentError("Gamma population means and variances must be finite and positive"))
    return _PackedDiagonalBank(
        means,
        packed.second_parameters,
        packed.lognormalizers,
        packed.logmasses,
        packed.cdf,
        packed.proposal_ids,
        packed.layout,
        _GammaPopulationFamily(packed.first_parameters, variances),
    )
end

_population_snapshot_proposal(proposal::_LaplaceProposal, location) =
    IndependentLaplace(location, proposal.scale)
function _population_snapshot_proposal(proposal::_GammaProposal, mean)
    variance = (proposal.shape .* proposal.scale) .* proposal.scale
    scale = variance ./ mean
    return IndependentGamma(mean ./ scale, scale)
end

@inline function _independent_density_bank(bank::_PackedLaplacePopulation)
    return _PackedIndependentBank(
        bank.family,
        bank.locations,
        bank.scales,
        bank.lognormalizers,
        bank.logmasses,
        bank.cdf,
        bank.proposal_ids,
        bank.layout,
    )
end

@inline function _independent_density_bank(bank::_PackedGammaPopulation)
    return _PackedIndependentBank(
        _GammaFamily(),
        bank.family.shapes,
        bank.scales,
        bank.lognormalizers,
        bank.logmasses,
        bank.cdf,
        bank.proposal_ids,
        bank.layout,
    )
end

@inline _native_proposal_coordinate(
    bank::_PackedIndependentPopulation,
    inputs,
    offset,
    coordinate,
    slot,
) = _native_proposal_coordinate(_independent_density_bank(bank), inputs, offset, coordinate, slot)
@inline _mis_proposal_logdensity(
    ::Type{T},
    bank::_PackedIndependentPopulation,
    sample,
    slot,
) where {T} = _mis_proposal_logdensity(T, _independent_density_bank(bank), sample, slot)

_allocate_radial_buffers(prototype, ::_LaplaceFamily, count) = _BoxFamily()
_allocate_radial_buffers(prototype, family::_GammaPopulationFamily, count) =
    _allocate_gamma_draw_buffers(prototype, size(family.shapes, 1), count)

function _prepare_mis_normals!(
    values,
    buffers::_GammaDrawBuffers,
    bank::_PackedGammaPopulation,
    assignments,
    failures,
    execution,
)
    return _prepare_mis_normals!(
        values,
        buffers,
        _independent_density_bank(bank),
        assignments,
        failures,
        execution,
    )
end

function _fill_population_normals!(
    rng,
    buffers::_PopulationNormalBuffers{N,F,_BoxFamily};
    threaded = false,
) where {N,F}
    return _rand_uniform!(rng, buffers.normals; threaded)
end

_fill_population_normals!(
    rng,
    buffers::_PopulationNormalBuffers{N,F,G};
    threaded = false,
) where {N,F,G<:_GammaDrawBuffers} = _fill_radial_buffers!(rng, buffers.radial; threaded)
_fill_population_normals!(
    rng,
    buffers::_DMPMCRandomBuffers{N,U,F,G};
    threaded = false,
) where {N,U,F,G<:_GammaDrawBuffers} = _fill_radial_buffers!(rng, buffers.radial; threaded)

function _population_with_locations(bank::_PackedGammaPopulation, locations)
    return _PackedDiagonalBank(
        locations,
        copy(bank.scales),
        copy(bank.lognormalizers),
        bank.logmasses,
        bank.cdf,
        bank.proposal_ids,
        bank.layout,
        _GammaPopulationFamily(copy(bank.family.shapes), bank.family.variances),
    )
end

function _reset_population_run!(
    state::Union{_PreparedAPIS,_PreparedDMPMC},
    run::_PackedGammaPopulation,
    committed,
)
    copyto!(run.locations, committed.locations)
    copyto!(run.scales, committed.scales)
    copyto!(run.lognormalizers, committed.lognormalizers)
    copyto!(run.family.shapes, committed.family.shapes)
    return nothing
end

function _commit_population_round!(
    state::Union{_PreparedAPIS,_PreparedDMPMC},
    run::_PackedGammaPopulation,
)
    copyto!(run.locations, state.workspace.candidate_locations)
    run.scales .= run.family.variances ./ run.locations
    run.family.shapes .= run.locations ./ run.scales
    _independent_logconstants!(run.lognormalizers, run.family.shapes, run.scales, _GammaFamily())
    return all(isfinite, run.lognormalizers)
end

_validate_population_commit!(bank, status, transfers) = nothing
function _validate_population_commit!(bank::_PackedGammaPopulation, status, transfers)
    valid = Bool(status)
    _record_device_scalar_transfer!(transfers, bank.locations, Bool, Val(:local_mean_validity))
    valid || throw(DomainError(nothing, "Gamma population moment parameters are not representable"))
    return nothing
end

# Unshifted Gamma is not the lower translation family assumed by LAIS.
function _prepare_method_state(
    algorithm::LAIS{<:ProposalBank{<:AbstractVector{<:_GammaProposal}}},
    prepared_target,
)
    throw(ArgumentError("LAIS requires a translation family, not an unshifted Gamma family"))
end

function _fill_population_normals!(
    rng,
    buffers::_DMPMCRandomBuffers{N,U,F,_BoxFamily};
    threaded = false,
) where {N,U,F}
    return _rand_uniform!(rng, buffers.normals; threaded)
end
