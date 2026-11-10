"""
    LAIS(bank; transition, rounds, round_size, adapt_scales=false)

Configure layered importance sampling. Each round advances the upper MCMC chains,
then draws equally from the lower Gaussian, Student-t or independent Laplace
proposals at their new centres. Use
current-round deterministic-mixture weights and retain all lower samples.

Independent [`RandomWalkMetropolis`](@ref)/[`RAM`](@ref) chains implement
PI-MAIS-style upper adaptation. [`SampleMetropolisHastings`](@ref) instead makes
interacting population replacements for I²-MAIS while retaining the same LAIS
result and lower-weighting API.

`round_size` is the total lower count per round, either an integer or a vector
with one entry per round. It must divide equally across positive, equal-mass
proposals. Each bank uses one translation family. Lower scales and Student-t degrees of freedom
stay fixed by default and independent of `transition`; every positive `nu` is allowed.
With `adapt_scales=true`, fit marginal widths from local DM-weighted lower draws
after each round. Blend old and fitted marginal variance equally, preserving
factor correlations. Student-t adaptation requires `nu > 2`. Degenerate local
fits retain the old widths. This optional extension does not adapt the upper
transition or its mass matrix. See [Width adaptation](@ref lower-width-adaptation).
"""
struct LAIS{B<:ProposalBank,K<:AbstractMCMCTransition,S} <: _FixedPopulationSampler
    bank::B
    transition::K
    rounds::Int
    round_size::S
    adapt_scales::Bool

    function LAIS(bank::B; transition::K, rounds, round_size, adapt_scales::Bool=false) where {B<:ProposalBank,K<:AbstractMCMCTransition}
        validated = _validate_adaptive_schedule(rounds, round_size)
        masses = bank.masses
        all(>(zero(eltype(masses))), masses) && all(==(first(masses)), masses) ||
            throw(ArgumentError("LAIS requires equal positive proposal masses"))
        count = length(bank.proposals)
        _validate_lais_transition(transition, bank, rounds, validated)
        all(n -> n % count == 0, _resolve_adaptive_schedule(rounds, validated)) ||
            throw(ArgumentError("LAIS round sizes must divide equally across proposals"))
        _validate_lower_widths(bank, adapt_scales)
        return new{B,K,typeof(validated)}(bank, transition, rounds, validated, adapt_scales)
    end
end

_validate_lais_transition(transition, bank, rounds, schedule) = nothing

_algorithm_proposal(algorithm::LAIS) = algorithm.bank
_algorithm_sample_budget(algorithm::LAIS) =
    _adaptive_sample_budget(algorithm.rounds, algorithm.round_size)

function _retarget_algorithm(sampler::_PreparedImportanceSampler{R,B,T,A}) where {R,B,T,A<:LAIS}
    algorithm = sampler.algorithm
    destination = MLDataDevices.cpu_device()
    transition = _with_backend_device(sampler.device) do
        retarget_transition(algorithm.transition, sampler.method_state.transition, destination)
    end
    return LAIS(current_proposal(destination, sampler);
        transition, rounds=algorithm.rounds, round_size=algorithm.round_size,
        adapt_scales=algorithm.adapt_scales)
end

mutable struct _PreparedLAIS{B,P,W,K}
    bank::B
    run_bank::B
    plan::P
    workspace::W
    transition::K
    run_transition::K
end

_lais_centres(bank::_PackedDiagonalBank{L,S,N,M,C,I,<:_ScalarGaussianLayout}) where {L,S,N,M,C,I} =
    vec(bank.locations)
_lais_centres(bank) = bank.locations

function _prepare_method_state(algorithm::LAIS, prepared_target)
    bank, plan = _prepare_fixed_population_state(algorithm)
    binding_sample = _population_binding_sample(bank)
    target = _bind_resolved_target(prepared_target, binding_sample)
    L = _resolve_packed_static_mis_logweight_type(target, bank, typeof(binding_sample))
    capacity = maximum(plan.schedule)
    prototype = bank.locations
    workspace = (
        round_samples=_allocate_packed_static_mis_samples(prototype, bank, capacity),
        round_logweights=similar(prototype, L, capacity),
        round_proposal_ids=similar(prototype, Int, capacity),
        solve_scratch=_allocate_mis_solve_scratch(prototype, bank, capacity),
        widths=_allocate_lower_widths(algorithm.adapt_scales, bank, plan, L),
    )
    committed = _prepare_lais_transition(algorithm.transition, _lais_centres(bank), target, prepared_target, L)
    run = deepcopy(committed)
    run_bank = _population_with_locations(_lower_width_bank(bank, workspace.widths),
        reshape(transition_centres(run), size(bank.locations)))
    return _PreparedLAIS(bank, run_bank, plan, workspace, committed, run)
end

_prepare_lais_transition(config, centres, target, original, L) =
    prepare_transition(config, centres, target, L)

function _allocate_random_buffers(::MLDataDevices.AbstractDevice, ::ProposalBank, state::_PreparedLAIS, sample_budget)
    normals = similar(state.bank.locations, eltype(state.bank.locations),
        size(state.bank.locations, 1) * maximum(state.plan.schedule))
    return _PopulationNormalBuffers(normals,
        _allocate_native_failure_scratch(
            normals, maximum(state.plan.schedule); capacity=sample_budget,
        ),
        _allocate_radial_buffers(normals, state.bank.family, maximum(state.plan.schedule)))
end

function current_proposal(sampler::_PreparedImportanceSampler{R,B,T,A}) where {R,B,T,A<:LAIS}
    return _current_population_proposal(sampler)
end

function current_proposal(destination::MLDataDevices.AbstractDevice,
    sampler::_PreparedImportanceSampler{R,B,T,A}) where {R,B,T,A<:LAIS}
    return _current_population_proposal(destination, sampler)
end

_copy_algorithm(::MLDataDevices.CPUDevice{Missing}, algorithm::LAIS) = deepcopy(algorithm)
function _copy_algorithm(device, algorithm::LAIS)
    return LAIS(_copy_population_bank(device, algorithm.bank);
        transition=deepcopy(algorithm.transition), rounds=algorithm.rounds, round_size=algorithm.round_size,
        adapt_scales=algorithm.adapt_scales)
end

_copy_accelerator_algorithm(device, algorithm::LAIS, ::_PreparedLAIS) = deepcopy(algorithm)

function _prepare_transferred_method_state(device, algorithm::LAIS, state::_PreparedLAIS, target)
    committed, run = _transfer_transition_pair(device, state.transition, state.run_transition, target)
    shape = size(state.bank.locations)
    bank = _copy_packed_bank(device, state.bank;
        locations=reshape(transition_centres(committed), shape))
    run_bank = _population_with_locations(_lower_width_bank(bank, state.workspace.widths),
        reshape(transition_centres(run), shape))
    workspace = map(value -> _copy_to_device(device, value), state.workspace)
    return _PreparedLAIS(bank, run_bank, _transfer_population_plan(device, state.plan),
        workspace, committed, run)
end

_transfer_transition_pair(device, committed, running, target) =
    (Adapt.adapt(device, committed), Adapt.adapt(device, running))

_transferred_backend_state(algorithm, state::_PreparedLAIS, target, random_buffers) =
    (state, target, random_buffers)
_prepared_backend_state(sampler, state::_PreparedLAIS) =
    (state, sampler.target, sampler.random_buffers, sampler.rng)
