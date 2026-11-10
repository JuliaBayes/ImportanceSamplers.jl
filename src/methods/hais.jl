"""
    HAIS(bank; transition, rounds, round_size)

Configure Hamiltonian adaptive importance sampling with a fixed native Gaussian,
Student-t, or independent Laplace population and a [`HamiltonianMonteCarlo`](@ref)
transition. Each round draws and weights lower samples before moving the upper
centres and selecting their ancestors independently with replacement.

Lower samples use the normalized current-round deterministic mixture. Cooperation
uses pointwise target-to-old-mixture ratios only as adaptation scores: HMC
endpoints need not follow that mixture, and cooperation is not target invariant.
The final adapted bank is retained even though it generated no returned draw.
Scales, Student-t degrees of freedom, and destination proposal IDs remain fixed.

`round_size` is a positive `Int` or one positive entry per round in a `Vector{Int}`.
Every round allocates equally across equal, positive-mass proposals. Optional HMC
warmup runs once before the first lower round and performs no cooperation.
"""
struct HAIS{B<:ProposalBank,K<:HamiltonianMonteCarlo,S} <: _FixedPopulationSampler
    bank::B
    transition::K
    rounds::Int
    round_size::S

    function HAIS(bank::B; transition::K, rounds,
                  round_size) where {B<:ProposalBank,K<:HamiltonianMonteCarlo}
        schedule = _validate_adaptive_schedule(rounds, round_size)
        masses = bank.masses
        all(>(zero(eltype(masses))), masses) && all(==(first(masses)), masses) ||
            throw(ArgumentError("HAIS requires equal positive proposal masses"))
        count = length(bank.proposals)
        sizes = schedule isa Int ? (schedule,) : schedule
        all(n -> n % count == 0, sizes) ||
            throw(ArgumentError("HAIS round sizes must divide equally across proposals"))
        _validate_hais_workcounts(count, rounds, schedule, transition)
        return new{B,K,typeof(schedule)}(bank, transition, rounds, schedule)
    end
end

function _validate_hais_workcounts(count, rounds, schedule, transition)
    budget = _adaptive_sample_budget(rounds, schedule)
    capacity = schedule isa Int ? schedule : maximum(schedule)
    try
        Base.checked_add(budget, 1)
        Base.checked_add(rounds, 1)
        Base.checked_mul(capacity, rounds)
        endpoints = Base.checked_mul(count, rounds)
        cooperation = Base.checked_mul(count, endpoints)
        lower = Base.checked_mul(count, budget)
        Base.checked_add(lower, cooperation)
        sweeps = Base.checked_add(rounds, transition.tuning.steps)
        upper = Base.checked_mul(count, Base.checked_mul(sweeps, transition.leapfrog_steps))
        Base.checked_add(budget, Base.checked_add(count, upper))
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("HAIS allocation or work count exceeds Int"))
    end
    return nothing
end

_algorithm_proposal(algorithm::HAIS) = algorithm.bank
function _algorithm_sample_budget(algorithm::HAIS)
    _adaptive_sample_budget(algorithm.rounds, algorithm.round_size)
end

mutable struct _PreparedHAIS{B,P,W,K}
    bank::B
    run_bank::B
    plan::P
    workspace::W
    transition::K
    run_transition::K
end

function _prepare_method_state(algorithm::HAIS, prepared_target)
    bank = _prepare_population_bank(algorithm.bank)
    _lais_centres(bank) isa AbstractMatrix ||
        throw(ArgumentError("HAIS requires vector-valued proposal centres"))
    budget = _algorithm_sample_budget(algorithm)
    capacity = algorithm.round_size isa Int ? algorithm.round_size :
               maximum(algorithm.round_size)
    try
        Base.checked_mul(size(bank.locations, 1), budget)
        Base.checked_mul(size(bank.locations, 1), capacity)
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("HAIS coordinate allocation exceeds Int"))
    end
    schedule = _resolve_adaptive_schedule(algorithm.rounds, algorithm.round_size)
    plan = _deterministic_allocation_plan(bank, algorithm.bank.masses[bank.proposal_ids],
                                          schedule)
    binding_sample = _population_binding_sample(bank)
    target = _bind_resolved_target(prepared_target, binding_sample)
    L = _resolve_packed_static_mis_logweight_type(target, bank, typeof(binding_sample))
    prototype = bank.locations
    count = size(prototype, 2)
    workspace = (round_samples=_allocate_packed_static_mis_samples(prototype, bank,
                                                                   capacity),
                 round_logweights=similar(prototype, L, capacity),
                 round_proposal_ids=similar(prototype, Int, capacity),
                 solve_scratch=_allocate_mis_solve_scratch(prototype, bank, capacity),
                 cooperation_scores=similar(prototype, L, count),
                 cooperation_cdf=similar(prototype, L, count),
                 cooperation_uniforms=similar(prototype, L, count),
                 ancestors=similar(prototype, Int, count),
                 cooperation_solve_scratch=_allocate_mis_solve_scratch(prototype, bank,
                                                                       count))
    # Unlike LAIS, the scoring bank must not alias the moving HMC endpoints.
    committed = _prepare_hmc_transition(algorithm.transition, copy(bank.locations),
                                        prepared_target, L)
    run = deepcopy(committed)
    return _PreparedHAIS(bank, _population_run_bank(bank), plan, workspace, committed, run)
end

function _prepare_method_state(algorithm::HAIS{<:ProposalBank{<:AbstractVector{<:_GammaProposal}}},
                               prepared_target)
    throw(ArgumentError("HAIS requires a translation family, not an unshifted Gamma family"))
end

function _allocate_random_buffers(::MLDataDevices.AbstractDevice, ::ProposalBank,
                                  state::_PreparedHAIS, sample_budget)
    normals = similar(state.bank.locations, eltype(state.bank.locations),
                      Base.checked_mul(size(state.bank.locations, 1),
                                       maximum(state.plan.schedule)))
    return _PopulationNormalBuffers(normals,
                                    _allocate_native_failure_scratch(normals,
                                                                     maximum(state.plan.schedule);
                                                                     capacity=sample_budget),
                                    _allocate_radial_buffers(normals, state.bank.family,
                                                             maximum(state.plan.schedule)))
end

function current_proposal(sampler::_PreparedImportanceSampler{R,B,T,A}) where {R,B,T,
                                                                               A<:HAIS}
    return _current_population_proposal(sampler)
end

function current_proposal(destination::MLDataDevices.AbstractDevice,
                          sampler::_PreparedImportanceSampler{R,B,T,A}) where {R,B,T,
                                                                               A<:HAIS}
    return _current_population_proposal(destination, sampler)
end

function _retarget_algorithm(sampler::_PreparedImportanceSampler{R,B,T,A}) where {R,B,T,
                                                                                  A<:HAIS}
    algorithm = sampler.algorithm
    destination = MLDataDevices.cpu_device()
    transition = _with_backend_device(sampler.device) do
        retarget_transition(algorithm.transition, sampler.method_state.transition,
                            destination)
    end
    return HAIS(current_proposal(destination, sampler);
                transition, rounds=algorithm.rounds, round_size=algorithm.round_size)
end

_copy_algorithm(::MLDataDevices.CPUDevice{Missing}, algorithm::HAIS) = deepcopy(algorithm)
function _copy_algorithm(device, algorithm::HAIS)
    return HAIS(_copy_population_bank(device, algorithm.bank);
                transition=deepcopy(algorithm.transition), rounds=algorithm.rounds,
                round_size=algorithm.round_size)
end

_copy_accelerator_algorithm(device, algorithm::HAIS, ::_PreparedHAIS) = deepcopy(algorithm)

# Device paths are enabled only after the full HMC and cooperation path is validated.

function _prepare_transferred_method_state(device, algorithm::HAIS, state::_PreparedHAIS,
                                           target)
    committed, run = _transfer_transition_pair(device, state.transition,
                                               state.run_transition, target)
    bank = _copy_packed_bank(device, state.bank)
    run_bank = _population_run_bank(bank)
    copyto!(run_bank.locations, state.run_bank.locations)
    workspace = map(value -> _copy_to_device(device, value), state.workspace)
    return _PreparedHAIS(bank, run_bank, _transfer_population_plan(device, state.plan),
                         workspace, committed, run)
end

function _transferred_backend_state(algorithm, state::_PreparedHAIS, target, random_buffers)
    (state, target, random_buffers)
end
function _prepared_backend_state(sampler, state::_PreparedHAIS)
    (state, sampler.target, sampler.random_buffers, sampler.rng)
end
