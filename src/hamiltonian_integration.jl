# Keep derivative metadata out of the legacy custom-transition interface.
function _validate_lais_transition(config::HamiltonianMonteCarlo, bank, rounds, schedule)
    count = length(bank.proposals)
    budget = _adaptive_sample_budget(rounds, schedule)
    proposal = first(bank.proposals)
    proposal isa Union{_NativeRadialProposal,_LaplaceProposal} ||
        throw(ArgumentError("HMC requires native Gaussian, Student-t or Laplace vector proposals"))
    proposal.location isa AbstractVector ||
        throw(ArgumentError("HMC requires vector-layout proposal centres"))
    Base.checked_mul(length(proposal.location), budget)
    moves = Base.checked_mul(count, Base.checked_add(rounds, config.tuning.steps))
    upper = Base.checked_add(count, Base.checked_mul(moves, config.leapfrog_steps))
    Base.checked_add(budget, upper)
    Base.checked_mul(count, budget)
    Base.checked_add(budget, 1)
    Base.checked_add(rounds, 1)
    return nothing
end

function _transfer_transition_pair(device, committed::_HamiltonianState,
                                   running::_HamiltonianState, target)
    # Placement is allowed only before first execution. Both numerical states
    # start equal, and their sequential runs share one backend AD preparation.
    transferred = _transfer_hmc_transition(device, committed, target)
    return transferred, deepcopy(transferred)
end

function _hmc_population_execution(sampler, threaded)
    execution = threaded ? _ThreadedCPUExecution() : _SerialCPUExecution()
    return sampler.device isa MLDataDevices.AbstractAcceleratorDevice ?
           _KernelExecution(execution) : execution
end
function _population_execution(sampler, threaded,
                               ::_PreparedLAIS{B,P,W,K}) where {B,P,W,K<:_HamiltonianState}
    _hmc_population_execution(sampler, threaded)
end
function _population_execution(sampler, threaded, ::_PreparedHAIS)
    _hmc_population_execution(sampler, threaded)
end

function _prepare_lais_transition(config::HamiltonianMonteCarlo, centres, target, original,
                                  L)
    _prepare_hmc_transition(config, centres, original, L)
end

_reset_transition_diagnostics!(state::_HamiltonianState) = _reset_hmc_diagnostics!(state)

function _transition_call_diagnostics(backend, after::_HamiltonianState,
                                      before::_HamiltonianState, transfers)
    _hmc_call_diagnostics(after, before, transfers)
end

function _transition_run_extras(::_HamiltonianState, counts)
    (gradient_evaluations=counts.initial_gradient_evaluations +
                          counts.warmup_gradient_evaluations +
                          counts.production_gradient_evaluations,)
end

function _before_population_round!(sampler,
                                   state::_PreparedLAIS{B,P,W,K}, target, round, execution,
                                   transfers) where {B,P,W,K<:_HamiltonianState}
    size = state.plan.schedule[round]
    _capture_population_round(sampler.algorithm, round, :warmup, size, round - 1) do
        _warmup_hmc!(state.run_transition, sampler.rng, execution, transfers)
    end
    _capture_population_round(sampler.algorithm, round, :hmc, size, round - 1) do
        _hmc_sweep!(state.run_transition, sampler.rng, execution, transfers)
    end
    return nothing
end
