"""A failed HAIS phase, retaining the pre-call bank and HMC state."""
struct HAISRoundError{E,D<:NamedTuple} <: Exception
    round::Int
    phase::Symbol
    cause::E
    diagnostics::D
end

function Base.showerror(io::IO, error::HAISRoundError)
    print(io, "HAIS failed in round ", error.round, " during ", error.phase, ": ")
    showerror(io, error.cause)
end

_population_round_error(::HAIS) = HAISRoundError
_population_round_phase(::HAIS) = :weight_summary
_population_mis_adaptation(::_PreparedHAIS, views) = nothing

function _population_round_views(state::_PreparedHAIS, round)
    count = state.plan.schedule[round]
    workspace = state.workspace
    return (round_size=count,
            samples=_sample_view(workspace.round_samples, 1:count),
            logweights=view(workspace.round_logweights, 1:count),
            proposal_ids=view(workspace.round_proposal_ids, 1:count),
            assignments=view(state.plan.assignments, 1:count, round))
end

function _reset_population_run!(state::_PreparedHAIS, run, committed)
    _reset_lower_widths!(run, committed, state.workspace.widths)
    copyto!(run.locations, committed.locations)
    copyto!(state.run_transition, state.transition)
    _reset_hmc_diagnostics!(state.run_transition)
    return nothing
end

function _before_population_round!(sampler, state::_PreparedHAIS, target, round, execution,
                                   transfers)
    if round == 1
        _capture_population_round(sampler.algorithm, round, :warmup,
                                  state.plan.schedule[round], 0) do
            result = _warmup_hmc!(state.run_transition, sampler.rng, execution, transfers)
            _retain_transition_rng!(sampler, result)
            copyto!(state.run_bank.locations, state.run_transition.centres)
        end
    end
    return nothing
end

@kernel function _hais_cooperation_scores_kernel!(scores, endpoints, logtargets, bank,
                                                  denominator, solve_scratch)
    endpoint = @index(Global, Linear)
    T = eltype(scores)
    sample = view(endpoints, :, endpoint)
    # All terms are evaluated at this endpoint in the unchanged pre-HMC bank.
    logmixture, _, _ = _mis_logdenominator_core(T, bank, denominator, 1,
                                                sample, solve_scratch, endpoint)
    score = logtargets[endpoint] - logmixture
    scores[endpoint] = isfinite(logmixture) && isfinite(score) ? score : T(NaN)
end

function _hais_cooperation_weights!(state, round, execution, transfers)
    workspace = state.workspace
    backend = KernelAbstractions.get_backend(workspace.cooperation_scores)
    scratch = _fused_mis_solve_scratch(workspace.cooperation_solve_scratch, backend)
    count = length(workspace.cooperation_scores)
    _hais_cooperation_scores_kernel!(backend)(workspace.cooperation_scores,
                                              state.run_transition.centres,
                                              state.run_transition.logtargets,
                                              state.run_bank,
                                              _population_denominator(state, round),
                                              scratch;
                                              ndrange=count,
                                              workgroupsize=_native_workgroupsize(execution,
                                                                                  count))
    KernelAbstractions.synchronize(backend)
    valid = all(isfinite, workspace.cooperation_scores)
    _record_device_scalar_transfer!(transfers, workspace.cooperation_scores, Bool)
    valid ||
        throw(ArgumentError("HAIS cooperation requires finite pointwise target-to-mixture scores"))
    _resampling_cdf!(workspace.cooperation_cdf, workspace.cooperation_scores, transfers)
    return nothing
end

function _hais_cooperation_resample!(state, rng, execution)
    workspace = state.workspace
    rng = _rand_uniform!(rng, workspace.cooperation_uniforms;
        threaded = _transition_threaded(execution)
    )
    try
    backend = KernelAbstractions.get_backend(workspace.ancestors)
    count = length(workspace.ancestors)
    _select_resampling_ancestors_kernel!(backend)(workspace.ancestors,
                                                  workspace.cooperation_uniforms,
                                                  workspace.cooperation_cdf,
                                                  workspace.cooperation_scores, count;
                                                  ndrange=count,
                                                  workgroupsize=_native_workgroupsize(execution,
                                                                                      count))
    KernelAbstractions.synchronize(backend)
    _gather_hmc_centres!(state.run_transition, workspace.ancestors)
    # Only locations change; each destination retains its own scale and stable ID.
    copyto!(state.run_bank.locations, state.run_transition.centres)
    return rng, nothing
    catch failure
        return rng, failure
    end
end

function _advance_prepared_population!(::Nothing, sampler, state::_PreparedHAIS, views,
                                       round, execution, transfers)
    # An all-zero lower round is retained; it does not drive upper adaptation.
    summary = _logweight_summary(views.logweights, transfers)
    completed = round
    _capture_population_round(sampler.algorithm, round, :hmc, views.round_size,
                              completed) do
        result = _hmc_sweep!(state.run_transition, sampler.rng, execution, transfers)
        _retain_transition_rng!(sampler, result)
    end
    _capture_population_round(sampler.algorithm, round, :cooperation_weights,
                              views.round_size, completed) do
        _hais_cooperation_weights!(state, round, execution, transfers)
    end
    _capture_population_round(sampler.algorithm, round, :cooperation_resampling,
                              views.round_size, completed) do
        result = _hais_cooperation_resample!(state, sampler.rng, execution)
        _retain_transition_rng!(sampler, result)
    end
    _capture_population_round(sampler.algorithm, round, :width_adaptation,
                              views.round_size, completed) do
        _adapt_lower_widths!(state.workspace.widths, state, views, round, execution)
    end
    return summary
end

_commit_population_round!(::_PreparedHAIS, bank) = nothing

function _commit_population_run!(state::_PreparedHAIS)
    state.bank, state.run_bank = state.run_bank, state.bank
    state.transition, state.run_transition = state.run_transition, state.transition
    return nothing
end

function _population_diagnostics(sampler, state::_PreparedHAIS, execution, schedule,
                                 round_ess, round_lognormalizers, transfers)
    transition = _hmc_call_diagnostics(state.run_transition, state.transition, transfers)
    count = size(state.run_bank.locations, 2)
    lower_count = _algorithm_sample_budget(sampler.algorithm)
    endpoints = Base.checked_mul(count, sampler.algorithm.rounds)
    cooperation_terms = Base.checked_mul(count, endpoints)
    return (method=:hais, execution=_execution_name(execution), threaded=sampler.threaded,
            factor_execution_policy=_use_factor_batch_mis_path(sampler.device,
                                                               state.run_bank,
                                                               _population_denominator(state,
                                                                                       firstindex(schedule)),
                                                               eltype(state.workspace.round_logweights),
                                                               sampler.factor_execution) ?
                                    :batched : :fused,
            rounds=sampler.algorithm.rounds, round_sizes=collect(schedule),
            round_ess=round_ess, round_lognormalizers=round_lognormalizers,
            target_evaluations=lower_count + transition.initial_target_evaluations +
                               transition.warmup_target_evaluations +
                               transition.production_target_evaluations,
            gradient_evaluations=transition.initial_gradient_evaluations +
                                 transition.warmup_gradient_evaluations +
                                 transition.production_gradient_evaluations,
            proposal_evaluations=Base.checked_add(Base.checked_mul(count, lower_count),
                                                  cooperation_terms),
            transition=transition,
            cooperation=(rounds=sampler.algorithm.rounds, endpoint_count=endpoints,
                         proposal_evaluations=cooperation_terms),
            failures=0, transfers=transfers)
end

function _importance_sample_cpu!(sampler, state::_PreparedHAIS, threaded)
    _importance_sample_fixed_population!(sampler, state, threaded)
end
