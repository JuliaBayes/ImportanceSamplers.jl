"""
    HamiltonianMonteCarlo(; step_size, leapfrog_steps, mass=I,
        tuning=WarmupTuning(0), target_acceptance=0.8, divergence_threshold=1000)

Native fixed-length, Metropolis-corrected HMC for vector proposal centres.
`step_size` is the positive integration step and `leapfrog_steps` is the number
of steps per trajectory. `mass` is the shared momentum covariance, not a lower
proposal covariance; use a positive scalar, scaled identity, `Diagonal`, SPD
matrix, or successful Cholesky factorization. No inverse is formed.

`WarmupTuning(W)` performs `W` upper-only sweeps and tunes one shared step size
by dual averaging, then freezes it across reuse. Mass and trajectory length
remain fixed. Scores use the existing `LogTarget` derivative interface in sampled
coordinates, including any configured transform. Scalar-layout banks are not
supported; a one-dimensional problem uses a one-element vector.

Finite energy errors above `divergence_threshold` (in log-density units) are
diagnostic only. Numerical integration failures and outside-support candidates
reject the trajectory. Invalid initial states, malformed scores and callback
failures fail the call without committing partial sampler state.
"""
struct HamiltonianMonteCarlo{S,M,T,A,D} <: AbstractMCMCTransition
    step_size::S
    leapfrog_steps::Int
    mass::M
    tuning::T
    target_acceptance::A
    divergence_threshold::D
    function HamiltonianMonteCarlo(; step_size::Real, leapfrog_steps,
                                   mass=LinearAlgebra.I,
                                   tuning::WarmupTuning=WarmupTuning(0),
                                   target_acceptance::Real=0.8,
                                   divergence_threshold::Real=1000)
        isfinite(step_size) && step_size > zero(step_size) ||
            throw(ArgumentError("HMC step size must be finite and positive"))
        leapfrog_steps isa Integer && !(leapfrog_steps isa Bool) &&
            0 < leapfrog_steps <= typemax(Int) ||
            throw(ArgumentError("HMC leapfrog steps must be a positive Int-representable integer"))
        isfinite(target_acceptance) && 0 < target_acceptance < 1 ||
            throw(ArgumentError("HMC target acceptance must lie in (0, 1)"))
        isfinite(divergence_threshold) &&
            divergence_threshold > zero(divergence_threshold) ||
            throw(ArgumentError("HMC divergence threshold must be finite and positive"))
        return new{typeof(step_size),typeof(mass),typeof(tuning),
                   typeof(target_acceptance),typeof(divergence_threshold)}(step_size,
                                                                           Int(leapfrog_steps),
                                                                           mass, tuning,
                                                                           target_acceptance,
                                                                           divergence_threshold)
    end
end

function _hmc_mass_root(mass::Real, ::Type{T}, dimension) where {T}
    value = T(mass)
    isfinite(value) && value > zero(T) ||
        throw(ArgumentError("HMC mass must remain finite and positive in sample precision"))
    return sqrt(value)
end
function _hmc_mass_root(mass::LinearAlgebra.UniformScaling, T, dimension)
    _hmc_mass_root(mass.λ, T, dimension)
end

function _hmc_mass_root(mass::LinearAlgebra.Diagonal, ::Type{T}, dimension) where {T}
    size(mass) == (dimension, dimension) ||
        throw(DimensionMismatch("HMC mass dimension mismatch"))
    values = Vector{T}(mass.diag)
    all(x -> isfinite(x) && x > zero(T), values) ||
        throw(ArgumentError("HMC diagonal mass must be finite and positive"))
    return sqrt.(values)
end

function _hmc_mass_root(mass::AbstractMatrix, ::Type{T}, dimension) where {T}
    size(mass) == (dimension, dimension) ||
        throw(DimensionMismatch("HMC mass dimension mismatch"))
    matrix = Matrix{T}(mass)
    all(isfinite, matrix) && LinearAlgebra.issymmetric(matrix) ||
        throw(ArgumentError("HMC mass must be finite and symmetric"))
    return _hmc_mass_root(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(matrix)), T,
                          dimension)
end

function _hmc_mass_root(mass::LinearAlgebra.Cholesky, ::Type{T}, dimension) where {T}
    size(mass) == (dimension, dimension) ||
        throw(DimensionMismatch("HMC mass dimension mismatch"))
    factor = Matrix{T}(mass.L)
    LinearAlgebra.issuccess(mass) && all(isfinite, factor) &&
        all(x -> x > zero(T), LinearAlgebra.diag(factor)) ||
        throw(ArgumentError("HMC mass factor must be finite and positive definite"))
    return LinearAlgebra.LowerTriangular(factor)
end
function _hmc_mass_root(mass, T, dimension)
    throw(ArgumentError("HMC mass must be a positive scalar, scaled identity, Diagonal, SPD matrix, or Cholesky factor"))
end

function _hmc_counts(::Type{E}) where {E}
    (initial_target_evaluations=0, initial_gradient_evaluations=0,
     warmup_target_evaluations=0, production_target_evaluations=0,
     warmup_gradient_evaluations=0, production_gradient_evaluations=0,
     warmup_proposals=0, production_proposals=0, accepted=0,
     warmup_accepted=0, production_accepted=0,
     warmup_acceptance_sum=zero(E), production_acceptance_sum=zero(E),
     warmup_divergences=0, production_divergences=0,
     warmup_support_rejections=0, production_support_rejections=0)
end

mutable struct _HamiltonianState{C,V,M,G,U,E,K,F,T,W,Q,B}
    centres::C
    logtargets::V
    gradients::C
    candidates::C
    candidate_values::V
    candidate_gradients::C
    normals::C
    momenta::C
    velocities::C
    whitened::C
    uniforms::U
    mass_root::M
    bound_gradient::G
    work::W
    acceptance::Q
    energy_error::Q
    failures::F
    counts::K
    cache_valid::Bool
    step_size::T
    initial_step_size::T
    leapfrog_steps::Int
    warmup_steps::Int
    n_tuned::Int
    target_acceptance::E
    divergence_threshold::E
    hbar::E
    log_step_average::E
    max_abs_energy_error::Union{Nothing,E}
    device_workspace::B
end

function _prepare_hmc_transition(config::HamiltonianMonteCarlo, centres::AbstractMatrix,
                                 prepared_target, ::Type{D}) where {D}
    T = eltype(centres)
    T <: Union{Float32,Float64} && D <: Union{Float32,Float64} ||
        throw(ArgumentError("native HMC requires Float32 or Float64 coordinates and log densities"))
    dimension, count = size(centres)
    dimension > 0 && count > 0 ||
        throw(ArgumentError("HMC requires nonempty vector centres"))
    Base.Checked.checked_mul(dimension, count)
    work = Base.Checked.checked_mul(count, config.leapfrog_steps)
    Base.Checked.checked_mul(work, config.tuning.steps)
    E = promote_type(T, D)
    step = T(config.step_size)
    acceptance, threshold = E(config.target_acceptance), E(config.divergence_threshold)
    isfinite(step) && step > zero(T) ||
        throw(ArgumentError("HMC step size is invalid in sample precision"))
    isfinite(acceptance) && zero(E) < acceptance < one(E) ||
        throw(ArgumentError("HMC target acceptance is invalid in energy precision"))
    isfinite(threshold) && threshold > zero(E) ||
        throw(ArgumentError("HMC divergence threshold is invalid in energy precision"))
    root = _hmc_mass_root(config.mass, T, dimension)
    bound = _prepare_bound_gradient(prepared_target, view(centres, :, 1),
                                    max(1, Threads.nthreads(:default)))
    return _HamiltonianState(centres, zeros(D, count), similar(centres),
                             similar(centres), Vector{D}(undef, count), similar(centres),
                             similar(centres), similar(centres), similar(centres),
                             similar(centres),
                             Vector{T}(undef, count), root, bound, zeros(Int, 5, count),
                             zeros(E, count), fill(E(NaN), count),
                             _allocate_native_target_failures(KernelAbstractions.CPU(),
                                                              count),
                             _hmc_counts(E), false, step, step, config.leapfrog_steps,
                             config.tuning.steps,
                             0, acceptance, threshold, zero(E), log(E(step)), nothing,
                             nothing)
end
function _prepare_hmc_transition(config::HamiltonianMonteCarlo, centres, target, D)
    throw(ArgumentError("HMC requires vector-layout proposal centres, including for dimension one"))
end

function prepare_transition(config::HamiltonianMonteCarlo, centres, target, D::Type)
    _prepare_hmc_transition(config, centres, target, D)
end
transition_centres(state::_HamiltonianState) = state.centres

# The two states run sequentially. Derivative preparations and the constant mass
# are shared; all position-dependent and tuning state has independent storage.
function Base.deepcopy_internal(state::_HamiltonianState, dict::IdDict)
    haskey(dict, state) && return dict[state]
    values = map(fieldnames(typeof(state))) do name
        value = getfield(state, name)
        name in (:bound_gradient, :mass_root) ? value : Base.deepcopy_internal(value, dict)
    end
    result = _HamiltonianState(values...)
    dict[state] = result
    return result
end

function Base.copyto!(destination::_HamiltonianState, source::_HamiltonianState)
    copyto!(destination.centres, source.centres)
    copyto!(destination.logtargets, source.logtargets)
    copyto!(destination.gradients, source.gradients)
    destination.counts = source.counts
    destination.cache_valid = source.cache_valid
    destination.step_size = source.step_size
    destination.initial_step_size = source.initial_step_size
    destination.n_tuned = source.n_tuned
    destination.hbar = source.hbar
    destination.log_step_average = source.log_step_average
    destination.max_abs_energy_error = source.max_abs_energy_error
    isnothing(destination.device_workspace) ||
        copyto!(destination.device_workspace, source.device_workspace)
    return destination
end

function _reset_hmc_diagnostics!(state::_HamiltonianState)
    (state.max_abs_energy_error = nothing; nothing)
end
function transition_diagnostics(state::_HamiltonianState)
    merge(state.counts,
          (max_abs_energy_error=state.max_abs_energy_error, step_size=state.step_size,
           leapfrog_steps=state.leapfrog_steps))
end

function _hmc_call_diagnostics(after::_HamiltonianState, before::_HamiltonianState,
                               transfers)
    names = keys(after.counts)
    differences = map(names) do name
        getproperty(after.counts, name) - getproperty(before.counts, name)
    end
    return merge(NamedTuple{names}(differences),
                 (max_abs_energy_error=after.max_abs_energy_error,
                  step_size=after.step_size, leapfrog_steps=after.leapfrog_steps))
end

_hmc_retarget_mass(root::Real, destination) = root * root
function _hmc_retarget_mass(root::AbstractVector, destination)
    LinearAlgebra.Diagonal(destination(Array(root)) .^ 2)
end
function _hmc_retarget_mass(root::LinearAlgebra.LowerTriangular, destination)
    LinearAlgebra.Cholesky(destination(Array(parent(root))), 'L', 0)
end
function retarget_transition(::HamiltonianMonteCarlo, state::_HamiltonianState, destination)
    return HamiltonianMonteCarlo(; step_size=state.step_size,
                                 leapfrog_steps=state.leapfrog_steps,
                                 mass=_hmc_retarget_mass(state.mass_root, destination),
                                 tuning=WarmupTuning(state.warmup_steps),
                                 target_acceptance=state.target_acceptance,
                                 divergence_threshold=state.divergence_threshold)
end

_hmc_foreach(f, count, ::_SerialCPUExecution) = foreach(i -> f(i, 1), 1:count)
_hmc_foreach(f, count, ::_ThreadedCPUExecution) = _threaded_foreach_worker(f, 1:count)
function _hmc_foreach(f, count, execution::_KernelExecution)
    _hmc_foreach(f, count, execution.cpu_execution)
end

struct _HamiltonianTrajectoryError{E} <: Exception
    centre::Int
    leapfrog_step::Int
    cause::E
end
function Base.showerror(io::IO, error::_HamiltonianTrajectoryError)
    print(io, "HMC centre ", error.centre, ", leapfrog step ", error.leapfrog_step, ": ")
    showerror(io, error.cause)
end

function _hmc_value_and_gradient!(gradient, bound, position, chain, leapfrog, slot)
    try
        fill!(gradient, eltype(gradient)(NaN))
        value = _value_and_gradient!(gradient, bound, position, slot)
        value == -Inf && return value
        isfinite(value) || throw(DomainError(value, "HMC target may not be NaN or +Inf"))
        all(isfinite, gradient) || throw(DomainError(gradient, "HMC score must be finite"))
        return value
    catch cause
        throw(_HamiltonianTrajectoryError(chain, leapfrog, cause))
    end
end

function _hmc_checked_foreach(f, state, execution)
    _reset_native_target_failures!(state.failures)
    _hmc_foreach(length(state.logtargets), execution) do chain, slot
        try
            f(chain, slot)
        catch cause
            _record_cpu_failure!(state.failures, chain, :hamiltonian, cause,
                                 catch_backtrace())
        end
    end
    failure = _take_first_native_target_failure!(state.failures)
    isnothing(failure) || throw(failure)
    return nothing
end

function _initialize_hmc!(state, execution)
    state.cache_valid && return nothing
    _hmc_checked_foreach(state, execution) do chain, slot
        position = view(state.centres, :, chain)
        all(isfinite, position) ||
            throw(DomainError(position, "initial HMC coordinates must be finite"))
        gradient = view(state.gradients, :, chain)
        value = _hmc_value_and_gradient!(gradient, state.bound_gradient, position, chain, 0,
                                         slot)
        isfinite(value) ||
            throw(DomainError(value, "initial HMC target and score must be finite"))
        state.logtargets[chain] = value
    end
    count = length(state.logtargets)
    old = state.counts
    state.counts = merge(old,
                         (initial_target_evaluations=Base.Checked.checked_add(old.initial_target_evaluations,
                                                                              count),
                          initial_gradient_evaluations=Base.Checked.checked_add(old.initial_gradient_evaluations,
                                                                                count)))
    state.cache_valid = true
    return nothing
end

function _hmc_momentum!(momentum, normal, root::Real)
    for row in eachindex(momentum)
        momentum[row] = root * normal[row]
    end
    return nothing
end
function _hmc_momentum!(momentum, normal, root::AbstractVector)
    for row in eachindex(momentum)
        momentum[row] = root[row] * normal[row]
    end
    return nothing
end
function _hmc_momentum!(momentum, normal, root::LinearAlgebra.LowerTriangular)
    LinearAlgebra.mul!(momentum, root, normal)
end

function _hmc_velocity!(velocity, whitened, momentum, root::Real)
    for row in eachindex(momentum)
        whitened[row] = momentum[row] / root
        velocity[row] = whitened[row] / root
    end
    return nothing
end
function _hmc_velocity!(velocity, whitened, momentum, root::AbstractVector)
    for row in eachindex(momentum)
        whitened[row] = momentum[row] / root[row]
        velocity[row] = whitened[row] / root[row]
    end
    return nothing
end
function _hmc_velocity!(velocity, whitened, momentum, root::LinearAlgebra.LowerTriangular)
    copyto!(whitened, momentum)
    LinearAlgebra.ldiv!(root, whitened)
    copyto!(velocity, whitened)
    LinearAlgebra.ldiv!(adjoint(root), velocity)
    return nothing
end
function _hmc_kinetic(whitened, ::Type{E}) where {E}
    value = zero(E)
    for x in whitened
        value += abs2(E(x))
    end
    return value / E(2)
end

function _hmc_trajectory!(state, chain, slot)
    E = eltype(state.acceptance)
    T = eltype(state.centres)
    position = view(state.candidates, :, chain)
    gradient = view(state.candidate_gradients, :, chain)
    momentum = view(state.momenta, :, chain)
    velocity = view(state.velocities, :, chain)
    whitened = view(state.whitened, :, chain)
    copyto!(position, view(state.centres, :, chain))
    copyto!(gradient, view(state.gradients, :, chain))
    _hmc_momentum!(momentum, view(state.normals, :, chain), state.mass_root)
    _hmc_velocity!(velocity, whitened, momentum, state.mass_root)
    initial_kinetic = _hmc_kinetic(whitened, E)
    if !(all(isfinite, momentum) && all(isfinite, velocity) && isfinite(initial_kinetic))
        state.work[4, chain] = 1
        return nothing
    end
    step = state.step_size
    halfstep = step / T(2)
    for row in eachindex(momentum)
        momentum[row] += halfstep * gradient[row]
    end
    value = state.logtargets[chain]
    for leapfrog in 1:(state.leapfrog_steps)
        _hmc_velocity!(velocity, whitened, momentum, state.mass_root)
        if !(all(isfinite, momentum) && all(isfinite, velocity))
            state.work[4, chain] = 1
            return nothing
        end
        for row in eachindex(position)
            position[row] += step * velocity[row]
        end
        if !all(isfinite, position)
            state.work[4, chain] = 1
            return nothing
        end
        value = _hmc_value_and_gradient!(gradient, state.bound_gradient, position, chain,
                                         leapfrog, slot)
        state.work[1, chain] += 1
        # Combined sources request a score even outside support; explicit sources
        # do not. Their shared helper records which operation was requested.
        state.work[2, chain] += _hmc_gradient_requested(state.bound_gradient, value)
        if value == -Inf
            state.work[5, chain] = 1
            return nothing
        end
        kick = leapfrog == state.leapfrog_steps ? halfstep : step
        for row in eachindex(momentum)
            momentum[row] += kick * gradient[row]
        end
    end
    _hmc_velocity!(velocity, whitened, momentum, state.mass_root)
    final_kinetic = _hmc_kinetic(whitened, E)
    error = (E(state.logtargets[chain]) - E(value)) + (final_kinetic - initial_kinetic)
    if !(all(isfinite, momentum) && all(isfinite, velocity) && isfinite(final_kinetic) &&
         isfinite(error))
        state.work[4, chain] = 1
        return nothing
    end
    state.energy_error[chain] = abs(error)
    state.work[4, chain] = abs(error) > state.divergence_threshold
    logacceptance = min(zero(E), -error)
    state.acceptance[chain] = exp(logacceptance)
    if log(E(state.uniforms[chain])) < logacceptance
        copyto!(view(state.centres, :, chain), position)
        copyto!(view(state.gradients, :, chain), gradient)
        state.logtargets[chain] = value
        state.work[3, chain] = 1
    end
    return nothing
end

_hmc_gradient_requested(::_BoundGradient, value) = isfinite(value) ? 1 : 0
_hmc_gradient_requested(::Union{_BoundDIGradient,_BoundLDPGradient}, value) = 1

function _hmc_add_sweep_counts!(state, count, values, gradients, accepted,
                                divergences, support, acceptance_sum, warmup)
    old = state.counts
    add = Base.Checked.checked_add
    if warmup
        state.counts = merge(old,
                             (warmup_target_evaluations=add(old.warmup_target_evaluations,
                                                            values),
                              warmup_gradient_evaluations=add(old.warmup_gradient_evaluations,
                                                              gradients),
                              warmup_proposals=add(old.warmup_proposals, count),
                              warmup_accepted=add(old.warmup_accepted, accepted),
                              warmup_divergences=add(old.warmup_divergences, divergences),
                              warmup_support_rejections=add(old.warmup_support_rejections,
                                                            support),
                              warmup_acceptance_sum=old.warmup_acceptance_sum +
                                                    acceptance_sum,
                              accepted=add(old.accepted, accepted)))
    else
        state.counts = merge(old,
                             (production_target_evaluations=add(old.production_target_evaluations,
                                                                values),
                              production_gradient_evaluations=add(old.production_gradient_evaluations,
                                                                  gradients),
                              production_proposals=add(old.production_proposals, count),
                              production_accepted=add(old.production_accepted, accepted),
                              production_divergences=add(old.production_divergences,
                                                         divergences),
                              production_support_rejections=add(old.production_support_rejections,
                                                                support),
                              production_acceptance_sum=old.production_acceptance_sum +
                                                        acceptance_sum,
                              accepted=add(old.accepted, accepted)))
    end
    return nothing
end

function _hmc_record_sweep!(state, warmup)
    values, gradients, accepted, divergences, support = ntuple(row -> sum(view(state.work,
                                                                               row, :)), 5)
    _hmc_add_sweep_counts!(state, length(state.logtargets), values, gradients,
                           accepted, divergences, support, sum(state.acceptance), warmup)
    for error in state.energy_error
        if isfinite(error)
            state.max_abs_energy_error = isnothing(state.max_abs_energy_error) ? error :
                                         max(state.max_abs_energy_error, error)
        end
    end
    return nothing
end

function _hmc_sweep!(state::_HamiltonianState, rng, execution, transfers; warmup=false)
    _initialize_hmc!(state, execution)
    Random.randn!(rng, state.normals)
    Random.rand!(rng, state.uniforms)
    fill!(state.work, 0)
    fill!(state.acceptance, zero(eltype(state.acceptance)))
    fill!(state.energy_error, eltype(state.energy_error)(NaN))
    _hmc_checked_foreach(state, execution) do chain, slot
        _hmc_trajectory!(state, chain, slot)
    end
    _hmc_record_sweep!(state, warmup)
    return nothing
end

function _hmc_tune_step!(state)
    E = eltype(state.acceptance)
    m = E(state.n_tuned)
    weight = inv(m + E(10))
    state.hbar = (one(E) - weight) * state.hbar +
                 weight * (state.target_acceptance -
                           sum(state.acceptance) / E(length(state.acceptance)))
    logstep = log(E(state.initial_step_size)) + log(E(10)) - sqrt(m) / E(0.05) * state.hbar
    average_weight = m^(-E(0.75))
    state.log_step_average = average_weight * logstep +
                             (one(E) - average_weight) * state.log_step_average
    next = state.n_tuned == state.warmup_steps ? exp(state.log_step_average) : exp(logstep)
    step = eltype(state.centres)(next)
    isfinite(next) && isfinite(step) && step > zero(step) ||
        throw(DomainError(next, "HMC dual averaging produced an unrepresentable step size"))
    state.step_size = step
    return nothing
end

function _warmup_hmc!(state::_HamiltonianState, rng, execution, transfers)
    _initialize_hmc!(state, execution)
    while state.n_tuned < state.warmup_steps
        _hmc_sweep!(state, rng, execution, transfers; warmup=true)
        state.n_tuned = Base.Checked.checked_add(state.n_tuned, 1)
        _hmc_tune_step!(state)
    end
    return nothing
end
function transition!(state::_HamiltonianState, target, rng, execution, transfers)
    _warmup_hmc!(state, rng, execution, transfers)
    _hmc_sweep!(state, rng, execution, transfers)
    return nothing
end

function _gather_hmc_centres!(state::_HamiltonianState, ancestors)
    source = (centres=state.centres, gradients=state.gradients, values=state.logtargets)
    scratch = (centres=state.candidates, gradients=state.candidate_gradients,
               values=state.candidate_values)
    backend = KernelAbstractions.get_backend(state.centres)
    _gather_resampled_samples_kernel!(backend)(scratch, source, ancestors;
                                               ndrange=length(ancestors), workgroupsize=64)
    KernelAbstractions.synchronize(backend)
    copyto!(state.centres, state.candidates)
    copyto!(state.gradients, state.candidate_gradients)
    copyto!(state.logtargets, state.candidate_values)
    return nothing
end
