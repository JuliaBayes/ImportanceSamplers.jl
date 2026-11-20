# Native device execution shares one phase engine between explicit scores and
# batched AD. Only the derivative evaluation differs between the two routes.
struct _HamiltonianDeviceWorkspace{T,I}
    tuning::T
    summary::I
    scalars::T
end

function Base.copyto!(destination::_HamiltonianDeviceWorkspace,
                      source::_HamiltonianDeviceWorkspace)
    copyto!(destination.tuning, source.tuning)
    return destination
end

_hmc_device_supported(device) = device isa MLDataDevices.AbstractCPUDevice

_hmc_trajectory_kernel(backend) = _hmc_device_trajectory_kernel!(backend)

function _preflight_transition(device, state::_HamiltonianState, target)
    _hmc_device_supported(device) ||
        throw(SamplerDeviceError(device, :hamiltonian_backend_unsupported))
    backend = KernelAbstractions.get_backend(state.centres)
    kernel = state.bound_gradient isa _BoundBatchGradient ?
             _hmc_device_start_kernel!(backend) : _hmc_trajectory_kernel(backend)
    _preflight_kernel_argument(device, kernel, _hmc_device_arrays(state))
    if !(state.bound_gradient isa _BoundBatchGradient)
        _preflight_kernel_argument(device, kernel, state.bound_gradient)
    end
    return nothing
end

function _transfer_hmc_transition(device, state::_HamiltonianState, target)
    _hmc_device_supported(device) ||
        throw(SamplerDeviceError(device, :hamiltonian_backend_unsupported))
    centres = _copy_to_device(device, state.centres)
    values = _copy_to_device(device, state.logtargets)
    bound = _prepare_accelerator_gradient(target, centres, values)
    E = eltype(state.acceptance)
    workspace = _HamiltonianDeviceWorkspace(_copy_to_device(device,
                                                            E[state.step_size, state.hbar,
                                                              state.log_step_average]),
                                            _copy_to_device(device, zeros(Int, 8)),
                                            _copy_to_device(device, zeros(E, 5)))
    fields = map(fieldnames(typeof(state))) do name
        value = getfield(state, name)
        name === :centres && return centres
        name === :logtargets && return values
        name === :bound_gradient && return bound
        name === :device_workspace && return workspace
        name === :work && return _copy_to_device(device, zeros(Int, 8, size(centres, 2)))
        name === :failures && return _NoNativeTargetFailures()
        value isa AbstractArray && return _copy_to_device(device, value)
        return value
    end
    return _HamiltonianState(fields...)
end

_hmc_resident_mass(root) = root
_hmc_resident_mass(root::LinearAlgebra.LowerTriangular) = parent(root)

function _hmc_resident_state(state::_HamiltonianState)
    (state.centres, state.logtargets, state.gradients, state.candidates,
     state.candidate_values, state.candidate_gradients, state.normals, state.momenta,
     state.velocities, state.whitened, state.uniforms, _hmc_resident_mass(state.mass_root),
     state.bound_gradient, state.work, state.acceptance, state.energy_error,
     state.device_workspace)
end

function _hmc_device_arrays(state)
    (centres=state.centres, logtargets=state.logtargets, gradients=state.gradients,
     candidates=state.candidates, candidate_values=state.candidate_values,
     candidate_gradients=state.candidate_gradients, normals=state.normals,
     momenta=state.momenta, velocities=state.velocities, whitened=state.whitened,
     uniforms=state.uniforms, mass_root=state.mass_root, work=state.work,
     acceptance=state.acceptance, energy_error=state.energy_error,
     tuning=state.device_workspace.tuning)
end

# Small dense shared masses use triangular substitution inside each trajectory.
# No per-centre factor copy or inverse is constructed.
function _hmc_device_momentum!(momentum, normal, root::LinearAlgebra.LowerTriangular)
    for row in eachindex(momentum)
        value = zero(eltype(momentum))
        for column = 1:row
            value += root[row, column] * normal[column]
        end
        momentum[row] = value
    end
    return nothing
end
_hmc_device_momentum!(momentum, normal, root) = _hmc_momentum!(momentum, normal, root)

function _hmc_device_velocity!(velocity, whitened, momentum,
                               root::LinearAlgebra.LowerTriangular)
    for row in eachindex(momentum)
        value = momentum[row]
        for column = 1:(row - 1)
            value -= root[row, column] * whitened[column]
        end
        whitened[row] = value / root[row, row]
    end
    for row in reverse(eachindex(momentum))
        value = whitened[row]
        for column = (row + 1):length(momentum)
            value -= root[column, row] * velocity[column]
        end
        velocity[row] = value / root[row, row]
    end
    return nothing
end
function _hmc_device_velocity!(velocity, whitened, momentum, root)
    _hmc_velocity!(velocity, whitened, momentum, root)
end

@inline function _hmc_device_deactivate!(arrays, chain, divergence, support)
    arrays.work[4, chain] = divergence
    arrays.work[5, chain] = support
    arrays.work[6, chain] = 0
    # Fixed-width AD must never revisit a nonfinite or out-of-support endpoint.
    copyto!(view(arrays.candidates, :, chain), view(arrays.centres, :, chain))
    return nothing
end

@inline function _hmc_device_start!(arrays, chain)
    E = eltype(arrays.acceptance)
    position = view(arrays.candidates, :, chain)
    gradient = view(arrays.candidate_gradients, :, chain)
    momentum = view(arrays.momenta, :, chain)
    velocity = view(arrays.velocities, :, chain)
    whitened = view(arrays.whitened, :, chain)
    copyto!(position, view(arrays.centres, :, chain))
    copyto!(gradient, view(arrays.gradients, :, chain))
    _hmc_device_momentum!(momentum, view(arrays.normals, :, chain), arrays.mass_root)
    _hmc_device_velocity!(velocity, whitened, momentum, arrays.mass_root)
    kinetic = _hmc_kinetic(whitened, E)
    arrays.energy_error[chain] = kinetic
    arrays.work[6, chain] = 1
    if all(isfinite, momentum) && all(isfinite, velocity) && isfinite(kinetic)
        halfstep = eltype(position)(arrays.tuning[1]) / eltype(position)(2)
        for row in eachindex(momentum)
            momentum[row] += halfstep * gradient[row]
        end
    else
        _hmc_device_deactivate!(arrays, chain, 1, 0)
        arrays.energy_error[chain] = E(NaN)
    end
end

@kernel function _hmc_device_start_kernel!(arrays)
    _hmc_device_start!(arrays, @index(Global, Linear))
end

@inline function _hmc_device_drift!(arrays, chain)
    if !iszero(arrays.work[6, chain])
        position = view(arrays.candidates, :, chain)
        momentum = view(arrays.momenta, :, chain)
        velocity = view(arrays.velocities, :, chain)
        _hmc_device_velocity!(velocity, view(arrays.whitened, :, chain), momentum,
                              arrays.mass_root)
        step = eltype(position)(arrays.tuning[1])
        for row in eachindex(position)
            position[row] += step * velocity[row]
        end
        if !(all(isfinite, momentum) && all(isfinite, velocity) && all(isfinite, position))
            _hmc_device_deactivate!(arrays, chain, 1, 0)
            arrays.energy_error[chain] = eltype(arrays.energy_error)(NaN)
        end
    end
end

@kernel function _hmc_device_drift_kernel!(arrays)
    _hmc_device_drift!(arrays, @index(Global, Linear))
end

@inline function _hmc_device_explicit_gradient!(values, gradients, locations, bound,
                                                work, chain, initial)
    if initial || !iszero(work[6, chain])
        fill!(view(gradients, :, chain), eltype(gradients)(NaN))
        value = _value_and_gradient!(view(gradients, :, chain), bound,
                                     view(locations, :, chain), chain)
        values[chain] = value
        work[1, chain] += 1
        work[2, chain] += _hmc_gradient_requested(bound, value)
    end
end

@kernel function _hmc_device_explicit_gradient_kernel!(values, gradients, locations, bound,
                                                       work, initial)
    _hmc_device_explicit_gradient!(values, gradients, locations, bound, work,
                                   @index(Global, Linear), initial)
end

function _hmc_device_evaluate!(state, bound, initial, backend)
    locations = initial ? state.centres : state.candidates
    values = initial ? state.logtargets : state.candidate_values
    gradients = initial ? state.gradients : state.candidate_gradients
    _hmc_device_explicit_gradient_kernel!(backend)(values, gradients, locations, bound,
                                                   state.work, initial;
                                                   ndrange=length(values), workgroupsize=64)
    return nothing
end

@kernel function _hmc_device_batch_work_kernel!(work)
    chain = @index(Global, Linear)
    work[1, chain] += 1
    work[2, chain] += 1
end

function _hmc_device_evaluate!(state, bound::_BoundBatchGradient, initial, backend)
    locations = initial ? state.centres : state.candidates
    values = initial ? state.logtargets : state.candidate_values
    gradients = initial ? state.gradients : state.candidate_gradients
    fill!(gradients, eltype(gradients)(NaN))
    _batch_value_and_gradient!(values, gradients, bound, locations)
    _hmc_device_batch_work_kernel!(backend)(state.work; ndrange=length(values),
                                            workgroupsize=64)
    return nothing
end

@kernel function _hmc_device_validate_initial_kernel!(arrays, coordinates_only)
    chain = @index(Global, Linear)
    if !all(isfinite, view(arrays.centres, :, chain))
        arrays.work[7, chain] = 1
    elseif !coordinates_only
        if !isfinite(arrays.logtargets[chain])
            arrays.work[7, chain] = 2
        elseif !all(isfinite, view(arrays.gradients, :, chain))
            arrays.work[7, chain] = 3
        end
    end
end

@inline function _hmc_device_kick!(arrays, chain, leapfrog, steps, threshold)
    if !iszero(arrays.work[6, chain])
        E = eltype(arrays.acceptance)
        value = arrays.candidate_values[chain]
        gradient = view(arrays.candidate_gradients, :, chain)
        if value == -Inf
            _hmc_device_deactivate!(arrays, chain, 0, 1)
            arrays.energy_error[chain] = E(NaN)
        elseif !isfinite(value) || !all(isfinite, gradient)
            arrays.work[7, chain] = isfinite(value) ? 3 : 2
            arrays.work[8, chain] = leapfrog
            _hmc_device_deactivate!(arrays, chain, 0, 0)
            arrays.energy_error[chain] = E(NaN)
        else
            momentum = view(arrays.momenta, :, chain)
            kick = eltype(momentum)(arrays.tuning[1]) /
                   (leapfrog == steps ? eltype(momentum)(2) : one(eltype(momentum)))
            for row in eachindex(momentum)
                momentum[row] += kick * gradient[row]
            end
            if leapfrog == steps
                velocity = view(arrays.velocities, :, chain)
                whitened = view(arrays.whitened, :, chain)
                _hmc_device_velocity!(velocity, whitened, momentum, arrays.mass_root)
                kinetic = _hmc_kinetic(whitened, E)
                error = (E(arrays.logtargets[chain]) - E(value)) +
                        (kinetic - arrays.energy_error[chain])
                arrays.energy_error[chain] = E(NaN)
                if !(all(isfinite, momentum) && all(isfinite, velocity) &&
                     isfinite(kinetic) && isfinite(error))
                    _hmc_device_deactivate!(arrays, chain, 1, 0)
                else
                    arrays.energy_error[chain] = abs(error)
                    arrays.work[4, chain] = abs(error) > threshold
                    logacceptance = min(zero(E), -error)
                    arrays.acceptance[chain] = exp(logacceptance)
                    if log(E(arrays.uniforms[chain])) < logacceptance
                        copyto!(view(arrays.centres, :, chain),
                                view(arrays.candidates, :, chain))
                        copyto!(view(arrays.gradients, :, chain), gradient)
                        arrays.logtargets[chain] = value
                        arrays.work[3, chain] = 1
                    end
                end
            end
        end
    end
end

@kernel function _hmc_device_kick_kernel!(arrays, leapfrog, steps, threshold)
    _hmc_device_kick!(arrays, @index(Global, Linear), leapfrog, steps, threshold)
end

@kernel function _hmc_device_trajectory_kernel!(arrays, bound, steps, threshold)
    chain = @index(Global, Linear)
    _hmc_device_start!(arrays, chain)
    for leapfrog = 1:steps
        _hmc_device_drift!(arrays, chain)
        _hmc_device_explicit_gradient!(arrays.candidate_values, arrays.candidate_gradients,
                                       arrays.candidates, bound, arrays.work, chain, false)
        _hmc_device_kick!(arrays, chain, leapfrog, steps, threshold)
    end
end

# Cheap explicit scores reuse the phase arithmetic in one resident trajectory;
# batched AD retains dependent launches across centres at each leapfrog step.
function _launch_hmc_device_sweep!(state, bound, backend)
    _hmc_trajectory_kernel(backend)(
        _hmc_device_arrays(state),
        bound,
        state.leapfrog_steps,
        state.divergence_threshold;
        ndrange = length(state.logtargets),
        workgroupsize = 64,
    )
    return nothing
end

function _launch_hmc_device_sweep!(state, bound::_BoundBatchGradient, backend)
    arrays, count = _hmc_device_arrays(state), length(state.logtargets)
    _hmc_device_start_kernel!(backend)(arrays; ndrange=count, workgroupsize=64)
    for leapfrog = 1:(state.leapfrog_steps)
        _hmc_device_drift_kernel!(backend)(arrays; ndrange=count, workgroupsize=64)
        _hmc_device_evaluate!(state, bound, false, backend)
        _hmc_device_kick_kernel!(backend)(arrays, leapfrog, state.leapfrog_steps,
                                          state.divergence_threshold; ndrange=count,
                                          workgroupsize=64)
    end
    return nothing
end

# One fixed-width summary is copied only at a complete sweep boundary. Pooling
# and dual averaging execute here on device, not from the copied statistics.
@kernel function _hmc_device_summary_kernel!(summary, scalars, arrays, warmup,
                                             tuned, warmup_steps, initial_step,
                                             target_acceptance)
    _ = @index(Global, Linear)
    E = eltype(scalars)
    for row in eachindex(summary)
        summary[row] = 0
    end
    acceptance = zero(E)
    maximum_error = E(NaN)
    for chain in eachindex(arrays.acceptance)
        for row = 1:5
            summary[row] += arrays.work[row, chain]
        end
        if iszero(summary[6]) && !iszero(arrays.work[7, chain])
            summary[6] = chain
            summary[7] = arrays.work[7, chain]
            summary[8] = arrays.work[8, chain]
        end
        acceptance += arrays.acceptance[chain]
        error = arrays.energy_error[chain]
        if isfinite(error)
            maximum_error = isnan(maximum_error) ? error : max(maximum_error, error)
        end
    end
    if warmup && iszero(summary[6])
        m = E(tuned)
        weight = inv(m + E(10))
        hbar = (one(E) - weight) * arrays.tuning[2] +
               weight * (target_acceptance - acceptance / E(length(arrays.acceptance)))
        logstep = log(E(initial_step)) + log(E(10)) - sqrt(m) / E(0.05) * hbar
        average_weight = m^(-E(0.75))
        average = average_weight * logstep + (one(E) - average_weight) * arrays.tuning[3]
        next = tuned == warmup_steps ? exp(average) : exp(logstep)
        step = eltype(arrays.centres)(next)
        if isfinite(next) && isfinite(step) && step > zero(step)
            arrays.tuning[1] = step
            arrays.tuning[2] = hbar
            arrays.tuning[3] = average
        else
            summary[6] = 1
            summary[7] = 4
        end
    end
    scalars[1] = acceptance
    scalars[2] = maximum_error
    for row = 1:3
        scalars[row + 2] = arrays.tuning[row]
    end
end

function _hmc_device_summary!(state, backend, transfers; warmup=false)
    _record_gradient_transfers!(transfers, state.bound_gradient)
    workspace = state.device_workspace
    tuned = warmup ? Base.Checked.checked_add(state.n_tuned, 1) : state.n_tuned
    _hmc_device_summary_kernel!(backend)(workspace.summary, workspace.scalars,
                                         _hmc_device_arrays(state), warmup, tuned,
                                         state.warmup_steps,
                                         state.initial_step_size, state.target_acceptance;
                                         ndrange=1, workgroupsize=1)
    KernelAbstractions.synchronize(backend)
    summary, scalars = Array(workspace.summary), Array(workspace.scalars)
    _record_device_scalar_transfer!(transfers, workspace.summary, NTuple{8,Int})
    _record_device_scalar_transfer!(transfers, workspace.scalars, NTuple{5,eltype(scalars)})
    if !iszero(summary[6])
        message = summary[7] == 1 ? "initial HMC coordinates must be finite" :
                  summary[7] == 2 ? "HMC target must be finite (except candidate -Inf)" :
                  summary[7] == 3 ? "HMC score must be finite" :
                  "HMC dual averaging produced an unrepresentable step size"
        throw(DomainError((centre=summary[6], leapfrog=summary[8]), message))
    end
    state.step_size = eltype(state.centres)(scalars[3])
    state.hbar, state.log_step_average = scalars[4], scalars[5]
    warmup && (state.n_tuned = tuned)
    return summary, scalars
end

function _initialize_hmc_device!(state, backend, transfers)
    state.cache_valid && return nothing
    fill!(state.work, 0)
    fill!(state.acceptance, zero(eltype(state.acceptance)))
    fill!(state.energy_error, eltype(state.energy_error)(NaN))
    arrays = _hmc_device_arrays(state)
    count = length(state.logtargets)
    _hmc_device_validate_initial_kernel!(backend)(arrays, true; ndrange=count)
    _hmc_device_summary!(state, backend, transfers)
    _hmc_device_evaluate!(state, state.bound_gradient, true, backend)
    _hmc_device_validate_initial_kernel!(backend)(arrays, false; ndrange=count)
    _hmc_device_summary!(state, backend, transfers)
    old = state.counts
    state.counts = merge(old,
                         (initial_target_evaluations=Base.Checked.checked_add(old.initial_target_evaluations,
                                                                              count),
                          initial_gradient_evaluations=Base.Checked.checked_add(old.initial_gradient_evaluations,
                                                                                count)))
    state.cache_valid = true
    return nothing
end

function _hmc_sweep!(state::_HamiltonianState, rng, execution::_KernelExecution, transfers;
                     warmup=false)
    # Kernel execution can also be selected for ordinary CPU proposal kernels.
    state.device_workspace === nothing && return invoke(_hmc_sweep!,
                                                        Tuple{_HamiltonianState,Any,Any,Any}, state, rng,
                                                        execution.cpu_execution, transfers; warmup)
    try
    backend = KernelAbstractions.get_backend(state.centres)
    _initialize_hmc_device!(state, backend, transfers)
        rng = _rand_normal!(rng, state.normals; threaded = _transition_threaded(execution))
        rng =
            _rand_uniform!(rng, state.uniforms; threaded = _transition_threaded(execution))
    fill!(state.work, 0)
    fill!(state.acceptance, zero(eltype(state.acceptance)))
    fill!(state.energy_error, eltype(state.energy_error)(NaN))
    count = length(state.logtargets)
    _launch_hmc_device_sweep!(state, state.bound_gradient, backend)
    summary, scalars = _hmc_device_summary!(state, backend, transfers; warmup)
    _hmc_add_sweep_counts!(state, count, summary[1], summary[2], summary[3],
                           summary[4], summary[5], scalars[1], warmup)
    if isfinite(scalars[2])
        state.max_abs_energy_error = isnothing(state.max_abs_energy_error) ? scalars[2] :
                                     max(state.max_abs_energy_error, scalars[2])
    end
    return rng, nothing
    catch failure
        return rng, failure
    end
end

function _warmup_hmc!(state::_HamiltonianState, rng, execution::_KernelExecution, transfers)
    state.device_workspace === nothing &&
        return _warmup_hmc!(state, rng, execution.cpu_execution, transfers)
    try
        _initialize_hmc_device!(state, KernelAbstractions.get_backend(state.centres), transfers)
    while state.n_tuned < state.warmup_steps
            rng, failure = _hmc_sweep!(state, rng, execution, transfers; warmup=true)
            isnothing(failure) || return rng, failure
        end
    return rng, nothing
    catch failure
        return rng, failure
    end
end
