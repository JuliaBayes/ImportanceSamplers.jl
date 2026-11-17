module ImportanceSamplersCUDAExt

import CUDA
import ImportanceSamplers
import KernelAbstractions
import LinearAlgebra
import MLDataDevices

ImportanceSamplers._backend_functional(::MLDataDevices.CUDADevice) =
    CUDA.functional()

ImportanceSamplers._factor_batch_supported(::MLDataDevices.CUDADevice) = true
ImportanceSamplers._hmc_device_supported(::MLDataDevices.CUDADevice) = true

function ImportanceSamplers._hmc_trajectory_kernel(backend::CUDA.CUDABackend)
    # Nested score reductions can overflow CUDA's cold, default device stack.
    options = map(
        name -> name === :always_inline ? true : getfield(backend, name),
        fieldnames(typeof(backend)),
    )
    inline_backend = CUDA.CUDABackend(options...)
    return ImportanceSamplers._hmc_device_trajectory_kernel!(inline_backend)
end

function ImportanceSamplers._with_backend_device(
    f,
    device::MLDataDevices.CUDADevice,
)
    requested = device.device
    return requested === nothing ? f() : CUDA.device!(f, requested)
end

function ImportanceSamplers._backend_state_resident(
    device::MLDataDevices.CUDADevice,
    state,
)
    requested = something(device.device, CUDA.device())
    return _cuda_state_resident(requested, state)
end

_cuda_state_resident(
    device,
    state::ImportanceSamplers._PreparedFirstOrderGRAMIS,
) = _cuda_state_resident(
    device,
    ImportanceSamplers._first_order_gramis_resident_state(state),
)

_cuda_state_resident(device, state::ImportanceSamplers._HamiltonianState) =
    _cuda_state_resident(device, ImportanceSamplers._hmc_resident_state(state))

_cuda_state_resident(device, array::CUDA.AnyCuArray) =
    CUDA.device(array) == device
_cuda_state_resident(device, range::AbstractRange) = true
_cuda_state_resident(device, array::Array) = false
function _cuda_state_resident(device, array::AbstractArray)
    # Only immutable wrappers can share checked storage or inline values.
    # Raw pointers do not identify their device and are rejected below.
    T = typeof(array)
    return !ismutabletype(T) && fieldcount(T) > 0 &&
           _cuda_fields_resident(device, array, Val(fieldcount(T)))
end
_cuda_state_resident(device, pointer::Union{Ptr,Core.LLVMPtr}) = false
_cuda_state_resident(
    device,
    state::Union{Number,AbstractString,Symbol,Type,Module,Nothing},
) = true
_cuda_state_resident(device, state::Tuple) =
    all(value -> _cuda_state_resident(device, value), state)
_cuda_state_resident(device, state::NamedTuple) =
    all(value -> _cuda_state_resident(device, value), values(state))

@inline function _cuda_state_resident(device, state::T) where {T}
    return _cuda_fields_resident(device, state, Val(fieldcount(T)))
end

_cuda_fields_resident(device, state, ::Val{0}) = true
@inline function _cuda_fields_resident(device, state::T, ::Val{N}) where {T,N}
    return isdefined(state, N) &&
           _cuda_state_resident(device, getfield(state, N)) &&
           _cuda_fields_resident(device, state, Val(N - 1))
end

ImportanceSamplers._owned_backend_rng(
    ::MLDataDevices.CUDADevice,
    seed::UInt64,
) = CUDA.RNG(seed)

function ImportanceSamplers._gaussian_potrf!(
    ::MLDataDevices.CUDADevice,
    factor::CUDA.StridedCuMatrix{T},
) where {T<:Union{Float32,Float64}}
    factor, info = CUDA.cuSOLVER.potrf!('L', factor)
    info == 0 || throw(LinearAlgebra.PosDefException(info))
    return factor
end

function ImportanceSamplers._factor_pooled_covariance!(
    pooled_covariance::CUDA.StridedCuMatrix{T},
) where {T<:Union{Float32,Float64}}
    _, info = CUDA.cuSOLVER.potrf!('L', pooled_covariance)
    iszero(info) || throw(
        ImportanceSamplers._FirstOrderGRAMISRepulsionError(
            0,
            :pooled_factorization_failed,
            info,
        ),
    )
    return nothing
end

end
