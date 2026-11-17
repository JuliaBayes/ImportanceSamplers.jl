using CUDA
using ImportanceSamplers
using MLDataDevices
using PureRNGs: Threefry4x32, rand_next, rngkey, rngposition
using Test

const IS = ImportanceSamplers

@testset "public preserving device dispatch" begin
    physical = CUDA.device()
    default_policy = MLDataDevices.CUDADevice(physical)
    preserving_policy =
        MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)

    @test eltype(default_policy) === Missing
    @test eltype(preserving_policy) === Nothing
    @test IS._backend_functional(preserving_policy) === CUDA.functional()
end

@testset "CUDA transfer preserves an advanced random stream" begin
    if CUDA.functional()
        physical = CUDA.device()
        device = MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
        _, rng = rand_next(Threefry4x32(0x63756461726e67), UInt64)
        proposal = SphericalGaussian(0.0f0, 1.0f0)
        source = prepare_sampler(rng, Returns(0.0f0), ImportanceSampling(proposal; nsamples=32))
        sampler = device(source)
        resident = current_rng(sampler)
        @test rngkey(resident) == rngkey(rng)
        @test rngposition(resident) == rngposition(rng)
        importance_sample!(sampler)
        @test rngposition(current_rng(sampler)) > rngposition(rng)
        @test current_rng(source) == rng
    else
        @test_skip CUDA.functional()
    end
end
