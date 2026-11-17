using Test
using CUDA
using ImportanceSamplers
using MLDataDevices
using Random
using Statistics

CUDA.allowscalar(false)

@testset "CUDA weighted statistics" begin
    shifted = WeightedSamples(CuArray(Float32[1, 3]), CuArray(fill(1.0f20, 2)))
    @test Array(normalized_weights(shifted)) == Float32[0.5, 0.5]
    @test (mean(shifted), var(shifted)) == (2.0f0, 1.0f0)

    logweights = CuArray(100.0f0 .+ log.(Float32[1, 2, 1]))
    samples = CuArray(Float32[1 3 10; 2 4 8])
    result = ImportanceSamplers._adopt_weighted_samples(samples, logweights)

    @test (@inferred mean(result)) isa CuArray
    @test (@inferred var(result)) isa CuArray
    @test (@inferred std(result)) isa CuArray
    @test (@inferred cov(result)) isa CuArray
    @test (@inferred mean(x -> x[1] + 2x[2], result)) ≈ 13.25f0
    @test (@inferred var(x -> x[1] + 2x[2], result)) ≈ 60.1875f0
    @test_throws ArgumentError quantile(result, 0.5)
end

@testset "CUDA weighted variance range" begin
    samples = Float32[0, 1.0f20]
    for logmass in (log(1.0f-20), -Inf32)
        mass = exp(BigFloat(logmass))
        expected = Float32(mass * BigFloat(samples[2])^2 / (1 + mass)^2)
        logweights = CuArray(Float32[0, logmass])
        scalar = WeightedSamples(CuArray(samples), logweights)
        matrix = WeightedSamples(CuArray(reshape(samples, 1, :)), logweights)
        @test var(scalar) ≈ expected rtol=8eps(Float32)
        @test only(Array(var(matrix))) ≈ expected rtol=8eps(Float32)
    end
end

@testset "CUDA incompatible RNG rejection preserves the CPU stream" begin
    device = MLDataDevices.CUDADevice{typeof(CUDA.device()),Nothing}(CUDA.device())
    fetch(@async begin
        Random.seed!(0x9114)
        task_rng = Random.default_rng()
        reference_rng = copy(task_rng)
            algorithm = ImportanceSampling(SphericalGaussian(0.0f0, 1.0f0); nsamples=8)
        target(x) = -abs2(x) / 2.0f0
        source = prepare_sampler(task_rng, target, algorithm)
            @test_throws SamplerDeviceError device(source)
        reference = prepare_sampler(reference_rng, target, algorithm)

            @test importance_sample!(source).samples ==
                  importance_sample!(reference).samples
            @test rand(task_rng, UInt64) == rand(reference_rng, UInt64)
    end)
end
