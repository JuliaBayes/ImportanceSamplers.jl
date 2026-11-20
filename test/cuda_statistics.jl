using Test
using CUDA
using ImportanceSamplers
using MLDataDevices
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
