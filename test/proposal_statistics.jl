import Statistics: mean, cov, var, std
import LinearAlgebra: Diagonal
import Random

@testset "native Gaussian moments and stored factors" begin
    scalar = SphericalGaussian(1.0f0, 2.0f0)
    @test (mean(scalar), cov(scalar), var(scalar), std(scalar)) ===
          (1.0f0, 4.0f0, 4.0f0, 2.0f0)
    spherical = SphericalGaussian(Float32[1, -1], 2.0f0)
    @test mean(spherical) == Float32[1, -1]
    @test cov(spherical) == Diagonal(Float32[4, 4])
    @test std(spherical) == Float32[2, 2]
    diagonal = DiagonalGaussian([1.0, -1.0], [2.0, 3.0])
    @test cov(diagonal) == Diagonal([4.0, 9.0])
    @test var(diagonal) == [4.0, 9.0]
    @test std(diagonal) == [2.0, 3.0]
    @test proposal_scale_factor(diagonal) == Diagonal([2.0, 3.0])
    factor = FactorGaussian(Float32[1, -1], Float32[2 0; 1 3])
    @test cov(factor) == Float32[4 2; 2 10]
    @test var(factor) == Float32[4, 10]
    @test std(factor) == sqrt.(Float32[4, 10])
    @test proposal_scale_factor(factor) * proposal_scale_factor(factor)' == cov(factor)
    # A finite standard deviation need not have a representable variance.
    large = FactorGaussian(zeros(Float32, 2), Float32[1e20 0; 1e20 1e20])
    @test std(large) ≈ Float32[1e20, sqrt(2.0f0)*1e20]
    @test std(SphericalGaussian(0.0f0, 1.0f20)) === 1.0f20
end

@testset "Student-t moments distinguish scale and covariance" begin
    proposal = FactorStudentT(4.0f0, Float32[1, -1], Float32[2 0; 1 3])
    @test mean(proposal) == Float32[1, -1]
    @test cov(proposal) ≈ Float32[8 4; 4 20] rtol=4eps(Float32)
    @test var(proposal) ≈ Float32[8, 20] rtol=4eps(Float32)
    @test std(proposal) == sqrt.(Float32[8, 20])
    @test proposal_scale_factor(proposal) * proposal_scale_factor(proposal)' ==
          Float32[4 2; 2 10]
    cauchy = SphericalStudentT(1.0f0, 2.0f0, 3.0f0)
    @test isnan(mean(cauchy)) && isnan(var(cauchy)) && isnan(cov(cauchy))
    heavy = DiagonalStudentT(2.0f0, Float32[1, -1], Float32[2, 3])
    @test mean(heavy) == Float32[1, -1]
    @test all(isnan, var(heavy)) && all(isnan, std(heavy))
    @test all(isnan, cov(heavy))
    scalar_heavy = SphericalStudentT(2.0f0, 1.0f0, 2.0f0)
    @test var(scalar_heavy) === Inf32 && std(scalar_heavy) === Inf32
    undefined = SphericalStudentT(0.5f0, Float32[1, -1], 2.0f0)
    @test all(isnan, mean(undefined)) && all(isnan, var(undefined))
    small = SphericalStudentT(nextfloat(2.0f0), 0.0f0, 1.0f-25)
    expected = Float32(
        Float64(1.0f-25)^2 * Float64(nextfloat(2.0f0)) / (Float64(nextfloat(2.0f0)) - 2),
    )
    @test var(small) ≈ expected rtol=4eps(Float32)
end

@testset "independent native proposal moments" begin
    gamma = IndependentGamma(Float32[2, 3], Float32[3, 2])
    @test mean(gamma) == Float32[6, 6]
    @test var(gamma) == Float32[18, 12]
    @test cov(gamma) == Diagonal(Float32[18, 12])
    @test std(gamma) == sqrt.(Float32[18, 12])
    laplace = IndependentLaplace(1.0f0, 2.0f0)
    @test (mean(laplace), var(laplace), cov(laplace)) === (1.0f0, 8.0f0, 8.0f0)
    @test std(laplace) ≈ sqrt(8.0f0)
    @test var(IndependentLaplace(0.0f0, 2.0f-23)) === Float32(2 * Float64(2.0f-23)^2)
    box = BoxUniform(Float32[-2, 2], Float32[4, 8])
    @test mean(box) == Float32[1, 5]
    @test var(box) ≈ Float32[3, 3] rtol=4eps(Float32)
    @test cov(box) ≈ Diagonal(Float32[3, 3]) rtol=4eps(Float32)
    @test std(box) ≈ sqrt.(Float32[3, 3]) rtol=4eps(Float32)
    @test var(IndependentGamma(1.0f-20, 1.0f20)) ≈ 1.0f20 rtol=4eps(Float32)
    narrow = BoxUniform(1.0f0, nextfloat(1.0f0))
    @test std(narrow) ≈ eps(Float32) / sqrt(12.0f0) rtol=4eps(Float32)
    wide = BoxUniform(-floatmax(Float32), floatmax(Float32))
    @test mean(wide) === 0.0f0
    @test std(wide) ≈ floatmax(Float32) / sqrt(3.0f0)
    @test var(wide) === Inf32
end

@testset "proposal moments retain independent reusable snapshots" begin
    initial = FactorGaussian(Float32[1, -1], Float32[2 0; 1 3])
    prepared = prepare_sampler(
        Random.Xoshiro(0x521),
        x -> -sum(abs2, x) / 2,
        AMIS(initial; rounds = 1, round_size = 32),
    )
    snapshot = current_proposal(prepared)
    proposal_scale_factor(snapshot)[1, 1] = 7.0f0
    mean(snapshot)[1] = 7.0f0
    @test var(current_proposal(prepared)) == Float32[4, 10]
    @test mean(current_proposal(prepared)) == Float32[1, -1]
    saved_mean, saved_cov = copy(mean(snapshot)), cov(snapshot)
    importance_sample!(prepared)
    @test mean(snapshot) == saved_mean && cov(snapshot) == saved_cov
    fitted = current_proposal(prepared)
    @test mean(fitted) != saved_mean
    importance_sample!(prepared)
    @test cov(snapshot) == saved_cov
end
