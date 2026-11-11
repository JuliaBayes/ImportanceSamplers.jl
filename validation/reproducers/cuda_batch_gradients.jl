using Test, CUDA, ImportanceSamplers, Random, LinearAlgebra
import MLDataDevices
include(joinpath(@__DIR__, "..", "..", "test", "support", "batch_gradients.jl"))
CUDA.allowscalar(false)
const BATCH_GRADIENT_DEVICE = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)

support_batch_value(x) = x[2]<0.5 || abs(x[1])<0.01 ? -sum(abs2, x)/2 : -Inf
function support_batch_scalar!(gradient, x)
    for i in eachindex(x)
        gradient[i]=-x[i]
    end
    return nothing
end
function support_batch!(values, gradients, samples)
    valid=(view(samples, 2, :) .< 0.5) .| (abs.(view(samples, 1, :)) .< 0.01)
    values .= ifelse.(valid, .-vec(sum(abs2, samples; dims = 1)) ./ 2, -Inf)
    gradients .= ifelse.(reshape(valid, 1, :), .-samples, NaN)
    return nothing
end

@testset "One unsupported batch column cannot poison another HMC chain" begin
    bank=ProposalBank([SphericalGaussian([0.0, marker], 0.8) for marker in (0.0, 1.0)])
    algorithm=LAIS(
        bank;
        rounds = 1,
        round_size = 64,
        transition = HamiltonianMonteCarlo(;
            step_size = 1.0,
            leapfrog_steps = 5,
            mass = Diagonal([1.0, 1e12]),
        ),
    )
    a=BATCH_GRADIENT_DEVICE(
        prepare_sampler(
            Xoshiro(35),
            LogTarget(support_batch_value; batch_grad = support_batch!),
            algorithm,
        ),
    )
    b=BATCH_GRADIENT_DEVICE(
        prepare_sampler(
            Xoshiro(35),
            LogTarget(support_batch_value; grad = support_batch_scalar!),
            algorithm,
        ),
    )
    actual, expected=importance_sample!(a), importance_sample!(b)
    @test actual.diagnostics.transition.production_support_rejections==1
    @test actual.diagnostics.transition.production_gradient_evaluations==10
    @test Array(actual.samples) ≈ Array(expected.samples)
    weights, reference=Array(actual.logweights), Array(expected.logweights)
    @test all((weights .== reference) .| isapprox.(weights, reference))
end

@testset "CUDA batched HMC scores preserve precision, reuse and retargeting" begin
    for T in (Float32, Float64)
        bank = ProposalBank([SphericalGaussian(fill(T(x), 3), T(0.8)) for x in (-0.2, 0.2)])
        algorithm = LAIS(
            bank;
            rounds = 2,
            round_size = 64,
            transition = HamiltonianMonteCarlo(;
                step_size = T(0.1),
                leapfrog_steps = 3,
                tuning = WarmupTuning(3),
            ),
        )
        batch = LogTarget(batch_score_value; batch_grad = batch_score!)
        scalar = LogTarget(batch_score_value; grad = batch_score_scalar!)
        context = (; centre = T[0.2, -0.1, 0.4])
        actual =
            BATCH_GRADIENT_DEVICE(prepare_sampler(Xoshiro(71), batch, context, algorithm))
        expected =
            BATCH_GRADIENT_DEVICE(prepare_sampler(Xoshiro(71), scalar, context, algorithm))
        tolerance = T === Float32 ? T(2e-3) : T(2e-10)
        for pass = 1:3
            if pass == 3
                shifted=(; centre = T[-0.2, 0.3, 0.1])
                actual=retarget(Xoshiro(72), actual, batch, shifted)
                expected=retarget(Xoshiro(72), expected, scalar, shifted)
            end
            a, b=importance_sample!(actual), importance_sample!(expected)
            @test Array(a.samples) ≈ Array(b.samples) rtol=tolerance atol=tolerance
            @test Array(a.logweights) ≈ Array(b.logweights) rtol=tolerance atol=tolerance
        end
    end
end

@testset "CUDA batched HAIS scores pull back mixed logical parameters" begin
    bank=ProposalBank([SphericalGaussian(fill(x, 4), 0.3) for x in (-0.1, 0.1)])
    algorithm=HAIS(
        bank;
        rounds = 2,
        round_size = 64,
        transition = HamiltonianMonteCarlo(; step_size = 0.05, leapfrog_steps = 3),
    )
    layout=(
        weights = (1:2=>SimplexTransform(3)),
        scale = (3=>PositiveTransform()),
        offset = 4,
    )
    context=(; powers = [2.0, 3.0, 4.0])
    a=BATCH_GRADIENT_DEVICE(
        prepare_sampler(
            Xoshiro(41),
            LogTarget(named_batch_score_value; batch_grad = named_batch_score!),
            context,
            algorithm;
            transform = layout,
        ),
    )
    b=BATCH_GRADIENT_DEVICE(
        prepare_sampler(
            Xoshiro(41),
            LogTarget(named_batch_score_value; grad = named_batch_score_scalar!),
            context,
            algorithm;
            transform = layout,
        ),
    )
    actual, expected=importance_sample!(a), importance_sample!(b)
    @test all(
        name->Array(getproperty(actual.samples, name)) ≈
              Array(getproperty(expected.samples, name)),
        keys(actual.samples),
    )
    @test Array(actual.logweights) ≈ Array(expected.logweights)
end

@testset "CUDA batched scores retain odd-dimensional coupling pullbacks" begin
    bank=ProposalBank([SphericalGaussian(zeros(3), 0.8)])
    algorithm=LAIS(
        bank;
        rounds = 1,
        round_size = 32,
        transition = HamiltonianMonteCarlo(; step_size = 0.1, leapfrog_steps = 3),
    )
    context=(; centre = [0.2, -0.1, 0.4])
    transform=QuadraticCoupling([0.3])
    a=BATCH_GRADIENT_DEVICE(
        prepare_sampler(
            Xoshiro(42),
            LogTarget(batch_score_value; batch_grad = batch_score!),
            context,
            algorithm;
            transform,
        ),
    )
    b=BATCH_GRADIENT_DEVICE(
        prepare_sampler(
            Xoshiro(42),
            LogTarget(batch_score_value; grad = batch_score_scalar!),
            context,
            algorithm;
            transform,
        ),
    )
    actual, expected=importance_sample!(a), importance_sample!(b)
    @test Array(actual.samples) ≈ Array(expected.samples)
    @test Array(actual.logweights) ≈ Array(expected.logweights)
end

@testset "CUDA GRAMIS reuses the batched derivative interface" begin
    bank=ProposalBank([
        FactorGaussian(Float32[x, 0], Matrix{Float32}(I, 2, 2)) for x in (-0.4, 0.4)
    ])
    algorithm=FirstOrderGRAMIS(
        bank;
        rounds = 2,
        round_size = 128,
        repulsion_strength = 0.0f0,
        covariance_rate = 0.1f0,
    )
    context=(; centre = Float32[0.1, -0.2])
    a=BATCH_GRADIENT_DEVICE(
        prepare_sampler(
            Xoshiro(73),
            LogTarget(batch_score_value; batch_grad = batch_score!),
            context,
            algorithm,
        ),
    )
    b=BATCH_GRADIENT_DEVICE(
        prepare_sampler(
            Xoshiro(73),
            LogTarget(batch_score_value; grad = batch_score_scalar!),
            context,
            algorithm,
        ),
    )
    actual, expected=importance_sample!(a), importance_sample!(b)
    @test Array(actual.samples) ≈ Array(expected.samples) rtol=2e-3 atol=2e-3
    @test Array(actual.logweights) ≈ Array(expected.logweights) rtol=2e-3 atol=2e-3
end
