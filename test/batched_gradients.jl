using Test, ImportanceSamplers, Random
import ADTypes, ForwardDiff, MLDataDevices
include("support/batch_gradients.jl")

@testset "Batch-only scores preserve wider log densities" begin
    value = x -> -1e40-Float64(sum(abs2, x))/2
    callback = function (values, gradients, samples)
        values .= .-vec(sum(abs2, samples; dims = 1)) ./ 2 .- 1e40
        gradients .= .-samples
    end
    algorithm=LAIS(
        ProposalBank([SphericalGaussian(zeros(Float32, 3), 0.8f0)]);
        rounds = 1,
        round_size = 16,
        transition = HamiltonianMonteCarlo(; step_size = 0.1f0, leapfrog_steps = 2),
    )
    actual=importance_sample(
        Xoshiro(18),
        LogTarget(value; batch_grad = callback),
        algorithm,
    )
    expected=importance_sample(
        Xoshiro(18),
        LogTarget(value; grad = (g, x)->(g .= .-x)),
        algorithm,
    )
    @test actual.samples ≈ expected.samples
    @test actual.logweights == expected.logweights
end

@testset "Batch wrappers preserve CPU closure-transfer semantics" begin
    centre=[0.2, -0.1, 0.4]
    value=x->-sum(abs2, x .- centre)/2
    scalar=(g, x)->(g .= centre .- x)
    callback=function (values, gradients, samples)
        residual=samples .- centre
        values .= .-vec(sum(abs2, residual; dims = 1)) ./ 2
        gradients .= .-residual
    end
    algorithm=LAIS(
        ProposalBank([SphericalGaussian(zeros(3), 0.8)]);
        rounds = 1,
        round_size = 16,
        transition = HamiltonianMonteCarlo(; step_size = 0.1, leapfrog_steps = 2),
    )
    device=MLDataDevices.cpu_device(Float32)
    a=device(
        prepare_sampler(
            Xoshiro(17),
            LogTarget(value; grad = scalar, batch_grad = callback),
            algorithm,
        ),
    )
    b=device(prepare_sampler(Xoshiro(17), LogTarget(value; grad = scalar), algorithm))
    actual, expected=importance_sample!(a), importance_sample!(b)
    @test actual.samples ≈ expected.samples
    @test actual.logweights ≈ expected.logweights
end

@testset "Context-free batch scores support the CPU fallback" begin
    value = x -> -sum(abs2, x)/2
    callback = function (values, gradients, samples)
        values .= .-vec(sum(abs2, samples; dims = 1)) ./ 2
        gradients .= .-samples
        return nothing
    end
    algorithm = LAIS(
        ProposalBank([SphericalGaussian(zeros(3), 0.8)]);
        rounds = 1,
        round_size = 16,
        transition = HamiltonianMonteCarlo(; step_size = 0.1, leapfrog_steps = 2),
    )
    actual =
        importance_sample(Xoshiro(19), LogTarget(value; batch_grad = callback), algorithm)
    expected = importance_sample(
        Xoshiro(19),
        LogTarget(value; grad = (g, x)->(g .= .-x)),
        algorithm,
    )
    @test actual.samples ≈ expected.samples
    @test actual.logweights ≈ expected.logweights
end

@testset "CPU scalar and automatic scores retain priority" begin
    algorithm = LAIS(
        ProposalBank([SphericalGaussian(zeros(3), 0.8)]);
        rounds = 1,
        round_size = 16,
        transition = HamiltonianMonteCarlo(; step_size = 0.1, leapfrog_steps = 2),
    )
    context = (; centre = [0.2, -0.1, 0.4])
    gpu_only = (args...) -> error("accelerator-only callback reached CPU")
    for (ad, gradient) in
        ((ADTypes.NoAutoDiff(), batch_score_scalar!), (ADTypes.AutoForwardDiff(), nothing))
        actual = importance_sample(
            Xoshiro(20),
            LogTarget(batch_score_value, ad; grad = gradient, batch_grad = gpu_only),
            context,
            algorithm,
        )
        expected = importance_sample(
            Xoshiro(20),
            LogTarget(batch_score_value, ad; grad = gradient),
            context,
            algorithm,
        )
        @test actual.samples ≈ expected.samples
        @test actual.logweights ≈ expected.logweights
    end
end

@testset "Batch gradients preserve CPU HMC transitions and retargeting" begin
    bank = ProposalBank([SphericalGaussian(fill(x, 3), 0.8) for x in (-0.2, 0.2)])
    algorithm = LAIS(
        bank;
        rounds = 2,
        round_size = 32,
        transition = HamiltonianMonteCarlo(;
            step_size = 0.1,
            leapfrog_steps = 3,
            tuning = WarmupTuning(3),
        ),
    )
    batch = LogTarget(batch_score_value; batch_grad = batch_score!)
    scalar = LogTarget(batch_score_value; grad = batch_score_scalar!)
    a = prepare_sampler(Xoshiro(71), batch, (; centre = [0.2, -0.1, 0.4]), algorithm)
    b = prepare_sampler(Xoshiro(71), scalar, (; centre = [0.2, -0.1, 0.4]), algorithm)
    for pass = 1:3
        if pass == 3
            a = retarget(Xoshiro(72), a, batch, (; centre = [-0.2, 0.3, 0.1]))
            b = retarget(Xoshiro(72), b, scalar, (; centre = [-0.2, 0.3, 0.1]))
        end
        actual, expected = importance_sample!(a), importance_sample!(b)
        @test actual.samples ≈ expected.samples
        @test actual.logweights ≈ expected.logweights
    end
end

@testset "Batch gradients retain named simplex, positive and identity pullbacks" begin
    bank = ProposalBank([SphericalGaussian(zeros(4), 0.3)])
    algorithm = HAIS(
        bank;
        rounds = 2,
        round_size = 32,
        transition = HamiltonianMonteCarlo(; step_size = 0.05, leapfrog_steps = 3),
    )
    transform = (
        weights = (1:2 => SimplexTransform(3)),
        scale = (3 => PositiveTransform()),
        offset = 4,
    )
    context = (; powers = [2.0, 3.0, 4.0])
    batch = LogTarget(named_batch_score_value; batch_grad = named_batch_score!)
    scalar = LogTarget(named_batch_score_value; grad = named_batch_score_scalar!)
    actual = importance_sample(Xoshiro(41), batch, context, algorithm; transform)
    expected = importance_sample(Xoshiro(41), scalar, context, algorithm; transform)
    @test all(
        name -> getproperty(actual.samples, name) ≈ getproperty(expected.samples, name),
        keys(actual.samples),
    )
    @test actual.logweights ≈ expected.logweights
end

@testset "Batch gradients preserve odd-dimensional coupling pullbacks" begin
    bank = ProposalBank([SphericalGaussian(zeros(3), 0.8)])
    algorithm = LAIS(
        bank;
        rounds = 1,
        round_size = 32,
        transition = HamiltonianMonteCarlo(; step_size = 0.1, leapfrog_steps = 3),
    )
    transform = QuadraticCoupling([0.3])
    context = (; centre = [0.2, -0.1, 0.4])
    actual = importance_sample(
        Xoshiro(42),
        LogTarget(batch_score_value; batch_grad = batch_score!),
        context,
        algorithm;
        transform,
    )
    expected = importance_sample(
        Xoshiro(42),
        LogTarget(batch_score_value; grad = batch_score_scalar!),
        context,
        algorithm;
        transform,
    )
    @test actual.samples ≈ expected.samples
    @test actual.logweights ≈ expected.logweights
end

@testset "An incomplete batch value cannot commit a stale HMC transition" begin
    calls = Ref(0)
    callback = function (values, gradients, samples, p)
        calls[] += 1
        if calls[] == 1
            batch_score!(values, gradients, samples, p)
        else
            gradients .= .-samples
        end
    end
    bank = ProposalBank([SphericalGaussian(zeros(3), 0.8)])
    algorithm = LAIS(
        bank;
        rounds = 1,
        round_size = 16,
        transition = HamiltonianMonteCarlo(; step_size = 0.1, leapfrog_steps = 2),
    )
    context = (; centre = zeros(3))
    prepared = prepare_sampler(
        Xoshiro(61),
        LogTarget(batch_score_value; batch_grad = callback),
        context,
        algorithm;
        threaded = false,
    )
    @test_throws LAISRoundError importance_sample!(prepared)
    @test only(current_proposal(prepared).proposals).location == zeros(3)
    repaired = retarget(
        Xoshiro(62),
        prepared,
        LogTarget(batch_score_value; batch_grad = batch_score!),
        context,
    )
    actual = importance_sample!(repaired)
    expected = importance_sample(
        Xoshiro(62),
        LogTarget(batch_score_value; grad = batch_score_scalar!),
        context,
        algorithm;
        threaded = false,
    )
    @test actual.samples ≈ expected.samples
    @test actual.logweights ≈ expected.logweights
end
