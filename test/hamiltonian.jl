using Test, ImportanceSamplers
import Random, LinearAlgebra, Statistics

if !isdefined(@__MODULE__, :LAISScriptedRNG)
    include("support/lais.jl")
end

@testset "HAIS retains zero lower weights and the final adapted population" begin
    target = LogTarget(x -> x[1] <= 0 ? -sum(abs2, x) / 2 : -Inf;
                       grad=(g, x) -> (g .= .-x))
    algorithm = HAIS(ProposalBank([SphericalGaussian([-1.0], 1.0)]);
                     transition=HamiltonianMonteCarlo(; step_size=0.1, leapfrog_steps=1),
                     rounds=1, round_size=2)
    rng = LAISScriptedRNG([[2.0, 2.0], [0.0]], [[0.0], [0.2]], 1, 1)
    prepared = prepare_sampler(rng, target, algorithm; threaded=false)
    result = importance_sample!(prepared)
    @test vec(result.samples) == [1.0, 1.0]
    @test all(==(-Inf), result.logweights)
    @test lognormalizer(result) == -Inf
    @test only(current_proposal(prepared).proposals).location ≈ [-0.995]
    @test result.diagnostics.cooperation ==
          (rounds=1, endpoint_count=1, proposal_evaluations=1)
end

@testset "HAIS scores the old mixture and gathers repeated ancestors coherently" begin
    target = LogTarget(x -> -sum(abs2, x) / 2; grad=(g, x) -> (g .= .-x))
    bank = ProposalBank([SphericalGaussian([-1.0], 1.0), SphericalGaussian([1.0], 2.0)])
    algorithm = HAIS(bank;
                     transition=HamiltonianMonteCarlo(;
                                                      step_size=1.0, leapfrog_steps=1),
                     rounds=1, round_size=2)
    # Old-bank P(ancestor=1)=0.5065, endpoint-bank P=0.4771. Uniform .49 separates them.
    rng = LAISScriptedRNG([zeros(2), [1.0, -0.2], zeros(2), zeros(2)],
                          [zeros(2), fill(0.49, 2), zeros(2), zeros(2)], 1, 1)
    prepared = prepare_sampler(rng, target, algorithm; threaded=false)
    result = importance_sample!(prepared)
    density(x, m, s) = exp(-abs2((x - m) / s) / 2) / (sqrt(2pi) * s)
    expected = [-x^2 / 2 - log((density(x, -1, 1) + density(x, 1, 2)) / 2) for x in (-1, 1)]
    @test vec(result.samples) == [-1.0, 1.0]
    @test result.logweights ≈ expected
    snapshot = current_proposal(prepared)
    @test [p.location[1] for p in snapshot.proposals] ≈ [0.5, 0.5]
    @test snapshot.proposals[2].scale == bank.proposals[2].scale
    @test result.provenance.proposal_id == [1, 2]
    importance_sample!(prepared)
    @test [p.location[1] for p in current_proposal(prepared).proposals] ≈ [0.25, 0.25]
end

@testset "A late HMC derivative failure preserves the pre-call population" begin
    calls = Ref(0)
    gradient! = function (g, x)
        calls[] += 1
        g .= calls[] == 5 ? NaN : .-x
        return nothing
    end
    target = LogTarget(x -> -sum(abs2, x) / 2; grad=gradient!)
    algorithm = LAIS(ProposalBank([SphericalGaussian([1.0], 1.0)]);
                     transition=HamiltonianMonteCarlo(; step_size=0.1, leapfrog_steps=3),
                     rounds=2, round_size=2)
    prepared = prepare_sampler(Random.Xoshiro(53), target, algorithm; threaded=false)
    @test_throws LAISRoundError importance_sample!(prepared)
    @test only(current_proposal(prepared).proposals).location == [1.0]
    retry = importance_sample!(prepared)
    @test retry.diagnostics.transition.initial_gradient_evaluations == 1
end

@testset "Native HMC follows factored-mass leapfrog and preserves weight scale" begin
    precision = [2.0 0.3; 0.3 0.7]
    mass = [1.5 -0.2; -0.2 0.8]
    start = [0.2, -0.4]
    momentum = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(mass)).L * [0.5, -0.3]
    endpoint = copy(start)
    epsilon = 0.3
    momentum -= epsilon / 2 * precision * endpoint
    for step in 1:3
        endpoint += epsilon * (mass \ momentum)
        momentum -= (step == 3 ? epsilon / 2 : epsilon) * precision * endpoint
    end
    results = map((0.0, 1000.0)) do shift
        target = LogTarget(x -> -LinearAlgebra.dot(x, precision * x) / 2 + shift;
                           grad=(g, x) -> LinearAlgebra.mul!(g, precision, x, -1.0, 0.0))
        rng = LAISScriptedRNG([[0.5, -0.3], zeros(2)], [[0.0]], 1, 1)
        algorithm = LAIS(ProposalBank([SphericalGaussian(start, 1.0)]);
                         transition=HamiltonianMonteCarlo(; step_size=epsilon,
                                                          leapfrog_steps=3,
                                                          mass, divergence_threshold=eps()),
                         rounds=1, round_size=1)
        prepared = prepare_sampler(rng, target, algorithm; threaded=false)
        samples = importance_sample!(prepared)
        @test samples.samples[:, 1] ≈ endpoint
        @test only(current_proposal(prepared).proposals).location ≈ endpoint
        @test samples.diagnostics.transition.production_gradient_evaluations == 3
        @test samples.diagnostics.transition.production_accepted == 1
        @test samples.diagnostics.transition.production_divergences == 1
        return samples
    end
    @test results[2].logweights .- results[1].logweights ≈ [1000.0]
    # Endpoint errors are -.09375 and +.125. A diagnostic threshold cannot
    # replace MH: the first accepts, the second rejects at uniform .9.
    for (start, normal, expected, accepted) in ((1.0, 0.0, 0.5, 1), (0.0, 1.0, 0.0, 0))
        target = LogTarget(x -> -sum(abs2, x) / 2; grad=(g, x) -> (g .= .-x))
        rng = LAISScriptedRNG([[normal], [0.0]], [[0.9]], 1, 1)
        algorithm = LAIS(ProposalBank([SphericalGaussian([start], 1.0)]);
                         transition=HamiltonianMonteCarlo(; step_size=1.0, leapfrog_steps=1,
                                                          divergence_threshold=0.01),
                         rounds=1, round_size=1)
        result = importance_sample(rng, target, algorithm; threaded=false)
        @test only(result.samples) == expected
        @test result.diagnostics.transition.production_accepted == accepted
    end
end

@testset "HMC warmup freezes, reuse retains state, and CPU threading replays" begin
    target = LogTarget(x -> -sum(abs2, x) / 2;
                       grad=(g, x) -> (g .= .-x))
    bank = ProposalBank([SphericalGaussian(fill(v, 3), 1.5) for v in (-1.0, 1.0)])
    algorithm = LAIS(bank;
                     transition=HamiltonianMonteCarlo(;
                                                      step_size=0.1, leapfrog_steps=4,
                                                      tuning=WarmupTuning(12)),
                     rounds=3, round_size=64)
    serial = prepare_sampler(Random.Xoshiro(72), target, algorithm; threaded=false)
    parallel = prepare_sampler(Random.Xoshiro(72), target, algorithm; threaded=true)
    first = importance_sample!(serial)
    other = importance_sample!(parallel)
    @test first.samples == other.samples
    @test first.logweights == other.logweights
    @test first.diagnostics.transition.warmup_proposals == 24
    step = first.diagnostics.transition.step_size
    retained = copy(first.samples)
    second = importance_sample!(serial)
    @test second.diagnostics.transition.warmup_proposals == 0
    @test second.diagnostics.transition.initial_gradient_evaluations == 0
    @test second.diagnostics.transition.step_size == step
    @test first.samples == retained
    changed = retarget(Random.Xoshiro(73), serial, target)
    @test importance_sample!(changed).diagnostics.transition.warmup_proposals == 24
end

@testset "HMC accepts explicit AD and named-coordinate scores" begin
    import ADTypes, ForwardDiff
    bank = ProposalBank([SphericalGaussian([0.0], 1.0)])
    algorithm = LAIS(bank;
                     transition=HamiltonianMonteCarlo(;
                                                      step_size=0.2, leapfrog_steps=3),
                     rounds=1, round_size=16)
    explicit = LogTarget(x -> -sum(abs2, x) / 2; grad=x -> -x)
    automatic = LogTarget(x -> -sum(abs2, x) / 2, ADTypes.AutoForwardDiff())
    a = importance_sample(Random.Xoshiro(32), explicit, algorithm)
    b = importance_sample(Random.Xoshiro(32), automatic, algorithm)
    @test a.samples ≈ b.samples
    @test a.logweights ≈ b.logweights
    logical = LogTarget(theta -> -theta.rate;
                        grad=(g, theta) -> (g.rate[] = -one(theta.rate)))
    latent = LogTarget(z -> -exp(z[1]) + z[1];
                       grad=(g, z) -> (g[1] = 1 - exp(z[1])))
    named = importance_sample(Random.Xoshiro(48), logical, algorithm;
                              transform=(rate=(1 => PositiveTransform()),))
    flat = importance_sample(Random.Xoshiro(48), latent, algorithm)
    @test vec(named.samples.rate) ≈ exp.(vec(flat.samples))
    @test named.logweights ≈ flat.logweights
    calls = Ref(0)
    partial_score! = function (g, theta)
        calls[] += 1
        g.left[] = -theta.left
        calls[] == 1 && (g.right[] = -theta.right)
        return nothing
    end
    incomplete = LogTarget(theta -> -(theta.left^2 + theta.right^2) / 2;
                           grad=partial_score!)
    two = LAIS(ProposalBank([SphericalGaussian([0.0, 1.0], 1.0)]);
               transition=HamiltonianMonteCarlo(; step_size=0.2, leapfrog_steps=2),
               rounds=1, round_size=2)
    prepared = prepare_sampler(Random.Xoshiro(19), incomplete, two;
                               transform=(left=1, right=2), threaded=false)
    @test_throws LAISRoundError importance_sample!(prepared)
    @test only(current_proposal(prepared).proposals).location == [0.0, 1.0]
end
