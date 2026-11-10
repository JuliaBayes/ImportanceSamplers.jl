using Test, ImportanceSamplers
import Random, LinearAlgebra, DensityInterface

if !isdefined(@__MODULE__, :LAISScriptedRNG)
    include("support/lais.jl")
end

@testset "Lower width fallbacks and failed calls preserve committed densities" begin
    transition = HamiltonianMonteCarlo(; step_size=0.1, leapfrog_steps=1)
    proposal = SphericalGaussian([0.0], 2.0)
    for (target, normals, expected_scale) in (
        (LogTarget(x -> -sum(abs2, x) / 8; grad=x -> .-x ./ 4), [-0.5, 0.5], sqrt(2.5)),
        (LogTarget(x -> -sum(abs2, x) / 2; grad=x -> -x), [0.0, 1.0], 2.0),
        (LogTarget(x -> abs(x[1]) < 1 ? -sum(abs2, x)/2 : -Inf; grad=x -> -x), [1.0, 1.0], 2.0),
    )
        rng = LAISScriptedRNG([zeros(1), normals], [zeros(1)], 1, 1)
        prepared = prepare_sampler(
            rng,
            target,
            LAIS(ProposalBank([proposal]); transition, rounds=1, round_size=2, adapt_scales=true);
            threaded=false,
        )
        importance_sample!(prepared)
        fitted = only(current_proposal(prepared).proposals)
        @test DensityInterface.logdensityof(fitted, [0.5]) ≈
              DensityInterface.logdensityof(SphericalGaussian([0.0], expected_scale), [0.5])
    end
    calls = Ref(0)
    gradient! = function (g, x)
        calls[] += 1
        g .= calls[] == 3 ? NaN : .-x
    end
    prepared = prepare_sampler(
        Random.Xoshiro(132),
        LogTarget(x -> -sum(abs2, x)/2; grad=gradient!),
        LAIS(ProposalBank([proposal]); transition, rounds=2, round_size=128, adapt_scales=true);
        threaded=false,
    )
    @test_throws LAISRoundError importance_sample!(prepared)
    @test DensityInterface.logdensityof(only(current_proposal(prepared).proposals), [0.5]) ==
          DensityInterface.logdensityof(proposal, [0.5])
    # One zero-variance coordinate retains the complete component, not one row.
    diagonal = DiagonalGaussian(zeros(2), fill(2.0, 2))
    target = LogTarget(x -> -sum(abs2, x)/8; grad=x -> .-x ./ 4)
    rng = LAISScriptedRNG([zeros(2), [-0.5, 0.0, 0.0, 0.0, 0.5, 0.0]], [zeros(1)], 1, 1)
    prepared = prepare_sampler(
        rng,
        target,
        LAIS(ProposalBank([diagonal]); transition, rounds=1, round_size=3, adapt_scales=true);
        threaded=false,
    )
    importance_sample!(prepared)
    @test DensityInterface.logdensityof(only(current_proposal(prepared).proposals), [0.5, 0.5]) ==
          DensityInterface.logdensityof(diagonal, [0.5, 0.5])
end

@testset "Lower widths follow local DM moments and retain factor correlations" begin
    factor = [2.0 0.0; 0.6 1.0]
    bank = ProposalBank([FactorGaussian([0.0, 0.0], factor), FactorGaussian([1.0, -1.0], factor)])
    target = LogTarget(x -> -sum(z -> z^4, x) / 4; grad=x -> .-x .^ 3)
    upper = HamiltonianMonteCarlo(; step_size=0.05, leapfrog_steps=2)
    prepared = prepare_sampler(
        Random.Xoshiro(423),
        target,
        HAIS(bank; transition=upper, rounds=1, round_size=256, adapt_scales=true),
    )
    result = importance_sample!(prepared)
    snapshot = current_proposal(prepared)
    for slot = 1:2
        group = findall(==(slot), result.provenance.proposal_id)
        lw = result.logweights[group]
        w = exp.(lw .- maximum(lw))
        w ./= sum(w)
        x = result.samples[:, group]
        mu = x * w
        variance = (x .- mu) .^ 2 * w
        old_variance = LinearAlgebra.diag(factor * factor')
        ratio = sqrt.((old_variance .+ variance) ./ (2 .* old_variance))
        expected = FactorGaussian(snapshot.proposals[slot].location, ratio .* factor)
        point = [0.4, -0.2]
        @test DensityInterface.logdensityof(snapshot.proposals[slot], point) ≈
              DensityInterface.logdensityof(expected, point)
    end
    next = importance_sample!(prepared)
    for sample in (1, 129)
        x = next.samples[:, sample]
        densities = exp.([DensityInterface.logdensityof(p, x) for p in snapshot.proposals])
        @test next.logweights[sample] ≈ target.logdensity(x) - log(sum(densities) / 2)
    end
end

@testset "Lower width fits use family variance and preserve reused proposals" begin
    target = LogTarget(x -> -sum(abs2, x) / 2; grad=x -> -x)
    cases = (
        (SphericalGaussian([0.0], 2.0), 1.0, DiagonalGaussian),
        (
            SphericalStudentT(6.0, [0.0], 2.0),
            1.5,
            (location, scale) -> DiagonalStudentT(6.0, location, scale),
        ),
        (IndependentLaplace([0.0], [2.0]), 2.0, IndependentLaplace),
    )
    for (proposal, multiplier, constructor) in cases
        algorithm = LAIS(
            ProposalBank([proposal]);
            transition=HamiltonianMonteCarlo(; step_size=0.01, leapfrog_steps=1),
            rounds=1,
            round_size=256,
            adapt_scales=true,
        )
        prepared = prepare_sampler(Random.Xoshiro(29), target, algorithm; threaded=false)
        result = importance_sample!(prepared)
        w = normalized_weights(result)
        x = vec(result.samples)
        variance = sum(w .* (x .- sum(w .* x)) .^ 2)
        scale = sqrt((4 * multiplier + variance) / (2 * multiplier))
        fitted = only(current_proposal(prepared).proposals)
        expected = constructor(fitted.location, [scale])
        @test DensityInterface.logdensityof(fitted, [0.5]) ≈
              DensityInterface.logdensityof(expected, [0.5])
        retargeted = retarget(Random.Xoshiro(31), prepared, target)
        @test DensityInterface.logdensityof(only(current_proposal(retargeted).proposals), [0.5]) ≈
              DensityInterface.logdensityof(fitted, [0.5])
    end
end

@testset "Zero-weight outliers cannot cancel a valid lower variance" begin
    # Explicit Float64 log weights with Float32 coordinates exercise mixed storage.
    target = LogTarget(
        x -> abs(x[1]) > 10.0f0 ? -Inf : -Float64(sum(abs2, x)) / 8;
        grad=x -> .-x ./ 4.0f0,
    )
    # Zero upper momentum keeps the centre at zero. Lower draws are 1e8, -1, +1.
    rng = LAISScriptedRNG([zeros(Float32, 1), Float32[5e7, -0.5, 0.5]], [zeros(Float32, 1)], 1, 1)
    algorithm = LAIS(
        ProposalBank([SphericalGaussian([0.0f0], 2.0f0)]);
        transition=HamiltonianMonteCarlo(; step_size=0.1f0, leapfrog_steps=1),
        rounds=1,
        round_size=3,
        adapt_scales=true,
    )
    prepared = prepare_sampler(rng, target, algorithm; threaded=false)
    importance_sample!(prepared)
    expected = SphericalGaussian([0.0f0], sqrt(2.5f0))
    @test DensityInterface.logdensityof(only(current_proposal(prepared).proposals), [1.0f0]) ≈
          DensityInterface.logdensityof(expected, [1.0f0])
end
