import DensityInterface: logdensityof
import LogExpFunctions: logsumexp
import MLDataDevices

@testset "Independent populations rebuild at explicit CPU precision" begin
    device = MLDataDevices.with_eltype(MLDataDevices.CPUDevice(), Float32)
    for q in (IndependentGamma([prevfloat(32.0)], [1.0]), IndependentLaplace([0.0], [1.0]))
        p = device(
            prepare_sampler(
                Threefry4x32(2042),
                q,
                APIS(ProposalBank([q]); rounds = 1, round_size = 128),
            ),
        )
        result = importance_sample!(p)
        @test eltype(result.samples) === Float32 &&
              maximum(abs, result.logweights) <= 64eps(Float32)
    end
end

gamma_population_logmixture(proposals, masses, x) =
    logsumexp(log.(masses) .+ [logdensityof(q, x) for q in proposals])

@testset "Gamma APIS fits local means at fixed actual variance" begin
    proposals = [IndependentGamma([2.0, 3.0], [0.5, 0.7]), IndependentGamma([4.0, 2.0], [0.5, 0.6])]
    bank = ProposalBank(proposals)
    target = IndependentGamma([3.0, 4.0], [0.6, 0.4])
    sampler = prepare_sampler(
        Threefry4x32(2032),
        target,
        APIS(bank; rounds = 2, round_size = [128, 192]),
    )
    result = importance_sample!(sampler)
    fitted = copy(proposals)
    variances = [(q.shape .* q.scale) .* q.scale for q in proposals]
    for round = 1:2
        indices = findall(==(round), result.provenance.round)
        expected = [
            logdensityof(target, x)-gamma_population_logmixture(fitted, bank.masses, x) for
            x in eachcol(result.samples[:, indices])
        ]
        @test result.logweights[indices] ≈ expected rtol=256eps(Float64) atol=256eps(Float64)
        next = similar(fitted)
        for slot in eachindex(fitted)
            local_indices = filter(i -> result.provenance.proposal_id[i] == slot, indices)
            points = result.samples[:, local_indices]
            logs = [logdensityof(target, x)-logdensityof(fitted[slot], x) for x in eachcol(points)]
            means = points * exp.(logs .- logsumexp(logs))
            scales = variances[slot] ./ means
            next[slot] = IndependentGamma(means ./ scales, scales)
        end
        fitted = next
    end
    learned = current_proposal(sampler)
    point = [1.0, 2.0]
    @test [logdensityof(q, point) for q in learned.proposals] ≈ [logdensityof(q, point) for q in fitted]
    @test [(q.shape .* q.scale) .* q.scale for q in learned.proposals] ≈ variances
end

@testset "Gamma DM-PMC resamples means and preserves variance across reuse" begin
    bank = ProposalBank([IndependentGamma(2.0f0, 0.5f0), IndependentGamma(4.0f0, 0.5f0)])
    target = IndependentGamma(3.0f0, 0.6f0)
    sampler = prepare_sampler(
        Threefry4x32(2033),
        target,
        DeterministicMixturePMC(bank; rounds = 1, round_size = 128, resampling = LocalResampling()),
    )
    result = importance_sample!(sampler)
    learned = current_proposal(sampler)
    for slot in eachindex(learned.proposals)
        q = learned.proposals[slot]
        points = result.samples[findall(==(slot), result.provenance.proposal_id)]
        @test any(x -> isapprox(q.shape*q.scale, x; rtol = 32eps(Float32)), points)
        initial = bank.proposals[slot]
        @test q.shape*q.scale^2 ≈ initial.shape*initial.scale^2
    end
    next = importance_sample!(sampler)
    expected = [
        logdensityof(target, x)-gamma_population_logmixture(learned.proposals, learned.masses, x) for x in next.samples
    ]
    @test next.logweights ≈ expected rtol=64eps(Float32) atol=64eps(Float32)
end

struct GammaPopulationExtremeRNG{R} <: TestScriptedRNG
    rng::R
    extreme::Base.RefValue{Bool}
end

function PureRNGs.randn_next!(rng::GammaPopulationExtremeRNG, values::AbstractArray; threaded=false)
    rng.extreme[] && return fill!(values, 1e102), rng
    _, next = randn_next!(rng.rng, values; threaded)
    return values, GammaPopulationExtremeRNG(next, rng.extreme)
end

function PureRNGs.rand_next!(rng::GammaPopulationExtremeRNG, values::AbstractArray; threaded=false)
    rng.extreme[] && return fill!(values, 0.0), rng
    _, next = rand_next!(rng.rng, values; threaded)
    return values, GammaPopulationExtremeRNG(next, rng.extreme)
end

@testset "Unrepresentable Gamma population fits retain committed density caches" begin
    q = IndependentGamma(2.0, 0.5)
    rng = GammaPopulationExtremeRNG(Threefry4x32(2034), Ref(false))
    sampler = prepare_sampler(
        rng,
        x -> x > 0 ? -2log1p(x) : -Inf,
        APIS(ProposalBank([q]); rounds = 1, round_size = 64),
    )
    result = importance_sample!(sampler)
    saved = current_proposal(sampler)
    weights = copy(result.logweights)
    rng.extreme[] = true
    failure = try
        importance_sample!(sampler)
    catch error
        error
    end
    @test failure isa APISRoundError &&
          failure.phase === :advance_population &&
          failure.cause isa DomainError
    @test logdensityof(only(current_proposal(sampler).proposals), 1.0) ==
          logdensityof(only(saved.proposals), 1.0)
    rng.extreme[] = false
    @test all(isfinite, importance_sample!(sampler).logweights) && result.logweights == weights
end
