import DensityInterface: logdensityof
import LogExpFunctions: logsumexp
import Random

laplace_population_logmixture(proposals, masses, x) =
    logsumexp(log.(masses) .+ [logdensityof(q, x) for q in proposals])
laplace_population_target(x, p) = p.valid[] ? logdensityof(p.law, x) : -Inf

@testset "Laplace APIS retains local mean updates and mixture weights" begin
    proposals =
        [IndependentLaplace([-1.0, 0.5], [0.8, 1.2]), IndependentLaplace([1.0, -0.5], [1.1, 0.7])]
    bank = ProposalBank(proposals)
    target = IndependentLaplace([0.3, -0.2], [0.9, 0.8])
    context = (; law = target, valid = Ref(true))
    sampler = prepare_sampler(
        Random.Xoshiro(2012),
        laplace_population_target,
        context,
        APIS(bank; rounds = 2, round_size = [128, 192]),
    )
    result = importance_sample!(sampler)
    fitted = copy(proposals)
    for round = 1:2
        indices = findall(==(round), result.provenance.round)
        expected = [
            logdensityof(target, x) - laplace_population_logmixture(fitted, bank.masses, x) for
            x in eachcol(result.samples[:, indices])
        ]
        @test result.logweights[indices] ≈ expected
        next = similar(fitted)
        for slot in eachindex(fitted)
            local_indices = filter(i -> result.provenance.proposal_id[i] == slot, indices)
            points = result.samples[:, local_indices]
            logs = [logdensityof(target, x)-logdensityof(fitted[slot], x) for x in eachcol(points)]
            weights = exp.(logs .- logsumexp(logs))
            next[slot] = IndependentLaplace(points * weights, proposals[slot].scale)
        end
        fitted = next
    end
    learned = current_proposal(sampler)
    @test [q.location for q in learned.proposals] ≈ [q.location for q in fitted]
    @test [q.scale for q in learned.proposals] == [q.scale for q in proposals]
    old_weights = copy(result.logweights)
    context.valid[] = false
    @test_throws APISRoundError importance_sample!(sampler)
    @test [q.location for q in current_proposal(sampler).proposals] == [q.location for q in learned.proposals]
    context.valid[] = true
    @test all(isfinite, importance_sample!(sampler).logweights) && result.logweights == old_weights
end

@testset "Laplace DM-PMC preserves ancestor scope, scales and inactive IDs" begin
    bank = ProposalBank(
        [
            IndependentLaplace(-1.0f0, 0.8f0),
            IndependentLaplace(100.0f0, 0.3f0),
            IndependentLaplace(1.0f0, 1.2f0),
        ],
        Float32[1, 0, 3],
    )
    target = IndependentLaplace(0.2f0, 0.9f0)
    for policy in (GlobalResampling(), LocalResampling())
        sampler = prepare_sampler(
            Random.Xoshiro(2013),
            target,
            DeterministicMixturePMC(bank; rounds = 1, round_size = 128, resampling = policy),
        )
        result = importance_sample!(sampler)
        expected = [
            logdensityof(target, x) - laplace_population_logmixture(bank.proposals, bank.masses, x) for
            x in result.samples
        ]
        @test result.logweights ≈ expected rtol=32eps(Float32) atol=32eps(Float32)
        learned = current_proposal(sampler)
        for slot in (1, 3)
            indices =
                policy isa GlobalResampling ? eachindex(result.samples) :
                findall(==(slot), result.provenance.proposal_id)
            @test learned.proposals[slot].location in result.samples[indices]
        end
        @test [q.scale for q in learned.proposals] == [q.scale for q in bank.proposals] &&
              learned.masses == bank.masses &&
              learned.proposals[2].location == 100.0f0
        next = importance_sample!(sampler)
        expected_next = [
            logdensityof(target, x) -
            laplace_population_logmixture(learned.proposals, learned.masses, x) for
            x in next.samples
        ]
        @test next.logweights ≈ expected_next rtol=32eps(Float32) atol=32eps(Float32)
    end
end

@testset "Laplace LAIS lower weights use the moved upper centres" begin
    proposals =
        [IndependentLaplace([-1.0, 0.5], [0.8, 1.2]), IndependentLaplace([1.0, -0.5], [1.1, 0.7])]
    bank = ProposalBank(proposals)
    logtarget(x) = -sum(abs2, x)/2
    sampler = prepare_sampler(
        Random.Xoshiro(2014),
        logtarget,
        LAIS(
            bank;
            transition = RandomWalkMetropolis([0.25 0.05; 0.05 0.2]),
            rounds = 1,
            round_size = 128,
        ),
    )
    result = importance_sample!(sampler)
    learned = current_proposal(sampler)
    expected = [
        logtarget(x)-laplace_population_logmixture(learned.proposals, learned.masses, x) for
        x in eachcol(result.samples)
    ]
    @test result.logweights ≈ expected
    @test [q.scale for q in learned.proposals] == [q.scale for q in proposals]
    @test any(learned.proposals[i].location != proposals[i].location for i in eachindex(proposals))
    learned.proposals[1].location .= 100
    @test current_proposal(sampler).proposals[1].location != learned.proposals[1].location
end
