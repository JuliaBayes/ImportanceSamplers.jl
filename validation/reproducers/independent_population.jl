using ImportanceSamplers, Random, Test, DensityInterface
using ImportanceSamplers.LogExpFunctions: logsumexp
import MLDataDevices

population_probe_logmixture(proposals, masses, x) =
    logsumexp(log.(masses) .+ [logdensityof(q, x) for q in proposals])

function check_independent_population(;
    device = MLDataDevices.CPUDevice(),
    isresident = x->x isa Array,
    T = Float32,
    count = 2048,
)
    rows = []
    tests = @testset "Independent populations on $(typeof(device)), $T" begin
        for family in (:gamma, :laplace)
            proposals, target = if family === :gamma
                (
                    [
                        IndependentGamma(T[20, 30], T[0.25, 0.25]),
                        IndependentGamma(T[24, 28], T[0.25, 0.25]),
                    ],
                    IndependentGamma(T[30, 36], T[0.2, 0.2]),
                )
            else
                (
                    [
                        IndependentLaplace(T[-1, 0.5], T[0.8, 1.2]),
                        IndependentLaplace(T[1, -0.5], T[1.1, 0.7]),
                    ],
                    IndependentLaplace(T[0.3, -0.2], T[0.9, 0.8]),
                )
            end
            bank = ProposalBank(proposals)
            algorithms = (
                APIS(bank; rounds = 1, round_size = count),
                DeterministicMixturePMC(bank; rounds = 1, round_size = count),
                DeterministicMixturePMC(
                    bank;
                    rounds = 1,
                    round_size = count,
                    resampling = LocalResampling(),
                ),
            )
            if family === :laplace
                algorithms = (
                    algorithms...,
                    LAIS(
                        bank;
                        transition = RandomWalkMetropolis(T[0.25 0.05; 0.05 0.2]),
                        rounds = 1,
                        round_size = count,
                    ),
                )
            end
            for algorithm in algorithms
                sampler = device(prepare_sampler(Xoshiro(2040), target, algorithm))
                result = importance_sample!(sampler)
                learned = current_proposal(MLDataDevices.CPUDevice(), sampler)
                samples, logs = Array(result.samples), Array(result.logweights)
                denominator = algorithm isa LAIS ? learned : bank
                expected = [
                    logdensityof(target, x)-population_probe_logmixture(
                        denominator.proposals,
                        denominator.masses,
                        x,
                    ) for x in eachcol(samples)
                ]
                @test isresident(result.samples) && isresident(result.logweights)
                @test logs ≈ expected rtol=256eps(T) atol=256eps(T)
                if algorithm isa APIS
                    for slot in eachindex(proposals)
                        indices = findall(==(slot), Array(result.provenance.proposal_id))
                        points = samples[:, indices]
                        local_logs = [
                            logdensityof(target, x)-logdensityof(proposals[slot], x) for
                            x in eachcol(points)
                        ]
                        mean = points * exp.(local_logs .- logsumexp(local_logs))
                        q = learned.proposals[slot]
                        got = family === :gamma ? q.shape .* q.scale : q.location
                        @test got ≈ mean rtol=256eps(T) atol=256eps(T)
                    end
                end
                if family === :gamma
                    @test [(q.shape .* q.scale) .* q.scale for q in learned.proposals] ≈ [(q.shape .* q.scale) .* q.scale for q in proposals]
                else
                    @test [q.scale for q in learned.proposals] == [q.scale for q in proposals]
                end
                saved_logs = copy(logs)
                next = importance_sample!(sampler)
                if !(algorithm isa LAIS)
                    expected_next = [
                        logdensityof(target, x)-population_probe_logmixture(
                            learned.proposals,
                            learned.masses,
                            x,
                        ) for x in eachcol(Array(next.samples))
                    ]
                    @test Array(next.logweights) ≈ expected_next rtol=512eps(T) atol=512eps(T)
                end
                @test Array(result.logweights) == saved_logs
                traffic = result.diagnostics.transfers
                push!(
                    rows,
                    (;
                        family,
                        method = nameof(typeof(algorithm)),
                        transfers = traffic.count,
                        transfer_bytes = traffic.bytes,
                    ),
                )
            end
        end
    end
    return (; rows, counts = Test.get_test_counts(tests))
end

function check_independent_population_precision(device)
    q = IndependentGamma([prevfloat(32.0)], [1.0])
    bank = ProposalBank([q])
    algorithms = (
        ImportanceSampling(q; nsamples = 256),
        ImportanceSampling(bank; nsamples = 256),
        AMIS(q; rounds = 1, round_size = 256),
        NPMC(q; rounds = 1, round_size = 256),
        APIS(bank; rounds = 1, round_size = 256),
        DeterministicMixturePMC(bank; rounds = 1, round_size = 256),
    )
    @testset "Independent prepared precision on $(typeof(device))" begin
        for algorithm in algorithms
            result = importance_sample!(device(prepare_sampler(Xoshiro(2043), q, algorithm)))
            @test eltype(result.samples) === eltype(result.logweights) === Float32 &&
                  maximum(abs, Array(result.logweights)) <= 256eps(Float32)
        end
        laplace = IndependentLaplace([0.0], [1.0])
        p = device(
            prepare_sampler(
                Xoshiro(2044),
                laplace,
                LAIS(
                    ProposalBank([laplace]);
                    transition = RandomWalkMetropolis(0.1),
                    rounds = 1,
                    round_size = 256,
                ),
            ),
        )
        result = importance_sample!(p)
        learned = current_proposal(MLDataDevices.CPUDevice(), p)
        expected = [
            logdensityof(IndependentLaplace(Float32[0], Float32[1]), x) -
            population_probe_logmixture(learned.proposals, learned.masses, x) for
            x in eachcol(Array(result.samples))
        ]
        @test eltype(result.samples) === Float32 && Array(result.logweights) ≈ expected
        invalid = ProposalBank([BoxUniform([1.0], [nextfloat(1.0)])])
        rng = Xoshiro(2045)
        source = prepare_sampler(rng, Returns(0.0f0), ImportanceSampling(invalid; nsamples = 8))
        snapshot = copy(rng)
        @test_throws ArgumentError device(source)
        @test rand(rng, UInt64) == rand(snapshot, UInt64)
    end
end
