using ImportanceSamplers, Random, Statistics, Test
import DensityInterface: logdensityof
import MLDataDevices

"Run normalized-density, MIS and residency oracles on an explicitly selected device."
function check_independent_proposals(;
    device = MLDataDevices.CPUDevice(),
    isresident = x -> x isa Array,
    T = Float64,
    count = 4096,
)
    cases = (
        (BoxUniform(T[-2, 0], T[2, 3]), BoxUniform(T[-3, -1], T[3, 4])),
        (IndependentGamma(T[0.5, 5], T[2, 0.2]), IndependentGamma(T[2, 3], T[1, 1])),
        (IndependentLaplace(T[0, 1], T[1, 0.5]), IndependentLaplace(T[0.5, -1], T[0.8, 2])),
    )
    @testset "independent proposals on $(typeof(device)), $T" begin
        for (target, other) in cases
            single =
                prepare_sampler(Xoshiro(1801), target, ImportanceSampling(target; nsamples = count))
            result = importance_sample!(device(single))
            @test isresident(result.samples) && isresident(result.logweights)
            host = MLDataDevices.CPUDevice()(result)
            @test maximum(abs, host.logweights) <= 128eps(T)
            bank = ProposalBank([target, other], T[0.3, 0.7])
            for scheme in (
                StratifiedMixture(),
                RandomMixture(),
                StandardMIS(),
                PartialDeterministicMixture([[1], [2]]),
            )
                sampler = prepare_sampler(
                    Xoshiro(1802),
                    target,
                    ImportanceSampling(bank; nsamples = count, mis_scheme = scheme),
                )
                samples = importance_sample!(device(sampler))
                @test isresident(samples.samples) && isresident(samples.logweights)
                cpu = MLDataDevices.CPUDevice()(samples)
                expected = map(eachindex(cpu.logweights)) do i
                    x = cpu.samples[:, i]
                    logdensities = logdensityof.((target, other), Ref(x))
                    generating = cpu.provenance.proposal_id[i]
                    denominator =
                        scheme isa Union{StandardMIS,PartialDeterministicMixture} ?
                        logdensities[generating] : log(sum(bank.masses .* exp.(logdensities)))
                    logdensityof(target, x) - denominator
                end
                @test cpu.logweights ≈ expected rtol=128eps(T) atol=128eps(T)
            end
        end
    end
    return nothing
end

independent_named_target(theta, p) = -theta.rate - theta.offset^2 / 2
independent_positive_target(x, base) = logdensityof(base, log(x)) - log(x)

"Check shared transforms and whole-call draw failure without scalar device reads."
function check_independent_proposal_edges(; device = MLDataDevices.CPUDevice(), T = Float32)
    @testset "independent proposal edges on $(typeof(device)), $T" begin
        for base in (BoxUniform(-one(T), one(T)), IndependentGamma(T(2), T(0.2)))
            q = TransformedProposal(base, PositiveTransform())
            sampler = prepare_sampler(
                Xoshiro(1803),
                independent_positive_target,
                base,
                ImportanceSampling(q; nsamples = 256),
            )
            result = MLDataDevices.CPUDevice()(importance_sample!(device(sampler)))
            @test maximum(abs, result.logweights) <= 128eps(T)
        end
        layout = (; rate = 1=>PositiveTransform(), offset = 2)
        for q in (BoxUniform(T[-1, -1], T[1, 1]), IndependentGamma(T[2, 2], T[0.2, 0.2]))
            sampler = prepare_sampler(
                Xoshiro(1804),
                independent_named_target,
                nothing,
                ImportanceSampling(q; nsamples = 256);
                transform = layout,
            )
            result = MLDataDevices.CPUDevice()(importance_sample!(device(sampler)))
            expected = map(eachindex(result.logweights)) do i
                theta = result[i].sample
                z = T[log(theta.rate), theta.offset]
                independent_named_target(theta, nothing) + log(theta.rate) - logdensityof(q, z)
            end
            @test result.logweights ≈ expected rtol=128eps(T) atol=128eps(T)
        end
        q = IndependentGamma(T(1e-10), one(T))
        sampler = device(prepare_sampler(Xoshiro(1805), q, ImportanceSampling(q; nsamples = 1)))
        failure = try
            importance_sample!(sampler)
        catch error
            error
        end
        @test failure isa SamplerExecutionError && failure.phase === :proposal_draw
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    check_independent_proposals()
    check_independent_proposal_edges()
end
