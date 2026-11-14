using Test, Random, LinearAlgebra, ImportanceSamplers
import MLDataDevices

@testset "Retained scalar coordinates use generic and native IS/MIS draws" begin
    for proposals in (
            [TestScalarProposal(0.0f0), TestScalarProposal(0.5f0)],
            [SphericalGaussian(0.0f0, 1.0f0), SphericalGaussian(0.5f0, 1.0f0)],
        ),
        mis in (false, true)

        proposal = mis ? ProposalBank(proposals) : first(proposals)
        algorithm = ImportanceSampling(proposal; nsamples = 32)
        target(x) = -abs2(x)
        retain(x) = (target(x), 2x, log(2.0f0))
        configured = LogTarget(target; retain = RetainCoordinates(retain, 0.0))
        expected = importance_sample(Xoshiro(851), target, algorithm; threaded = false)
        result = importance_sample(Xoshiro(851), configured, algorithm; threaded = false)
        @test (result.samples, result.logweights, result.provenance) ==
              (expected.samples, expected.logweights, expected.provenance)
        @test (
            result.retained.coordinates,
            eltype(result.retained.coordinates),
            eltype(result.retained.logjacobians),
        ) == (2 .* result.samples, eltype(result.samples), eltype(result.samples))
        @test result.logtargets .- result.retained.logjacobians ≈
              target.(result.samples) .- log(2.0f0) rtol=2eps(Float32)
    end
end

@testset "Retained histories exclude upper MCMC and gradient trials" begin
    proposal = FactorGaussian(zeros(2), Matrix{Float64}(I, 2, 2))
    bank = ProposalBank([
        FactorGaussian([-1.0, 0.0], Matrix{Float64}(I, 2, 2)),
        FactorGaussian([1.0, 0.0], Matrix{Float64}(I, 2, 2)),
    ])
    target(x) = -sum(abs2, x) / 2
    gradient!(out, x) = (out .= .-x)
    for algorithm in (
        AMIS(proposal; rounds = 3, round_size = [24, 32, 40]),
        NPMC(proposal; rounds = 2, round_size = 32),
        LAIS(bank; transition = RandomWalkMetropolis(0.3), rounds = 2, round_size = 32),
        FirstOrderGRAMIS(bank; rounds = 2, round_size = 32, repulsion_strength = 0.1),
    )
        calls = Threads.Atomic{Int}(0)
        retain(x) = (Threads.atomic_add!(calls, 1); (target(x), x, 0.0))
        configured = LogTarget(
            target;
            grad = gradient!,
            retain = RetainCoordinates(retain, zeros(2)),
        )
        baseline =
            importance_sample(Xoshiro(852), LogTarget(target; grad = gradient!), algorithm)
        result = importance_sample(Xoshiro(852), configured, algorithm)
        @test (result.samples, result.logweights, result.provenance) ==
              (baseline.samples, baseline.logweights, baseline.provenance)
        @test calls[] == length(result)
        @test result.retained.coordinates == result.samples
        @test result.logtargets == [target(x) for x in eachcol(result.samples)]
    end
end

@testset "Retained batch context and named Jacobian composition" begin
    target(x, p) = -p.rate * x.scale
    retain(x, p) = (target(x, p), x, 0.0)
    function batch!(out, coordinates, jacobians, samples, p)
        out .= .-p.rate .* samples.scale
        coordinates.scale .= samples.scale
        fill!(jacobians, 0.0)
    end
    algorithm = ImportanceSampling(SphericalGaussian([0.0], 0.5); nsamples = 32)
    transform = (scale = 1 => PositiveTransform(),)
    policy = RetainCoordinates(retain, (; scale = 0.0); batch = batch!)
    configured = LogTarget(target; retain = policy)
    result =
        importance_sample(Xoshiro(853), configured, (; rate = 2.0), algorithm; transform)
    scalar = importance_sample(
        Xoshiro(853),
        LogTarget(target; retain = RetainCoordinates(retain, (; scale = 0.0))),
        (; rate = 2.0),
        algorithm;
        transform,
    )
    @test result.logweights == scalar.logweights
    @test result.retained == scalar.retained
    @test result.retained.coordinates.scale == result.samples.scale
    @test result.retained.logjacobians ≈ log.(result.samples.scale)
    @test result.logtargets .- result.retained.logjacobians ≈ -2 .* result.samples.scale
end

@testset "Retained buffers preserve alias, lifecycle and indexing contracts" begin
    target(x) = -sum(abs2, x)
    policy = RetainCoordinates(x -> (target(x), x, 0.0), zeros(2))
    sampler = prepare_sampler(
        Xoshiro(854),
        LogTarget(target; retain = policy),
        ImportanceSampling(SphericalGaussian(zeros(2), 1.0); nsamples = 32),
    )
    result = importance_sample!(sampler)
    snapshot = copy(result.retained.coordinates)
    importance_sample!(sampler)
    @test result.retained.coordinates == snapshot
    result.samples[1, 1] += 1
    @test result.retained.coordinates == snapshot
    selected = result[[7, 2, 5]]
    @test selected.retained.coordinates == snapshot[:, [7, 2, 5]]
    copied = MLDataDevices.cpu_device()(selected)
    copied.retained.coordinates[1, 1] += 1
    @test selected.retained.coordinates == snapshot[:, [7, 2, 5]]
    @test psis(result)[[7, 2, 5]].retained == selected.retained
end
