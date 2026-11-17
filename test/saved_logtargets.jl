using Test, Random, ImportanceSamplers
import MLDataDevices

@testset "saved targets use existing scalar and batch evaluations" begin
    target(x) = x < 0 ? -Inf : -abs2(x)
    for proposal in (TestScalarProposal(0.0), SphericalGaussian(0.0, 1.0))
        calls = Ref(0)
        counted(x) = (calls[] += 1; target(x))
        algorithm = ImportanceSampling(proposal; nsamples=32)
        rng = Xoshiro(91)
        baseline = importance_sample(copy(rng), target, algorithm; threaded=false)
        actual = importance_sample(rng, counted, algorithm; threaded=false, save_logtargets=true)
        @test calls[] == length(actual)
        @test actual.logtargets == target.(actual.samples)
        @test any(==(-Inf), actual.logtargets)
        @test baseline.logtargets === nothing
        @test actual.samples == baseline.samples
        @test actual.logweights == baseline.logweights
        expected_rng = Xoshiro(91)
        importance_sample(expected_rng, target, algorithm; threaded=false)
        @test rand(rng) == rand(expected_rng)
    end

    batches = Ref(0)
    scalar_calls = Ref(0)
    scalar(x) = (scalar_calls[] += 1; target(x))
    batch!(out, xs) = (batches[] += 1; out .= target.(xs))
    result = importance_sample(
        Xoshiro(92),
        LogTarget(scalar; batch=batch!),
        ImportanceSampling(SphericalGaussian(0.0, 1.0); nsamples=32);
        save_logtargets=true,
        threaded=false,
    )
    @test batches[] == 1
    @test scalar_calls[] == 0
    @test result.logtargets == target.(result.samples)
end

@testset "saved static MIS batch values precede denominator arithmetic" begin
    for proposals in (
        [TestScalarProposal(-0.5), TestScalarProposal(0.5)],
        [SphericalGaussian(-0.5, 1.0), SphericalGaussian(0.5, 1.0)],
    )
        calls = Ref(0)
        scalar(x) = (calls[] += 1; -1e-20)
        batch!(out, xs) = fill!(out, -1e-20)
        result = importance_sample(
            Xoshiro(96),
            LogTarget(scalar; batch=batch!),
            ImportanceSampling(ProposalBank(proposals); nsamples=24);
            save_logtargets=true,
            threaded=false,
        )
        @test calls[] == 0
        @test result.logtargets == fill(-1e-20, length(result))
    end
end

@testset "saved targets retain adaptive history and ownership" begin
    target(x) = -sum(abs2, x) / 2
    proposal = SphericalGaussian(0.0, 2.0)
    bank = ProposalBank([SphericalGaussian(x, 1.5) for x in (-0.5, 0.5)])
    algorithms = (
        AMIS(proposal; rounds=2, round_size=[32, 24]),
        NPMC(proposal; rounds=2, round_size=[32, 24]),
        APIS(bank; rounds=2, round_size=[32, 24]),
        LAIS(bank; transition=RandomWalkMetropolis(0.3), rounds=2, round_size=[32, 24]),
    )
    for algorithm in algorithms
        sampler = prepare_sampler(Xoshiro(93), target, algorithm; save_logtargets=true, threaded=false)
        baseline = prepare_sampler(Xoshiro(93), target, algorithm; threaded=false)
        result = importance_sample!(sampler)
        expected = importance_sample!(baseline)
        saved = copy(result.logtargets)
        @test result.logtargets == target.(result.samples)
        @test result.samples == expected.samples
        @test result.logweights == expected.logweights
        @test result.provenance.round == vcat(fill(1, 32), fill(2, 24))
        next = importance_sample!(sampler)
        @test result.logtargets == saved
        @test next.logtargets == target.(next.samples)
    end

    factor_bank = ProposalBank([FactorGaussian([x, -x], [1.5 0.0; 0.2 1.2]) for x in (-0.5, 0.5)])
    grad!(g, x) = (g .= .-x)
    result = importance_sample(
        Xoshiro(94),
        LogTarget(target; grad=grad!),
        FirstOrderGRAMIS(factor_bank; rounds=2, round_size=[32, 24], repulsion_strength=0.0);
        save_logtargets=true,
        threaded=false,
    )
    @test result.logtargets == [target(x) for x in eachcol(result.samples)]
end

@testset "saved targets include sampling-coordinate Jacobians and aligned views" begin
    algorithm = ImportanceSampling(SphericalGaussian([0.0], 0.5); nsamples=32)
    target(x) = -x.scale
    transform = (scale=1 => PositiveTransform(),)
    result =
        importance_sample(Xoshiro(95), target, algorithm; transform, save_logtargets=true, threaded=false)
    coordinates = importance_sample(Xoshiro(95), x -> 0.0, algorithm; threaded=false).samples
    @test result.logtargets == [-exp(z) + z for z in vec(coordinates)]
    batch!(out, xs) = (out .= .-xs.scale)
    batched = importance_sample(
        Xoshiro(95),
        LogTarget(target; batch=batch!),
        algorithm;
        transform,
        save_logtargets=true,
        threaded=false,
    )
    @test batched.logtargets == result.logtargets
    selected = result[3:8]
    @test selected.logtargets == result.logtargets[3:8]
    @test keys(first(selected)) == (:sample, :logweight, :provenance)
    copied = MLDataDevices.cpu_device()(selected)
    @test copied.logtargets == selected.logtargets
    @test copied.logtargets !== selected.logtargets
    copied.logtargets[1] = 100
    @test result.logtargets[3] == selected.logtargets[1]
    smoothed = psis(result)
    @test smoothed[3:8].logtargets == selected.logtargets
    @test MLDataDevices.cpu_device()(smoothed).logtargets == result.logtargets
end
