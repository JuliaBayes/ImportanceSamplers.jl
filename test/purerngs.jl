using PureRNGs
import MLDataDevices

@testset "Native random stream lifecycle" begin
    rng = Threefry4x32(42)
    _, rng = rand_next(rng, Float64)
    algorithm = ImportanceSampling(SphericalGaussian(zeros(3), 1.0); nsamples = 17)
    target(x) = -sum(abs2, x) / 2
    sampler = prepare_sampler(rng, target, algorithm)
    placed = MLDataDevices.CPUDevice()(sampler)
    @test rngkey(current_rng(placed)) == rngkey(rng)
    @test rngposition(current_rng(placed)) == rngposition(rng)
    @test rngposition(current_rng(sampler)) == rngposition(rng)
    expected, next_rng = randn_next(rng, Float64, 3, 17)
    first = importance_sample!(placed)
    @test first.samples == expected
    @test rngposition(current_rng(placed)) == rngposition(next_rng)
    @test importance_sample(rng, target, algorithm).samples == expected
    expected, next_rng = randn_next(next_rng, Float64, 3, 17)
    @test importance_sample!(placed).samples == expected
    @test rngposition(current_rng(placed)) == rngposition(next_rng)

    unweighted, continued = resample_next(next_rng, first)
    replay, replay_rng = resample_next(next_rng, first, length(first))
    @test unweighted.samples == replay.samples
    @test rngposition(continued) == rngposition(replay_rng) > rngposition(next_rng)
end

@testset "Generic transform failures retain completed draws" begin
    rng = Threefry4x32(1)
    base = ProductProposal((
        x = SphericalGaussian(100.0f0, 1.0f0),
        g = IndependentGamma(2.0f0, 1.0f0),
    ))
    proposal = TransformedProposal(base, (x = PositiveTransform(),))
    prefix = BoxUniform(0.0f0, 1.0f0)
    for (input, control) in (
        (proposal, base),
        (ProposalBank([proposal]), ProposalBank([base])),
        (
            ProductProposal((prefix = prefix, inner = proposal)),
            ProductProposal((prefix = prefix, inner = base)),
        ),
    )
        sampler =
            prepare_sampler(rng, Returns(0.0f0), ImportanceSampling(input; nsamples = 1))
        reference =
            prepare_sampler(rng, Returns(0.0f0), ImportanceSampling(control; nsamples = 1))
        importance_sample!(reference)
        failure = try
            importance_sample!(sampler)
            nothing
        catch error
            error
        end
        @test failure isa SamplerExecutionError &&
              failure.phase == :proposal_draw &&
              failure.captured.ex isa InvalidTransformError
        @test current_rng(sampler) == current_rng(reference)
    end
end
