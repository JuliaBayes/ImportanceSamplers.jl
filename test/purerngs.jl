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
