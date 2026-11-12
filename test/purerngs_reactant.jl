using ImportanceSamplers, PureRNGs, Reactant, CUDA, Test

Reactant.set_default_backend("cpu")

function purerngs_compiled_uniforms(rng, out)
    _, rng = rand_next!(rng, out)
    return copy(out), rng
end

purerngs_reactant_target(x) = -(x[1]^2 + x[2]^2 + x[3]^2) / 2.0f0

@testset "Reactant explicit RNG placement" begin
    client = Reactant.XLA.client("cpu")
    previous = Reactant.XLA.default_backend()
    default = CUDA.functional() ? Reactant.XLA.client("gpu") : previous
    Reactant.set_default_backend(default)
    try
        physical = last(Reactant.XLA.addressable_devices(client))
        sharding = Reactant.Sharding.NoSharding()
        device = ImportanceSamplers.MLDataDevices.with_eltype(
            ImportanceSamplers.MLDataDevices.ReactantDevice(client, physical, sharding), nothing)
        seed = Threefry4x32(typemax(UInt128) - 17)
        _, seed = rand_next(seed, UInt32, 7)
        algorithm = ImportanceSampling(SphericalGaussian(zeros(Float32, 3), 1f0); nsamples=17)
        sampler = device(prepare_sampler(seed, purerngs_reactant_target, algorithm))
        snapshot = current_rng(sampler)
        actual = ImportanceSamplers.MLDataDevices.get_device(snapshot)
        @test actual.client == client && actual.device == physical
        result = importance_sample!(sampler)
        @test Array(result.logweights) ≈ fill(1.5f0 * log(2f0 * Float32(pi)), 17)
        @test Array(normalized_weights(result[1:3])) ≈ fill(1f0 / 3, 3)
        inherited = ImportanceSamplers.MLDataDevices.with_eltype(actual, nothing)
        replay = inherited(prepare_sampler(seed, purerngs_reactant_target, algorithm))
        @test Array(importance_sample!(replay).samples) == Array(result.samples)
        out = Reactant.to_rarray(zeros(Float32, 17); client, device=physical, sharding)
        step = Reactant.@compile client=client purerngs_compiled_uniforms(snapshot, out)
        values, _ = step(snapshot, out)
        expected, _ = rand_next(seed, Float32, 17)
        @test Array(values) == expected
        point_mass = device(WeightedSamples(Float32[1, 2, 3], Float32[-Inf, 0, -Inf]))
        @test Array(first(resample_next(current_rng(sampler), point_mass, 7)).samples) == fill(2f0, 7)
        @test Reactant.XLA.default_backend() == default
    finally
        Reactant.set_default_backend(previous)
    end
end

@testset "Reactant runtime RNG and prepared continuation" begin
    device = ImportanceSamplers.MLDataDevices.with_eltype(
        ImportanceSamplers.MLDataDevices.ReactantDevice(),
        nothing,
    )
    seed = Threefry4x32(42)
    _, seed = rand_next(seed, UInt32, 7)
    carrier = Reactant.to_rarray(seed)
    uniforms = Reactant.to_rarray(zeros(Float32, 17))
    step = Reactant.@compile purerngs_compiled_uniforms(carrier, uniforms)
    first, continued = step(carrier, uniforms)
    second, _ = step(continued, uniforms)
    expected_first, reference = rand_next(seed, Float32, 17)
    expected_second, _ = rand_next(reference, Float32, 17)
    @test Array(first) == expected_first
    @test Array(second) == expected_second
    replay, _ = step(deepcopy(continued), uniforms)
    @test Array(replay) == expected_second
    changed, _ = step(Reactant.to_rarray(Threefry4x32(43)), uniforms)
    expected, _ = rand_next(Threefry4x32(43), Float32, 17)
    @test Array(changed) == expected

    proposal = SphericalGaussian(zeros(Float32, 3), 1.0f0)
    algorithm = ImportanceSampling(proposal; nsamples = 17)
    sampler = device(prepare_sampler(seed, purerngs_reactant_target, algorithm))
    first_result = importance_sample!(sampler)
    saved = Array(first_result.samples)
    second_result = importance_sample!(sampler)
    @test Array(first_result.samples) == saved
    @test Array(second_result.samples) != saved
    @test Array(first_result.logweights) ≈ fill(1.5f0 * log(2.0f0 * Float32(pi)), 17)
    _, reference = randn_next(seed, Float32, 51)
    _, reference = randn_next(reference, Float32, 51)
    expected, _ = rand_next(reference, Float32, 17)
    snapshot = current_rng(sampler)
    actual, _ = step(snapshot, uniforms)
    @test Array(actual) == expected
    importance_sample!(sampler)
    replay, _ = step(snapshot, uniforms)
    @test Array(replay) == expected

    failed = device(prepare_sampler(seed, x -> NaN32, algorithm))
    @test_throws SamplerExecutionError importance_sample!(failed)
    actual, _ = step(current_rng(failed), uniforms)
    _, reference = randn_next(seed, Float32, 51)
    expected, _ = rand_next(reference, Float32, 17)
    @test Array(actual) == expected
end

@testset "Reactant continued retargeting and result resampling" begin
    device = ImportanceSamplers.MLDataDevices.with_eltype(
        ImportanceSamplers.MLDataDevices.ReactantDevice(),
        nothing,
    )
    proposal = SphericalGaussian(zeros(Float32, 3), 1.0f0)
    bank = ProposalBank([proposal, proposal])
    sampler = device(
        prepare_sampler(
            Threefry4x32(51),
            purerngs_reactant_target,
            DeterministicMixturePMC(bank; rounds = 2, round_size = 18),
        ),
    )
    result = importance_sample!(sampler)
    snapshot = current_rng(sampler)
    retargeted = retarget(snapshot, sampler, purerngs_reactant_target)
    @test size(importance_sample!(retargeted).samples) == (3, 36)
    out = Reactant.to_rarray(zeros(Float32, 11))
    step = Reactant.@compile purerngs_compiled_uniforms(snapshot, out)
    original, _ = step(snapshot, out)
    unchanged, _ = step(current_rng(sampler), out)
    @test Array(original) == Array(unchanged)
    point_mass = device(WeightedSamples(Float32[1, 2, 3], Float32[-Inf, 0, -Inf]))
    draws, next_rng = resample_next(snapshot, point_mass, 11)
    @test Array(draws.samples) == fill(2.0f0, 11)
    _, expected_rng = step(snapshot, out)
    actual, _ = step(next_rng, out)
    expected, _ = step(expected_rng, out)
    @test Array(actual) == Array(expected)
    @test length(first(resample_next(snapshot, result))) == length(result)

    failed = device(
        prepare_sampler(
            Threefry4x32(52),
            x -> iszero(x[1]) ? 0.0f0 : NaN32,
            LAIS(
                bank;
                transition = RAM(0.25f0; tuning = WarmupTuning(0)),
                rounds = 2,
                round_size = 18,
            ),
        ),
    )
    @test_throws LAISRoundError importance_sample!(failed)
    _, reference = randn_next(Threefry4x32(52), Float32, 6)
    _, reference = rand_next(reference, Float32, 2)
    expected, _ = rand_next(reference, Float32, 11)
    actual, _ = step(current_rng(failed), out)
    @test Array(actual) == expected
end
