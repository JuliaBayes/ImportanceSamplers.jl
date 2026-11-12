using Test, ImportanceSamplers, PureRNGs
import Random

struct ContinuationTestTransition <: AbstractMCMCTransition end

mutable struct ContinuationTestState{C}
    centres::C
    steps::Int
end

ImportanceSamplers.prepare_transition(
    ::ContinuationTestTransition,
    centres,
    target,
    ::Type,
) = ContinuationTestState(centres, 0)
ImportanceSamplers.transition_centres(state::ContinuationTestState) = state.centres
function Base.copyto!(destination::ContinuationTestState, source::ContinuationTestState)
    copyto!(destination.centres, source.centres)
    destination.steps = source.steps
    return destination
end
ImportanceSamplers.transition_diagnostics(state::ContinuationTestState) = (
    initial_target_evaluations = 0,
    warmup_target_evaluations = 0,
    production_target_evaluations = 0,
    warmup_proposals = 0,
    production_proposals = state.steps,
    accepted = state.steps,
)
ImportanceSamplers.retarget_transition(::ContinuationTestTransition, state, destination) =
    ContinuationTestTransition()
function ImportanceSamplers.transition!(
    state::ContinuationTestState,
    target,
    rng::Random.AbstractRNG,
    execution,
    transfers,
)
    state.centres .+= Random.randn(rng, length(state.centres))
    state.steps += 1
    return nothing
end
function ImportanceSamplers.transition!(
    state::ContinuationTestState,
    target,
    rng::PureRNGs.AbstractPureRNG,
    execution,
    transfers,
)
    normals, rng = PureRNGs.randn_next(rng, Float64, length(state.centres))
    state.centres .+= normals
    state.steps += 1
    return rng, nothing
end

function transition_reference_fill(rng, normals, uniforms; T = Float32)
    _, rng = PureRNGs.randn_next!(rng, Vector{T}(undef, normals))
    _, rng = PureRNGs.rand_next!(rng, Vector{T}(undef, uniforms))
    return rng
end

@testset "Custom transitions retain pure and original mutable CPU contracts" begin
    algorithm = LAIS(
        ProposalBank([SphericalGaussian(0.0, 1.0)]);
        transition = ContinuationTestTransition(),
        rounds = 1,
        round_size = 3,
    )
    for initial in (Threefry4x32(83), Random.Xoshiro(83))
        sampler = prepare_sampler(initial, Returns(0.0), algorithm; threaded = false)
        reference = deepcopy(initial)
        centre = 0.0
        for call = 1:2
            if reference isa Random.AbstractRNG
                centre += only(Random.randn(reference, 1))
                lower = Random.randn(reference, 3)
            else
                upper, reference = PureRNGs.randn_next(reference, Float64, 1)
                centre += only(upper)
                lower, reference = PureRNGs.randn_next(reference, Float64, 3)
            end
            result = importance_sample!(sampler)
            @test result.samples ≈ centre .+ lower
            snapshot = current_rng(sampler)
            if reference isa Random.AbstractRNG
                @test Random.rand(snapshot, 3) == Random.rand(deepcopy(reference), 3)
            else
                @test snapshot == reference
            end
        end
    end
end

@testset "Pure LAIS random walks follow direct draws across reuse" begin
    initial = Threefry4x32(41)
    algorithm = LAIS(
        ProposalBank([SphericalGaussian(zeros(Float32, 3), 0.7f0)]);
        transition = RandomWalkMetropolis(0.25f0),
        rounds = 2,
        round_size = 3,
    )
    serial = prepare_sampler(initial, Returns(0.0f0), algorithm; threaded = false)
    threaded = prepare_sampler(initial, Returns(0.0f0), algorithm; threaded = true)
    reference = initial
    centre = zeros(Float32, 3)
    for call = 1:2
        expected = Matrix{Float32}(undef, 3, 6)
        expected_weights = Vector{Float32}(undef, 6)
        for round = 1:2
            increments, reference = PureRNGs.randn_next!(reference, zeros(Float32, 3))
            _, reference = PureRNGs.rand_next!(reference, zeros(Float32, 1))
            centre .+= 0.5f0 .* increments
            normals, reference = PureRNGs.randn_next!(reference, zeros(Float32, 3, 3))
            columns = (3round-2):(3round)
            expected[:, columns] .= centre .+ 0.7f0 .* normals
            expected_weights[columns] .=
                3.0f0 * log(0.7f0 * sqrt(2.0f0 * Float32(pi))) .+
                vec(sum(abs2, normals; dims = 1)) ./ 2.0f0
        end
        actual = importance_sample!(serial)
        parallel = importance_sample!(threaded)
        @test actual.samples ≈ expected
        @test actual.logweights ≈ expected_weights
        @test parallel.samples == actual.samples
        @test current_rng(serial) == current_rng(threaded) == reference
    end
end

@testset "Pure transition fills retain warmup, radial, and cooperation streams" begin
    bank = ProposalBank([SphericalGaussian(zeros(Float32, 3), 0.7f0)])
    target = LogTarget(x -> -sum(abs2, x) / 2.0f0; grad = (g, x) -> (g .= .-x))
    hmc = HamiltonianMonteCarlo(;
        step_size = 0.1f0,
        leapfrog_steps = 2,
        tuning = WarmupTuning(2),
    )
    algorithms = (
        LAIS(
            bank;
            transition = RAM(0.25f0; tuning = WarmupTuning(2)),
            rounds = 2,
            round_size = 3,
        ),
        LAIS(
            bank;
            transition = RAM(0.25f0; tuning = ContinuousTuning()),
            rounds = 2,
            round_size = 3,
        ),
        LAIS(
            bank;
            transition = SampleMetropolisHastings(
                SphericalStudentT(1.0f0, zeros(Float32, 3), 1.0f0);
                moves = 3,
            ),
            rounds = 2,
            round_size = 3,
        ),
        LAIS(bank; transition = hmc, rounds = 2, round_size = 3),
        HAIS(bank; transition = hmc, rounds = 2, round_size = 3),
    )
    for (index, algorithm) in enumerate(algorithms)
        initial = Threefry4x32(50 + index)
        sampler = prepare_sampler(initial, target, algorithm; threaded = false)
        reference = initial
        for call = 1:2
            if call == 1 && index in (1, 4, 5)
                for warmup = 1:2
                    reference = transition_reference_fill(reference, 3, 1)
                end
            end
            for round = 1:2
                if index == 5
                    reference = transition_reference_fill(reference, 9, 0)
                    reference = transition_reference_fill(reference, 3, 1)
                    reference = transition_reference_fill(reference, 0, 1)
                else
                    reference =
                        index == 3 ? transition_reference_fill(reference, 12, 6) :
                        transition_reference_fill(reference, 3, 1)
                    reference = transition_reference_fill(reference, 9, 0)
                end
            end
            result = importance_sample!(sampler)
            @test current_rng(sampler) == reference
            @test all(isfinite, result.logweights)
            @test result.diagnostics.transition.warmup_proposals ==
                  (call == 1 && index in (1, 4, 5) ? 2 : 0)
        end
    end
end

@testset "Pure transition failures retain completed draws from later warmup" begin
    for hamiltonian in (false, true)
        calls = Ref(0)
        target_value = x -> begin
            calls[] += 1
            calls[] == 3 && error("later warmup target failure")
            return -sum(abs2, x) / 2.0f0
        end
        target = LogTarget(target_value; grad = (g, x) -> (g .= .-x))
        transition =
            hamiltonian ?
            HamiltonianMonteCarlo(;
                step_size = 0.1f0,
                leapfrog_steps = 1,
                tuning = WarmupTuning(3),
            ) : RAM(0.25f0; tuning = WarmupTuning(3))
        initial = Threefry4x32(63)
        sampler = prepare_sampler(
            initial,
            target,
            LAIS(
                ProposalBank([SphericalGaussian(zeros(Float32, 3), 0.7f0)]);
                transition,
                rounds = 1,
                round_size = 3,
            );
            threaded = false,
        )
        calls[] = 0
        @test_throws LAISRoundError importance_sample!(sampler)
        reference = transition_reference_fill(initial, 3, 1)
        reference = transition_reference_fill(reference, 3, 1)
        @test current_rng(sampler) == reference
        @test only(current_proposal(sampler).proposals).location == zeros(Float32, 3)
    end
end

@testset "Pure SMH failures consume the complete candidate draw group" begin
    calls = Ref(0)
    target = x -> begin
        calls[] += 1
        calls[] == 3 && error("candidate target failure")
        return -sum(abs2, x) / 2.0f0
    end
    initial = Threefry4x32(74)
    sampler = prepare_sampler(
        initial,
        target,
        LAIS(
            ProposalBank([SphericalGaussian(zeros(Float32, 3), 0.7f0)]);
            transition = SampleMetropolisHastings(
                SphericalGaussian(zeros(Float32, 3), 1.0f0);
                moves = 3,
            ),
            rounds = 1,
            round_size = 3,
        );
        threaded = false,
    )
    calls[] = 0
    @test_throws LAISRoundError importance_sample!(sampler)
    @test current_rng(sampler) == transition_reference_fill(initial, 9, 6)
    @test only(current_proposal(sampler).proposals).location == zeros(Float32, 3)
end

@testset "Pure batched transition failures retain their decision draws" begin
    calls = Ref(0)
    batch! = (values, samples) -> begin
        calls[] += 1
        calls[] == 2 && error("candidate batch failure")
        fill!(values, 0.0f0)
    end
    initial = Threefry4x32(95)
    sampler = prepare_sampler(
        initial,
        LogTarget(Returns(0.0f0); batch = batch!),
        LAIS(
            ProposalBank([SphericalGaussian(zeros(Float32, 3), 0.7f0)]);
            transition = RandomWalkMetropolis(0.25f0),
            rounds = 1,
            round_size = 3,
        );
        threaded = false,
    )
    @test_throws LAISRoundError importance_sample!(sampler)
    @test current_rng(sampler) == transition_reference_fill(initial, 3, 1)
    @test only(current_proposal(sampler).proposals).location == zeros(Float32, 3)
end
