using Test
using ImportanceSamplers
import DensityInterface: logdensityof
import PureRNGs
import LogDensityProblems

struct PureMISInterval
    lower::Float64
end

logdensityof(q::PureMISInterval, x::Real) = q.lower <= x <= q.lower + 1 ? 0.0 : -Inf

function PureRNGs.rand_next(rng::PureRNGs.AbstractPureRNG, q::PureMISInterval)
    uniform, rng = PureRNGs.rand_next(rng, Float64)
    return q.lower + uniform, rng
end

pure_adaptive_target(x) = -sum(abs2, x) / 2 - length(x) * log(2pi) / 2
pure_adaptive_gradient!(out, x) = (out .= -x)
pure_adaptive_switch(x, p) = p.fail[] ? error("target failure") : logdensityof(p.q, x)
pure_adaptive_switch_gradient!(out, x, p) = (out .= -x)

struct PureProductDensity end
LogDensityProblems.capabilities(::Type{PureProductDensity}) =
    LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(::PureProductDensity) = 17
LogDensityProblems.logdensity(::PureProductDensity, x) =
    -sum(v -> v isa Number ? abs2(v) : sum(abs2, v), values(x)) / 2

@testset "pure generic MIS selection and proposal continuation" begin
    rng = PureRNGs.Threefry4x32(2401)
    bank = ProposalBank([PureMISInterval(0.0), PureMISInterval(2.0)])
    sampler = prepare_sampler(
        rng,
        x -> -log(2.0),
        ImportanceSampling(bank; nsamples = 17, mis_scheme = RandomMixture()),
    )
    for _ = 1:2
        assignments = Vector{Int}(undef, 17)
        for i in eachindex(assignments)
            uniform, rng = PureRNGs.rand_next(rng, Float64)
            assignments[i] = uniform <= 0.5 ? 1 : 2
        end
        expected = Vector{Float64}(undef, 17)
        for i in eachindex(expected)
            uniform, rng = PureRNGs.rand_next(rng, Float64)
            expected[i] = 2(assignments[i] - 1) + uniform
        end
        result = importance_sample!(sampler)
        @test result.samples == expected
        @test result.provenance.proposal_id == assignments
        @test result.logweights ≈ zeros(17) atol=8eps(Float64)
        @test current_rng(sampler) == rng
    end
end

@testset "pure moment rounds retain bulk continuation across reuse" begin
    q = SphericalGaussian(zeros(3), 1.0)
    for method in (AMIS, NPMC)
        rng = PureRNGs.Threefry4x32(2402)
        sampler = prepare_sampler(rng, q, method(q; rounds = 2, round_size = [17, 23]))
        for run = 1:2
            normal = Vector{Float64}(undef, 3 * 23)
            _, rng = PureRNGs.randn_next!(rng, normal)
            first_draws = copy(normal[1:(3*17)])
            _, rng = PureRNGs.randn_next!(rng, normal)
            result = importance_sample!(sampler)
            run == 1 && @test result.samples[:, 1:17] == reshape(first_draws, 3, 17)
            @test all(isfinite, result.logweights)
            @test current_rng(sampler) == rng
        end
    end
end

@testset "pure fixed populations continue proposals and resampling" begin
    bank = ProposalBank([
        SphericalGaussian([-1.0, 0.0, 0.0], 1.0),
        SphericalGaussian([1.0, 0.0, 0.0], 1.0),
    ])
    target = LogTarget(pure_adaptive_target; grad = pure_adaptive_gradient!)
    algorithms = (
        APIS(bank; rounds = 2, round_size = [18, 24]),
        CAIS(bank; rounds = 2, round_size = [18, 24]),
        DeterministicMixturePMC(bank; rounds = 2, round_size = [18, 24]),
        DeterministicMixturePMC(
            bank;
            rounds = 2,
            round_size = [18, 24],
            resampling = LocalResampling(),
        ),
        FirstOrderGRAMIS(bank; rounds = 2, round_size = [18, 24], repulsion_strength = 0.1),
    )
    for algorithm in algorithms
        rng = PureRNGs.Threefry4x32(2403)
        sampler = prepare_sampler(rng, target, algorithm)
        for run = 1:2
            for count in (18, 24)
                normal_count = algorithm isa FirstOrderGRAMIS ? 3count : 3 * 24
                _, rng = PureRNGs.randn_next!(rng, Vector{Float64}(undef, normal_count))
                if algorithm isa DeterministicMixturePMC
                    _, rng = PureRNGs.rand_next!(rng, Vector{Float64}(undef, 2))
                end
            end
            result = importance_sample!(sampler)
            @test current_rng(sampler) == rng
            @test all(isfinite, result.logweights)
            if run == 1
                expected = map(
                    eachcol(result.samples[:, 1:18]),
                    result.provenance.proposal_id[1:18],
                ) do x, id
                    mixture = log(sum(exp(logdensityof(q, x)) for q in bank.proposals) / 2)
                    denominator =
                        algorithm isa CAIS ? logdensityof(bank.proposals[id], x) : mixture
                    pure_adaptive_target(x) - denominator
                end
                @test result.logweights[1:18] ≈ expected atol=64eps(Float64)
            end
        end
    end
end

@testset "pure independent and Student-t adaptive fills replay and continue" begin
    for (q, independent) in (
        (IndependentGamma([2.0, 3.0, 4.0], [1.0, 0.5, 0.75]), true),
        (IndependentLaplace(zeros(3), ones(3)), true),
        (SphericalStudentT(7.0, zeros(3), 1.0), false),
    )
        rng = PureRNGs.Threefry4x32(2404)
        bank = ProposalBank([q, q])
        algorithms = (
            ImportanceSampling(bank; nsamples = 33),
            AMIS(q; rounds = 2, round_size = [32, 48]),
            APIS(bank; rounds = 2, round_size = [32, 48]),
            DeterministicMixturePMC(bank; rounds = 2, round_size = [32, 48]),
        )
        for algorithm in (independent ? algorithms : (algorithms[3],))
            serial = prepare_sampler(rng, q, algorithm; threaded = false)
            threaded = prepare_sampler(rng, q, algorithm; threaded = true)
            previous = rng
            for _ = 1:2
                left, right = importance_sample!(serial), importance_sample!(threaded)
                @test left.samples == right.samples
                @test left.logweights ≈ right.logweights atol=128eps(Float64)
                @test current_rng(serial) == current_rng(threaded) != previous
                previous = current_rng(serial)
            end
        end
    end
end

@testset "pure generic named products preserve component laws and transforms" begin
    base = ProductProposal((
        gaussian = SphericalGaussian(0.2f0, 1.1f0),
        student_vector = SphericalStudentT(1.0f0, zeros(Float32, 3), 1.0f0),
        gaussian_vector = FactorGaussian(
            zeros(Float32, 3),
            Float32[1 0 0; 0.2 1 0; 0 0.1 1],
        ),
        student = SphericalStudentT(7.0f0, 0.0f0, 1.0f0),
        gamma = IndependentGamma(0.7f0, 1.0f0),
        gamma_vector = IndependentGamma(Float32[2, 3], Float32[1, 0.5]),
        box = BoxUniform(0.0f0, 1.0f0),
        box_vector = BoxUniform(zeros(Float32, 2), ones(Float32, 2)),
        laplace = IndependentLaplace(0.0f0, 1.0f0),
        laplace_vector = IndependentLaplace(zeros(Float32, 2), ones(Float32, 2)),
    ))
    proposal = TransformedProposal(base, (gaussian = PositiveTransform(),))
    target = PureProductDensity()
    algorithm = ImportanceSampling(proposal; nsamples = 17)
    sampler = prepare_sampler(PureRNGs.Threefry4x32(2408), target, algorithm)
    for _ = 1:2
        snapshot = current_rng(sampler)
        normal, reference = PureRNGs.randn_next(snapshot, Float32)
        student_normals, reference = PureRNGs.randn_next(reference, Float32, 3)
        radial_normal, reference = PureRNGs.randn_next(reference, Float32)
        result = importance_sample!(sampler)
        @test result.samples.gaussian[1] == exp(0.2f0 + 1.1f0 * normal)
        @test result.samples.student_vector[:, 1] ==
              student_normals .* inv(abs(radial_normal))
        expected = map(1:length(result)) do i
            x = map(array -> ndims(array) == 1 ? array[i] : array[:, i], result.samples)
            coordinate = merge(x, (gaussian = log(x.gaussian),))
            logproposal =
                sum(map(logdensityof, base.blocks, coordinate)) - log(x.gaussian)
            LogDensityProblems.logdensity(target, x) - logproposal
        end
        @test result.logweights ≈ expected rtol=64eps(Float32)
        replay = importance_sample(snapshot, target, algorithm)
        @test result.samples == replay.samples
        @test current_rng(sampler) != snapshot
    end
end

@testset "pure static box fills retain selection and coordinate streams" begin
    q = BoxUniform(zeros(3), ones(3))
    rng = PureRNGs.Threefry4x32(2406)
    sampler =
        prepare_sampler(rng, q, ImportanceSampling(ProposalBank([q, q]); nsamples = 17))
    for _ = 1:2
        _, rng = PureRNGs.rand_next!(rng, Vector{Float64}(undef, 17))
        coordinates = Vector{Float64}(undef, 3 * 17)
        _, rng = PureRNGs.rand_next!(rng, coordinates)
        result = importance_sample!(sampler)
        @test result.samples == reshape(coordinates, 3, 17)
        @test result.logweights ≈ zeros(17) atol=8eps(Float64)
        @test current_rng(sampler) == rng
    end
end

@testset "pure adaptive target failures retain the completed draw phase" begin
    q = SphericalGaussian(zeros(3), 1.0)
    bank = ProposalBank([q, q])
    for algorithm in (
        AMIS(q; rounds = 2, round_size = [17, 23]),
        APIS(bank; rounds = 2, round_size = [18, 24]),
        DeterministicMixturePMC(bank; rounds = 2, round_size = [18, 24]),
    )
        rng = PureRNGs.Threefry4x32(2405)
        parameters = (; q, fail = Ref(false))
        sampler = prepare_sampler(rng, pure_adaptive_switch, parameters, algorithm)
        parameters.fail[] = true
        @test_throws Union{AMISRoundError,APISRoundError,DMPMCRoundError} importance_sample!(
            sampler,
        )
        maximum_count = algorithm isa AMIS ? 23 : 24
        _, rng = PureRNGs.randn_next!(rng, Vector{Float64}(undef, 3maximum_count))
        @test current_rng(sampler) == rng
        parameters.fail[] = false
        @test all(isfinite, importance_sample!(sampler).logweights)
    end
    distinct_bank = ProposalBank([
        SphericalGaussian([-1.0, 0.0, 0.0], 1.0),
        SphericalGaussian([1.0, 0.0, 0.0], 1.0),
    ])
    rng = PureRNGs.Threefry4x32(2407)
    parameters = (; q, fail = Ref(false))
    sampler = prepare_sampler(
        rng,
        LogTarget(pure_adaptive_switch; grad = pure_adaptive_switch_gradient!),
        parameters,
        FirstOrderGRAMIS(
            distinct_bank;
            rounds = 2,
            round_size = [18, 24],
            repulsion_strength = 0.1,
        ),
    )
    parameters.fail[] = true
    @test_throws FirstOrderGRAMISRoundError importance_sample!(sampler)
    _, rng = PureRNGs.randn_next!(rng, Vector{Float64}(undef, 3 * 18))
    @test current_rng(sampler) == rng
    parameters.fail[] = false
    @test all(isfinite, importance_sample!(sampler).logweights)
end
