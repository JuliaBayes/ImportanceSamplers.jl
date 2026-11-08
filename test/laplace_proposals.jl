import DensityInterface: logdensityof
import Distributions
import Random
import Statistics

struct LaplaceEndpointRNG{T} <: Random.AbstractRNG
    value::T
end
Random.rand(rng::LaplaceEndpointRNG{T}, ::Type{T}) where {T} = rng.value
Random.rand!(rng::LaplaceEndpointRNG, values::AbstractArray) = fill!(values, rng.value)

@testset "Laplace density, independent sampling law and parameter ownership" begin
    location, scale = Float32[-1, 2], Float32[0.5, 1.5]
    proposal = IndependentLaplace(location, scale)
    location .= 100
    scale .= 100
    oracles = Distributions.Laplace.(Float32[-1, 2], Float32[0.5, 1.5])
    point = Float32[-1.4, 3]
    @test logdensityof(proposal, point) ≈ sum(Distributions.logpdf.(oracles, point))
    @test logdensityof(proposal, Float32[Inf, 2]) == -Inf32

    # Seed 1720, 20,000 draws. Mean/variance tolerances exceed six analytic SEs.
    result = importance_sample(
        Random.Xoshiro(1720),
        proposal,
        ImportanceSampling(proposal; nsamples = 20_000),
    )
    @test all(iszero, result.logweights)
    @test maximum(abs.(Statistics.mean(result) .- Float32[-1, 2])) < 0.1f0
    @test Statistics.var(result) ≈ Float32[0.5, 4.5] rtol=0.12
    @test abs(Statistics.cov(result)[1, 2]) < 0.07f0
    @test eltype(result.samples) === eltype(result.logweights) === Float32
    # The non-Gaussian law has one-quarter of its mass beyond log(2) scales.
    @test count(<(-1.0f0 - 0.5f0 * log(2.0f0)), result.samples[1, :]) / 20_000 ≈ 0.25 atol=0.02
    @test count(>(2.0f0 + 1.5f0 * log(2.0f0)), result.samples[2, :]) / 20_000 ≈ 0.25 atol=0.02
end

@testset "Laplace endpoint uniforms produce finite scalar and native draws" begin
    for T in (Float32, Float64)
        proposal = IndependentLaplace(T(1), T(2))
        for uniform in (zero(T), T(0.5), one(T))
            @test rand(LaplaceEndpointRNG(uniform), proposal) == one(T)
            result = importance_sample(
                LaplaceEndpointRNG(uniform),
                proposal,
                ImportanceSampling(proposal; nsamples = 2),
            )
            @test result.samples == fill(one(T), 2) && all(iszero, result.logweights)
        end
        @test rand(LaplaceEndpointRNG(T(0.25)), proposal) ≈ one(T) - T(2) * log(T(2))
        @test rand(LaplaceEndpointRNG(T(0.75)), proposal) ≈ one(T) + T(2) * log(T(2))
        @test isfinite(rand(LaplaceEndpointRNG(prevfloat(one(T))), proposal))
    end
    bound = floatmax(Float64)
    extreme = IndependentLaplace(-bound, bound)
    expected = setprecision(BigFloat, 128) do
        -log(2BigFloat(bound)) - 2
    end
    @test logdensityof(extreme, bound) ≈ Float64(expected)

    overflow = IndependentLaplace(bound, bound)
    failure = try
        importance_sample(
            LaplaceEndpointRNG(0.75),
            overflow,
            ImportanceSampling(overflow; nsamples = 1),
        )
    catch error
        error
    end
    @test failure isa SamplerExecutionError && failure.phase === :proposal_draw
end

@testset "Laplace static MIS denominators use independent coordinate densities" begin
    locations = ([-1.0, 2.0], [2.0, -1.0])
    scales = ([0.5, 1.5], [2.0, 0.7])
    proposals = [IndependentLaplace(locations[j], scales[j]) for j = 1:2]
    oracles = [Distributions.Laplace.(locations[j], scales[j]) for j = 1:2]
    masses = [0.3, 0.7]
    target = IndependentLaplace([0.0, 0.0], [1.0, 1.0])
    target_oracles = Distributions.Laplace.([0.0, 0.0], [1.0, 1.0])
    bank = ProposalBank(proposals, masses)
    for scheme in (StratifiedMixture(), StandardMIS())
        result = importance_sample(
            Random.Xoshiro(1721),
            target,
            ImportanceSampling(bank; nsamples = 256, mis_scheme = scheme),
        )
        expected = map(eachindex(result.logweights)) do i
            point = result.samples[:, i]
            densities = [sum(Distributions.logpdf.(q, point)) for q in oracles]
            slot = result.provenance.proposal_id[i]
            denominator =
                scheme isa StandardMIS ? densities[slot] : log(sum(masses .* exp.(densities)))
            sum(Distributions.logpdf.(target_oracles, point)) - denominator
        end
        @test result.logweights ≈ expected
    end
end

@testset "transformed Laplace density includes the change of variables" begin
    proposal = TransformedProposal(IndependentLaplace(0.5f0, 0.7f0), PositiveTransform())
    oracle = Distributions.Laplace(0.5f0, 0.7f0)
    result = importance_sample(
        Random.Xoshiro(1722),
        _ -> 0.0f0,
        ImportanceSampling(proposal; nsamples = 128),
    )
    expected = [-Distributions.logpdf(oracle, log(x)) + log(x) for x in result.samples]
    @test result.logweights ≈ expected rtol=8eps(Float32)
end
