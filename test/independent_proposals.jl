import DensityInterface: logdensityof
import Distributions
import Random
import Statistics
import SpecialFunctions

struct BoxEndpointRNG{T} <: Random.AbstractRNG
    value::T
end
Random.rand(rng::BoxEndpointRNG{T}, ::Type{T}) where {T} = rng.value

mutable struct RejectingGammaRNG <: Random.AbstractRNG
    filled_normals::Bool
    filled_uniforms::Bool
end
function Random.randn!(rng::RejectingGammaRNG, values::AbstractArray{T}) where {T}
    rng.filled_normals = true
    return fill!(values, T(-100))
end
function Random.rand!(rng::RejectingGammaRNG, values::AbstractArray{T}) where {T}
    rng.filled_uniforms = true
    return fill!(values, T(0.5))
end

@testset "box density, sampling law and input ownership" begin
    lower, upper = Float32[-2, 1], Float32[4, 5]
    proposal = BoxUniform(lower, upper)
    lower .= 100
    upper .= 101
    oracle = [Distributions.Uniform(-2.0f0, 4.0f0), Distributions.Uniform(1.0f0, 5.0f0)]
    point = Float32[0, 2]
    @test logdensityof(proposal, point) ≈ sum(Distributions.logpdf.(oracle, point))
    @test logdensityof(proposal, Float32[-3, 2]) == -Inf32

    # Seed 1701, 20,000 draws. Each mean tolerance exceeds six analytic SEs.
    samples = importance_sample(
        Random.Xoshiro(1701),
        proposal,
        ImportanceSampling(proposal; nsamples = 20_000),
    )
    @test all(iszero, samples.logweights)
    @test maximum(abs.(Statistics.mean(samples) .- Float32[1, 3])) < 0.08f0
    @test Statistics.var(samples) ≈ Float32[3, 4/3] rtol=0.04
    @test eltype(samples.samples) === eltype(samples.logweights) === Float32
end

@testset "Gamma normalization and independent-coordinate sampling" begin
    shape, scale = Float32[0.5, 5], Float32[2, 0.2]
    proposal = IndependentGamma(shape, scale)
    oracle = Distributions.Gamma.(shape, scale)
    point = Float32[0.7, 1.3]
    @test logdensityof(proposal, point) ≈ sum(Distributions.logpdf.(oracle, point)) rtol=2e-6
    @test logdensityof(proposal, Float32[-1, 0]) == -Inf32
    # Seed 1710, 20,000 draws; tolerances exceed six analytic mean/variance SEs.
    samples = importance_sample(
        Random.Xoshiro(1710),
        proposal,
        ImportanceSampling(proposal; nsamples = 20_000),
    )
    @test all(iszero, samples.logweights)
    @test maximum(abs.(Statistics.mean(samples) .- shape .* scale)) < 0.08f0
    @test Statistics.var(samples) ≈ shape .* scale .^ 2 rtol=0.18
    @test abs(Statistics.cov(samples)[1, 2]) < 0.03f0
end

@testset "Gamma density retains finite extreme-range terms" begin
    cases = (
        (1.0f6, 1.0f0, 1.0f6),
        (1e300, 1.0, 1e300),
        (1e308, 1e-308, 2.0),
        (32.0f0, floatmax(Float32), floatmin(Float32)),
        (2.0f38, 1.0f0, 2.0f38 * exp(-2.0f0)),
    )
    for (shape, scale, x) in cases
        expected = setprecision(BigFloat, 2048) do
            a, theta, value = BigFloat(shape), BigFloat(scale), BigFloat(x)
            (a-1)*log(value) - value/theta - SpecialFunctions.loggamma(a) - a*log(theta)
        end
        actual = logdensityof(IndependentGamma(shape, scale), x)
        @test isfinite(actual) && isapprox(actual, typeof(x)(expected); rtol = 32eps(typeof(x)))
    end
    exponential = IndependentGamma(1.0, 2.0)
    @test logdensityof(exponential, 0.0) == -log(2.0)
end

@testset "Gamma target accepts samples in another precision" begin
    target = IndependentGamma(2.0f0, 1.0f0)
    proposal = SphericalGaussian(0.0, 2.0)
    samples = importance_sample(
        Random.Xoshiro(1714),
        target,
        ImportanceSampling(proposal; nsamples = 128),
    )
    oracle = Distributions.Gamma(2.0, 1.0)
    expected =
        [Distributions.logpdf(oracle, x) - logdensityof(proposal, x) for x in samples.samples]
    @test samples.logweights ≈ expected
end

@testset "Gamma MIS uses independent vector densities" begin
    shapes = ([0.7, 2.0], [3.0, 5.0])
    scales = ([2.0, 1.0], [1.0, 0.5])
    proposals = [IndependentGamma(shapes[j], scales[j]) for j = 1:2]
    oracles = [Distributions.Gamma.(shapes[j], scales[j]) for j = 1:2]
    target = IndependentGamma([2.0, 3.0], [1.0, 1.0])
    masses = [0.3, 0.7]
    bank = ProposalBank(proposals, masses)
    for scheme in (
        StratifiedMixture(),
        RandomMixture(),
        StandardMIS(),
        PartialDeterministicMixture([[1], [2]]),
    )
        samples = importance_sample(
            Random.Xoshiro(1711),
            target,
            ImportanceSampling(bank; nsamples = 256, mis_scheme = scheme),
        )
        expected = map(eachindex(samples.logweights)) do i
            x = samples.samples[:, i]
            densities = [sum(Distributions.logpdf.(q, x)) for q in oracles]
            slot = samples.provenance.proposal_id[i]
            denominator =
                scheme isa Union{StandardMIS,PartialDeterministicMixture} ? densities[slot] :
                log(sum(masses .* exp.(densities)))
            logdensityof(target, x) - denominator
        end
        @test samples.logweights ≈ expected
    end
    scalar = Distributions.Gamma(2.0, 3.0)
    samples =
        importance_sample(Random.Xoshiro(1712), scalar, ImportanceSampling(scalar; nsamples = 128))
    @test maximum(abs, samples.logweights) < 1e-13
end

@testset "unrepresentable Gamma draws fail the call and advance its RNG" begin
    proposal = IndependentGamma(1.0f-10, 1.0f0)
    rng = Random.Xoshiro(1713)
    before = copy(rng)
    sampler = prepare_sampler(rng, proposal, ImportanceSampling(proposal; nsamples = 1))
    failure = try
        importance_sample!(sampler)
    catch error
        error
    end
    @test failure isa SamplerExecutionError && failure.phase == :proposal_draw
    @test rand(rng, UInt64) != rand(before, UInt64)
end

@testset "rejected Gamma candidates fail without returning partial samples" begin
    proposal = IndependentGamma(2.0f0, 1.0f0)
    rng = RejectingGammaRNG(false, false)
    sampler = prepare_sampler(rng, proposal, ImportanceSampling(proposal; nsamples = 1))
    failure = try
        importance_sample!(sampler)
    catch error
        error
    end
    @test rng.filled_normals &&
          rng.filled_uniforms &&
          failure isa SamplerExecutionError &&
          failure.phase === :proposal_draw
end

@testset "finite extreme interval and scalar draws" begin
    bound = floatmax(Float64)
    proposal = BoxUniform(-bound, bound)
    expected = -Float64(log(2BigFloat(bound)))
    @test logdensityof(proposal, 0.0) ≈ expected
    samples = importance_sample(
        Random.Xoshiro(1702),
        proposal,
        ImportanceSampling(proposal; nsamples = 256),
    )
    @test all(isfinite, samples.samples)
    @test all(x -> -bound <= x <= bound, samples.samples)
    @test all(iszero, samples.logweights)
    for T in (Float32, Float64)
        lower, upper = T(3), nextfloat(T(3))
        narrow = BoxUniform(lower, upper)
        result = importance_sample(
            Random.Xoshiro(1715),
            narrow,
            ImportanceSampling(narrow; nsamples = 1024),
        )
        uniform = T === Float32 ? T(0.00015) : T(0.0006)
        forced = rand(BoxEndpointRNG(uniform), narrow)
        @test lower <= forced <= upper &&
              all(x -> lower <= x <= upper, result.samples) &&
              all(iszero, result.logweights)
    end
end

@testset "box MIS denominators and stable proposal identifiers" begin
    proposals = [BoxUniform(0.0, 2.0), BoxUniform(1.0, 4.0)]
    oracles = [Distributions.Uniform(0.0, 2.0), Distributions.Uniform(1.0, 4.0)]
    target = Distributions.Uniform(1.0, 2.0)
    masses = [0.3, 0.7]
    bank = ProposalBank(proposals, masses)
    for scheme in (
        StratifiedMixture(),
        RandomMixture(),
        StandardMIS(),
        PartialDeterministicMixture([[1], [2]]),
    )
        samples = importance_sample(
            Random.Xoshiro(1703),
            target,
            ImportanceSampling(bank; nsamples = 256, mis_scheme = scheme),
        )
        expected = map(eachindex(samples.samples)) do i
            x = samples.samples[i]
            q =
                scheme isa Union{StandardMIS,PartialDeterministicMixture} ?
                Distributions.pdf(oracles[samples.provenance.proposal_id[i]], x) :
                sum(masses .* Distributions.pdf.(oracles, x))
            Distributions.logpdf(target, x) - log(q)
        end
        @test samples.logweights ≈ expected
    end
    disjoint = ProposalBank([Distributions.Uniform(0.0, 1.0), Distributions.Uniform(1.0, 2.0)])
    samples = importance_sample(
        Random.Xoshiro(1704),
        Distributions.Uniform(0.0, 2.0),
        ImportanceSampling(disjoint; nsamples = 256),
    )
    @test all(iszero, samples.logweights)
end

@testset "independent proposal preparation and execution preserve the RNG contract" for proposal in
                                                                                        (
    BoxUniform([-2.0, -1.0], [2.0, 1.0]),
    IndependentGamma([0.5, 3.0], [2.0, 1.0]),
)
    rng = Random.Xoshiro(1705)
    reference = copy(rng)
    prepare_sampler(rng, proposal, ImportanceSampling(proposal; nsamples = 256))
    @test rand(rng, UInt64) == rand(reference, UInt64)
    algorithm = ImportanceSampling(proposal; nsamples = 256)
    serial = importance_sample(Random.Xoshiro(1706), proposal, algorithm; threaded = false)
    threaded = importance_sample(Random.Xoshiro(1706), proposal, algorithm; threaded = true)
    @test serial.samples == threaded.samples
    @test serial.logweights == threaded.logweights
end
