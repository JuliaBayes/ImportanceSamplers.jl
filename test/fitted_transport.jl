using Test, ImportanceSamplers, Random, Statistics, LinearAlgebra
import DensityInterface
import ADTypes, ForwardDiff
include("../validation/reproducers/fitted_transport.jl")

transport_target(x) = -(x[1]^2 + (x[2] - 1.25 * (x[1]^2 - 1))^2) / 2 - log(2pi)
transport_latent(x) = -sum(abs2, x) / 2 - log(2pi)
function transport_gradient!(g, x)
    residual = x[2] - 1.25 * (x[1]^2 - 1)
    g[1] = -x[1] + 2.5 * x[1] * residual
    g[2] = -residual
    return g
end

@testset "Unpaired coordinates retain their value and gradient" begin
    FittedTransportValidation.main(; d = 1)
    FittedTransportValidation.main(; d = 3)
    algorithm = ImportanceSampling(SphericalGaussian(0.0, 1.0); nsamples = 32)
    target(x) = -abs2(x) / 2
    plain = importance_sample(Xoshiro(19), target, algorithm)
    identity = importance_sample(
        Xoshiro(19),
        target,
        algorithm;
        transform = QuadraticCoupling(Float64[]),
    )
    @test identity.samples == plain.samples
    @test identity.logweights == plain.logweights
end
function transport_batch!(logs, samples)
    logs .= transport_target.(eachcol(samples))
end

@testset "Quadratic transport measure and batch evaluation" begin
    transform = QuadraticCoupling([1.25])
    base = SphericalGaussian(zeros(2), 1.0)
    proposal = TransformedProposal(base, transform)
    point = rand(Xoshiro(17), proposal)
    @test DensityInterface.logdensityof(proposal, point) ≈ transport_target(point)

    algorithm = ImportanceSampling(base; nsamples = 64)
    options = (; transform, threaded = false)
    scalar = importance_sample(Xoshiro(29), transport_target, algorithm; options...)
    batch_target = LogTarget(transport_target; batch = transport_batch!)
    batch = importance_sample(Xoshiro(29), batch_target, algorithm; options...)
    @test maximum(abs, scalar.logweights) < 16eps(Float64)
    @test batch.samples == scalar.samples
    @test batch.logweights ≈ scalar.logweights atol = 16eps(Float64)
    identity = importance_sample(
        Xoshiro(29),
        transport_latent,
        algorithm;
        transform = QuadraticCoupling([0.0]),
        threaded = false,
    )
    plain = importance_sample(Xoshiro(29), transport_latent, algorithm; threaded = false)
    @test identity.samples == plain.samples
    @test identity.logweights == plain.logweights
    huge = TransformedProposal(
        SphericalGaussian([floatmax(Float64), 0.0], 1.0),
        QuadraticCoupling([0.0]),
    )
    @test all(isfinite, rand(Xoshiro(31), huge))
end

@testset "Weighted pilot fits preserve covariance convention" begin
    for T in (Float32, Float64)
        # Translation and a linear nuisance term must not change the curvature fit.
        rng = Xoshiro(7129)
        samples = randn(rng, T, 2, 256)
        samples[1, :] .+= T(2)
        samples[2, :] .+=
            T(1.25) .* (samples[1, :] .^ 2 .- one(T)) .+ T(1.75) .* samples[1, :] .+ T(4)
        scores = similar(samples)
        scores[2, :] .=
            -samples[2, :] .+ T(1.25) .* (samples[1, :] .^ 2 .- one(T)) .+
            T(1.75) .* samples[1, :] .+ T(4)
        scores[1, :] .= zero(T) # The fit uses only even-coordinate scores.
        logweights = collect(range(T(-2), zero(T); length = 256))
        pilot = WeightedSamples(samples, logweights)
        weights = normalized_weights(pilot)
        fit = fit_transport(pilot; scores, dof = 7, threaded = false)
        @test fit.transform.coefficients ≈ T[1.25] rtol = 100eps(T)
        latent = copy(samples)
        latent[2, :] .-= fit.transform.coefficients[1] .* (samples[1, :] .^ 2 .- one(T))
        covariance = cov(WeightedSamples(latent, logweights))
        covariance[diagind(covariance)] .+= T(1e-6) * max(tr(covariance) / 2, one(T))
        @test fit.proposal.location ≈ latent * weights
        reference = FactorStudentT(
            T(7),
            latent * weights,
            sqrt(T(5) / T(7)) * Matrix(cholesky(Symmetric(covariance)).L),
        )
        @test DensityInterface.logdensityof(fit.proposal, T[0.3, 4.5]) ≈
              DensityInterface.logdensityof(reference, T[0.3, 4.5])
        moment = fit_transport(pilot; threaded = false)
        design = hcat(ones(T, 256), samples[1, :], samples[1, :] .^ 2)
        beta = (sqrt.(weights) .* design) \ (sqrt.(weights) .* samples[2, :])
        @test moment.transform.coefficients ≈ T[beta[3]] rtol = 100eps(T)
        shifted = fit_transport(
            WeightedSamples(samples, logweights .+ T(100));
            scores,
            dof = 7,
            threaded = false,
        )
        @test shifted.transform.coefficients ≈ fit.transform.coefficients rtol = 100eps(T)
        @test pilot.samples == samples
        if T === Float32
            mixed = fit_transport(
                WeightedSamples(samples, Float64.(logweights));
                scores,
                dof = 7,
            )
            run = importance_sample(
                Xoshiro(71),
                transport_target,
                ImportanceSampling(mixed.proposal; nsamples = 64);
                transform = mixed.transform,
            )
            @test eltype(run.samples) === T && all(isfinite, run.logweights)
        end
        if T === Float64
            extended = fit_transport(
                WeightedSamples(hcat(samples, [1e200, 0.0]), vcat(logweights, -Inf));
                scores = hcat(scores, [0.0, 0.0]),
                dof = 7,
                threaded = false,
            )
            @test extended.transform.coefficients ≈ fit.transform.coefficients
            @test DensityInterface.logdensityof(extended.proposal, T[0.3, 4.5]) ≈
                  DensityInterface.logdensityof(fit.proposal, T[0.3, 4.5])
        end
    end
end

@testset "Gaussian linear dependence is not curvature" begin
    for T in (Float32, Float64)
        FittedTransportValidation.gaussian_null(; T)
    end
end

@testset "Latent adaptation and target reuse" begin
    transform = QuadraticCoupling([1.25])
    base = FactorStudentT(5.0, zeros(2), Matrix{Float64}(I, 2, 2))
    algorithm = AMIS(base; rounds = 2, round_size = 128)
    options = (; transform, threaded = false)
    source = prepare_sampler(Xoshiro(7), transport_target, algorithm; options...)
    mapped = importance_sample!(source)
    latent = importance_sample(Xoshiro(7), transport_latent, algorithm; threaded = false)
    recovered = copy(mapped.samples)
    recovered[2, :] .-= 1.25 .* (recovered[1, :] .^ 2 .- 1)
    @test recovered ≈ latent.samples
    @test mapped.logweights ≈ latent.logweights
    reused = retarget(Xoshiro(11), source, transport_target)
    adapted = AMIS(current_proposal(source); rounds = 2, round_size = 128)
    rebuilt = prepare_sampler(Xoshiro(11), transport_target, adapted; options...)
    @test importance_sample!(reused).samples == importance_sample!(rebuilt).samples
end

@testset "Physical gradients pull back through the coupling" begin
    # Compare full adaptation against its analytic latent target, not private buffers.
    transform = QuadraticCoupling([1.25])
    bank = ProposalBank([
        SphericalGaussian([-0.4, 0.0], 1.0),
        SphericalGaussian([0.4, 0.0], 1.0),
    ])
    algorithm =
        FirstOrderGRAMIS(bank; rounds = 2, round_size = 128, repulsion_strength = 0.1)
    reference = LogTarget(transport_latent; grad = (g, x) -> (g .= .-x))
    flat = importance_sample(Xoshiro(18), reference, algorithm; threaded = false)
    for target in (
        LogTarget(transport_target; grad = transport_gradient!),
        LogTarget(transport_target, ADTypes.AutoForwardDiff()),
    )
        mapped =
            importance_sample(Xoshiro(18), target, algorithm; transform, threaded = false)
        @test mapped.logweights ≈ flat.logweights rtol = 1e-10 atol = 1e-10
    end
end

@testset "Undefined coupling fits do not produce a proposal" begin
    constant = WeightedSamples([fill(2.0, 5)'; collect(1.0:5.0)'], zeros(5))
    @test_throws ArgumentError fit_transport(constant)
    binary = WeightedSamples([-1.0 1 -1 1; 0 1 2 3], zeros(4))
    @test_throws ArgumentError fit_transport(binary)
    rng = Xoshiro(43)
    samples = randn(rng, 2, 32)
    pilot = WeightedSamples(samples, zeros(32))
    # A positive score slope cannot represent a normal conditional density.
    @test_throws ArgumentError fit_transport(pilot; scores = samples)
    samples[2, :] .= samples[1, :] .^ 2 .- 1
    collinear = WeightedSamples(samples, zeros(32))
    @test_throws ArgumentError fit_transport(collinear; scores = .-samples)
end
