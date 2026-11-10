module FittedTransportValidation

using ImportanceSamplers, Random, Statistics, LinearAlgebra, Test
import DensityInterface, MLDataDevices

function logtarget(x)
    bend = oftype(x[1], 1.25)
    return -(x[1]^2 + (x[2] - bend * (x[1]^2 - 1))^2) / 2 - log(oftype(x[1], 2pi))
end

function gradient!(g, x)
    bend = oftype(x[1], 1.25)
    residual = x[2] - bend * (x[1]^2 - 1)
    g[1] = -x[1] + 2bend * x[1] * residual
    g[2] = -residual
    return g
end

batch!(values, samples) = (values .= logtarget.(eachcol(samples)); nothing)

function main(; device = MLDataDevices.cpu_device(), T = Float64)
    tolerance = T(200) * eps(T)
    points = randn(Xoshiro(7129), T, 2, 512)
    points[2, :] .+= T(1.25) .* (points[1, :] .^ 2 .- one(T))
    scores = similar(points)
    for j in axes(points, 2)
        gradient!(view(scores, :, j), view(points, :, j))
    end
    pilot = device(WeightedSamples(points, zeros(T, 512)))
    cached = device(scores)
    host_fit = fit_transport(WeightedSamples(points, zeros(T, 512)); scores)
    fit = fit_transport(pilot; scores = cached)
    @testset "Fitted transport $T on $(typeof(device))" begin
        @test fit.transform.coefficients ≈ T[1.25] rtol = tolerance
        @test DensityInterface.logdensityof(fit.proposal, T[0.3, -0.4]) ≈
              DensityInterface.logdensityof(host_fit.proposal, T[0.3, -0.4]) rtol =
            tolerance
        @test Array(pilot.samples) == points
        base = SphericalGaussian(zeros(T, 2), one(T))
        transform = QuadraticCoupling(T[1.25])
        batch_target = LogTarget(logtarget; batch = batch!)
        algorithm = ImportanceSampling(base; nsamples = 256)
        prepared = prepare_sampler(Xoshiro(17), batch_target, algorithm; transform)
        result = importance_sample!(device(prepared))
        @test maximum(abs, Array(result.logweights)) < tolerance
        algorithm = AMIS(fit.proposal; rounds = 2, round_size = 256)
        production =
            prepare_sampler(Xoshiro(31), logtarget, algorithm; transform = fit.transform)
        sampled = importance_sample!(device(production))
        @test all(isfinite, Array(sampled.logweights))
        bank = ProposalBank([
            SphericalGaussian(T[-0.4, 0], one(T)),
            SphericalGaussian(T[0.4, 0], one(T)),
        ])
        algorithm =
            FirstOrderGRAMIS(bank; rounds = 2, round_size = 128, repulsion_strength = 0.1)
        target = LogTarget(logtarget; grad = gradient!)
        prepared = prepare_sampler(Xoshiro(43), target, algorithm; transform)
        mapped = importance_sample!(device(prepared))
        target = LogTarget(
            x -> -sum(abs2, x) / 2 - log(oftype(x[1], 2pi));
            grad = (g, x) -> (g .= .-x),
        )
        prepared = prepare_sampler(Xoshiro(43), target, algorithm)
        flat = importance_sample!(device(prepared))
        @test Array(mapped.logweights) ≈ Array(flat.logweights) rtol = sqrt(eps(T)) atol =
            tolerance
    end
    return (; T, coefficients = fit.transform.coefficients, device = typeof(device))
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    FittedTransportValidation.main()
end
