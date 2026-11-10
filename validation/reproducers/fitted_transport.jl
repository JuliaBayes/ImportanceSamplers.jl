module FittedTransportValidation

using ImportanceSamplers, Random, Statistics, LinearAlgebra, Test
import DensityInterface, MLDataDevices

function logtarget(x)
    bend = oftype(x[1], 1.25)
    radius = zero(eltype(x))
    for odd = 1:2:(length(x)-1)
        radius += x[odd]^2 + (x[odd+1] - bend * (x[odd]^2 - 1))^2
    end
    isodd(length(x)) && (radius += (x[end] - oftype(x[1], 0.4))^2)
    return -radius / 2 - length(x) * log(oftype(x[1], 2pi)) / 2
end

function gradient!(g, x)
    bend = oftype(x[1], 1.25)
    for odd = 1:2:(length(x)-1)
        residual = x[odd+1] - bend * (x[odd]^2 - 1)
        g[odd] = -x[odd] + 2bend * x[odd] * residual
        g[odd+1] = -residual
    end
    isodd(length(x)) && (g[end] = oftype(x[1], 0.4) - x[end])
    return g
end

function latent_target(x)
    radius = sum(abs2, x)
    isodd(length(x)) && (radius += (x[end] - oftype(x[1], 0.4))^2 - x[end]^2)
    return -radius / 2 - length(x) * log(oftype(x[1], 2pi)) / 2
end
function latent_gradient!(g, x)
    g .= .-x
    isodd(length(x)) && (g[end] += oftype(x[1], 0.4))
    return g
end

batch!(values, samples) = (values .= logtarget.(eachcol(samples)); nothing)

function gaussian_null(; device = MLDataDevices.cpu_device(), T = Float64)
    nodes, mass = T[-sqrt(3), 0, sqrt(3)], T[1/6, 2/3, 1/6]
    points = stack(T[3+2u, -4+5u+T(0.7)*v] for u in nodes for v in nodes)
    scores = stack(T[-u/2+5v/(2T(0.7)), -v/T(0.7)] for u in nodes for v in nodes)
    pilot = device(WeightedSamples(points, [log(a * b) for a in mass for b in mass]))
    for cached in (nothing, device(scores))
        fit = fit_transport(pilot; scores = cached)
        @test abs(only(fit.transform.coefficients)) <= 64eps(T) * T(5) / 4
    end
end

function main(; device = MLDataDevices.cpu_device(), T = Float64, d = 2)
    tolerance = T(200) * eps(T)
    points = randn(Xoshiro(7129), T, d, 512)
    for odd = 1:2:(d-1)
        points[odd+1, :] .+= T(1.25) .* (points[odd, :] .^ 2 .- one(T))
    end
    isodd(d) && (points[end, :] .+= T(0.4))
    scores = similar(points)
    for j in axes(points, 2)
        gradient!(view(scores, :, j), view(points, :, j))
    end
    pilot = device(WeightedSamples(points, zeros(T, 512)))
    cached = device(scores)
    host_fit = fit_transport(WeightedSamples(points, zeros(T, 512)); scores)
    fit = fit_transport(pilot; scores = cached)
    @testset "Fitted transport $T on $(typeof(device))" begin
        @test fit.transform.coefficients ≈ fill(T(1.25), d ÷ 2) rtol = tolerance
        point = fill(T(0.3), d)
        @test DensityInterface.logdensityof(fit.proposal, point) ≈
              DensityInterface.logdensityof(host_fit.proposal, point) rtol = tolerance
        @test Array(pilot.samples) == points
        location = zeros(T, d)
        isodd(d) && (location[end] = T(0.4))
        base = SphericalGaussian(location, one(T))
        transform = QuadraticCoupling(fill(T(1.25), d ÷ 2))
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
        left, right = zeros(T, d), zeros(T, d)
        left[1], right[1] = T(-0.4), T(0.4)
        bank = ProposalBank([
            SphericalGaussian(left, one(T)),
            SphericalGaussian(right, one(T)),
        ])
        algorithm =
            FirstOrderGRAMIS(bank; rounds = 2, round_size = 128, repulsion_strength = 0.1)
        target = LogTarget(logtarget; grad = gradient!)
        prepared = prepare_sampler(Xoshiro(43), target, algorithm; transform)
        mapped = importance_sample!(device(prepared))
        target = LogTarget(latent_target; grad = latent_gradient!)
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
