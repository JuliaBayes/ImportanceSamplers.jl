module ControlVariateValidation

using ImportanceSamplers, Random, Statistics, LinearAlgebra, Test
import MLDataDevices

score(x) = -x + oftype(x, 2) * tanh(oftype(x, 2) * x)
functional(x) = oftype(x[1], 2) + score(x[1])

function main(; device=MLDataDevices.cpu_device(), T=Float64)
    x = reshape(collect(range(T(-6), T(6); length=65)), 1, :)
    logs = [-v^2 / 2 + log(cosh(2v)) for v in vec(x)]
    pilot = WeightedSamples(x, logs)
    reference = fit_control_variate(functional, pilot; scores=score.(x), ridge=0)
    device_pilot = device(pilot)
    @assert eltype(device_pilot.samples) === T "Select a device preserving T"
    scores = score.(device_pilot.samples)
    fit = fit_control_variate(functional, device_pilot; scores, ridge=0)
    production_x = reshape(randn(Xoshiro(73), T, 128), 1, :)
    production = device(WeightedSamples(production_x, log.(cosh.(2vec(production_x)))))
    production_scores = score.(production.samples)
    @testset "Score controls $T on $(typeof(device))" begin
        @test Array(fit.coefficients) ≈ reference.coefficients rtol = 500eps(T) atol = 500eps(T)
        @test control_variate_mean(functional, production, fit;
                                   scores=production_scores) ≈ T(2) atol = 500eps(T)
        values = T(2) .+ vec(production_scores)
        @test control_variate_mean(values, production, device(reference);
                                   scores=production_scores) ≈ T(2) atol = 500eps(T)
        @test Array(device_pilot.logweights) == pilot.logweights
        extended = device(WeightedSamples(hcat(x, T(Inf)), vcat(logs, T(-Inf))))
        augmented = fit_control_variate(functional, extended;
                                        scores=device(hcat(score.(x), T(NaN))), ridge=0)
        @test Array(augmented.coefficients) ≈ Array(fit.coefficients) rtol = 500eps(T) atol = 500eps(T)

        # Three-dimensional full-rank fit and real device-side production sampling.
        proposal = SphericalGaussian(zeros(T, 3), T(1.4))
        logtarget(x) = -sum(abs2, x) / 2
        pilot = importance_sample(Xoshiro(8), logtarget,
                                  ImportanceSampling(proposal; nsamples=4096))
        host = fit_control_variate(x -> sum(abs2, x), pilot; scores=-pilot.samples)
        pilot = device(pilot)
        fit = fit_control_variate(x -> sum(abs2, x), pilot; scores=-pilot.samples)
        @test Array(fit.coefficients) ≈ host.coefficients rtol = 500eps(T) atol = 500eps(T)
        prepared = prepare_sampler(Xoshiro(19), logtarget,
                                   ImportanceSampling(proposal; nsamples=4096))
        production = importance_sample!(device(prepared))
        adjusted = control_variate_mean(x -> sum(abs2, x), production, fit;
                                        scores=-production.samples)
        host_production = MLDataDevices.cpu_device(T)(production)
        expected = control_variate_mean(x -> sum(abs2, x), host_production, host;
                                        scores=-host_production.samples)
        @test adjusted ≈ expected rtol = 500eps(T)
        @test adjusted isa T
        @test eltype(fit.coefficients) === T
    end
    return nothing
end

end
