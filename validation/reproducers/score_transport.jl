module ScoreTransportValidation

using ImportanceSamplers, Random, Test
import DensityInterface, MLDataDevices
include(joinpath(@__DIR__, "..", "..", "benchmark", "score_transport.jl"))
const Experiment = ScoreTransportExperiment

function main(; device = MLDataDevices.cpu_device(), T = Float64)
    rng = Xoshiro(7129)
    tolerance = T === Float32 ? T(2e-4) : T(2e-11)
    model = Experiment.BentFunnel(T(1.25), zero(T))
    points = T(1.5) .* randn(rng, T, 4, 256)
    coefficients = T[0.75, -0.4]
    resident = device(points)
    mapped = Experiment.couple(resident, device(coefficients))
    recovered = Experiment.couple(mapped, device(coefficients); inverse = true)
    base = SphericalGaussian(zeros(T, 4), T(1.5))
    logs = [
        Experiment.logtarget(view(points, :, j), model) -
        DensityInterface.logdensityof(base, view(points, :, j)) for j in axes(points, 2)
    ]
    fitted = Experiment.fit_coefficients(
        device(WeightedSamples(points, logs)),
        model,
        Val(:score),
    )
    @testset "Additive coupling measure and score fit" begin
        @test Array(recovered) ≈ points rtol = tolerance atol = tolerance
        @test Array(mapped)[2, :] ≈
              points[2, :] .+ coefficients[1] .* (points[1, :] .^ 2 .- 1)
        @test Array(fitted) ≈ fill(model.bend, 2) rtol = tolerance atol = tolerance
        exact = Array(Experiment.couple(resident, device(fill(model.bend, 2))))
        unit_base = SphericalGaussian(zeros(T, 4), one(T))
        @test [Experiment.logtarget(view(exact, :, j), model) for j in axes(exact, 2)] ≈ [
            DensityInterface.logdensityof(unit_base, view(points, :, j)) for
            j in axes(points, 2)
        ] rtol = tolerance atol = tolerance
    end
    return (; T, points = size(points, 2), fitted = Array(fitted))
end

function analytic_checks()
    @testset "Non-Gaussian target oracles" begin
        z = [0.7, -0.4, -0.3, 1.2]
        for model in (Experiment.BentFunnel(1.25, 0.0), Experiment.BentFunnel(1.25, 0.45))
            x = copy(z)
            for i = 1:2:length(z)
                x[i+1] = z[i+1] * exp(model.logscale * z[i]) + model.bend * (z[i]^2 - 1)
            end
            standard_logdensity = -(sum(abs2, z) + length(z) * log(2pi)) / 2
            @test Experiment.logtarget(x, model) + model.logscale * sum(z[1:2:end]) ≈
                  standard_logdensity
        end
        for model in (
            Experiment.BentFunnel(1.25, 0.0),
            Experiment.BentFunnel(1.25, 0.45),
            Experiment.Bimodal(2.5, 0.7),
        )
            gradient = similar(z)
            Experiment.score!(gradient, z, model)
            difference = map(eachindex(z)) do i
                h = cbrt(eps(Float64))
                upper, lower = copy(z), copy(z)
                upper[i] += h
                lower[i] -= h
                (Experiment.logtarget(upper, model) - Experiment.logtarget(lower, model)) / (2h)
            end
            @test gradient ≈ difference rtol = 1e-8 atol = 1e-8
        end
    end
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    ScoreTransportValidation.analytic_checks()
    ScoreTransportValidation.main()
end
