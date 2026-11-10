using Test, ImportanceSamplers, Random, Statistics, LinearAlgebra

@testset "Control fitting follows the squared-weight regression" begin
    for T in (Float32, Float64)
        x = randn(Xoshiro(49), T, 3, 96) .+ T[2, -1, 3]
        scores = .-(x .- T[1, 0, 2])
        logs = collect(range(T(-4), zero(T); length=size(x, 2)))
        pilot = WeightedSamples(x, logs)
        f(x) = x[1]^2 + x[2] * x[3]
        values = f.(eachcol(x))
        ridge = T(0.01)
        fit = fit_control_variate(f, pilot; scores, ridge, threaded=false)
        weights = normalized_weights(pilot)
        center = x * weights
        controls = vcat(scores, one(T) .+ (x .- center) .* scores)
        design = transpose(controls) .* (weights ./ maximum(weights))
        response = (weights ./ maximum(weights)) .* (values .- dot(weights, values))
        norms = sqrt.(vec(sum(abs2, design; dims=1)))
        # Independent augmented least-squares oracle, not the Gram solve.
        reference = vcat(design, sqrt(ridge) * Diagonal(norms)) \
                    vcat(response, zeros(T, length(norms)))
        @test fit.coefficients ≈ reference rtol = 100eps(T) atol = 100eps(T)
        @test fit.center ≈ center
        @test fit_control_variate(values, pilot; scores, ridge).coefficients ≈
              fit.coefficients
        shifted = WeightedSamples(x, logs .+ T(80))
        @test fit_control_variate(values, shifted; scores, ridge).coefficients ≈
              fit.coefficients rtol = 1000eps(T) atol = 1000eps(T)
        corrected = control_variate_mean(f, pilot, fit; scores, threaded=false)
        @test corrected ≈ dot(weights, values .- transpose(controls) * fit.coefficients)
        @test control_variate_mean(values, pilot, fit; scores) ≈ corrected
        @test pilot.samples == x && pilot.logweights == logs
        @test eltype(fit.coefficients) === T && corrected isa T
    end
end

@testset "Gaussian polynomial correction uses a frozen pilot" begin
    for T in (Float32, Float64)
        x = reshape(T[-sqrt(T(3)), 0, sqrt(T(3))], 1, :)
        pilot = WeightedSamples(x, log.(T[1 / 6, 2 / 3, 1 / 6]))
        f(x) = T(3) * x[1]^2 + T(2) * x[1] + T(5)
        fit = fit_control_variate(f, pilot; scores=-x, ridge=0)
        production = importance_sample(Xoshiro(17), x -> -sum(abs2, x) / 2,
                                       ImportanceSampling(SphericalGaussian(zeros(T, 1),
                                                                            T(1.4));
                                                          nsamples=128))
        before = copy(fit.coefficients)
        @test control_variate_mean(f, production, fit; scores=-production.samples) ≈
              T(8) rtol = 100eps(T)
        @test fit.coefficients == before
    end
end

@testset "Non-Gaussian Stein identity and zero-weight rows" begin
    # Symmetric two-normal mixture: score = -x + 2tanh(2x).
    score(x) = -x + 2tanh(2x)
    f(x) = 2 + score(x[1])
    x = reshape(collect(range(-6.0, 6.0; length=65)), 1, :)
    logs = [-v^2 / 2 + log(cosh(2v)) for v in vec(x)]
    pilot = WeightedSamples(x, logs)
    scores = score.(x)
    fit = fit_control_variate(f, pilot; scores, ridge=0)
    production_x = reshape(randn(Xoshiro(73), 128), 1, :)
    production = WeightedSamples(production_x, log.(cosh.(2vec(production_x))))
    @test control_variate_mean(f, production, fit; scores=score.(production_x)) ≈ 2 atol = 1e-12
    null = fit_control_variate(x -> 3.0, pilot; scores)
    @test control_variate_mean(x -> 3.0, production, null; scores=score.(production_x)) ≈ 3
    extended = WeightedSamples(hcat(x, Inf), vcat(logs, -Inf))
    # The callable must never evaluate the discarded row.
    guarded(x) = isfinite(x[1]) ? f(x) : error("zero-weight row evaluated")
    augmented = fit_control_variate(guarded, extended; scores=hcat(scores, NaN), ridge=0)
    @test augmented.coefficients ≈ fit.coefficients
    @test control_variate_mean(guarded, extended, fit; scores=hcat(scores, NaN)) ≈
          control_variate_mean(f, pilot, fit; scores)
    @test control_variate_mean(vcat(f.(eachcol(x)), NaN), extended, fit;
                               scores=hcat(scores, NaN)) ≈
          control_variate_mean(f, pilot, fit; scores)
    degenerate = WeightedSamples(zeros(1, 3), zeros(3))
    inactive = fit_control_variate(x -> 3.0, degenerate; scores=zeros(1, 3))
    @test control_variate_mean(x -> 3.0, production, inactive;
                               scores=score.(production_x)) ≈ 3
end

@testset "Weighted residuals preserve a finite Float32 fit" begin
    pilot = WeightedSamples(reshape(Float32[-1, 1], 1, :), log.(Float32[0.01, 0.99]))
    fit = fit_control_variate(Float32[-1e38, 3e38], pilot; scores=Float32[1 -1])
    @test all(isfinite, fit.coefficients)
end
