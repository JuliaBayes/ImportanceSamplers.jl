module PSISValidation

using ImportanceSamplers, Random, Statistics

"""
    main(; n=2_000, repetitions=400, seed=20261109)

Compare raw IS and PSIS for an Exp(1) target and Exp(rate) proposals.
The true target mean is one and the ratio-tail shape is 1 - 1/rate.
Use paired samples and report the Monte Carlo error of the squared-error gain.
This is an accuracy experiment, not a stochastic package test or a timing test.
"""
function main(; n = 2_000, repetitions = 400, seed = 20261109)
    return map((1.5, 3.0, 5.0)) do rate
        raw_errors = Float64[]
        psis_errors = Float64[]
        shapes = Float64[]
        unreliable = 0
        for repetition = 1:repetitions
            rng = Xoshiro(seed + repetition)
            x = randexp(rng, n) ./ rate
            samples = WeightedSamples(x, (rate - 1) .* x .- log(rate))
            smoothed = psis(samples; threaded = false)
            push!(raw_errors, mean(samples) - 1)
            push!(psis_errors, mean(smoothed) - 1)
            push!(shapes, smoothed.diagnostics.pareto_k)
            unreliable += !smoothed.diagnostics.reliable
        end
        gain = raw_errors .^ 2 .- psis_errors .^ 2
        (;
            rate,
            true_shape = 1 - inv(rate),
            n,
            repetitions,
            seed,
            raw_rmse = sqrt(mean(abs2, raw_errors)),
            psis_rmse = sqrt(mean(abs2, psis_errors)),
            raw_bias = mean(raw_errors),
            psis_bias = mean(psis_errors),
            mse_gain = mean(gain),
            mse_gain_mcse = std(gain) / sqrt(repetitions),
            mean_shape = mean(shapes),
            unreliable_fraction = unreliable / repetitions,
        )
    end
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    foreach(println, PSISValidation.main())
end
