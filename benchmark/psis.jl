using BenchmarkTools, ImportanceSamplers, Statistics
import MLDataDevices

"""
    benchmark_psis(; device=MLDataDevices.cpu_device(), T=Float64,
        sizes=(10_000, 100_000, 1_000_000), threaded=true)

Time complete PSIS calls after setup and compilation. Device selection is explicit.
Report minimum and median milliseconds, host bytes, allocations, and Pareto shape.
Device calls finish their scalar diagnostics before returning.
"""
function benchmark_psis(;
    device = MLDataDevices.cpu_device(),
    T = Float64,
    sizes = (10_000, 100_000, 1_000_000),
    threaded = true,
)
    return map(sizes) do n
        p = ((0:(n-1)) .+ 0.5) ./ n
        weights = T.(log1p.(expm1.(-0.7 .* log1p.(-p)) ./ 0.7))
        input = device(weights[mod.(37 .* (0:(n-1)), n).+1])
        result = psis(input; threaded)
        trial =
            @benchmark psis($input; threaded = $threaded) samples = 20 evals = 1 seconds = 5
        (;
            n,
            precision = string(T),
            device = string(typeof(device)),
            threaded,
            minimum_ms = minimum(trial).time / 1e6,
            median_ms = median(trial).time / 1e6,
            host_bytes = minimum(trial).memory,
            allocations = minimum(trial).allocs,
            pareto_k = result.diagnostics.pareto_k,
            status = result.diagnostics.status,
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    foreach(println, benchmark_psis())
end
