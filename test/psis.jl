using Test, ImportanceSamplers, Statistics
import LogExpFunctions
import MLDataDevices

# Numeric oracles from the pinned loo/posterior equations, independently
# evaluated with scalar Python and NumPy/SciPy. Shape is the input GPD shape.
# https://github.com/stan-dev/loo/blob/7ecc6bcf7975bf2f689421986073ce662cbc93dd/R/psis.R
# https://github.com/stan-dev/posterior/blob/c7145f7adf3e14b4e9a2eadfea106712957e8191/R/gpd.R
const PSIS_REFERENCE_CASES = (
    (
        shape = 0.7,
        k = 0.6227881402611476,
        tail = (1.40024365308168, 1.9131489294474737, 4.056565586448096),
        logsum = 6.133634780544687,
    ),
    (
        shape = -0.27,
        k = 0.004630651275242315,
        tail = (0.8416306719427399, 1.00412618580241, 1.3545396721235734),
        logsum = 5.437458679230809,
    ),
    (
        shape = -0.7,
        k = -0.2831624387822257,
        tail = (0.6794541971921931, 0.7675427635839941, 0.875101222741876),
        logsum = 5.316775824483093,
    ),
)

function psis_reference_input(shape, ::Type{T} = Float64; n = 128) where {T}
    return map(0:(n-1)) do i
        p = (T(mod(37i, n)) + T(0.5)) / T(n)
        log1p(expm1(-T(shape) * log1p(-p)) / T(shape))
    end
end

@testset "PSIS reference tails" begin
    for case in PSIS_REFERENCE_CASES
        raw = psis_reference_input(case.shape)
        output = @inferred psis(raw; threaded = false)
        order = sortperm(raw)
        @test output.diagnostics.pareto_k ≈ case.k atol = 2e-11
        @test output.logweights[order[[104, 116, 128]]] ≈ collect(case.tail) atol = 2e-11
        @test LogExpFunctions.logsumexp(output.logweights) ≈ case.logsum atol = 2e-11
        @test output.logweights[order[1:103]] == raw[order[1:103]]
        @test maximum(output.logweights) <= maximum(raw)
    end
end

@testset "PSIS weight and result semantics" begin
    for T in (Float32, Float64)
        raw = psis_reference_input(0.7, T)
        original = copy(raw)
        samples =
            WeightedSamples((x = T.(1:128),), raw; provenance = (round = fill(2, 128),))
        raw_normalizer = lognormalizer(samples)
        smoothed = @inferred psis(samples)
        weights = normalized_weights(smoothed)
        tolerance = T === Float32 ? 3e-5 : 2e-11
        @test mean(smoothed).x ≈ sum(weights .* samples.samples.x)
        @test var(x -> x.x, smoothed) ≈
              sum(weights .* (samples.samples.x .- mean(smoothed).x) .^ 2)
        @test samples.logweights == original && raw == original
        @test lognormalizer(samples) == raw_normalizer
        @test_throws ArgumentError lognormalizer(smoothed)
        @test smoothed.samples === samples.samples
        @test smoothed.provenance === samples.provenance
        @test smoothed[1].logweight == smoothed.logweights[1]
        @test mean(smoothed[1:20]).x ≈ sum(weights[1:20] .* T.(1:20)) / sum(weights[1:20])
        reversed = psis(reverse(raw); threaded = false).logweights
        @test exp.(reversed .- LogExpFunctions.logsumexp(reversed)) ≈ reverse(weights) rtol =
            tolerance
        offset = T === Float32 ? T(100) : T(10000)
        shifted = psis(raw .+ offset).logweights
        @test exp.(shifted .- LogExpFunctions.logsumexp(shifted)) ≈ weights rtol = tolerance
        host = MLDataDevices.cpu_device()(smoothed)
        @test mean(host).x ≈ mean(smoothed).x
    end
    raw = psis_reference_input(0.7; n = 512)
    @test (
        psis(raw).diagnostics.tail_length,
        psis(raw; reff = 0.5).diagnostics.tail_length,
    ) == (67, 96)
end

@testset "PSIS unavailable tail diagnostics" begin
    for (raw, status) in (
        ([0.0, -1.0, -Inf], :insufficient_tail),
        (zeros(128), :degenerate_tail),
        (vcat(fill(-Inf, 127), 0.0), :degenerate_tail),
        (vcat(fill(-Inf, 83), range(-4.0, 0.0; length = 17)), :degenerate_tail),
        (vcat(range(-745.0, -740.0; length = 127), 0.0), :failed_fit),
    )
        output = psis(raw)
        @test output.logweights == raw
        @test isnan(output.diagnostics.pareto_k) && !output.diagnostics.reliable
        @test output.diagnostics.status === status
    end
    @test_throws AllZeroWeightsError psis(fill(-Inf, 128))
end

@testset "PSIS tied mass is independent of sample order" begin
    logs = collect(range(-5.0, 0.0; length = 100))
    logs[87:92] .= -0.5
    x = zeros(100)
    x[87:92] .= 1:6
    permutation = vcat(1:86, 92:-1:87, 93:100)
    original = psis(WeightedSamples(x, logs))
    permuted = psis(WeightedSamples(x[permutation], logs[permutation]))
    @test mean(original) ≈ mean(permuted)
    @test all(==(original.logweights[87]), original.logweights[87:92])
    @test original.logweights ≈ permuted.logweights[invperm(permutation)]
    logs[80:82] .= -1
    boundary = psis(logs)
    @test boundary.logweights == logs
    @test boundary.diagnostics.status === :degenerate_tail
end
