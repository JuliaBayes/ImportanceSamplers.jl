using ImportanceSamplers, Random, Statistics, Test
import DensityInterface: logdensityof
import MLDataDevices

function check_adaptive_independent(;
    device = MLDataDevices.CPUDevice(),
    isresident = x->x isa Array,
    T = Float32,
    count = 8192,
)
    summaries = []
    cases = (
        (IndependentGamma(T[2, 3], T[1, 0.5]), IndependentGamma),
        (IndependentLaplace(T[0, 1], T[1, 0.5]), IndependentLaplace),
    )
    @testset "adaptive independent on $(typeof(device)), $T" begin
        for (target, factory) in cases, method in (AMIS, NPMC)
            sampler = device(
                prepare_sampler(
                    Xoshiro(1910),
                    target,
                    method(target; rounds = 3, round_size = count),
                ),
            )
            result = importance_sample!(sampler)
            @test isresident(result.samples) && isresident(result.logweights)
            host = MLDataDevices.CPUDevice()(result)
            learned = current_proposal(MLDataDevices.CPUDevice(), sampler)
            indices = method === AMIS ? eachindex(host.logweights) : (2count+1):3count
            logs = host.logweights[indices]
            method === NPMC && (logs .= min.(logs, sort(logs)[length(logs)-isqrt(length(logs))+1]))
            weights = exp.(logs .- maximum(logs))
            weights ./= sum(weights)
            values = host.samples[:, indices]
            center = values * weights
            variance = vec(sum((values .- center) .^ 2 .* reshape(weights, 1, :); dims = 2))
            expected =
                factory === IndependentGamma ?
                factory(center .^ 2 ./ variance, variance ./ center) :
                factory(center, sqrt.(variance ./ T(2)))
            point = T[1, 2]
            @test logdensityof(learned, point) ≈ logdensityof(expected, point) rtol=512eps(T) atol=512eps(
                T,
            )
            saved = copy(host.samples)
            @test all(isfinite, Array(importance_sample!(sampler).logweights))
            @test Array(result.samples) == saved
            push!(
                summaries,
                (;
                    family = factory,
                    method,
                    transfers = result.diagnostics.transfers.count,
                    transfer_bytes = result.diagnostics.transfers.bytes,
                ),
            )
        end
    end
    return summaries
end

adaptive_independent_switch(x, p) = iszero(p.active[1]) ? oftype(x, -Inf) : logdensityof(p.law, x)

function adaptive_independent_scalar_target(x, p)
    value = zero(eltype(x))
    for i in eachindex(x)
        value -= abs2(x[i]-p[i])/2
    end
    return value
end

function adaptive_independent_batch!(values, samples, p)
    values .= .-vec(sum(abs2, samples .- p; dims = 1)) ./ 2
    return nothing
end

function check_adaptive_independent_batch(; device = MLDataDevices.CPUDevice(), T = Float32)
    q = IndependentLaplace(T[0, 1], T[1, 0.5])
    p = T[0.5, -0.5]
    target = LogTarget(adaptive_independent_scalar_target; batch = adaptive_independent_batch!)
    result = importance_sample!(
        device(prepare_sampler(Xoshiro(1914), target, p, NPMC(q; rounds = 1, round_size = 512))),
    )
    host = MLDataDevices.CPUDevice()(result)
    expected =
        [adaptive_independent_scalar_target(x, p)-logdensityof(q, x) for x in eachcol(host.samples)]
    @test host.logweights ≈ expected rtol=128eps(T) atol=128eps(T)
    return nothing
end

function check_adaptive_independent_edges(; device = MLDataDevices.CPUDevice(), T = Float32)
    @testset "adaptive scalar reuse on $(typeof(device)), $T" begin
        for (q, method) in
            ((IndependentGamma(T(2), T(1)), AMIS), (IndependentLaplace(T(0), T(1)), NPMC))
            context = (; law = q, active = T[1])
            sampler = device(
                prepare_sampler(
                    Xoshiro(1911),
                    adaptive_independent_switch,
                    context,
                    method(q; rounds = 2, round_size = 256),
                ),
            )
            result = importance_sample!(sampler)
            saved = Array(result.samples)
            learned = current_proposal(MLDataDevices.CPUDevice(), sampler)
            copyto!(sampler.target.context.active, T[0])
            failure = try
                importance_sample!(sampler)
            catch error
                error
            end
            @test failure isa Union{AMISRoundError,NPMCRoundError} && failure.phase === :moment
            after = current_proposal(MLDataDevices.CPUDevice(), sampler)
            @test logdensityof(after, T(1)) == logdensityof(learned, T(1))
            copyto!(sampler.target.context.active, T[1])
            @test all(isfinite, Array(importance_sample!(sampler).logweights))
            @test Array(result.samples) == saved
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    check_adaptive_independent()
    check_adaptive_independent_edges()
    check_adaptive_independent_batch()
end
