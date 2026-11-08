import DensityInterface: logdensityof
import Distributions
import LogExpFunctions: logaddexp
import MLDataDevices
import Random
import Statistics

function independent_fit_oracle(family, samples, logweights, clipped)
    logs = copy(logweights)
    clipped && (logs .= min.(logs, sort(logs)[length(logs)-isqrt(length(logs))+1]))
    weights = exp.(logs .- maximum(logs))
    weights ./= sum(weights)
    center = samples * weights
    variance = vec(sum((samples .- center) .^ 2 .* reshape(weights, 1, :); dims = 2))
    return family === Distributions.Gamma ?
           Distributions.Gamma.(center .^ 2 ./ variance, variance ./ center) :
           Distributions.Laplace.(center, sqrt.(variance ./ 2))
end

@testset "explicit device precision converts independent scalar caches" begin
    device = MLDataDevices.with_eltype(MLDataDevices.CPUDevice(), Float32)
    for (q, method) in (
        (BoxUniform([0.0, 1.0], [2.0, 4.0]), ImportanceSampling),
        (IndependentGamma([2.0, 3.0], [1.0, 0.5]), AMIS),
        (IndependentLaplace([0.0, 1.0], [1.0, 0.5]), NPMC),
    )
        algorithm =
            method === ImportanceSampling ? method(q; nsamples = 128) :
            method(q; rounds = 1, round_size = 128)
        result = importance_sample!(device(prepare_sampler(Random.Xoshiro(1913), q, algorithm)))
        @test eltype(result.samples) === eltype(result.logweights) === Float32 &&
              maximum(abs, result.logweights) <= 64eps(Float32)
    end
    rounded = device(IndependentGamma(prevfloat(32.0), 1.0))
    @test logdensityof(rounded, 32.0f0) ≈ Distributions.logpdf(Distributions.Gamma(32.0f0), 32.0f0)
    for q in (
        BoxUniform(1.0, nextfloat(1.0)),
        IndependentGamma(1e-300, 1.0),
        IndependentLaplace(0.0, 1e-300),
    )
        @test_throws ArgumentError device(q)
    end
    @test_throws ArgumentError MLDataDevices.with_eltype(device, Float16)(
        IndependentLaplace(0.0, 1.0),
    )
end

@testset "independent moment fits and two-round weighting laws" begin
    cases = (
        (
            IndependentGamma([2.0, 3.0], [1.0, 0.5]),
            IndependentGamma([3.0, 5.0], [0.6, 0.3]),
            [1.0, 2.0],
            Distributions.Gamma,
        ),
        (
            IndependentLaplace([0.0, 1.0], [1.0, 0.5]),
            IndependentLaplace([0.4, 0.8], [0.7, 0.8]),
            [-0.2, 1.3],
            Distributions.Laplace,
        ),
    )
    for (proposal, target, point, family) in cases, method in (AMIS, NPMC)
        count = 256
        first = prepare_sampler(
            Random.Xoshiro(1901),
            target,
            method(proposal; rounds = 1, round_size = count),
        )
        pilot = importance_sample!(first)
        learned = current_proposal(first)
        oracle = independent_fit_oracle(family, pilot.samples, pilot.logweights, method === NPMC)
        @test logdensityof(learned, point) ≈ sum(Distributions.logpdf.(oracle, point)) rtol=128eps(
            Float64,
        )

        sizes = [128, 256]
        prepared = prepare_sampler(
            Random.Xoshiro(1901),
            target,
            method(proposal; rounds = 2, round_size = sizes),
        )
        result = importance_sample!(prepared)
        @test result.samples[:, 1:sizes[1]] == pilot.samples[:, 1:sizes[1]]
        fitted = independent_fit_oracle(
            family,
            pilot.samples[:, 1:sizes[1]],
            pilot.logweights[1:sizes[1]],
            method === NPMC,
        )
        expected = map(eachcol(result.samples), result.provenance.round) do x, round
            fitted_logdensity = sum(Distributions.logpdf.(fitted, x))
            denominator =
                method === AMIS ?
                logaddexp(
                    log(sizes[1])+logdensityof(proposal, x),
                    log(sizes[2])+fitted_logdensity,
                ) - log(sum(sizes)) :
                (round == 1 ? logdensityof(proposal, x) : fitted_logdensity)
            logdensityof(target, x) - denominator
        end
        @test result.logweights ≈ expected rtol=256eps(Float64) atol=256eps(Float64)
    end
end

independent_switch_target(x, p) = p.valid[] ? logdensityof(p.law, x) : -Inf

@testset "independent learned proposals survive failure and retarget" begin
    for (proposal, method, point) in (
        (IndependentGamma([2.0, 3.0], [1.0, 0.5]), AMIS, [1.0, 2.0]),
        (IndependentLaplace([0.0, 1.0], [1.0, 0.5]), NPMC, [0.0, 1.0]),
    )
        p = (; law = proposal, valid = Ref(true))
        sampler = prepare_sampler(
            Random.Xoshiro(1902),
            independent_switch_target,
            p,
            method(proposal; rounds = 2, round_size = 256),
        )
        importance_sample!(sampler)
        saved = current_proposal(sampler)
        p.valid[] = false
        failure = try
            importance_sample!(sampler)
        catch error
            error
        end
        @test failure isa Union{AMISRoundError,NPMCRoundError} && failure.phase === :moment
        @test logdensityof(current_proposal(sampler), point) == logdensityof(saved, point)
        p.valid[] = true
        @test all(isfinite, importance_sample!(sampler).logweights)
        changed = retarget(Random.Xoshiro(1903), sampler, proposal)
        @test logdensityof(current_proposal(changed), point) ==
              logdensityof(current_proposal(sampler), point)
    end
end

@testset "degenerate independent fits fail without committing" begin
    for (proposal, method, sizes) in
        ((IndependentGamma(2.0, 1.0), AMIS, [1]), (IndependentLaplace(0.0, 1.0), NPMC, [16, 1]))
        sampler = prepare_sampler(
            Random.Xoshiro(1904),
            proposal,
            method(proposal; rounds = length(sizes), round_size = sizes),
        )
        failure = try
            importance_sample!(sampler)
        catch error
            error
        end
        @test failure isa Union{AMISRoundError,NPMCRoundError} &&
              failure.round == length(sizes) &&
              failure.phase === :moment &&
              failure.cause isa DomainError
        @test logdensityof(current_proposal(sampler), 1.0) == logdensityof(proposal, 1.0)
    end
end

@testset "independent adaptive serial and threaded replay" begin
    for q in (IndependentGamma([2.0, 3.0], [1.0, 0.5]), IndependentLaplace([0.0, 1.0], [1.0, 0.5]))
        algorithm = AMIS(q; rounds = 3, round_size = [96, 128, 64])
        serial = importance_sample(Random.Xoshiro(1905), q, algorithm; threaded = false)
        threaded = importance_sample(Random.Xoshiro(1905), q, algorithm; threaded = true)
        @test serial.samples == threaded.samples && serial.logweights == threaded.logweights
    end
end
