using Test, CUDA, ImportanceSamplers, Random, LinearAlgebra
import MLDataDevices, DensityInterface

CUDA.allowscalar(false)

function width_cuda_value(x)
    value = zero(eltype(x))
    for z in x
        value -= z^2 / 2 + z^4 / 16
    end
    return value
end
function width_cuda_score!(g, x)
    for row in eachindex(x)
        g[row] = -x[row] - x[row]^3 / 4
    end
    return nothing
end

@testset "CUDA width fitting supports non-Hamiltonian LAIS and retained HAIS" begin
    device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)
    target = LogTarget(width_cuda_value; grad=width_cuda_score!)
    bank = ProposalBank([SphericalGaussian(zeros(Float32, 3), 2.0f0)])
    # The random-walk execution tag must not route device reductions through CPU loops.
    walk = LAIS(
        bank;
        transition=RandomWalkMetropolis(0.1f0),
        rounds=1,
        round_size=128,
        adapt_scales=true,
    )
    prepared = device(prepare_sampler(Xoshiro(44), target, walk))
    result = importance_sample!(prepared)
    learned = only(current_proposal(MLDataDevices.cpu_device(), prepared).proposals)
    x, lw = Array(result.samples), Array(result.logweights)
    weights = exp.(lw .- maximum(lw))
    weights ./= sum(weights)
    variance = (x .- x*weights) .^ 2*weights
    expected = DiagonalGaussian(learned.location, sqrt.((4.0f0 .+ variance) ./ 2.0f0))
    @test DensityInterface.logdensityof(learned, zeros(Float32, 3)) ≈
          DensityInterface.logdensityof(expected, zeros(Float32, 3)) rtol=64eps(Float32)
    algorithm = HAIS(
        bank;
        transition=HamiltonianMonteCarlo(; step_size=0.1f0, leapfrog_steps=2),
        rounds=1,
        round_size=128,
        adapt_scales=true,
    )
    prepared = device(prepare_sampler(Xoshiro(45), target, algorithm))
    importance_sample!(prepared)
    fitted = only(current_proposal(MLDataDevices.cpu_device(), prepared).proposals)
    retargeted = retarget(Xoshiro(46), prepared, target)
    @test DensityInterface.logdensityof(
        only(current_proposal(MLDataDevices.cpu_device(), retargeted).proposals),
        zeros(Float32, 3),
    ) ≈ DensityInterface.logdensityof(fitted, zeros(Float32, 3))
    next = importance_sample!(prepared)
    point = Array(next.samples)[:, 1]
    @test Array(next.logweights)[1] ≈
          width_cuda_value(point) - DensityInterface.logdensityof(fitted, point)
end

@testset "CUDA lower widths preserve local moments, densities and storage precision" begin
    device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)
    target = LogTarget(width_cuda_value; grad=width_cuda_score!)
    for T in (Float32, Float64), method in (LAIS, HAIS)
        factor = T[1.5 0 0; 0.3 0.8 0; 0.1 -0.2 1.2]
        cases = (
            (DiagonalGaussian, T(1), T[1.5, 0.8, 1.2]),
            (FactorGaussian, T(1), factor),
            ((m, s)->FactorStudentT(T(6), m, s), T(1.5), factor),
            (IndependentLaplace, T(2), T[1.5, 0.8, 1.2]),
        )
        for (constructor, multiplier, storage) in cases
            bank = ProposalBank([constructor(fill(T(slot-1)/4, 3), storage) for slot = 1:2])
            upper = HamiltonianMonteCarlo(; step_size=T(0.1), leapfrog_steps=2)
            algorithm = method(bank; transition=upper, rounds=1, round_size=512, adapt_scales=true)
            prepared = device(prepare_sampler(Xoshiro(231), target, algorithm))
            fixed = method(bank; transition=upper, rounds=1, round_size=512)
            reference = importance_sample!(device(prepare_sampler(Xoshiro(231), target, fixed)))
            result = importance_sample!(prepared)
            snapshot = current_proposal(MLDataDevices.cpu_device(), prepared)
            x, lw, ids = Array(result.samples),
            Array(result.logweights),
            Array(result.provenance.proposal_id)
            @test x == Array(reference.samples)
            @test lw == Array(reference.logweights)
            @test result.diagnostics.transfers.count == reference.diagnostics.transfers.count
            old_variance =
                multiplier .* (storage isa AbstractVector ? storage .^ 2 : diag(storage*storage'))
            for slot = 1:2
                group = findall(==(slot), ids)
                weights = exp.(lw[group] .- maximum(lw[group]))
                weights ./= sum(weights)
                points = x[:, group]
                mean = points*weights
                variance = (points .- mean) .^ 2*weights
                ratios = sqrt.((old_variance .+ variance) ./ (2 .* old_variance))
                expected = constructor(snapshot.proposals[slot].location, ratios .* storage)
                point = T[0.4, -0.2, 0.7]
                @test DensityInterface.logdensityof(snapshot.proposals[slot], point) ≈
                      DensityInterface.logdensityof(expected, point) rtol=64eps(T)
            end
        end
    end
end
