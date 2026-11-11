using Test, CUDA, ImportanceSamplers, Random, LinearAlgebra
import DensityInterface, MLDataDevices

CUDA.allowscalar(false)

function cold_regression(g, x, p)
    d = size(p.X, 2)
    value = -sum(abs2, view(x, 1:d)) / 8
    if g !== nothing
        for j = 1:d
            g[j] = -x[j] / 4
        end
    end
    for i in eachindex(p.y)
        eta = zero(eltype(x))
        for j = 1:d
            eta += p.X[i, j] * x[j]
        end
        residual = p.y[i] - eta
        value -= residual^2 / 2
        if g !== nothing
            for j = 1:d
                g[j] += p.X[i, j] * residual
            end
        end
    end
    return value
end
cold_regression_value(x, p) = cold_regression(nothing, x, p)
cold_regression_score!(g, x, p) = (cold_regression(g, x, p); nothing)

# Run this file in a fresh process: earlier lower draws can grow CUDA's stack.
@testset "Cold CUDA HMC supports reduction-based regression targets" begin
    BLAS.set_num_threads(1)
    rng = Xoshiro(101)
    X = randn(rng, 1024, 25)
    y = X * [0.5cos(j) / 5 for j = 1:25] + randn(rng, 1024)
    context = (; X, y)
    precision = Symmetric(X'X + I / 4)
    location = precision \ (X'y)
    covariance = cholesky(precision) \ Matrix{Float64}(I, 25, 25)
    root = Matrix(cholesky(Symmetric(covariance)).L)
    bank = ProposalBank([
        FactorStudentT(7.0, location + root * [0.2sin(i + 2j) for i = 1:25], 0.85root)
        for j = 1:16
    ])
    algorithm = LAIS(
        bank;
        transition = HamiltonianMonteCarlo(;
            step_size = 0.15,
            leapfrog_steps = 8,
            mass = precision,
            tuning = WarmupTuning(1),
        ),
        rounds = 1,
        round_size = 16,
    )
    target = LogTarget(cold_regression_value; grad = cold_regression_score!)
    device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)
    prepared = device(prepare_sampler(Xoshiro(705), target, context, algorithm))
    result = importance_sample!(prepared)
    CUDA.synchronize()
    samples, logweights = Array(result.samples), Array(result.logweights)
    final_bank = current_proposal(MLDataDevices.cpu_device(), prepared)
    expected = map(eachcol(samples)) do x
        logq = [DensityInterface.logdensityof(proposal, x) for proposal in final_bank.proposals]
        largest = maximum(logq)
        cold_regression_value(x, context) - largest - log(sum(exp.(logq .- largest)) / 16)
    end
    @test logweights ≈ expected rtol=1e-10
end
