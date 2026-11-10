using Test, CUDA, Enzyme, ImportanceSamplers, Random, LinearAlgebra
import ADTypes, MLDataDevices

CUDA.allowscalar(false)

function hmc_cuda_logtarget(x, context)
    value = zero(eltype(x))
    for row in eachindex(x)
        z = x[row] - context.centre[row]
        value -= z^2 / 2 + z^4 / 16
    end
    return value
end

function hmc_cuda_gradient!(gradient, x, context)
    for row in eachindex(x)
        z = x[row] - context.centre[row]
        gradient[row] = -z - z^3 / 4
    end
    return nothing
end

function hmc_cuda_algorithm(method, T, mass; warmup=4, dimension=3, count=3, steps=3)
    bank = ProposalBank([SphericalGaussian(fill(T(slot - 2) / 4, dimension), T(0.8))
                         for slot in 1:count])
    transition = HamiltonianMonteCarlo(; step_size=T(0.1), leapfrog_steps=steps,
                                       mass, tuning=WarmupTuning(warmup))
    return method(bank; transition, rounds=2, round_size=6count)
end

@testset "Native CUDA HMC derivative routes preserve reuse and retargeting" begin
    device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)
    ad = ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
    explicit = LogTarget(hmc_cuda_logtarget; grad=hmc_cuda_gradient!)
    automatic = LogTarget(hmc_cuda_logtarget, ad)
    for T in (Float32, Float64), method in (LAIS, HAIS)
        for mass in (I, Diagonal(T[1, 2, 3]), T[1 0.1 0; 0.1 2 0.2; 0 0.2 1])
            algorithm = hmc_cuda_algorithm(method, T, mass)
            context = (; centre=T[0.2, -0.3, 0.4])
            generated = device(prepare_sampler(Xoshiro(231), automatic, context, algorithm))
            reference = device(prepare_sampler(Xoshiro(231), explicit, context, algorithm))
            tolerance = T === Float32 ? T(2e-3) : T(2e-10)
            for pass in 1:3
                if pass == 3
                    shifted = (; centre=T[-0.2, 0.3, -0.4])
                    generated = retarget(Xoshiro(83), generated, automatic, shifted)
                    reference = retarget(Xoshiro(83), reference, explicit, shifted)
                end
                actual, expected = importance_sample!(generated),
                                   importance_sample!(reference)
                @test Array(actual.samples) ≈ Array(expected.samples) rtol = tolerance atol = tolerance
                @test Array(actual.logweights) ≈ Array(expected.logweights) rtol = tolerance atol = tolerance
                @test actual.diagnostics.transition.step_size ≈
                      expected.diagnostics.transition.step_size rtol = tolerance
                @test actual.diagnostics.transition.warmup_proposals == (pass == 2 ? 0 : 12)
                @test actual.samples isa CUDA.AnyCuArray
            end
        end
    end
end

function hmc_cuda_support_target(x)
    abs(x[1]) < eltype(x)(0.01) || return eltype(x)(-Inf)
    value = zero(eltype(x))
    for row in eachindex(x)
        value -= x[row]^2 / 2
    end
    return value
end
function hmc_cuda_support_gradient!(gradient, x)
    for row in eachindex(x)
        gradient[row] = -x[row]
    end
    return nothing
end

@testset "Native CUDA HMC inactive trajectories use safe batch columns" begin
    device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)
    ad = ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
    target = LogTarget(hmc_cuda_support_target, ad)
    algorithm = LAIS(ProposalBank([SphericalGaussian(zeros(Float32, 3), 0.1f0)
                                   for _ in 1:3]);
                     transition=HamiltonianMonteCarlo(; step_size=1.0f0, leapfrog_steps=5),
                     rounds=1, round_size=6)
    result = importance_sample!(device(prepare_sampler(Xoshiro(35), target, algorithm)))
    @test result.diagnostics.transition.production_support_rejections == 3
    # Fixed-width Enzyme evaluates safe replacement columns at later steps.
    @test result.diagnostics.transition.production_target_evaluations == 15
    @test result.diagnostics.transition.production_gradient_evaluations == 15
end

@testset "Native CUDA HMC status transfers are sweep bounded" begin
    device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)
    target = LogTarget(hmc_cuda_logtarget; grad=hmc_cuda_gradient!)
    summaries = map(((3, 1, 1), (9, 5, 7))) do (count, dimension, steps)
        algorithm = hmc_cuda_algorithm(LAIS, Float32, I;
                                       warmup=2, dimension, count, steps)
        context = (; centre=zeros(Float32, dimension))
        result = importance_sample!(device(prepare_sampler(Xoshiro(91), target, context,
                                                           algorithm)))
        transfers = result.diagnostics.transfers
        return (transfers.count, transfers.bytes)
    end
    @test summaries[1] == summaries[2]
end

@testset "Native CUDA HAIS retains translated Student-t and Laplace proposals" begin
    device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)
    target = LogTarget(hmc_cuda_logtarget; grad=hmc_cuda_gradient!)
    banks = (ProposalBank([SphericalStudentT(3.0f0, fill(Float32(slot - 2) / 4, 3), 0.8f0)
                           for slot in 1:3]),
             ProposalBank([IndependentLaplace(fill(Float32(slot - 2) / 4, 3),
                                              fill(0.8f0, 3))
                           for slot in 1:3]))
    for bank in banks
        algorithm = HAIS(bank;
                         transition=HamiltonianMonteCarlo(;
                                                          step_size=0.1f0,
                                                          leapfrog_steps=3), rounds=2,
                         round_size=18)
        result = importance_sample!(device(prepare_sampler(Xoshiro(88), target,
                                                           (; centre=zeros(Float32, 3)),
                                                           algorithm)))
        @test size(result.samples) == (3, 36)
        @test all(isfinite, Array(result.logweights))
        @test result.diagnostics.cooperation.rounds == 2
    end
end

hmc_cuda_named_target(theta) = -theta.scale - sum(abs2, theta.offset) / 2
function hmc_cuda_named_gradient!(gradient, theta)
    gradient.scale[] = -one(theta.scale)
    for row in eachindex(theta.offset)
        gradient.offset[row] = -theta.offset[row]
    end
    return nothing
end

@testset "Native CUDA HAIS explicit named scores retain their transform" begin
    device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)
    bank = ProposalBank([SphericalGaussian(fill(Float32(slot - 2) / 4, 3), 0.8f0)
                         for slot in 1:3])
    algorithm = HAIS(bank;
                     transition=HamiltonianMonteCarlo(;
                                                      step_size=0.1f0, leapfrog_steps=3),
                     rounds=2, round_size=18)
    layout = (scale=(1 => PositiveTransform()), offset=(2:3 => IdentityTransform()))
    prepared = device(prepare_sampler(Xoshiro(92),
                                      LogTarget(hmc_cuda_named_target;
                                                grad=hmc_cuda_named_gradient!), algorithm;
                                      transform=layout))
    for _ in 1:2
        result = importance_sample!(prepared)
        @test all(>(0), Array(result.samples.scale))
        @test size(result.samples.offset) == (2, 36)
        @test all(isfinite, Array(result.logweights))
    end
end
