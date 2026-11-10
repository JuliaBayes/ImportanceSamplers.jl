using ImportanceSamplers, LinearAlgebra, Random, Statistics

# A normalized curved distribution. If z1 ~ Normal(0, scale), then
# z2 + bend*(z1^2-scale^2) and z3 are independent standard normals.
function hamiltonian_logtarget(z, p)
    T = eltype(z)
    curved = z[2] + p.bend * (z[1]^2 - p.scale^2)
    quadratic = (z[1] / p.scale)^2 + curved^2 + z[3]^2
    return -quadratic / T(2) - log(p.scale) -
           T(3) / T(2) * log(T(2) * T(pi))
end

function hamiltonian_score!(g, z, p)
    curved = z[2] + p.bend * (z[1]^2 - p.scale^2)
    g[1] = -z[1] / p.scale^2 - 2 * p.bend * z[1] * curved
    g[2] = -curved
    g[3] = -z[3]
    return nothing
end

T = Float64
p = (; scale=T(2), bend=T(0.15))
target = LogTarget(hamiltonian_logtarget; grad=hamiltonian_score!)
bank = ProposalBank([SphericalStudentT(T(7), T[-2, 0, -1], T(1.5)),
                     SphericalStudentT(T(7), T[-2, 0, 1], T(1.5)),
                     SphericalStudentT(T(7), T[2, 0, -1], T(1.5)),
                     SphericalStudentT(T(7), T[2, 0, 1], T(1.5))])
upper = HamiltonianMonteCarlo(; step_size=T(0.12), leapfrog_steps=12,
                              mass=I, tuning=WarmupTuning(100))
algorithm = HAIS(bank; transition=upper, rounds=20, round_size=4096)
prepared = prepare_sampler(Xoshiro(42), target, p, algorithm)
samples = importance_sample!(prepared)

# These are finite-run estimates, not exact equalities. A high weight ESS does
# not prove coverage. Independent whole runs assess Monte Carlo variability.
function summarize(result)
    (; mean=mean(result), variance=var(result),
     normalizer=exp(lognormalizer(result)))
end
replicates = [summarize(samples)]
for seed in 101:108
    push!(replicates, summarize(importance_sample(Xoshiro(seed), target, p, algorithm)))
end
estimates = (; mean=mean(getproperty.(replicates, :mean)),
             variance=mean(getproperty.(replicates, :variance)),
             normalizer=mean(getproperty.(replicates, :normalizer)))
standard_errors = (; mean=std(getproperty.(replicates, :mean)) / sqrt(length(replicates)),
                   variance=std(getproperty.(replicates, :variance)) /
                            sqrt(length(replicates)),
                   normalizer=std(getproperty.(replicates, :normalizer)) /
                              sqrt(length(replicates)))
reference = (; mean=zeros(T, 3),
             variance=T[p.scale^2, 1 + 2p.bend^2 * p.scale^4, 1], normalizer=one(T))
display((; estimates, standard_errors, reference,
         transition=samples.diagnostics.transition))

# LAIS(bank; transition=upper, rounds=20, round_size=4096) uses the same HMC
# transition without cooperation, and moves centres before each lower round.
# Reuse retains learned centres and the frozen step, but returns fresh draws:
# next_samples = importance_sample!(prepared)
# snapshot = current_proposal(prepared) # An independent lower-proposal bank.
