import DensityInterface
import LogDensityProblems

struct StaticMISDimensionTarget{D} end

LogDensityProblems.capabilities(::Type{<:StaticMISDimensionTarget}) =
    LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(::StaticMISDimensionTarget{D}) where {D} = D
LogDensityProblems.logdensity(::StaticMISDimensionTarget, sample) = -sum(abs2, sample)

struct StaticMISAssignmentFailure <: Exception
    sample_index::Int
end

struct StaticMISFailingAssignmentRNG{R<:PureRNGs.AbstractPureRNG} <: TestScriptedRNG
    inner::R
    uniform_count::Base.RefValue{Int}
    fail_at::Int
end

function PureRNGs.rand_next(rng::StaticMISFailingAssignmentRNG, ::Type{Float64})
    rng.uniform_count[] += 1
    rng.uniform_count[] == rng.fail_at && throw(
        StaticMISAssignmentFailure(rng.uniform_count[]),
    )
    value, inner = rand_next(rng.inner, Float64)
    return value, StaticMISFailingAssignmentRNG(inner, rng.uniform_count, rng.fail_at)
end

mutable struct StaticMISGaussian{T<:AbstractFloat}
    location::T
    draw_count::Base.RefValue{Int}
    density_count::Base.RefValue{Int}
end

StaticMISGaussian(location::T) where {T<:AbstractFloat} =
    StaticMISGaussian(location, Ref(0), Ref(0))

function PureRNGs.rand_next(rng::PureRNGs.AbstractPureRNG, proposal::StaticMISGaussian{T}) where {T}
    proposal.draw_count[] += 1
    offset, rng = randn_next(rng, T)
    return proposal.location + offset, rng
end

function DensityInterface.logdensityof(proposal::StaticMISGaussian, sample::Real)
    proposal.density_count[] += 1
    offset = sample - proposal.location
    return -oftype(offset, 0.5) * abs2(offset) - oftype(offset, 0.5 * log(2pi))
end

function static_mis_gaussian_logdensity(location, sample)
    offset = sample - location
    return -0.5 * abs2(offset) - 0.5 * log(2pi)
end

struct StaticMISRecordingRNG{R<:PureRNGs.AbstractPureRNG} <: TestScriptedRNG
    inner::R
    events::Vector{Symbol}
end

function PureRNGs.rand_next(rng::StaticMISRecordingRNG, ::Type{Float64})
    push!(rng.events, :random)
    value, inner = rand_next(rng.inner, Float64)
    return value, StaticMISRecordingRNG(inner, rng.events)
end

mutable struct StaticMISUniformProposal
    location::Float64
    draw_count::Base.RefValue{Int}
    events::Vector{Symbol}
end

function PureRNGs.rand_next(rng::StaticMISRecordingRNG, proposal::StaticMISUniformProposal)
    proposal.draw_count[] += 1
    push!(proposal.events, :draw)
    offset, rng = rand_next(rng, Float64)
    return proposal.location + offset, rng
end

DensityInterface.logdensityof(::StaticMISUniformProposal, ::Float64) = 0.0

mutable struct StaticMISTableProposal{T<:AbstractFloat}
    proposal_id::Int
    draw_value::T
    logdensities::Matrix{T}
    draw_count::Base.RefValue{Int}
    density_count::Base.RefValue{Int}
    fail_draw::Bool
end

function StaticMISTableProposal(
    proposal_id::Int,
    draw_value::T,
    logdensities::Matrix{T};
    fail_draw=false,
) where {T<:AbstractFloat}
    return StaticMISTableProposal(
        proposal_id,
        draw_value,
        logdensities,
        Ref(0),
        Ref(0),
        fail_draw,
    )
end

function PureRNGs.rand_next(rng::PureRNGs.AbstractPureRNG, proposal::StaticMISTableProposal)
    proposal.draw_count[] += 1
    proposal.fail_draw && error("intentional static-MIS draw failure")
    return proposal.draw_value, rng
end

function DensityInterface.logdensityof(
    proposal::StaticMISTableProposal,
    sample::Real,
)
    proposal.density_count[] += 1
    return proposal.logdensities[proposal.proposal_id, Int(sample)]
end

struct StaticMISTableTarget{T<:AbstractFloat}
    logdensities::Vector{T}
end

(target::StaticMISTableTarget)(sample::Real) = target.logdensities[Int(sample)]

mutable struct StaticMISThreadState
    draw_count::Base.RefValue{Int}
    draw_tasks::Vector{Task}
    target_tasks::Vector{Task}
    density_tasks::Matrix{Task}
    density_draw_counts::Matrix{Int}
end

function StaticMISThreadState(nsamples, nproposals)
    return StaticMISThreadState(
        Ref(0),
        Vector{Task}(undef, nsamples),
        Vector{Task}(undef, nsamples),
        Matrix{Task}(undef, nproposals, nsamples),
        Matrix{Int}(undef, nproposals, nsamples),
    )
end

struct StaticMISThreadProposal
    proposal_id::Int
    state::StaticMISThreadState
end

function PureRNGs.rand_next(rng::PureRNGs.AbstractPureRNG, proposal::StaticMISThreadProposal)
    state = proposal.state
    state.draw_count[] += 1
    sample_index = state.draw_count[]
    state.draw_tasks[sample_index] = current_task()
    return Float64(sample_index), rng
end

function DensityInterface.logdensityof(
    proposal::StaticMISThreadProposal,
    sample::Float64,
)
    sample_index = Int(sample)
    state = proposal.state
    state.density_tasks[proposal.proposal_id, sample_index] = current_task()
    state.density_draw_counts[proposal.proposal_id, sample_index] = state.draw_count[]
    return -sample
end

struct StaticMISThreadTarget
    state::StaticMISThreadState
end

function (target::StaticMISThreadTarget)(sample::Float64)::Float64
    target.state.target_tasks[Int(sample)] = current_task()
    return -sample
end
