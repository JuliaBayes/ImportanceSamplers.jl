import DensityInterface
import MLDataDevices

abstract type TestScriptedRNG <: PureRNGs.AbstractPureRNG end

(::MLDataDevices.CPUDevice)(rng::TestScriptedRNG) = rng

function PureRNGs.rand_next!(rng::TestScriptedRNG, values::AbstractArray; threaded=false)
    isempty(values) || error("unexpected uniform fill")
    return values, rng
end

function PureRNGs.randn_next!(rng::TestScriptedRNG, values::AbstractArray; threaded=false)
    isempty(values) || error("unexpected normal fill")
    return values, rng
end

function check_static_mis_effective_coefficients(cdf, coefficient_sets...)
    boundaries = BigFloat.(cdf)
    interval_masses = diff(vcat(zero(BigFloat), boundaries))
    @test sum(interval_masses) == one(BigFloat)
    @test all(>(zero(BigFloat)), interval_masses)
    for logcoefficients in coefficient_sets
        masses = exp.(BigFloat.(logcoefficients))
        @test all(isapprox.(
            masses,
            interval_masses;
            rtol=BigFloat(8eps(Float32)),
            atol=zero(BigFloat),
        ))
        # For a target supported only by the last disjoint proposal, its
        # proposal density cancels from this exact linear-normalizer oracle.
        @test isapprox(
            interval_masses[end] / masses[end],
            one(BigFloat);
            atol=BigFloat(16eps(Float64)),
            rtol=zero(BigFloat),
        )
    end
    return nothing
end

struct TestScalarProposal{T<:AbstractFloat}
    location::T
    draw_count::Base.RefValue{Int}
    history::Vector{T}
end

TestScalarProposal(location::T) where {T<:AbstractFloat} =
    TestScalarProposal(location, Ref(0), T[])

function PureRNGs.rand_next(rng::PureRNGs.AbstractPureRNG, proposal::TestScalarProposal{T}) where {T}
    offset, rng = randn_next(rng, T)
    sample = proposal.location + offset
    proposal.draw_count[] += 1
    push!(proposal.history, sample)
    return sample, rng
end

function DensityInterface.logdensityof(proposal::TestScalarProposal, sample::Real)
    offset = sample - proposal.location
    return -oftype(offset, 0.5) * abs2(offset) - oftype(offset, 0.5 * log(2pi))
end

struct TestVectorProposal{T<:AbstractFloat}
    location::Vector{T}
    draw_count::Base.RefValue{Int}
    history::Vector{Vector{T}}
end

TestVectorProposal(location::Vector{T}) where {T<:AbstractFloat} =
    TestVectorProposal(location, Ref(0), Vector{T}[])

function PureRNGs.rand_next(rng::PureRNGs.AbstractPureRNG, proposal::TestVectorProposal{T}) where {T}
    offset, rng = randn_next(rng, T, length(proposal.location))
    sample = proposal.location + offset
    proposal.draw_count[] += 1
    push!(proposal.history, sample)
    return sample, rng
end

function DensityInterface.logdensityof(proposal::TestVectorProposal, sample::AbstractVector)
    offset = sample - proposal.location
    return -oftype(first(offset), 0.5) * sum(abs2, offset) -
           oftype(first(offset), 0.5 * length(offset) * log(2pi))
end

struct TestNamedProposal{T<:AbstractFloat}
    draw_count::Base.RefValue{Int}
    history::Vector{NamedTuple}
end

TestNamedProposal(::Type{T}=Float64) where {T<:AbstractFloat} =
    TestNamedProposal{T}(Ref(0), NamedTuple[])

function PureRNGs.rand_next(rng::PureRNGs.AbstractPureRNG, proposal::TestNamedProposal{T}) where {T}
    location, rng = randn_next(rng, T)
    position, rng = randn_next(rng, T, 2)
    scale, rng = randn_next(rng, T)
    sample = (; location, state=(; position, scale))
    proposal.draw_count[] += 1
    push!(proposal.history, sample)
    return sample, rng
end

function DensityInterface.logdensityof(::TestNamedProposal{T}, sample::NamedTuple) where {T}
    square_sum = abs2(sample.location) + sum(abs2, sample.state.position) +
                 abs2(sample.state.scale)
    return -T(0.5) * square_sum - T(2 * log(2pi))
end

struct TestAbstractElementVectorProposal
    draw_count::Base.RefValue{Int}
end

TestAbstractElementVectorProposal() = TestAbstractElementVectorProposal(Ref(0))

function PureRNGs.rand_next(rng::PureRNGs.AbstractPureRNG, proposal::TestAbstractElementVectorProposal)
    proposal.draw_count[] += 1
    first_value, rng = rand_next(rng, Float64)
    second_value, rng = rand_next(rng, Float64)
    return Real[first_value, Float32(second_value)], rng
end

function DensityInterface.logdensityof(
    ::TestAbstractElementVectorProposal, sample::AbstractVector
)
    return -sum(abs2, sample)
end

struct TestChangingVectorLengthProposal
    draw_count::Base.RefValue{Int}
end

TestChangingVectorLengthProposal() = TestChangingVectorLengthProposal(Ref(0))

function PureRNGs.rand_next(rng::PureRNGs.AbstractPureRNG, proposal::TestChangingVectorLengthProposal)
    proposal.draw_count[] += 1
    sample_length = isone(proposal.draw_count[]) ? 2 : 3
    return randn_next(rng, Float64, sample_length)
end

function DensityInterface.logdensityof(
    ::TestChangingVectorLengthProposal, sample::AbstractVector
)
    return -0.5 * sum(abs2, sample)
end

struct TestEmptyNamedProposal
    draw_count::Base.RefValue{Int}
end

TestEmptyNamedProposal() = TestEmptyNamedProposal(Ref(0))

function PureRNGs.rand_next(rng::PureRNGs.AbstractPureRNG, proposal::TestEmptyNamedProposal)
    _, rng = rand_next(rng, Float64)
    proposal.draw_count[] += 1
    return NamedTuple(), rng
end

DensityInterface.logdensityof(::TestEmptyNamedProposal, sample::NamedTuple) = 0.0

struct TestNestedEmptyNamedProposal
    draw_count::Base.RefValue{Int}
end

TestNestedEmptyNamedProposal() = TestNestedEmptyNamedProposal(Ref(0))

function PureRNGs.rand_next(rng::PureRNGs.AbstractPureRNG, proposal::TestNestedEmptyNamedProposal)
    proposal.draw_count[] += 1
    value, rng = rand_next(rng, Float64)
    return (; value, empty=NamedTuple()), rng
end

function DensityInterface.logdensityof(
    ::TestNestedEmptyNamedProposal, sample::NamedTuple
)
    return -abs2(sample.value)
end

struct TestReusedVectorBufferProposal{T<:AbstractFloat}
    buffer::Vector{T}
    draw_count::Base.RefValue{Int}
    history::Vector{Vector{T}}
end

function TestReusedVectorBufferProposal(buffer::Vector{T}) where {T<:AbstractFloat}
    return TestReusedVectorBufferProposal(buffer, Ref(0), Vector{T}[])
end

function PureRNGs.rand_next(rng::PureRNGs.AbstractPureRNG, proposal::TestReusedVectorBufferProposal{T}) where {T}
    for sample_index in eachindex(proposal.buffer)
        proposal.buffer[sample_index], rng = randn_next(rng, T)
    end
    proposal.draw_count[] += 1
    push!(proposal.history, copy(proposal.buffer))
    return proposal.buffer, rng
end

function DensityInterface.logdensityof(
    ::TestReusedVectorBufferProposal, sample::AbstractVector
)
    return -0.5 * sum(abs2, sample)
end
