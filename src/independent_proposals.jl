abstract type _NativeIndependentProposal end

# Keep scalar caches and array parameters under the same explicit device policy.
MLDataDevices.isleaf(::_NativeIndependentProposal) = true
MLDataDevices.get_device(proposal::_NativeIndependentProposal) =
    MLDataDevices.get_device(_independent_parameters(proposal))
MLDataDevices.get_device_type(proposal::_NativeIndependentProposal) =
    MLDataDevices.get_device_type(_independent_parameters(proposal))

struct _BoxFamily <: AbstractProposalFamily end
struct _GammaFamily <: AbstractProposalFamily end

struct _BoxUniform{P,T} <: _NativeIndependentProposal
    lower::P
    upper::P
    lognormalizer::T
end

struct _GammaProposal{P,T} <: _NativeIndependentProposal
    shape::P
    scale::P
    logconstants::P
    lognormalizer::T
end

_independent_parameters(proposal::_BoxUniform) = (proposal.lower, proposal.upper)
_independent_parameters(proposal::_GammaProposal) = (proposal.shape, proposal.scale)
_independent_family(::_BoxUniform) = _BoxFamily()
_independent_family(::_GammaProposal) = _GammaFamily()

function _gamma_logconstant(shape::T, scale::T) where {T}
    shape < T(32) && return -SpecialFunctions.loggamma(shape) - shape * log(scale)
    return _gamma_center_logconstant(shape, scale)
end

function _gamma_center_logconstant(shape::T, scale::T) where {T}
    # DLMF 5.11.1: the next Stirling term is <= 1/(156*32^13) < 2e-22.
    # Cache the density at the mean without subtracting O(shape*log(shape)) terms.
    r = inv(shape)
    correction =
        r * evalpoly(
            r*r,
            (
                T(1)/T(12),
                -T(1)/T(360),
                T(1)/T(1260),
                -T(1)/T(1680),
                T(1)/T(1188),
                -T(691)/T(360360),
            ),
        )
    return -log(scale) - T(0.5) * (log(T(2)*T(pi)) + log(shape)) - correction
end

"""
    IndependentGamma(shape, scale)

Normalized Gamma proposal with shape–scale parameters, or a product of
independent Gamma coordinates. Parameters must be finite, strictly positive,
matching `Float32` or `Float64` scalars or vectors. Vector inputs are copied.
The mean is `shape .* scale` and variance is `shape .* scale.^2`.

Plain IS and homogeneous static MIS banks use fixed parameters. [`AMIS`](@ref)
and [`NPMC`](@ref) fit weighted marginal moments, then use
`shape = mean^2 / variance` and `scale = variance / mean`. This is moment
matching, not maximum likelihood. Correlations are not fitted. Degenerate or
unrepresentable fits fail without replacing the committed proposal.
[`APIS`](@ref) and [`DeterministicMixturePMC`](@ref) instead update each mean
at fixed coordinate variance, changing both shape and scale. Gamma proposals
require a target supported on positive coordinates. Adaptation can produce tiny
shapes whose draws underflow at the chosen precision.
Native execution uses eight
rejection candidates per coordinate. Exhaustion or a nonpositive/nonfinite
rounded draw fails the entire call, without discarding selected samples.
"""
function IndependentGamma(shape, scale)
    a, theta = _independent_parameters(shape, scale)
    all(>(0), a) && all(>(0), theta) ||
        throw(ArgumentError("Gamma shape and scale must be positive"))
    constants = _gamma_logconstant.(a, theta)
    all(isfinite, constants) || throw(ArgumentError("Gamma normalization is not representable"))
    return _GammaProposal(a, theta, constants, sum(constants))
end

Adapt.@adapt_structure _GammaProposal

_independent_constructor(::_GammaProposal) = IndependentGamma
_independent_fields(proposal::_GammaProposal) =
    (proposal.shape, proposal.scale, proposal.logconstants, proposal.lognormalizer)

function _adapt_independent_fields(device, proposal)
    policy = applicable(eltype, device) ? eltype(device) : Nothing
    policy isa Type &&
        policy <: AbstractFloat &&
        !(policy <: _NativeGaussianFloat) &&
        throw(ArgumentError("native independent proposals require Float32 or Float64"))
    T = policy === Float32 || policy === Float64 ? policy : typeof(proposal.lognormalizer)
    if T !== typeof(proposal.lognormalizer)
        parameters = _independent_parameters(proposal)
        source_device = MLDataDevices.get_device(proposal)
        if source_device !== nothing && !(source_device isa MLDataDevices.CPUDevice)
            device isa MLDataDevices.CPUDevice || throw(
                ArgumentError(
                    "change independent proposal precision on CPU before accelerator transfer",
                ),
            )
            parameters = map(field -> Adapt.adapt(device, field), parameters)
        end
        # Revalidate rounded parameters and rebuild representation-dependent caches on CPU.
        proposal = _independent_constructor(proposal)(T.(parameters[1]), T.(parameters[2]))
    end
    return map(field -> Adapt.adapt(device, field), _independent_fields(proposal))
end

function Adapt.adapt_structure(device::MLDataDevices.AbstractDevice, proposal::_GammaProposal)
    return _GammaProposal(_adapt_independent_fields(device, proposal)...)
end
DensityInterface.DensityKind(::_GammaProposal) = DensityInterface.HasDensity()
_proposal_dimension(proposal::_GammaProposal) = _gaussian_dimension(proposal.shape)
_default_mass_type(proposal::_GammaProposal) = typeof(proposal.lognormalizer)

@inline function _gamma_logdensity(shape::T, scale::T, constant::T, x::T) where {T}
    isnan(x) && return x
    (x < zero(T) || isinf(x)) && return T(-Inf)
    iszero(x) && return shape == one(T) ? -log(scale) : (shape < one(T) ? T(Inf) : T(-Inf))
    y = x / scale
    if shape < T(32)
        return constant + (shape - one(T)) * log(x) - y
    elseif isfinite(y) && y > zero(T)
        delta = (y - shape) / shape
        if abs(delta) <= T(0.5)
            return constant + shape * LogExpFunctions.log1pmx(delta) - log1p(delta)
        end
        ratio = y / shape
        logratio = ratio > zero(T) ? log(ratio) : log(x) - log(scale) - log(shape)
        return delta < zero(T) ? constant + shape * (logratio - delta) - logratio :
               constant + (shape - one(T)) * logratio - (y - shape)
    end
    # The first division may avoid an overflowing x/scale without subtracting logs.
    ratio = (x / shape) / scale
    if isfinite(ratio) && ratio > zero(T)
        logratio = log(ratio)
        return constant + shape * (logratio - (ratio - one(T))) - logratio
    end
    logratio = log(x) - log(scale) - log(shape)
    return constant + shape * (logratio - expm1(logratio)) - logratio
end

@inline function _gamma_logdensity(shape, scale, constant, x)
    return _gamma_logdensity(promote(shape, scale, constant, x)...)
end

function DensityInterface.logdensityof(proposal::_GammaProposal, sample)
    dimension = _proposal_dimension(proposal)
    length(sample) == dimension ||
        throw(DimensionMismatch("sample dimension differs from Gamma proposal"))
    sample_type = sample isa Number ? typeof(sample) : eltype(sample)
    result = zero(promote_type(typeof(proposal.lognormalizer), sample_type))
    for i = 1:dimension
        value = _gamma_logdensity(
            _packed_sample_coordinate(proposal.shape, i),
            _packed_sample_coordinate(proposal.scale, i),
            _packed_sample_coordinate(proposal.logconstants, i),
            _packed_sample_coordinate(sample, i),
        )
        value == -Inf && return value
        result += value
    end
    return result
end

function _gamma_draw(rng, shape, scale)
    value = _rand_unit_gamma(rng, shape) * scale
    isfinite(value) && value > zero(value) ||
        throw(DomainError(value, "Gamma draw is not finite and positive in the chosen precision"))
    return value
end

Random.rand(rng::Random.AbstractRNG, proposal::_GammaProposal{T}) where {T<:_NativeGaussianFloat} =
    _gamma_draw(rng, proposal.shape, proposal.scale)

function Random.rand(
    rng::Random.AbstractRNG,
    proposal::_GammaProposal{<:AbstractVector{T}},
) where {T}
    return [
        _gamma_draw(rng, proposal.shape[i], proposal.scale[i]) for i in eachindex(proposal.shape)
    ]
end

function _independent_parameters(first, second)
    # Native parameter validation also copies scalar/vector inputs without RNG use.
    a = _validated_native_parameter(first)
    b = _validated_native_parameter(second)
    typeof(a) === typeof(b) || throw(
        ArgumentError(
            "independent proposal parameters must share precision and scalar/vector layout",
        ),
    )
    length(a) == length(b) ||
        throw(DimensionMismatch("independent proposal parameters must have the same length"))
    return a, b
end

"""
    BoxUniform(lower, upper)

Normalized uniform proposal on a scalar interval or an independent vector box.
Bounds must be finite `Float32` or `Float64` values of matching precision and
shape, with `lower < upper` in every coordinate. Vector inputs are copied.
Density evaluation includes the endpoints and returns `-Inf` outside the box.
Finite intervals with an overflowing `upper - lower` remain supported.

The proposal is fixed. It does not use Gaussian covariance adaptation.
"""
function BoxUniform(lower, upper)
    a, b = _independent_parameters(lower, upper)
    all(a .< b) || throw(ArgumentError("box bounds must satisfy lower < upper"))
    T = _gaussian_float_type(a)
    lognormalizer = zero(T)
    for i = 1:_gaussian_dimension(a)
        lognormalizer -= _log_positive_difference(
            _packed_sample_coordinate(b, i),
            _packed_sample_coordinate(a, i),
        )
    end
    return _BoxUniform(a, b, lognormalizer)
end

Adapt.@adapt_structure _BoxUniform
_independent_constructor(::_BoxUniform) = BoxUniform
_independent_fields(proposal::_BoxUniform) =
    (proposal.lower, proposal.upper, proposal.lognormalizer)
function Adapt.adapt_structure(device::MLDataDevices.AbstractDevice, proposal::_BoxUniform)
    return _BoxUniform(_adapt_independent_fields(device, proposal)...)
end
DensityInterface.DensityKind(::_BoxUniform) = DensityInterface.HasDensity()

_proposal_dimension(proposal::_BoxUniform) = _gaussian_dimension(proposal.lower)
_default_mass_type(proposal::_BoxUniform) = typeof(proposal.lognormalizer)

@inline function _box_coordinate(proposal::_BoxUniform, uniform, coordinate)
    return _bounded_interval_value(
        _packed_sample_coordinate(proposal.lower, coordinate),
        _packed_sample_coordinate(proposal.upper, coordinate),
        uniform,
    )
end

function Random.rand(
    rng::Random.AbstractRNG,
    proposal::_BoxUniform{T},
) where {T<:_NativeGaussianFloat}
    return _box_coordinate(proposal, Random.rand(rng, T), 1)
end

function Random.rand(rng::Random.AbstractRNG, proposal::_BoxUniform{<:AbstractVector{T}}) where {T}
    sample = Random.rand(rng, T, _proposal_dimension(proposal))
    for i in eachindex(sample)
        sample[i] = _box_coordinate(proposal, sample[i], i)
    end
    return sample
end

@inline function DensityInterface.logdensityof(proposal::_BoxUniform, sample)
    dimension = _proposal_dimension(proposal)
    length(sample) == dimension || throw(DimensionMismatch("sample dimension differs from box"))
    for i = 1:dimension
        x = _packed_sample_coordinate(sample, i)
        isnan(x) && return oftype(proposal.lognormalizer, NaN)
        lower = _packed_sample_coordinate(proposal.lower, i)
        upper = _packed_sample_coordinate(proposal.upper, i)
        lower <= x <= upper || return oftype(proposal.lognormalizer, -Inf)
    end
    return proposal.lognormalizer
end
