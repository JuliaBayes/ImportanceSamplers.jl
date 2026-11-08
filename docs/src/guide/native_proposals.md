# [Proposals](@id proposals-guide)

A proposal supplies normalized draws and their log density.
Its support must cover the regions that contribute to the target integral.

## Gaussian proposals

```@example proposals
using ImportanceSamplers, Random, LinearAlgebra

scalar = SphericalGaussian(0.0, 2.0)
spherical = SphericalGaussian(zeros(3), 2.0)
diagonal = DiagonalGaussian(zeros(3), [0.5, 1.0, 2.0])

C = [1.0 0.6 0.0; 0.6 2.0 0.2; 0.0 0.2 1.0]
correlated = FactorGaussian(zeros(3), cholesky(Symmetric(C)))

rand(Xoshiro(7), correlated)
```

| Constructor | Scale argument | Gaussian covariance |
|:--|:--|:--|
| `SphericalGaussian(mu, sigma)` | Standard deviation | ``\sigma^2 I`` |
| `DiagonalGaussian(mu, scales)` | Marginal standard deviations | `Diagonal(scales .^ 2)` |
| `FactorGaussian(mu, L)` | Lower factor or `Cholesky` | `L * L'` |

The spherical form accepts a scalar or a vector location.
The diagonal and factor forms take vector locations.

Factor a covariance once. Do not pass the covariance itself as `L`.
The implementation uses multiplication and triangular solves, not a covariance inverse.

## Student-t proposals

```@example proposals
proposal = FactorStudentT(
    5.0, zeros(3),
    cholesky(Symmetric(C)),
)
rand(Xoshiro(7), proposal)
```

Student-t constructors follow the same layouts:

```julia
SphericalStudentT(nu, mu, sigma)
DiagonalStudentT(nu, mu, scales)
FactorStudentT(nu, mu, L)
```

The first argument is the positive degrees of freedom `nu`.
The factors define a scale matrix ``S``, not the covariance:

```math
\operatorname{Cov}(X)=\frac{\nu}{\nu-2}S,\qquad \nu>2.
```

To obtain a desired covariance factor `Lcov`, use
`sqrt((nu - 2) / nu) * Lcov` as the Student-t factor.

A multivariate Student-t draw uses one shared random radial scale.
A diagonal scale therefore does not make its coordinates independent.

The adaptive methods keep `nu` fixed.
AMIS, NPMC, CAIS, and FirstOrderGRAMIS require `nu > 2` because they fit covariance.
Location-only methods accept any finite `nu > 0`.
None fits `nu` by maximum likelihood.

## Uniform boxes

`BoxUniform(lower, upper)` defines a scalar interval or a product of coordinate
intervals. Bounds must be finite, ordered, and have matching precision and shape.
The proposal copies vector bounds and has constant density within the box.

```@example proposals
box = BoxUniform([-2.0, 0.0], [2.0, 3.0])
box_samples = importance_sample(
    Xoshiro(17), box, ImportanceSampling(box; nsamples=256),
)
extrema(box_samples.logweights)
```

The target equals the normalized proposal, so every log weight is zero.
Scalar bounds, such as `BoxUniform(0.0, 1.0)`, produce scalar samples.
Density evaluation includes the endpoints and returns `-Inf` outside the box.
Intervals with a finite pair of bounds remain supported even when their width
overflows the chosen floating-point type.

Plain IS and static MIS accept fixed boxes. Covariance-adaptive methods do not
adapt their bounds. A box cannot cover an unbounded target. For overlapping or
disjoint boxes, the MIS scheme determines whether support coverage is required
from the complete mixture, each proposal, or each group.

## Independent Gamma proposals

`IndependentGamma(shape, scale)` uses the shape–scale convention, not shape–rate.
Scalar parameters define one Gamma variable. Matching vectors define independent
coordinates with means `shape .* scale` and variances `shape .* scale.^2`.

```@example proposals
gamma = IndependentGamma([0.5, 5.0], [2.0, 0.2])
gamma_samples = importance_sample(
    Xoshiro(18), gamma, ImportanceSampling(gamma; nsamples=256),
)
extrema(gamma_samples.logweights)
```

Parameters are fixed in plain IS and homogeneous static MIS banks. AMIS and N-PMC
fit Gamma shapes and scales from weighted marginal means and variances.
Direct Gamma proposals differ from exponentiated Gaussian proposals, especially near zero.

Native sampling reserves eight rejection candidates per coordinate. Gamma
random buffers occupy `17 * dimension * nsamples` floating-point elements;
packed MIS also retains a coordinate draw buffer. Exhaustion, underflow to zero,
or a nonfinite draw fails the complete call. No sample is discarded or clipped.
Very small shapes may therefore be unusable at a chosen precision.

## Independent Laplace proposals

`IndependentLaplace(location, scale)` defines a scalar Laplace law or independent
coordinates. Its mean is `location` and its coordinate variances are `2 .* scale.^2`.
Each coordinate uses one uniform draw, without rejection sampling.

```@example proposals
laplace = IndependentLaplace([0.0, 1.0], [1.0, 0.5])
prepared = prepare_sampler(Xoshiro(19), laplace,
    AMIS(laplace; rounds=3, round_size=1_000))
laplace_samples = importance_sample!(prepared)
current_proposal(prepared)
```

AMIS and N-PMC fit independent Gamma and Laplace proposals by marginal moment
matching. Gamma uses `shape = mean^2 / variance` and `scale = variance / mean`.
Laplace uses `location = mean` and `scale = sqrt(variance / 2)`.
These fits do not learn correlations and are not maximum-likelihood fits.
Nonpositive variances or unrepresentable fitted parameters fail the call without
replacing the last committed proposal. No parameter clipping or Gaussian ridge is applied.

| Family | Fixed IS/static MIS | AMIS/N-PMC | Population methods |
|:--|:--|:--|:--|
| Gaussian, Student-t | Yes | Yes | Yes |
| Box-uniform | Yes | No | No |
| Independent Gamma, Laplace | Yes | Marginal moment fits | No |

## Proposal populations

```@example proposals
bank = ProposalBank([
    SphericalGaussian([-2.0, 0.0, 0.0], 1.0),
    SphericalGaussian([ 2.0, 0.0, 0.0], 1.5),
], [1.0, 2.0])

bank.masses
```

`ProposalBank` copies the inputs and normalizes the masses.
Omitting masses gives equal masses. Some adaptive methods require equal positive masses.

A bank is sampler configuration, not a distribution.
It has no standalone `rand` or `logdensityof` method.
The [MIS scheme](@ref static-mis) defines assignment and weighting.

Native banks use one scalar/vector layout, dimension, floating type, and proposal family.
Mixed-family banks are not supported.
Student-t components may have different degrees of freedom.

## Precision

Native proposals support `Float32` and `Float64` and copy their input arrays.
Use matching types for locations, scales, factors, and Student-t degrees of freedom.

Omitted bank masses use the proposals' precision for native and Distributions.jl
inputs. Unknown generic proposals default to `Float64` masses. Explicit masses
retain their precision, so use single-precision masses for a wholly
single-precision weighted bank:

```@example proposals
bank32 = ProposalBank([
    SphericalGaussian(Float32[-1, 0], 1f0),
    SphericalGaussian(Float32[ 1, 0], 1f0),
], Float32[1, 1])

eltype(bank32.masses)
```

Masses participate in numerical promotion.
A device changes precision only when you request that conversion explicitly.
For boxes, independent Gamma, and independent Laplace, a precision change
revalidates rounded parameters and rebuilds density constants on CPU.
Choose precision before accelerator transfer. To change an already resident
proposal's precision, explicitly return it to CPU first.

## Distributions.jl interface

Loading Distributions.jl activates native conversion for `Uniform`, `Gamma`, `Laplace`,
`Normal`, conventional `MvNormal` forms, `TDist`, `Cauchy`, and supported
multivariate Student-t forms. Uniform remains fixed. Gamma and Laplace also work
with the AMIS/N-PMC marginal moment fits described above.

```julia
using Distributions, ImportanceSamplers

proposal = MvNormal(zeros(3), [1.0 0.6 0.0; 0.6 2.0 0.2; 0.0 0.2 1.0])
algorithm = AMIS(proposal; rounds=4, round_size=2_000)
```

Other normalized distributions remain generic CPU proposals.
They do not gain adaptive or GPU support merely by loading the extension.

A mixture distribution remains one atomic proposal.
To expose its components to MIS, construct
`ProposalBank(components(mixture), probs(mixture))` explicitly.

## Custom CPU proposals

Implement these two operations for the same normalized measure:

```julia
Random.rand(rng, proposal)
DensityInterface.logdensityof(proposal, sample)
```

A draw may be a scalar, vector, or named tuple of scalar/vector leaves.
Every draw must keep the same structure, numeric types, and dimensions.

See [Custom proposals and transitions](@ref extension-guide) for a complete example.
Use [transforms](@ref transforms-guide) for constrained support.
