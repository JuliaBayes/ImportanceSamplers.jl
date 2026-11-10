# [Method comparison](@id choosing-method)

The initial proposal must cover the target.
Adaptation can improve it but cannot recover regions that the run never visits.

## Choosing a method

The proposal and available model operations determine a useful starting point.

| Situation | Starting point | Main consideration |
|:--|:--|:--|
| A fixed proposal already covers the target well | Plain IS | Also useful for production after an independent pilot |
| Several fixed proposals cover different regions | Static MIS | Full-mixture weighting evaluates every active density |
| One proposal needs moment fitting | AMIS or N-PMC | AMIS recycles history; N-PMC fits each round using clipped weights |
| Population locations need to move, but widths are already suitable | DM-PMC or APIS | Neither tunes covariance; resampling and local mean fitting behave differently |
| MCMC moves can locate useful population centres | LAIS | Upper moves and tuning add work beyond the returned samples |
| Population widths also need fitting | CAIS | Each proposal needs enough local samples for a viable fit |
| Reliable target gradients are available | FirstOrderGRAMIS | Gradient and backtracking costs must be worthwhile |

AMIS updates earlier denominators, which adds proposal-density work and history storage.
N-PMC leaves earlier weights unchanged. A single fitted proposal may still miss separated modes.
The [DM-PMC example](@ref dmpmc-method) checks an asymmetric two-mode target.

Use an independent fixed-proposal production run when you want the usual independent-draw
[Monte Carlo error analysis](@ref logistic-mc-error).
No method dominates for every target, budget, and estimand.

## Adaptation and weighting

| Method | Adapted state | Weight denominator |
|:--|:--|:--|
| [Plain IS](@ref plain-is) | Nothing | Single proposal |
| [Static MIS](@ref static-mis) | Nothing | Chosen fixed-bank scheme |
| [AMIS](@ref amis-method) | One proposal, by moment matching | Mixture of all generating proposals |
| [N-PMC](@ref npmc-method) | One proposal, by clipped-weight moment matching | Generating proposal |
| [DM-PMC](@ref dmpmc-method) | Population centres, by resampling | Current realized population mixture |
| [GR-PMC](@ref dmpmc-method) | Centres, by global resampling with equal allocation | Current equal population mixture |
| [LR-PMC](@ref dmpmc-method) | Centres, by local resampling with equal allocation | Current equal population mixture |
| [APIS](@ref apis-method) | Population means | Current equal population mixture |
| [CAIS](@ref cais-method) | Population means and covariances | Generating proposal |
| [LAIS](@ref lais-method) | Population centres, using upper MCMC | Current equal population mixture |
| [HAIS](@ref hais-method) | Population centres, using HMC and cooperation | Current equal population mixture |
| [First-order GRAMIS-CAIS](@ref gramis-method) | Gradient-driven means and local covariances | Current realized population mixture |

GR-PMC and LR-PMC use `DeterministicMixturePMC` with different resampling policies.
FirstOrderGRAMIS is a first-order hybrid, not the Hessian-based GRAMIS paper algorithm.

## Preparation and execution

```julia
prepared = prepare_sampler(rng, logtarget, p, algorithm)
samples = importance_sample!(prepared)
```

For a single run, `importance_sample(rng, logtarget, p, algorithm)` combines both steps.
Omit `p` for a context-free target.

Adaptive algorithms require `rounds` and `round_size`.
An integer repeats the total round size.
A vector gives one total count per round.
All rounds contribute to the returned result.

Only AMIS retrospectively changes earlier weights.
The other adaptive methods retain each sample's original round weight.

## Allocation requirements

| Method | Per-round allocation requirement |
|:--|:--|
| AMIS, N-PMC | One proposal with a positive round size and a viable moment fit |
| DM-PMC | At least one draw for each active proposal |
| APIS | Equal positive masses, divisible round size, at least two draws per proposal |
| LAIS | Equal positive masses, divisible round size, at least one draw per proposal |
| HAIS | Equal positive masses, divisible round size, at least one draw per proposal |
| CAIS | Equal positive masses, divisible round size, at least `d + 2` draws per proposal |
| FirstOrderGRAMIS | At least two distinct initial means, equal positive masses, at least `d + 2` draws per proposal |

These are admissibility conditions, not recommended statistical budgets.
Covariance fitting and tail coverage often need substantially more samples.

## Budgets and checks

For `M` equally allocated population proposals, each generates `round_size / M` draws.
Proposal-local fits use only their own draws. A larger population leaves fewer draws
per local fit at a fixed round size.
Choose initial widths that cover each region; more rounds do not tune fixed widths.

The [logistic tutorial](@ref logistic-tutorial) uses 8,000 pilot draws and 20,000
independent production draws. Its separate budgets are illustrative, not precision guarantees.
Time the pilot, inspect its [diagnostics](@ref logistic-diagnostics), then check the
production estimate across independent streams and larger budgets.
A high weight ESS does not establish coverage of missed modes or tails.
Count pilot work, proposal-density evaluations, and LAIS upper moves in total cost.

## Proposal support

All population methods accept native Gaussian or Student-t proposals.
DM-PMC and APIS also accept independent Gamma and Laplace populations.
LAIS also accepts independent Laplace lower proposals.
AMIS and N-PMC support independent Gamma and Laplace moment fits.
[Method compatibility](@ref proposal-method-support) lists proposal families;
[Proposals](@ref proposals-guide) explains layouts and the Student-t covariance convention.
[Devices](@ref devices-guide) lists backend support.
