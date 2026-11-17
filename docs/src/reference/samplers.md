# [Samplers and transitions](@id sampler-reference)

See [Method comparison](@ref choosing-method) for adaptation, weighting, and allocation requirements.

## Algorithm configurations

```@docs
AbstractImportanceSampler
ImportanceSampling
AMIS
NPMC
DeterministicMixturePMC
APIS
CAIS
LAIS
HAIS
FirstOrderGRAMIS
current_rng
```

## Static MIS schemes

```@docs
AbstractMISScheme
StratifiedMixture
RandomMixture
StandardMIS
PartialDeterministicMixture
```

## Population resampling

```@docs
AbstractPMCResamplingPolicy
GlobalResampling
LocalResampling
```

## LAIS transitions and tuning

```@docs
AbstractMCMCTransition
RandomWalkMetropolis
HamiltonianMonteCarlo
RAM
WarmupTuning
ContinuousTuning
SampleMetropolisHastings
```

## Factor execution

```@docs
FusedFactorExecution
BatchedFactorExecution
```
