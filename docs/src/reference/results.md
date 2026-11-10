# [Results and statistics](@id result-reference)

See [Weighted samples](@ref results-guide) for weighted estimates, indexing, and device rules.
See [Pareto smoothing](@ref psis-guide) for a raw-IS comparison and tail diagnostics.

## Containers and weights

```@docs
WeightedSamples
WeightedSampleView
ParetoSmoothedSamples
UnweightedSamples
normalized_weights
lognormalizer
psis
```

## Statistics

```@docs
Statistics.mean
Statistics.var
Statistics.std
Statistics.cov
Statistics.quantile
Statistics.median
```

## Control variates

```@docs
SteinControlVariate
fit_control_variate
control_variate_mean
```

## Resampling

```@docs
AbstractResamplingMethod
MultinomialResampling
resample
```
