# [Benchmarks](@id benchmarks-guide)

The repository contains model comparisons, analytic accuracy checks, and per-method performance scripts.
Benchmark dependencies remain separate from package and documentation dependencies.

## Model comparison

Use Julia 1.13 or later. Run the comparison from the repository root:

```sh
julia --project=benchmark/comparison -e 'using Pkg; Pkg.instantiate()'
julia --threads=16 --project=benchmark/comparison benchmark/comparison/compare.jl --cuda
```

The runner prints tables and records the environment, source hashes, hardware, timings, and diagnostics.
It refuses to overwrite an existing result.
See the [comparison guide](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/benchmark/comparison/README.md)
for CPU-only runs, selected methods, pilot settings, and resumption.

To print an existing table without sampling:

```sh
julia --project=benchmark/comparison benchmark/comparison/compare.jl --report benchmark/comparison/results-2026-09-16-comparison.toml
```

## Models and samplers

| Model | Parameters | Data |
|:--|--:|:--|
| Linear regression | 32 | 1,024 observations |
| Logistic regression | 12 | 1,024 observations |
| Poisson regression | 12 | 1,024 observations |
| Robust regression | 13 | 512 observations |
| Eight schools | 10 | Eight groups |
| Signal-background | 9 | 37 events across five detectors |

The comparison includes fixed and adaptive IS, AdvancedHMC NUTS, AdvancedMH,
SliceSampling, and EnsembleMCMC.
The [model source](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/benchmark/comparison/models.jl)
defines data, coordinates, priors, and target functions.

The [combined report](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/benchmark/comparison/results-2026-09-22-refresh-comparison.md)
retains timing ranges and accuracy warnings.
Some rows reuse archived measurements. They are not simultaneous measurements and cannot establish a speedup by themselves.

## Effective sample size per second

| Sampler class | Reported ESS |
|:--|:--|
| Importance sampling | Weight concentration, ``1/\sum_i\bar w_i^2`` |
| Independent MCMC chains | Minimum rank-normalized bulk ESS across coordinates |
| EnsembleMCMC | Minimum coordinate-mean ESS from within-sweep walker averages |

These are different diagnostics, not a common accuracy score.
Weight ESS does not detect missed modes or tails.
Ensemble walkers are not treated as independent chains.

CPU and GPU tables remain separate.
Pilot and adaptation costs belong in end-to-end timing.
Accuracy checks and ESS calculations occur outside the timed interval.

Use the full reports to inspect moment errors, divergences, R-hat, and per-seed variation.
A high ESS/s cell does not override a failed accuracy check.

## [Recorded results](@id benchmark-results)

Effective sample size per second (ESS/s)\* for six posterior models. Parentheses
give the number of parameters. Measurements use Float64 and include initialization,
warmup or adaptation, sampling, and posterior-mean estimation.
Each table lists importance samplers first, followed by other measured methods.

### CPU

| Sampler | Linear (32) | Logistic (12) | Poisson (12) | Robust (13) | Eight schools (10) | Signal-background (9) |
|:--|--:|--:|--:|--:|--:|--:|
| IS | **5.1e+04** | 7.4e+04 | 1.8e+05 | 1.6e+05 | 2.9e+04§ | 7.3e+04 |
| AMIS | 4.9e+04 | **8.6e+04** | 1.8e+05 | **1.9e+05** | **7.6e+05** | **2.3e+05** |
| DM-PMC | 49¶ | 7.7e+03 | 1.6e+04 | 1.4e+04 | 4.8e+04 | 6e+04 |
| CAIS | 2e+04 | 6.7e+04 | 1.5e+05 | 1.4e+05 | 1.8e+05§ | 1.8e+04§¶ |
| LAIS-RAM | 12¶ | 4.9e+03 | 1.2e+04 | 8.3e+03 | 7e+04 | 6.8e+04 |
| First-order GRAMIS-CAIS | 4.4e+04 | 8.3e+04 | **1.9e+05** | 1.6e+05 | 1.9e+04 | 9.7e+04 |
| AdvancedHMC NUTS | 1.5e+04 | 9e+03 | 2.1e+04 | 5e+04 | 1.3e+05† | 5.2e+04 |
| AdvancedMH RWMH | 8.5e+02‡ | 2.6e+03 | 5.2e+03 | 4.5e+03 | 1.8e+04 | 1.4e+04 |
| SliceSampling | 7.7e+02 | 1.9e+03 | 5.6e+03 | 4.4e+03 | 1.6e+05 | 1.5e+04 |
| EnsembleMCMC (selected) | 3e+04 | 2.4e+04 | 4.9e+04 | 5e+04 | 1e+04 | 3.5e+03‡ |

### GPU (CUDA)

Only importance samplers have GPU measurements in this comparison.

| Sampler | Linear (32) | Logistic (12) | Poisson (12) | Robust (13) | Eight schools (10) | Signal-background (9) |
|:--|--:|--:|--:|--:|--:|--:|
| IS | 5.4e+05 | 2.6e+06 | 3.3e+06 | 3.7e+06 | 4.8e+04 | 7.3e+05 |
| AMIS | 3.5e+05 | 9.4e+05 | 1.4e+06 | 1.3e+06 | 1.6e+06 | 2.1e+06 |
| DM-PMC | 1.8e+02¶ | 9.2e+04 | 2.2e+05 | 1.8e+05 | 3.3e+05 | 3.4e+05 |
| CAIS | 1.9e+05 | 2.1e+06 | 2.4e+06 | 2.9e+06 | 3.5e+05 | 1.7e+06 |
| LAIS-RAM | 4.8¶ | 7.5e+03 | 1.2e+04 | 1.4e+04 | 2.3e+05 | 6.2e+04¶ |
| First-order GRAMIS-CAIS | 3.6e+05 | 1.4e+06 | 2e+06 | 2.1e+06 | 1.5e+05 | 4.1e+05 |
| IS, AMIS-fitted‖ | 3.5e+05 | — | — | 1.6e+06 | — | — |
| LAIS-RWM, AMIS-fitted‖ | 2.7e+05 | — | — | 1.1e+06 | — | — |

Bold denotes the highest measured CPU ESS/s per model. GPU results appear separately.

The EnsembleMCMC baseline uses the highest screening ESS/s among accuracy-eligible
candidates: DE for eight schools and Gaussian replacement with shrinkage 1 for
the other models. The table reports held-out results. All four moves remain
in the detailed report.

\* Importance sampling uses weight ESS. Independent-chain MCMC uses minimum
bulk ESS. EnsembleMCMC uses minimum mean ESS from the sweep-average process.
These diagnostics do not define an equal-accuracy comparison.

† At least one retained NUTS transition diverged.

‡ At least one run had maximum R-hat above 1.01. Ensemble R-hat splits the
sweep-average time series, not the walkers.

§ Weight ESS varied by more than a factor of ten across seeds.

¶ At least one run exceeded a mean error of 0.2 posterior standard deviations
or a marginal variance error of 30%. Archived rows lack variance checks.

‖ Independent 262,144-draw AMIS pilot, then 262,144 production draws, with both
costs included. Gaussian LAIS-RWM uses 16 proposals, four rounds, and no extra
MCMC warmup. Two seeds per model, 128 host threads, and the shared A100.
Dashes denote unmeasured cases. [Protocol, results, and reproducer](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/benchmark/comparison/lais.md).

The main comparison uses three seeds, 16 threads on a 128-thread EPYC CPU, and one A100.
Independent-chain MCMC retains 16,384 draws per chain in 16 chains. Importance
samplers retain 262,144 draws. Ensemble methods use 4d walkers and retain
16,384 sweeps per walker. Model-specific move widths come from a separate
three-seed screen, then remain fixed for three held-out seeds. Linear starts
at stationarity without warmup; other models use 1,024 warmup sweeps.
DM-PMC and LAIS include an independent 4,096-draw width pilot.
Compilation and post-run diagnostics are excluded. Unchanged rows reuse archived
measurements. The report records each row's source and timing protocol.
The September 22 EnsembleMCMC refresh used a shared host. Other rows retain
their original measurement windows. These timings do not form a paired comparison.

Signal-background uses the nine-parameter BAT paper model and its original
37-event dataset. Its CAIS CPU and LAIS-RAM CUDA runs include failed moment
checks, marked above. The detailed report also retains ensemble snooker's
and DE's failed checks. Offline settings search is not timed.

The linear posterior is Gaussian, so a fitted Gaussian proposal leaves little
room for adaptation to improve weight ESS. Follow-up CUDA runs used an independent
AMIS fit and short Gaussian LAIS-RWM runs without extra MCMC warmup. Throughput
improved, but matched static IS remained faster on linear and robust regression.
Longer LAIS runs degraded the fitted proposal. These findings concern a different
configuration from the LAIS-RAM rows above, which remain for comparison.

[Reproducer and recorded environment](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/benchmark/comparison/README.md) ·
[Timings, ranges, and accuracy checks](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/benchmark/comparison/results-2026-09-22-refresh-comparison.md) ·
[Scalar/batch measurements](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/benchmark/comparison/README.md#scalar-and-batch-regression-targets) ·
[Models and methodology](@ref benchmarks-guide)

## Scalar and batch evaluation

The paired [CPU report](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/benchmark/comparison/batch-cpu-2026-09-28-paired.md)
and [CUDA report](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/benchmark/comparison/batch-cuda-2026-09-28-paired.md)
compare scalar and batch evaluation.

Shared machines require interleaved comparisons.
Separate collection windows cannot establish a regression or speedup by themselves.

The [batch API guide](@ref batch-guide) explains the callback contract.
The comparison README gives the paired-run commands.

## Signal-background coverage

The model-specific conditional proposal has a separate runner:

```sh
julia --threads=16 --project=benchmark/comparison benchmark/comparison/signal_background_conditional.jl --cuda conditional-results.toml
julia --project=benchmark/comparison benchmark/comparison/signal_background_conditional.jl --report conditional-results.toml
```

It fits a conditional proposal from a CPU pilot, then compares independent production IS with LAIS.
It is benchmark-local code, not a new public sampler.

Its weight ESS is not tail-functional ESS.
The archived moment reference and the independent coordinate-specific tail checks answer different questions.
Passing those moment checks does not establish coverage of every tail functional.

## Additional benchmarks

- [`benchmark/time_to_accuracy.jl`](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/benchmark/time_to_accuracy.jl)
  compares normalizer, mean, and second-moment errors against independent references.
- [`benchmark/plain_is.jl`](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/benchmark/plain_is.jl)
  measures preparation, warmed execution, and host allocations.
- [`benchmark/logistic_gramis.jl`](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/benchmark/logistic_gramis.jl)
  separates preparation, first-call cost, warmed timing, and profiles.

Record the exact source and environment rather than copying a published timing to another machine.
