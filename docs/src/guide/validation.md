# [Validation and support](@id validation-guide)

The manual executes its CPU teaching examples during the build.
The package test suite and standalone reproducers check mathematical and device-specific contracts.

A successful example is not proof of accuracy for every model.
A checked-in GPU script is not evidence that it passed on the current machine.

## Executable support table

This table executes each example during the build. Every row checks 256 retained
samples, finite values, sample and log-weight precision, and normalized weights.
Adaptive examples use two rounds. The scalar Gaussian row also checks the exact
self-target weight identity. These are execution checks, not accuracy benchmarks.

```@eval
import Markdown
include(joinpath(@__DIR__, "..", "..", "..", "validation", "support_table.jl"))
Markdown.parse(SamplerSupportTable.checked_support_table())
```

The table covers only the stated examples. CUDA, Metal, and Reactant are **not
validated by this CPU build**. A missing hardware run means unvalidated, not unsupported.
`passed` denotes execution in this build. `not run` denotes no result from this build.
`restricted` denotes a documented preflight restriction, not a hardware test.
An optional-backend run reports `rejected` only after checking the expected exception reason.
[Devices](@ref devices-guide) lists deliberate restrictions, including
Reactant CPU cooperative kernels and generic accelerator proposals.

Use Julia 1.13 or later. From the repository root, prepare the validation environment:

```sh
julia --project=validation -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
```

Run the same checks on CUDA with `julia --project=validation`:

```julia
using CUDA, MLDataDevices
include("validation/support_table.jl")
CUDA.allowscalar(false)
device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)
print(SamplerSupportTable.checked_support_table(
    backend=:cuda, device, isresident=x -> x isa CUDA.AnyCuArray,
))
```

The checker verifies residency before explicit CPU readback. Any failed check
stops table generation. Optional-backend checks below cover deeper numerical,
reuse, statistics, and failure contracts.

## Package tests

From a checkout:

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

CPU CI covers Linux, macOS, and Windows.
The CI configuration defines the tested Julia environments.

Accelerator checks are separate, explicit hardware runs.
They do not run on the default hosted CPU CI jobs.

## Equation and backend reproducers

Use the prepared validation environment:

```sh
julia --project=validation validation/reproducers/plain_is.jl
```

| Subject | Script under `validation/reproducers/` |
|:--|:--|
| Plain IS identities | `plain_is.jl` |
| Native Gaussian formulas | `native_gaussian.jl` |
| Simplex measure and Jacobian | `simplex_transform.jl` |
| Static MIS schemes | `static_mis.jl` |
| AMIS recurrence | `amis.jl` |
| DM-PMC recurrence | `dm_pmc_global.jl` |
| First-order GRAMIS equations | `first_order_gramis.jl` |
| CUDA samplers | `cuda_plain_is.jl`, `cuda_static_mis.jl`, `cuda_amis.jl`, `cuda_dm_pmc.jl` |
| CUDA population methods | `cuda_apis.jl`, `cuda_cais.jl`, `cuda_npmc.jl`, `cuda_lais.jl` |
| CUDA HMC and HAIS | `cuda_hamiltonian.jl`, `cuda_lower_widths.jl` |
| Cold CUDA HMC | `cuda_hmc_cold.jl` (separate fresh process) |
| Batched gradients and transformed scores | `cuda_batch_gradients.jl` |
| CUDA GRAMIS | `cuda_first_order_gramis.jl` |
| Student-t devices | `cuda_student_t.jl`, `metal_student_t.jl` |
| Batch callbacks | `batch_targets.jl` |
| Fitted quadratic transport | `fitted_transport.jl` |
| Reactant execution | `reactant_execution.jl` |

Read each script's setup before running optional-backend cases.
Run `cuda_hmc_cold.jl` in a fresh Julia process before other CUDA samplers.
Earlier kernel launches can increase the device stack and hide cold-start failures.
Metal and Reactant checks need their respective packages and hardware.
CPU LAIS recurrence checks are in
[`test/lais.jl`](https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/test/lais.jl).

The CUDA GRAMIS wrapper additionally records a supplied source revision:

```sh
git status --short
git rev-parse HEAD
IMPORTANCE_SAMPLERS_SOURCE_COMMIT=PASTE_FULL_COMMIT_HERE julia --project=validation validation/reproducers/cuda_first_order_gramis.jl
```

Inspect the checkout and loaded package path first.
The environment variable records a claim. It does not verify the loaded source.

## Documentation build

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Serve `docs/build` with an HTTP server to preview its directory-style URLs.
The build executes CPU examples and doctests and checks exported docstrings and internal links.
Check external links separately with `DOCUMENTER_LINKCHECK=true julia --project=docs docs/make.jl`.
CI reports external-link failures without blocking deployment.

Repository source links are checked against local files to avoid repeated GitHub rate limits.
Rendered source links use the build's Git commit. Local checks do not verify remote availability.
CI deploys development documentation from `dev` and versioned documentation from release tags.

## Validation records

Record the commit and tracked diff, `Project.toml`, `Manifest.toml`, command, seed,
precision, counts, threads, device, and complete result.

For GPU checks, disable scalar indexing and verify array residency.
Use a profiler to establish transfer or kernel claims.
Source-level transfer counters cannot observe every runtime or vendor-library transfer.

For performance, also record compilation boundaries and machine load.
[Benchmarks](@ref benchmarks-guide) separates accuracy, timing, and ESS diagnostics.
