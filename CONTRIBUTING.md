# Contributing

Keep sampler changes narrow. Preserve the sampling law, log weights, precision,
sample structure, random streams, and device residency. Use analytic oracles for
numerical changes and interleaved BenchmarkTools runs for performance claims.

## Branches

Use `dev` for integration and `main` for tested releases.

- Target feature and fix pull requests at `dev`.
- Promote changes to `main` only through a pull request from `dev`.
- Keep `dev` after a promotion.
- Do not push directly to `main`.

The source-branch rule is a project convention, not an automated merge check.

CI builds and checks documentation on both branches. Development documentation
deploys from `dev`. Tagged releases supply stable and versioned documentation.

## Tests

From the repository root:

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

CI tests Julia 1.13 on Linux and the latest stable Julia on Linux, macOS, and Windows.
It also runs the Reactant CPU checks. Codecov upload failures do not block CI.
GPU execution requires separate hardware checks; passing CPU CI does not prove
GPU support. See the [validation guide](https://juliabayes.org/ImportanceSamplers.jl/dev/guide/validation/).

## Documentation

Build the complete site, examples, and doctests:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Serve `docs/build` with an HTTP server to preview the directory-style URLs.
The build is strict: broken internal links, undocumented exports, and failing
examples or doctests fail it. External links have a separate, non-gating CI check.

Write runnable examples before describing options. Define each term before
using it. Keep guides separate from the API reference. `DocumenterCodeBlocks`
provides the code highlighting and reference links used by Wren and EnsembleMCMC.

### Versioned deployment

`docs/make.jl` only builds the site. `docs/deploy.jl` retains versions on
`gh-pages`. After tests and docs pass, CI publishes that full branch through
GitHub Pages Actions:

| Source | Documentation |
| --- | --- |
| `dev` | `/dev/` |
| A release tag such as `v0.1.0` | `/v0.1.0/` |
| Latest release | `/stable/` |

The root URL redirects to the latest release, or to `dev` before any release.
Links between manual pages stay within the selected version. Repository source
links use the build's Git commit, including links to examples and reproducers.
Pull requests build an artifact but do not deploy. Keep Pages configured to use
GitHub Actions. Only the deployment job receives repository write permission.

TagBot creates tags and GitHub releases after registration. Its workflow
explicitly dispatches `CI.yml` for release tags without archived documentation,
because tags created with `GITHUB_TOKEN` do not trigger push workflows.
No SSH deploy key is required. A manually pushed release tag also triggers CI.
See [Documenter's hosting guide](https://documenter.juliadocs.org/stable/man/hosting/).

## Releases

Set `Project.toml` to the intended release version and merge after CI passes.
Request registration with `@JuliaRegistrator register` on the release commit.
After registration merges, TagBot creates the tag and GitHub release; CI then
publishes its versioned documentation. Verify the release and `/stable/` site.
The Registrator GitHub App must have access to this repository.

Dependabot groups GitHub Actions updates into a weekly pull request.
