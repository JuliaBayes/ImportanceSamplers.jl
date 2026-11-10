using Documenter
using DocumenterCodeBlocks
using ImportanceSamplers
using Statistics

include("source_links.jl")

# Documenter discovers the shared logo in its source assets directory.
mkpath(joinpath(@__DIR__, "src", "assets"))
cp(joinpath(@__DIR__, "..", "assets", "logo.svg"),
   joinpath(@__DIR__, "src", "assets", "logo.svg"); force=true)

makedocs(
    root=@__DIR__,
    repo=Documenter.Remotes.GitHub("JuliaBayes", "ImportanceSamplers.jl"),
    modules=[ImportanceSamplers],
    sitename="ImportanceSamplers.jl",
    format=Documenter.HTML(
        prettyurls=true,
        canonical="https://juliabayes.org/ImportanceSamplers.jl/stable/",
        edit_link=:commit,
        repolink="https://github.com/JuliaBayes/ImportanceSamplers.jl",
        assets=["assets/logo.css"],
    ),
    build=joinpath(@__DIR__, "build"),
    pages=[
        "Getting started" => "index.md",
        "Tutorials" => [
            "Weighted estimation" => "tutorials/first_estimate.md",
            "Logistic regression" => "tutorials/logistic_regression.md",
            "Numerical integration" => "tutorials/integration.md",
        ],
        "User guide" => [
            "Targets and data" => "guide/targets.md",
            "Proposals" => "guide/native_proposals.md",
            "Constraints and named parameters" => "guide/transforms.md",
            "Weighted samples" => "guide/results.md",
            "Pareto smoothing" => "guide/psis.md",
            "Fitted transport" => "guide/transport.md",
            "Score control variates" => "guide/control_variates.md",
            "Adaptation and reuse" => "guide/reuse.md",
        ],
        "Methods" => [
            "Method comparison" => "methods/index.md",
            "Plain IS" => "methods/importance_sampling.md",
            "Static MIS" => "methods/static_mis.md",
            "AMIS" => "methods/amis.md",
            "N-PMC" => "methods/npmc.md",
            "DM-PMC, GR-PMC, LR-PMC" => "methods/dm_pmc.md",
            "APIS" => "methods/apis.md",
            "CAIS" => "methods/cais.md",
            "LAIS" => "methods/lais.md",
            "First-order GRAMIS-CAIS" => "methods/first_order_gramis.md",
        ],
        "Execution" => [
            "Batch targets" => "guide/batch_targets.md",
            "Gradients" => "guide/gradients.md",
            "Devices" => "guide/accelerators.md",
            "Performance" => "guide/performance.md",
        ],
        "Reference" => [
            "Targets and execution" => "reference.md",
            "Samplers and transitions" => "reference/samplers.md",
            "Proposals and transforms" => "reference/proposals.md",
            "Results and statistics" => "reference/results.md",
            "Errors" => "reference/errors.md",
        ],
        "Further information" => [
            "Changelog" => "changelog.md",
            "Custom proposals and transitions" => "guide/extensions.md",
            "Troubleshooting" => "guide/troubleshooting.md",
            "Validation" => "guide/validation.md",
            "Benchmarks" => "guide/benchmarks.md",
        ],
    ],
    doctest=true,
    plugins=[CodeBlocks()],
    checkdocs=:exports,
    linkcheck=get(ENV, "DOCUMENTER_LINKCHECK", "false") == "true",
    linkcheck_ignore=[
        r"^https://github\.com/JuliaBayes/ImportanceSamplers\.jl/blob/[0-9a-f]{40}/",
    ],
    warnonly=false,
)
