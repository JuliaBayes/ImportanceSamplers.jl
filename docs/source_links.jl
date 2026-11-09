# Keep source links readable on GitHub, but pin rendered links to the docs revision.
abstract type VersionedSourceLinks <: Documenter.Builder.DocumentPipeline end

Documenter.Selectors.order(::Type{VersionedSourceLinks}) = 2.5
Documenter.Selectors.matcher(::Type{VersionedSourceLinks}, doc::Documenter.Document) = true

function Documenter.Selectors.runner(::Type{VersionedSourceLinks}, doc::Documenter.Document)
    prefix = "https://github.com/JuliaBayes/ImportanceSamplers.jl/blob/main/"
    for page in values(doc.blueprint.pages),
        node in Documenter.AbstractTrees.PreOrderDFS(page.mdast)

        link = node.element
        link isa Documenter.MarkdownAST.Link || continue
        startswith(link.destination, prefix) || continue
        path, fragment = Documenter.splitfragment(chopprefix(link.destination, prefix))
        source = joinpath(doc.user.root, "..", path)
        isfile(source) || error("Missing repository link: $(link.destination)")
        url = Documenter.edit_url(doc, source; rev=nothing)
        link.destination = isempty(fragment) ? url : "$url#$fragment"
    end
    return nothing
end
