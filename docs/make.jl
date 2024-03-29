using Cairn
using Documenter

DocMeta.setdocmeta!(Cairn, :DocTestSetup, :(using Cairn); recursive=true)

makedocs(;
    modules=[Cairn],
    authors="Joanna Zou, Spencer Wyant, and contributors",
    repo="https://github.com/cesmix-mit/Cairn.jl/blob/{commit}{path}#{line}",
    sitename="Cairn.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cesmix-mit.github.io/Cairn.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cesmix-mit/Cairn.jl",
    devbranch="main",
    push_preview = true,
)
