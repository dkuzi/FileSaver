using Documenter
using Literate

makedocs(;
    modules=[AVI],
    sitename="AVI.jl",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true", collapselevel=1),
    pages=[
        "Home" => "index.md",
        "How does it work?" => "basics.md",
        "Oracles" => ["reference/2_oracles.md", "basics.md"],
        "API Reference" => ["reference/0_reference.md", "reference/1_algorithms.md", "reference/2_oracles.md"]
    ])

deploydocs(; repo="https://github.com/Pustey/FileSaver.git", push_preview=true);