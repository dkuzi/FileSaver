push!(LOAD_PATH,"../src/")
using Documenter
using AVI

# makedocs(
#     sitename="AVI.jl",
#     modules=[AVI],
#     format=Documenter.HTML(repolink="https://github.com/ZIB-IOL/AVI.jl.git",
#                             prettyurls = get(ENV, "CI", nothing) == "true",
#     ),
#     pages=[
#         "Home" => "index.md",
#         "How does it work?" => "basics.md",
#         "Obtaining a transformation" => [
#                                         "how_to_run.md",
#                                         "docs_oavi_transformation.md",
#                                         "docs_custom_oracle.md",
#                                         "docs_vca_transformation.md"
#                                         ],
#         "Printing polynomials" => "print_polynomials.md",
#         "API Reference" => ["reference/0_reference.md",
#                             "reference/1_algorithms.md",
#                             "reference/2_oracles.md",
#                             "reference/3_border.md",
#                             "reference/4_terms_and_polys.md",
#                             "reference/5_auxiliary_funcs.md"
#                             ]
#     ],
#     warnonly=true
# )


# deploydocs(
#             repo="github.com/ZIB-IOL/AVI.jl.git",
#             devbranch="main",
#             push_preview=true
# )

makedocs(
    sitename="AVI.jl",
    modules=[AVI],
    format=Documenter.HTML(repolink="https://github.com/dkuzi/FileSaver.git",
                            prettyurls = get(ENV, "CI", nothing) == "true",
                            edit_link="main",
    ),
    pages=[
        "Home" => "index.md",
        "How does it work?" => "basics.md",
        "Obtaining a transformation" => [
                                        "how_to_run.md",
                                        "docs_oavi_transformation.md",
                                        "docs_custom_oracle.md",
                                        "docs_vca_transformation.md"
                                        ],
        "Printing polynomials" => "print_polynomials.md",
        "API Reference" => ["reference/0_reference.md",
                            "reference/1_algorithms.md",
                            "reference/2_oracles.md",
                            "reference/3_border.md",
                            "reference/4_terms_and_polys.md",
                            "reference/5_auxiliary_funcs.md"
                            ]
    ],
    warnonly=true
)


deploydocs(
            repo="github.com/dkuzi/FileSaver.git",
            devbranch="main",
            push_preview=true,
            target="build",

)