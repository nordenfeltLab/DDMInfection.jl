using DDMInfection
using Documenter

DocMeta.setdocmeta!(DDMInfection, :DocTestSetup, :(using DDMInfection); recursive=true)

makedocs(;
    modules=[DDMInfection],
    authors="Oscar Andre <bmp13oan@student.lu.se> and contributors",
    repo="https://github.com/nordenfeltLab/DDMTransfection.jl/blob/{commit}{path}#{line}",
    sitename="DDMInfection.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nordenfeltLab.github.io/DDMInfection.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Microscopes" => ["microscopes/nikon.md"]
    ],
)

deploydocs(;
    repo="github.com/nordenfeltLab/DDMInfection.jl",
    devbranch="main",
)
