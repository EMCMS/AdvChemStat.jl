#using Base
using Documenter, AdvChemStat

makedocs(modules = [AdvChemStat],
    build = "build",
    clean = true,
    sitename="AdvChemStat.jl",
    pages = [
        "Home" => "index.md",
        "SVD" => "svd.md",
        "HCA" => "HCA.md"
        ]
    )

deploydocs(
        repo = "github.com/EMCMS/AdvChemStat.jl.git",
        target = "build",
        branch = "gh-pages",
        #push_preview = true,
)


# include("/Users/saersamanipour/Desktop/dev/pkgs/DataSci4Chem.jl/docs/make.jl") 