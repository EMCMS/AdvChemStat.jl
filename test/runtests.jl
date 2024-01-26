using AdvChemStat
using Test
@sk_import decomposition: PCA

@testset "AdvChemStat.jl" begin
    # Write your tests here.

    iris = dataset("datasets", "iris")
    #println(iris)

    X = Matrix(iris[:,1:3])
    s = iris[!,"Species"]
    #pca = sk_pca(n_components=2)
    pca = PCA(n_components=2)
    pca.fit(X)
    #println(pca.explained_variance_ratio_)

    @test pca.explained_variance_ratio_[1] > 0.9
end
