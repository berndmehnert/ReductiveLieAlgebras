using ReductiveLieAlgebras
using LinearAlgebra
using Test
include("../src/ExampleLieAlgebras.jl")

@testset "Test definitions and basic functions" begin
    Pmat = P(diagm([1.0, 1, 3, 2, 2, 2, 5, 4, 4, 5]))
    B1 = 𝔨(10, 10)
    B2 = 𝔭(10, 10)
    A, _ = getFixSpace(B1, B2, Pmat)
    nsp, dimnsp, range, rank =
        computeLinearFunctionData(B1, B2, Y -> ad(Y, Pmat), false, 1e-13)
    B3 = [
        reshape([1.0, 0, 0, 0], 4, 1),
        reshape([0, 1.0, 0, 0], 4, 1),
        reshape([0, 0, 1.0, 0], 4, 1),
        reshape([0, 0, 0, 1.0], 4, 1),
    ]
    B4 = [reshape([1.0, 0], 2, 1), reshape([0, 1.0], 2, 1)]
    C = [1 2.0 0 0; 0 0 0 0]
    nsp2, dimnsp2, range2, rank2 =
        computeLinearFunctionData(B3, B4, Y -> C * Y, false, 1e-13)
    B5 = [
        [0.0 1.0; -1.0 0.0] / sqrt(2),
        [1.0 0.0; 0.0 0.0],
        [0.0 0.0; 0.0 1.0],
        [0.0 1.0; 1.0 0.0] / sqrt(2),
    ]
    B51 = [
        [1.0 1.7; 1.0 1.0] / sqrt(1 + 1.7^2 + 1 + 1),
        [1.0 1.0/1.7; -0.5 -1.5] / sqrt(1 + 1.7^(-2) + 0.5^2 + 1.5^2),
    ]
    M = [1.0 2; 3 4]
    v = [B5[1] ⋅ M, B5[2] ⋅ M, B5[3] ⋅ M, B5[4] ⋅ M]
    f = projectionOntoSubspace(B5, B51)
    w = rand(2, 2)
    Pw = f(w)
    PPw = f(Pw)
    x = w - Pw
    @test ReductiveLieAlgebras.getElement(B2, ReductiveLieAlgebras.getVector(B2, Pmat)) ≈ Pmat rtol = 1e-14
    @test ad(A[1], Pmat) ≈ zeros(20, 20) atol = 1e-14
    @test length(A) == 6
    @test dimnsp + rank == length(B1)
    @test rank2 == 1
    @test dimnsp2 == 3
    @test ReductiveLieAlgebras.getVector(B5, M) ≈ v atol = 1e-14
    @test ReductiveLieAlgebras.getElement(B5, v) ≈ M atol = 1e-14
    @test w ⋅ w ≈ Pw ⋅ Pw + x ⋅ x atol = 1e-14
    @test PPw ≈ Pw atol = 1e-14
end

@testset "some more tests" begin
    A = [[1.0 2.0 0]/3, [0 1.1 0]/1.1]
    B = [[0 5.0 0]/5, [0 1.4 6.7]/sqrt(1.4^2 + 6.7^2)]
    S = [[1.0 0 0], [0 1.0 0], [0 0 1.0]]
    C = transpose(B[1])
    DSM = DirectSumMat(A[1],C)
    pr1 = x -> x - projectionOntoSubspace(S, A)(x)
    pr2 = x -> x - projectionOntoSubspace(S, B)(x)
    @test computeKernel(S, ⊕(S,S), X -> DirectSumMat(pr1(X), pr2(X)))[2] == 1
    @test ReductiveLieAlgebras.firstArg(1, 3, DSM) ≈ A[1] atol = 1e-14
    @test ReductiveLieAlgebras.lastArg(3, 1, DSM) ≈ C atol = 1e-14
end

@testset "rootspaces etc." begin
    B = [[1.0 0;0 0.0], [0.0 1.0;0.0 0.0], [0.0 0.0;1.0 0], [0 0;0.0 1.0]]
    A = [[1.0 0;0 0.0], [0 0;0.0 1.0]]
    decomposition = computeRootspaceDecomposition(A, B)
    for rs in decomposition
        λ = getRootFunction(A, rs.root)
        H = ReductiveLieAlgebras.getElement(A, rand(2))
        X = rs.basis[1]
        @test ad(H, X) ≈ λ(H)*X atol = 1e-14
    end
    rsp = computeRootspaceDecomposition(a(3), gl(3))
    Y = [1.0 0.0 0.0;0.0 0.5 0.0;0.0 0.0 -1.0]
    pos_root_spaces = getPositiveRootSpaces(rsp, a(3), Y)
    @test length(pos_root_spaces) == 3
end

@testset "further basic tests" begin
    R = ReductiveLieAlgebra(gl(2))
    V = [[1.0 2.0;2.0 0.0]/3, [0.0 0; 0 1.0]]
    W = [[0.0 2.0;2.0 0.0]/2, [0.0 0; 0 1.0]]
    X, dimX = vectorSpaceIntersection(gl(2), V, W)
    @test length(R.g) == length(R.p) + length(R.k)
    @test Φ(R.p[1]) ≈ -R.p[1] atol = 1e-14
    @test Φ(R.k[1]) ≈ R.k[1] atol = 1e-14
    @test dimX == 1
end