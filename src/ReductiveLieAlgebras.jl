module ReductiveLieAlgebras
using LinearAlgebra
export bracket, ad, Ad, Φ, Dot, RootSpace, computeRootspaceDecomposition, ⊕, eigenvs, vectorSpaceIntersection, DirectSumMat, computeKernel, 
       computeLinearFunctionData, projectionOntoSubspace, getFixSpace, getRootFunction, getPositiveRootSpaces

       # Lie algebras are given as normalized(!) basis of real n × n matrices!

bracket(A :: AbstractMatrix{Float64}, B :: AbstractMatrix{Float64}) = A*B - B*A
ad(A :: AbstractMatrix{Float64}, B :: AbstractMatrix{Float64}) = bracket(A,B)
Ad(g :: AbstractMatrix{Float64}, X :: AbstractMatrix{Float64}, compact = true) = compact ? g*X*transpose(g) : g*X*g^(-1)
Φ(A :: AbstractMatrix{Float64}) = -transpose(A)
Dot(A :: AbstractMatrix{Float64}, B :: AbstractMatrix{Float64}) = -tr(A*Φ(B))

struct RootSpace 
    root :: Vector{Float64}
    basis :: Array{<:AbstractMatrix{Float64}}
end

getRootFunction(𝔞, v) = H -> getVector(𝔞, H)⋅v

"""
Compute the root-spaces belonging to positive roots.
rsp : the root space decomposition of 𝔞,𝔤
Y : a regular element in 𝔞

As result we get the rootspaces rspace in rsp for which rspace.root(Y) > 0.
"""
getPositiveRootSpaces(Σ :: Vector{RootSpace}, 𝔞 :: Array{<:AbstractMatrix{Float64}}, Y :: Matrix{Float64}) = filter(rspace -> getRootFunction(𝔞, rspace.root)(Y) > 0.0, Σ)

"""
Compute the rootspace decomposition of the pair (𝔞, 𝔤) where 𝔞 is a maximal abelian subalgebra of 𝔭
The result is a list of type RootSpace.
Note that at the moment as RootSpace.root the zero vector is allowed. 
"""
function computeRootspaceDecomposition(𝔞 :: Array{<:AbstractMatrix{Float64}}, 𝔤 :: Array{<:AbstractMatrix{Float64}})
    if length(𝔞) == 0
        return [RootSpace(Float64[], 𝔤)]
    end
    resultDecomposition = RootSpace[]
    A = 𝔞[1]
    EIGS = eigenvs(𝔤, 𝔤, Y -> ad(A, Y))
    for λ in EIGS
        V, _ = eigspace(λ, A, 𝔤)
        subDecomposition :: Array{RootSpace} = computeRootspaceDecomposition(𝔞[2:end], V)
        for rootspace in subDecomposition
            insert!(rootspace.root, 1, λ)
        end
        append!(resultDecomposition, subDecomposition)
    end
    return resultDecomposition
end

function eigenvs(B1, B2, f)
    A = getMatrix(B1, B2, f)
    vals = eigvals(A)
    if length(vals) == 0
        return Float64[]
    end
    result = [vals[1]]
    for i in 2:length(vals)
        if abs(vals[i]-vals[i-1]) > 1e-14
            push!(result, vals[i])
        end
    end
    return result
end
eigspace(x, A, V) = computeKernel(V, V, Y -> ad(A,Y) - x*Y)

"""
Direct sum of Lie algebras by "adding" their "canonical" basis ..
"""
function ⊕(B1, B2)
    R1, C1 = size(B1[1])
    R2, C2 = size(B2[1])
    result = AbstractMatrix{Float64}[]
    for b1 in B1
        push!(result, [b1 zeros(R1, C2); zeros(R2, C1 + C2)])
    end
    for b2 in B2
        push!(result, [zeros(R1, C1 + C2); zeros(R2, C1) b2])
    end
    result
end

function firstArg(R, C, X)
    return X[1:R, 1:C]
end

function lastArg(R, C, X)
    R1,C1 = size(X)
    return X[R1-R + 1: R1, C1-C + 1: C1]
end

"""
Given vectorspace  V and subspaces V1, V2 with basis B, B1, B2. Compute a basis for V1 ∩ V2.  
"""
function vectorSpaceIntersection(B :: Array{<:AbstractMatrix{Float64}}, B1 :: Array{<:AbstractMatrix{Float64}}, B2 :: Array{<:AbstractMatrix{Float64}})
    Pr1 = x -> x - projectionOntoSubspace(B, B1)(x)
    Pr2 = x -> x - projectionOntoSubspace(B, B2)(x)
    N, dimN = computeKernel(B, ⊕(B, B), x -> DirectSumMat(Pr1(x), Pr2(x)))
    return N, dimN
end 

function  ToN(n :: Int, B)
    if n == 0
        return AbstractMatrix{Float64}[]
    end
    if n == 1
        return B
    end
    return ⊕(B, ToN(n - 1, B))
end

function DirectSumMat(X, Y)
    r1, c1 = size(X)
    r2, c2 = size(Y)
    return [X zeros(r1, c2); zeros(r2, c1) Y]
end

function DirectSumMatN(matList)
    if length(matList) == 0
        error("At least one matrix should be in the list ...")
    end
    if length(matList) == 1
        return matList[1]
    end
    return DirectSumMat(matList[1], DirectSumMatN(matList[2:end]))
end

"""
Let f be a linear map, compute the kernel of f.
B1: basis of domain of f
B2: basis of codomain of f
asElements: true: return dimension and basis of nullspace
            false: return dimension and a matrix representing nullspace basis in form of a matrix
"""
function computeKernel(B1, B2, f, asElements :: Bool = true)
    if length(B1)*length(B2) == 0
        error("There should be no trivial vector spaces involved ...")
    end
    M = getMatrix(B1, B2, f)
    N = nullspace(M, atol = 1e-12)
    dimN = size(N)[2]
    if asElements
        return getElements(B1, N), dimN
    end
    return N, dimN
end

function factorSpace(𝔪, 𝔩)
    pr = projectionOntoSubspace(𝔪, 𝔩)
    F, dimF = computeKernel(𝔪, 𝔪, pr)
    return F, dimF, pr
end

"""
Given a Lie algebra 𝔪 and a subset S of a Lie algebra 𝔞 whose elements take 𝔪 to 𝔫 via Y -> α(X,Y) (X ∈ 𝔞), where by default α = ad. Let us compute ker(Y -> [α(S_1, Y) ... α(S_r, Y)]).
𝔪, 𝔫 are given as an array of basis. S ⊂ 𝔞 is a finite set of elements in 𝔞.
"""
function fixSpace(S, 𝔪, 𝔫, α = ad)
    cardS = length(S)
    dimm = length(𝔪)
    if cardS > 0
        f = Y -> DirectSumMatN([α(L,Y) for L in S])
        F, dimF = computeKernel(𝔪, ToN(cardS, 𝔫), f)
    else
        F, dimF = 𝔪, dimm
    end
end

"""
Given a linear map f : 𝔞 → 𝔟 between Lie algebras 𝔞,𝔟 with basis B1, B2 respectively, we compute the nullspace and the range of f.
"""
function computeLinearFunctionData(B1 :: Array{<:AbstractMatrix{Float64}}, B2 :: Array{<:AbstractMatrix{Float64}}, f, asElements :: Bool = true, absTolMachine = 1e-13)
    M = getMatrix(B1, B2, f)
    SVD = svd(M, full = true)
    largeSV = filter(s -> s > absTolMachine, SVD.S)
    rankM = length(largeSV)
    NspaceM = transpose(SVD.Vt)[:, rankM + 1 : end]
    dimKerM = size(NspaceM)[2]
    RangeM = SVD.U[:,1:rankM]
    if asElements
        return getElements(B1, NspaceM), dimKerM, getElements(B2, RangeM), rankM
    end
    return NspaceM, dimKerM, RangeM, rankM
end

getFixSpace(B1, B2, X) = computeKernel(B1, B2, Y -> ad(Y,X))

function ProjectMat(A :: AbstractMatrix{Float64}, orthonormalColumns = false) 
    if orthonormalColumns
        return A*transpose(A)
    else
        return A*(transpose(A)*A)^(-1)*transpose(A)
    end
end

"""
B, B_sub ⊂ AbstractMatrix[] basis, where W := <B_sub> ⊂ V := <B>, compute the linear map pr : V -> V such that pr ∘ pr = pr and <pr(x),y> = <x,pr(y)> and image(pr) = W. 
"""
function projectionOntoSubspace(B :: Array{<:AbstractMatrix{Float64}}, B_sub :: Array{<:AbstractMatrix{Float64}}, orthSubspaceBasis = false)
    M = getMatrix(B_sub, B, X -> X)
    P = ProjectMat(M, orthSubspaceBasis)
    v = X :: AbstractMatrix{Float64} -> getVector(B, X)
    pr = X :: AbstractMatrix{Float64} ->  getElement(B, P*v(X))
    return pr
end

function getVector(basis, A :: AbstractMatrix{Float64})
    result = Float64[]
    for v in basis
        push!(result, v⋅A)
    end
    return result
end

function getElement(basis, v :: AbstractVector{Float64})
    return sum([v[i] * basis[i] for i in 1:length(v)])
end

function getElements(basis, N :: AbstractMatrix{Float64})
    result = AbstractMatrix{Float64}[]
    for i in 1:size(N)[2]
         push!(result, getElement(basis, N[:,i]))
    end
    return result
end

"""
compute the matrix with respect to the given basis
"""
function getMatrix(B1, B2, f)
    domD = length(B1)
    codomD = length(B2)
    result = Matrix{Float64}(undef, codomD, domD)
    for i in 1:domD
        result[:,i] = getVector(B2, f(B1[i]))
    end
    return result
end
end # module
