using SparseArrays

"""
define gl=gl_2, o_2, sym_2,... via their normalized orthonormal basis 
"""
sym_2 = [[1.0 0.0;0.0 0.0], [0.0 0.0;0.0 1.0], [0.0 1.0;1.0 0.0]/sqrt(2)]
o_2 = [[0.0 1.0;-1.0 0.0]/sqrt(2)]
gl_2 = [sym_2; o_2]
diag_2 = [[1.0 0.0;0.0 0.0], [0.0 0.0;0.0 1.0]]
k(α) = [cos(α) -sin(α);sin(α) cos(α)]

"""
define gl(n) = Lie(GL(n,IR))
"""
function gl(n :: Int)
    result = Matrix{Float64}[]
    for i in 1:n
        for j in 1:n
            M = zeros(Int, n,n)
            M[i,j] = 1.0
            push!(result, M)
        end
    end
    return result
end

function a(n :: Int)
    result = Matrix{Float64}[]
    for i in 1:n
        M = zeros(Int, n,n)
        M[i,i] = 1.0
        push!(result, M)
    end
    return result
end

"""
define Lie algebra basis with sparse matrices (here fore the special case of Lorenz matrices)
"""
BasisMatrix(m::Int, n::Int, i::Int, j::Int, α::Float64, β::Float64) =
    sparse([i, j], [j, i], [α, β], m + n, m + n)

"""
define 𝔤(m,n), 𝔨(m,n), 𝔭(m,n) via their normalized orthonormal basis 
"""
𝔭(m, n) = [BasisMatrix(m, n, i, j, 1.0, 1.0) / sqrt(2) for i ∈ 1:m, j ∈ m+1:m+n][1:end]
𝔨(m, n) = begin
    result = AbstractMatrix{Float64}[]
    for j ∈ 2:m
        for i ∈ 1:j-1
            push!(result, BasisMatrix(m, n, i, j, 1.0, -1.0) / sqrt(2))
        end
    end
    for j ∈ m+2:m+n
        for i ∈ m+1:j-1
            push!(result, BasisMatrix(m, n, i, j, 1.0, -1.0) / sqrt(2))
        end
    end
    result
end
𝔤(m, n) = [𝔭(m, n); 𝔨(m, n)]

P(A) = [zeros(size(A)[1], size(A)[1]) A; transpose(A) zeros(size(A)[2], size(A)[2])]
_e(n::Int, i::Int) = begin
    v = zeros(n)
    v[i] = 1
    return v
end
𝔞(m, n) = [P(diagm(m, n, _e(min(m, n), i))) for i ∈ 1:min(m, n)]