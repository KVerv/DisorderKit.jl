# Shared helpers: the algorithm supertype, the finalizer of a ground-state run, robust SVDs and space-generic helpers

abstract type AbstractAlgorithm end

# Function f!(ψ, Hs) called after every step of a ground-state algorithm, its results (of type E) are collected
struct Finalizer
    E::Type
    f!::Function

    function Finalizer(E::Type, f::Function)
        return new(E, f)
    end
end

# svd_trunc with the default algorithm (LAPACK divide-and-conquer, gesdd), falling back to the slower but more robust
# QR iteration (gesvd) when gesdd does not converge (LAPACKException). A matrix with NaN/Inf entries is not retried:
# that is an upstream problem, reported as such
function robust_svd_trunc(A::AbstractTensorMap; kwargs...)
    return svd_with_fallback(A) do alg
        isnothing(alg) ? svd_trunc(A; kwargs...) : svd_trunc(A; alg, kwargs...)
    end
end

# f(nothing) runs the SVD with the default algorithm, f(alg) with alg; see robust_svd_trunc
function svd_with_fallback(f, A::AbstractTensorMap)
    try
        return f(nothing)
    catch e
        e isa LinearAlgebra.LAPACKException || rethrow()
        all(((_, b),) -> all(isfinite, b), blocks(A)) ||
            throw(DomainError(A, "SVD of a matrix with non-finite entries (NaN or Inf)"))
        @warn "gesdd did not converge ($(sprint(showerror, e))), retrying the SVD with QR iteration" maxlog = 10
        return f(LAPACK_QRIteration())
    end
end

#==========================================================================================
 Space-generic helpers: the same code runs on ComplexSpace (no symmetry) and on graded spaces of any sector type
 (abelian such as U₁, non-abelian such as SU₂, anyonic such as Fibonacci)
==========================================================================================#

# Integer size of a (bond) space for diagnostics and caps: the dimension dim(V) when it is an integer (groups), the number
# of multiplets Σ_c n_c otherwise (anyons have irrational quantum dimensions, e.g. dim(τ) = φ for Fibonacci)
function bond_dim(V::Union{ElementarySpace, ProductSpace})
    d = dim(V)
    return d isa Integer ? Int(d) : Int(sum(c -> dim(fuse(V), c), sectors(fuse(V)); init = 0))
end

# Whether dense arrays (convert(Array, t)) represent tensors of sector type I faithfully: groups with bosonic braiding,
# which have fusion tensors (Trivial, U₁, SU₂, Z_N, …), not fermions or anyons
dense_supported(::Type{I}) where {I<:Sector} = BraidingStyle(I) isa Bosonic && I <: Union{Trivial, AbstractIrrep}
dense_supported(t::AbstractTensorMap) = dense_supported(sectortype(t))

# Space of n dimensions in the trivial sector of the space type S: ℂ^n for ComplexSpace, S(unit => n) for a graded space.
# Disorder-only objects (partition function, filter, sketch rows of the partition) carry no charge and live on these
trivial_space(::Type{ComplexSpace}, n::Integer) = ℂ^n
trivial_space(::Type{S}, n::Integer) where {S<:GradedSpace} = S(one(sectortype(S)) => n)
trivial_space(V::ElementarySpace, n::Integer) = trivial_space(typeof(V), n)
trivial_space(t::AbstractTensorMap, n::Integer) = trivial_space(spacetype(t), n)

# Block of the trivial sector (the whole data for a ComplexSpace tensor)
dense_block(t::AbstractTensorMap) = block(t, one(sectortype(t)))

# Tensor with the dense data `data` (codomain legs first, every leg fastest-first) between trivial-sector spaces of the
# space type S, e.g. trivial_tensor(M, S, (m,), (n,)) : trivial_space(S, m) ← trivial_space(S, n). Unlike
# TensorMap(data, V), this does not need fusion tensors, so it works for every sector type (also anyons)
function trivial_tensor(data::AbstractArray, S::Type{<:ElementarySpace}, cod::Tuple, dom::Tuple = ())
    V = ProductSpace{S, length(cod)}(map(n -> trivial_space(S, n), cod)) ← ProductSpace{S, length(dom)}(map(n -> trivial_space(S, n), dom))
    t = zeros(promote_type(eltype(data), TensorKit.sectorscalartype(sectortype(S))), V)   # complex for anyons
    dense_block(t) .= reshape(data, prod(cod; init = 1), prod(dom; init = 1))
    return t
end

# Basis ket e_u of ℂ^q (trivial sector of the space type S)
basis_ket(S::Type{<:ElementarySpace}, q::Int, u::Int, T::Type = Float64) = trivial_tensor(T[j == u ? 1 : 0 for j in 1:q], S, (q,))

# Basis order of a graded space with multiplicity one sectors: the natural basis state k has sector charges[k], and
# TensorKit orders the basis by sector (for U₁: 0, 1, -1, 2, -2, …). Returns p with TensorKit basis state j = natural
# basis state p[j], so that data[p, p] is a natural operator in the TensorKit order
function basis_order(V::ElementarySpace, charges::AbstractVector{<:Sector})
    order = Int[]
    for c in sectors(V)
        idx = findall(==(c), charges)
        length(idx) == dim(V, c) || throw(ArgumentError("charges do not match the space $V in sector $c"))
        append!(order, idx)
    end
    return order
end
