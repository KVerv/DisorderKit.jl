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
 Space-generic helpers: the same code runs on ComplexSpace (no symmetry) and on graded spaces (e.g. U₁)
==========================================================================================#

# Space of n dimensions in the trivial sector of the space type S: ℂ^n for ComplexSpace, S(unit => n) for a graded space.
# Disorder-only objects (partition function, filter, sketch rows of the partition) carry no charge and live on these
trivial_space(::Type{ComplexSpace}, n::Integer) = ℂ^n
trivial_space(::Type{S}, n::Integer) where {S<:GradedSpace} = S(one(sectortype(S)) => n)
trivial_space(V::ElementarySpace, n::Integer) = trivial_space(typeof(V), n)
trivial_space(t::AbstractTensorMap, n::Integer) = trivial_space(spacetype(t), n)

# Block of the trivial sector (the whole data for a ComplexSpace tensor)
dense_block(t::AbstractTensorMap) = block(t, one(sectortype(t)))

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
