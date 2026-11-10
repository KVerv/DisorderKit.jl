function make_DiagonalBlockTensorMap(ps::Vector{<:Number})
    P = spzeros(ComplexF64, BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...), BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...))
    for (i,p) in enumerate(ps)
        P[i,i] = TensorMap([p],ℂ^1,ℂ^1)
    end
    return P
end

# Entanglement spectrum of MPO
function entanglement_spectrum(Os::InfiniteMPO, i::Int)
    unit_cell = length(Os)
    transfer_l = transfer_left_mpo(Os[i+1])
    transfer_r = transfer_right_mpo(Os[i])
    for j = i+2:i+unit_cell
        transfer_l = transfer_left_mpo(Os[j]) ∘ transfer_l
    end
    for j = i-1:-1:i-unit_cell+1
        transfer_r = transfer_right_mpo(Os[j]) ∘ transfer_r
    end

    Dl = space(Os[i+1], 1)
    Dr = space(Os[i+1], 1)

    ρl0 = rand(ComplexF64, Dl, Dl)
    ρr0 = rand(ComplexF64, Dr, Dr)

    _, ρls, infol = eigsolve(transfer_l, ρl0, 1, :LM)
    _, ρrs, infor = eigsolve(transfer_r, ρr0, 1, :LM)

    S = svd_vals((ρls[1] * ρrs[1]))
    es = S.data
    es /= sum(es)
    return es
end

function all_combinations(A::AbstractVector, L::Integer)
    @assert L ≥ 0 "L must be non-negative"
    combos = Vector{Vector{eltype(A)}}()
    for t in Base.Iterators.product(ntuple(_ -> A, L)...)
        push!(combos, collect(t))
    end
    return combos
end

function mixed_mpo_right_transfer(A::AbstractMPOTensor, B::AbstractMPOTensor)
    function ftransfer(vr)
        @tensor vr[-1; -2] := A[-1 4; 3 1] * conj(B[-2 4; 3 2]) * vr[1; 2]
        return vr
    end
    return ftransfer
end

function mixed_mpo_left_transfer(A::AbstractMPOTensor, B::AbstractMPOTensor)
    function ftransfer(vl)
        @tensor vl[-1; -2] := A[1 4; 3 -2] * conj(B[2 4; 3 -1]) * vl[2; 1]
        return vl
    end
    return ftransfer
end

function mpo_overlap(O1::AbstractMPOTensor, O2::AbstractMPOTensor)
    @assert space(O1, 1) == space(O2, 1) "Physical spaces must match"
    @assert space(O1, 2) == space(O2, 2) "Physical spaces must match"


    v0 = rand(ComplexF64, space(O1, 1), space(O2, 1))
    mixed_transfer_r = mixed_mpo_right_transfer(O1, O2) 
    λ, _ = eigsolve(mixed_transfer_r, v0, 1, :LM)

    return λ[1]
end
function mpo_fidelity(O1::AbstractMPOTensor, O2::AbstractMPOTensor)
    @assert space(O1, 1) == space(O2, 1) "Physical spaces must match"
    @assert space(O1, 2) == space(O2, 2) "Physical spaces must match"

    overlap12 = mpo_overlap(O1, O2)
    overlap11 = mpo_overlap(O1, O1)
    overlap22 = mpo_overlap(O2, O2)
    return norm(overlap12) / sqrt(norm(overlap11) * norm(overlap22))
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

