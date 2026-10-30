# Successive randomized compression (SRC) of the implicit products √P(w) Z(w) of W ψ and f ⊙ (W ψ), without forming them.
# The step numbers refer to the SRC chapter of docs/FiniteNormalizedITE (Steps 0-7 and the section on the first cut i = L).
#
# Design: the SRC algorithm (Steps 1-7) is written as if the input were an explicit MPS with site tensors
#   M_i : V_i ← X_i ⊗ R_i
# and only touches site i of the source P through the two primitives apply_suffix(P, i, S) and
# apply_prefix(P, i, C, Ω) (section "Site tensors"), which act as the explicit M_i would without forming it.
# The composite bonds V_i are kept as multi-leg ProductSpaces: TensorKit's *, ' and qr work on them directly, so
# no fusing isomorphism is needed.
#
# Dimensions (docs, table "Dimensions and symbols"): d spin dimension, q number of disorder sectors, D_i the bond at cut i
# of the MPS that SRC builds (capped by D_max = max_rank or maxD), n_i = dim X_i the local dimension of the compressed
# tensor (q for the partition, d q for the state), K the number of test vectors (sketch columns), m the number of basis
# vectors, rows = n_i D_i.
#
# Conventions (as in the docs, "Spaces and objects"):
#   V_i = left_space(P, i)        composite left bond of site i (product of the layer bonds)
#   R_i = right_space(P, i)       composite right bond of site i (= V_{i+1})
#   X_i = physical_space(P, i)    physical legs: ℂ^q (partition) or (ℂ^d)' ⊗ ℂ^q (physical, spin s fastest)
#   site tensor  M_i : V_i ← X_i ⊗ R_i
#   suffix       S_i : R_i ← ℂ^{D_i}
#   cut-local    J_i : V_i ← X_i ⊗ ℂ^{D_i}      (J_i = M_i S_i)
#   prefix       C_{i-1} : ℂ^K ← V_i            (one row per Khatri-Rao test vector)
#   sketch       Z : X_i ⊗ ℂ^{D_i} ← ℂ^K        (Z = (C_{i-1} J_i)')
#   basis        Q : X_i ⊗ ℂ^{D_i} ← ℂ^m
#   core         G_i : ℂ^{D_{i-1}} ⊗ X_i' ← ℂ^{D_i}   (MPS tensor: virtual ⊗ physical ← virtual, D_{i-1} = m)
# Layer conventions (MPSKit): A[a s; b], W[h s; t j] (s output, t input), f[x; y].
#
# Note: DisorderKit defines its own `norm`, so this file always writes LinearAlgebra.norm.

#==========================================================================================
 Step 0: the sources (implicit products) and the scales
==========================================================================================#

# √P(w) Z(w) of W A from the four layers A, W, W̄, Ā; physical index the sector u (n_i = q).
# A[i][u], W[i][u]: layers of site i in sector u; g[u] = √p_u
struct PartitionSource
    A::Vector{<:Vector{<:AbstractMPSTensor}}
    W::Vector{<:Vector{<:AbstractMPOTensor}}
    g::Vector{Float64}
end

# Σ_w √P(w) |w⟩ ⊗ f(w) W_w|A_w⟩ from the three layers A, W, f; physical index (s, u), s fastest (n_i = d q)
struct PhysicalSource
    A::Vector{<:Vector{<:AbstractMPSTensor}}
    W::Vector{<:Vector{<:AbstractMPOTensor}}
    f::Vector{<:Vector{<:AbstractBondTensor}}
    g::Vector{Float64}
end

const Source = Union{PartitionSource, PhysicalSource}

Base.length(P::Source) = length(P.A)

# Composite left bond V_i of site i: a ⊗ h ⊗ h' ⊗ a' (partition), a ⊗ h ⊗ x (physical)
left_space(P::PartitionSource, i::Int) = space(P.A[i][1], 1) ⊗ space(P.W[i][1], 1) ⊗ space(P.W[i][1], 1)' ⊗ space(P.A[i][1], 1)'
left_space(P::PhysicalSource, i::Int) = space(P.A[i][1], 1) ⊗ space(P.W[i][1], 1) ⊗ space(P.f[i][1], 1)

# Composite right bond R_i of site i (= V_{i+1}); built from the domains so that R_L (one-dimensional) works too
right_space(P::PartitionSource, i::Int) = domain(P.A[i][1])[1] ⊗ domain(P.W[i][1])[2] ⊗ domain(P.W[i][1])[2]' ⊗ domain(P.A[i][1])[1]'
right_space(P::PhysicalSource, i::Int) = domain(P.A[i][1])[1] ⊗ domain(P.W[i][1])[2] ⊗ domain(P.f[i][1])[1]

# Physical legs X_i of site i: ℂ^q (partition) or (ℂ^d)' ⊗ ℂ^q (physical)
physical_space(P::PartitionSource, i::Int) = ProductSpace(ℂ^length(P.g))
physical_space(P::PhysicalSource, i::Int) = space(P.W[i][1], 2)' ⊗ ℂ^length(P.g)

# Dimension n_i of the physical legs X_i
physical_dim(P::Source, i::Int) = dim(physical_space(P, i))

# Divide every site of a layer by its largest entry (over all sectors).
# Returns (scaled layers, Σ log(max entry)); the scale is put back in the entry points (Step 7)
function rescale_layers(layers::Vector{<:Vector{<:AbstractTensorMap}})
    log_scale = 0.0
    scaled = similar(layers)
    for (i, X) in enumerate(layers)                             # X[u]: layer of site i in sector u
        m = maximum(T -> LinearAlgebra.norm(T, Inf), X)          # largest |entry| over all sectors of site i
        all(T -> all(isfinite, convert(Array, T)), X) || throw(DomainError(m, "nonfinite layer at site $i"))
        if m > 0
            X = X ./ m
            log_scale += log(m)
        end
        scaled[i] = X
    end
    return scaled, log_scale
end

# Identity MPO layers W[1 s; t 1] = δ_st for every site and sector, for the partition of an unevolved ψ
function identity_layers(ψ::FiniteDisorderMPS)
    q = length(ψ.ps)
    return map(1:length(ψ)) do i
        s = space(ψ[i, 1], 2)                                    # physical space of site i
        W = permute(id(ComplexF64, ℂ^1 ⊗ s), ((1, 2), (4, 3)))  # (ℂ^1 ⊗ s ← ℂ^1 ⊗ s) to W[1 s; t 1]
        fill(W, q)
    end
end

#==========================================================================================
 Site tensors: the interface between the SRC algorithm and the source
==========================================================================================#

# Intermediates of the site contractions are freed right away instead of being left to the GC
const site_allocator = TensorKit.TensorOperations.ManualAllocator()

# Dense array of a tensor between trivially graded spaces, as a view on its data: codomain legs first, every leg
# fastest-first as in convert(Array, t)
dense_view(t::AbstractTensorMap, dims...) = reshape(block(t, Trivial()), dims...)

# Step 1 primitive: J = M_i S, i.e. J[v; x r] = Σ_w M_i[v; x w] S[w; r].  S: R_i ← ℂ^{D_i}, J: V_i ← X_i ⊗ ℂ^{D_i}.
# Partition: Σ_u e_u ⊗ g_u (A_u W_u W̄_u Ā_u S), e_u the sector-u basis covector on X_i.
# Contract S with the four layers of sector u (bond legs unfused), weight g_u, write it into physical index u of J.
function apply_suffix(P::PartitionSource, i::Int, S::AbstractTensorMap)
    q = length(P.g)
    T = promote_type(scalartype(S), scalartype(P.A[i][1]), scalartype(P.W[i][1]))
    J = TensorMap{T}(undef, left_space(P, i) ← physical_space(P, i) ⊗ domain(S))   # every sector u is written below
    Jd = dense_view(J, dim(codomain(J)), q, dim(domain(S)))                       # J[v, u, r]
    for (u, g) in enumerate(P.g)
        A, W = P.A[i][u], P.W[i][u]
        # explicit order, 5-25% faster than opt=true at D_i = 4..16
        @tensor allocator=site_allocator X[-1 -2 -3 -4; -5] := g * ((((S[1 3 5 7; -5] * A[-1 2; 1]) * W[-2 4; 2 3]) * conj(W[-3 4; 6 5])) * conj(A[-4 6; 7]))
        Jd[:, u, :] .= block(X, Trivial())
    end
    return J
end

# Physical source: per sector u contract S with A_u, W_u, f_u, keep the spin leg s open,
# weight g_u, write it into physical index (s, u) of J
function apply_suffix(P::PhysicalSource, i::Int, S::AbstractTensorMap)
    q = length(P.g)
    d = dim(space(P.W[i][1], 2))
    T = promote_type(scalartype(S), scalartype(P.A[i][1]), scalartype(P.W[i][1]), scalartype(P.f[i][1]))
    J = TensorMap{T}(undef, left_space(P, i) ← physical_space(P, i) ⊗ domain(S))   # V_i ← (ℂ^d)' ⊗ ℂ^q ⊗ ℂ^{D_i}
    Jd = dense_view(J, dim(codomain(J)), d, q, dim(domain(S)))                    # J[v, s, u, r]
    for (u, g) in enumerate(P.g)
        A, W, f = P.A[i][u], P.W[i][u], P.f[i][u]
        # X[a h x; s r]: S[b j y; r] with A[a t; b], W[h s; t j], f[x; y]; the output spin s of W stays open
        @tensor opt=true allocator=site_allocator X[-1 -2 -3; -4 -5] := g * S[1 3 5; -5] * A[-1 2; 1] * W[-2 -4; 2 3] * f[-3; 5]
        Jd[:, :, u, :] .= dense_view(X, dim(codomain(J)), d, dim(domain(S)))
    end
    return J
end

# Step 3 primitive (Khatri-Rao): C′[k; w] = Σ_{v,x} C[k; v] M_i[v; x w] Ω[x, k]; row k uses column k of Ω.
# C: ℂ^K ← V_i, Ω: n_i × K (x linear, s fastest), C′: ℂ^K ← R_i.
# Partition: Σ_u (Diag(g_u Ω[u, :]) C) contracted with the four layers of u; the row weights commute with the
# contraction, so they are applied to C (K × dim V_i) instead of the result.
# The sector δ of M_i becomes the loop over u; no copy tensor needed
function apply_prefix(P::PartitionSource, i::Int, C::AbstractTensorMap, Ω::AbstractMatrix)
    K = codomain(C)[1]
    T = promote_type(scalartype(C), eltype(Ω), scalartype(P.A[i][1]), scalartype(P.W[i][1]))
    out = zeros(T, K ← right_space(P, i))
    for (u, g) in enumerate(P.g)
        A, W = P.A[i][u], P.W[i][u]
        Cw = DiagonalTensorMap(g .* Ω[u, :], K) * C   # row k of C scaled by g_u Ω[u, k]
        @tensor opt=true allocator=site_allocator out[-1; -2 -3 -4 -5] += Cw[-1; 1 3 5 7] * A[1 2; -2] * W[3 4; 2 -3] * conj(W[5 4; 6 -4]) * conj(A[7 6; -5])
    end
    return out
end

# Physical source: per u contract C with A_u, W_u with s open, then f_u, then
# out[k; ⋯] += Σ_s g_u Ω[(u-1) d + s, k] X[k; ⋯ s] in place on the dense arrays
function apply_prefix(P::PhysicalSource, i::Int, C::AbstractTensorMap, Ω::AbstractMatrix)
    K = codomain(C)[1]
    d = dim(space(P.W[i][1], 2))
    T = promote_type(scalartype(C), eltype(Ω), scalartype(P.A[i][1]), scalartype(P.W[i][1]), scalartype(P.f[i][1]))
    out = zeros(T, K ← right_space(P, i))
    outd = block(out, Trivial())                      # out[k, w]
    for (u, g) in enumerate(P.g)
        A, W, f = P.A[i][u], P.W[i][u], P.f[i][u]
        # X[k; b j y s]: C[k; a h x] with A[a t; b], W[h s; t j], f[x; y]; the output spin s of W stays open
        @tensor opt=true allocator=site_allocator X[-1; -2 -3 -4 -5] := C[-1; 1 3 4] * A[1 2; -2] * W[3 -5; 2 -3] * f[4; -4]
        Xd = dense_view(X, dim(K), size(outd, 2), d)  # X[k, w, s]
        for s in 1:d
            weights = g .* view(Ω, (u - 1) * d + s, :)   # weight of row k: g_u Ω[(s, u), k], s fastest
            outd .+= weights .* view(Xd, :, :, s)
        end
    end
    return out
end

# The explicit site tensor M_i : V_i ← X_i ⊗ R_i, for tests on small systems only
function dense(P::Source, i::Int)
    R, X = right_space(P, i), physical_space(P, i)
    iso = isomorphism(fuse(R), R)                      # apply_suffix takes S with a single domain leg
    return apply_suffix(P, i, iso') * (id(X) ⊗ iso)
end

#==========================================================================================
 Step 3: Khatri-Rao prefix sketches
==========================================================================================#

# Prefix sketches shared by all cuts: C[j] holds C_{j-1}: ℂ^width ← V_j (C_0 = 1), logs[j] the log scale of every row.
# The prefixes are built with apply_prefix only
mutable struct PrefixStore{TP<:Source, R<:AbstractRNG}
    P::TP
    rng::R
    C::Vector{Any}
    logs::Vector{Vector{Float64}}
    width::Int
end

PrefixStore(P::Source, rng::AbstractRNG) = PrefixStore(P, rng, Any[nothing for _ in 1:length(P)], [Float64[] for _ in 1:length(P)], 0)

# Append ncols fresh Gaussian rows to the prefixes of the sites 1..top (the first call builds all sites 1..L):
# C_i = apply_prefix(P, i, C_{i-1}, randn(physical_dim(P, i), ncols)), rescaled per row
function extend_prefixes!(store::PrefixStore, ncols::Int, top::Int)
    P, rng = store.P, store.rng
    C = ones(ComplexF64, ℂ^ncols ← left_space(P, 1))            # C_0 = 1 for every new row (V_1 is one-dimensional)
    logs = zeros(ncols)                                           # log scale of every new row
    for i in 1:top
        if i > 1
            n = physical_dim(P, i - 1)                            # n_{i-1}
            Ω = (randn(rng, n, ncols) + im * randn(rng, n, ncols)) / sqrt(2)   # complex Gaussian test vectors
            C = apply_prefix(P, i - 1, C, Ω)                      # C_{i-1} from C_{i-2}
            s = row_scales(C)
            all(isfinite, s) || throw(DomainError(s, "nonfinite prefix sketch at site $i"))
            s[s .== 0] .= 1.0
            C = DiagonalTensorMap(1 ./ s, ℂ^ncols) * C           # every row rescaled to max |entry| 1
            logs = logs .+ log.(s)
        end
        store.C[i] = store.width == 0 ? C : catcodomain(store.C[i], C)   # new rows below the old ones
        append!(store.logs[i], logs)
    end
    store.width += ncols
    return store
end

# Largest |entry| of every row of C: ℂ^K ← V (trivial symmetry)
row_scales(C::AbstractTensorMap) = vec(maximum(abs, reshape(convert(Array, C), dim(codomain(C)), :); dims = 2))

# Row selection Π: ℂ^|rows| ← ℂ^w that picks the rows `rows` out of w
function select_rows(rows, w::Int)
    Π = zeros(ComplexF64, length(rows), w)
    for (k, r) in enumerate(rows)
        Π[k, r] = 1
    end
    return TensorMap(Π, ℂ^length(rows) ← ℂ^w)
end

# Sketch columns of cut i for the prefix rows cols: Z = (C_{i-1}[cols, :] J)': X_i ⊗ ℂ^{D_i} ← ℂ^|cols|
function sketch_columns(store::PrefixStore, J::AbstractTensorMap, i::Int, cols)
    Ci = select_rows(cols, store.width) * store.C[i]              # C_{i-1}[cols, :]: ℂ^|cols| ← V_i
    return copy((Ci * J)')
end

#==========================================================================================
 Step 4: orthonormal basis of the sketch
==========================================================================================#

const SketchMatrix = tensormaptype(ComplexSpace, 1, 1, ComplexF64)

# State of one cut: all sketch columns Z: X_i ⊗ ℂ^{D_i} ← ℂ^K, stored relative to the common log scale `shift`
# (true column j = exp(shift) Z[:, j]), and their QR factorization Z = Q R, recomputed after every block of columns.
#   rows       dim(X_i ⊗ ℂ^{D_i}), the maximal number of basis vectors
#   threshold  relative singular value cutoff for the numerical rank of R
#   Q          orthonormal X_i ⊗ ℂ^{D_i} ← ℂ^m, m = min(rows, K)
#   R          upper triangular ℂ^m ← ℂ^K
#   rank       numerical rank of R
# m can exceed the rank: for a deficient sketch the Householder QR still returns m orthonormal columns
mutable struct SketchBasis{TZ<:AbstractTensorMap, TQ<:AbstractTensorMap}
    rows::Int
    threshold::Float64
    Z::TZ
    Q::TQ
    R::SketchMatrix
    shift::Float64
    rank::Int
end

# Empty basis on the space V = X_i ⊗ ℂ^{D_i}
SketchBasis(V; threshold::Real = 1e-13) =
    SketchBasis(dim(V), Float64(threshold), zeros(ComplexF64, V ← ℂ^0), zeros(ComplexF64, V ← ℂ^0),
                zeros(ComplexF64, ℂ^0 ← ℂ^0), -Inf, 0)

sketch_width(basis::SketchBasis) = dim(domain(basis.R))       # K, number of stored sketch columns
basis_size(basis::SketchBasis) = dim(codomain(basis.R))       # m, number of basis vectors

# Singular values of a matrix between trivially graded spaces (trivial symmetry)
matrix_svdvals(t::AbstractTensorMap) = dim(t) == 0 ? Float64[] : svdvals(convert(Array, t))

# Append the sketch columns Znew: X_i ⊗ ℂ^{D_i} ← ℂ^nnew with column log scales logs; update Z, Q, R and rank
function append_columns!(basis::SketchBasis, Znew::AbstractTensorMap, logs::AbstractVector{Float64})
    old, nnew = sketch_width(basis), dim(domain(Znew))

    # 1. Common scale: all columns relative to exp(shift), shift the largest log scale so far
    shift = max(basis.shift, maximum(logs))
    scale = old > 0 ? exp(basis.shift - shift) : 1.0           # old columns to the new reference (≤ 1)
    factors = exp.(logs .- shift)                                # new columns relative to the new reference (≤ 1)
    (any(iszero, factors) || scale == 0) && throw(DomainError(shift, "sketch column scale exceeds numerical range"))
    basis.shift = shift
    Zs = Znew * DiagonalTensorMap(factors, ℂ^nnew)
    basis.Z = catdomain(basis.Z * scale, Zs)

    # 2. Householder QR of all columns: Q orthonormal even for a deficient sketch, m = min(rows, K).
    # Rank and leave-one-out residuals only depend on R'R = Z'Z, so recomputing gives the same results
    basis.Q, basis.R = qr_compact(basis.Z)

    # 3. Numerical rank
    s = matrix_svdvals(basis.R)
    basis.rank = (isempty(s) || s[1] == 0) ? 0 : count(>(basis.threshold * s[1]), s)
    return basis
end

#==========================================================================================
 Step 5: error estimate and stopping
==========================================================================================#

# Leave-one-out residuals of all columns of R (rank r): residual of column j against the span of the other columns,
# i.e. the fresh-column test ‖v - QQ'v‖ of column j against the basis of the others. Q is an isometry, so these are the
# residuals of the sketch columns. With R = U Σ V' truncated to rank r (V: K × r) the leverage h_j = ‖V[j, :]‖² is the
# diagonal of the projector V V' onto the row space of R (0 ≤ h_j ≤ 1, Σ h_j = r). For independent columns
# ρ_j = |R_KK| after moving column j to the end = 1/‖row j of R⁺‖ (docs, Step 5); with R⁺ = V Σ⁻¹ U':
#   - leverage ‖V[j, :]‖² < 1: e_j is not in the row space, column j is a combination of the others, residual 0
#   - leverage ‖V[j, :]‖² = 1: column j is needed for the span, residual 1/‖Σ⁻¹ V[j, :]'‖ (= 1/‖R^{-†}[:, j]‖ at full rank)
function loo_residuals(R::AbstractTensorMap, rank::Int; leverage_tol::Real = 1e-8)
    _, Σ, Vh = svd_trunc(R; trunc = truncrank(max(rank, 1)))
    V = copy(convert(Array, Vh)')                                 # K × r
    σ = diag(convert(Array, Σ))
    return [sum(abs2, v) > 1 - leverage_tol ? 1 / LinearAlgebra.norm(v ./ σ) : 0.0 for v in eachrow(V)]
end

# Returns (rank, err, nrm, saturated): numerical rank of the sketch, leave-one-out error estimate, norm estimate,
# and whether Q spans all rows (then err = 0). The cut is accepted when saturated || err <= rtol*nrm
function estimate_error(basis::SketchBasis)
    K = sketch_width(basis)
    nrm = LinearAlgebra.norm(basis.R) / sqrt(K)
    K >= basis.rows && return basis.rank, 0.0, nrm, true
    err = LinearAlgebra.norm(loo_residuals(basis.R, basis.rank)) / sqrt(K)
    return basis.rank, err, nrm, false
end

# Double check of a rank-deficient sketch with fresh columns: ‖V - QQ'V‖ <= rtol ‖V‖. Returns a Bool.
# The fresh columns are the prefix rows right after the current sketch; they are appended to the basis afterwards,
# so the prefix rows and the sketch columns stay aligned
function fresh_check(store::PrefixStore, basis::SketchBasis, J::AbstractTensorMap, i::Int; rtol::Real, nfresh::Int = 3)
    K = sketch_width(basis)
    cols = K+1:K+nfresh
    store.width < last(cols) && extend_prefixes!(store, last(cols) - store.width, i)
    V, logs = sketch_columns(store, J, i, cols), store.logs[i][cols]
    Vs = V * DiagonalTensorMap(exp.(logs .- maximum(logs)), ℂ^nfresh)     # relative scales of the fresh columns
    Q = basis.Q
    passed = LinearAlgebra.norm(Vs - Q * (Q' * Vs)) <= rtol * LinearAlgebra.norm(Vs)
    append_columns!(basis, V, logs)
    return passed
end

#==========================================================================================
 Step 6: core and next suffix
==========================================================================================#

# Basis Q: X_i ⊗ ℂ^{D_i} ← ℂ^m to the right-orthonormal MPS core G_i: ℂ^m ⊗ X_i' ← ℂ^{D_i} (m becomes D_{i-1})
function core_from_basis(Q::AbstractTensorMap)
    t = copy(Q')                                                  # ℂ^m ← X_i ⊗ ℂ^{D_i}
    N = numind(t)
    return permute(t, (Tuple(1:N-1), (N,)))
end

# Orthonormal basis of the numerical range of the sketch: Q itself when R has full row rank, else Q U with U the
# leading rank left singular vectors of R (drops the completion directions of a deficient sketch). At least 1 vector
function trimmed_basis(basis::SketchBasis)
    r = max(basis.rank, 1)
    r == basis_size(basis) && return basis.Q
    U, = svd_trunc(basis.R; trunc = truncrank(r))
    return basis.Q * U
end

# Next suffix S_{i-1} = J Q: V_i ← ℂ^m
next_suffix(J::AbstractTensorMap, Q::AbstractTensorMap) = J * Q

#==========================================================================================
 Steps 1-6 for one cut, and the sweep
==========================================================================================#

# One cut i (also the first cut i = L, where S = 1: R_L ← ℂ^1): J = apply_suffix(P, i, S), sketch adaptively
# up to cap until the error target rtol is met, and return (G_i, S_{i-1}, final sketch width, target met).
# The sketch starts at 2 columns, or at 4/5 of the final width κ of this cut in the previous time step (warm start)
function compress_cut!(P::Source, store::PrefixStore, S::AbstractTensorMap, i::Int; rtol::Real, cap::Int,
                       rank_threshold::Real = 1e-13, previous_width = nothing, to::TimerOutput = TimerOutput())
    @timeit to "contract_suffix" J = apply_suffix(P, i, S)
    basis = SketchBasis(domain(J); threshold = rank_threshold)
    k0 = isnothing(previous_width) ? 2 : min(basis.rows, max(2, (4 * previous_width) ÷ 5))
    k = min(k0, cap)
    while true
        store.width < k && @timeit to "extend_prefixes" extend_prefixes!(store, k - store.width, i)
        K = sketch_width(basis)
        if K < k
            @timeit to "sketch" Z = sketch_columns(store, J, i, K+1:k)
            @timeit to "qr_update" append_columns!(basis, Z, store.logs[i][K+1:k])
        end
        @timeit to "estimates" rank, err, nrm, saturated = estimate_error(basis)
        met = saturated || err <= rtol * nrm
        if met && !saturated && rank < sketch_width(basis)       # deficient sketch: double check with fresh columns
            met = @timeit to "fresh_check" fresh_check(store, basis, J, i; rtol = rtol)
            k = sketch_width(basis)
        end
        if met || k >= cap
            Q = trimmed_basis(basis)
            @timeit to "project" S = next_suffix(J, Q)
            return core_from_basis(Q), S, sketch_width(basis), met
        end
        k = min(k + max(3, k ÷ 4), cap)                          # geometric growth: few, wide blocks
    end
end

# Right-to-left sweep over the cuts i = L..2, first core G_1 = J_1, optional rounding (Step 7).
# previous_widths[i-1] is the final sketch width of cut i in the previous call (warm start), or nothing.
# Returns (cores, info) with info (pre_round_ranks, sketch_widths, target_met, cap_unmet, rounding_error, rounding_met)
function run_src(P::Source, rng::AbstractRNG; rtol::Real, sketch_cap::Int, rank_threshold::Real = 1e-13,
                 round_rank = nothing, round_tol::Real = 0.0, previous_widths = nothing, to::TimerOutput = TimerOutput())
    L = length(P)
    cap = max(sketch_cap, 2)
    store = PrefixStore(P, rng)
    cores = Vector{AbstractTensorMap}(undef, L)
    widths, met = zeros(Int, L-1), trues(L-1)
    S = ones(ComplexF64, right_space(P, L) ← ℂ^1)                # R_L is one-dimensional
    for i in L:-1:2
        previous_width = isnothing(previous_widths) ? nothing : previous_widths[i-1]
        cores[i], S, widths[i-1], met[i-1] = compress_cut!(P, store, S, i; rtol = rtol, cap = cap, rank_threshold = rank_threshold,
                                                           previous_width = previous_width, to = to)
        store.C[i] = nothing                                      # later cuts never use the prefixes of site i
    end
    J1 = isomorphism(ComplexF64, ℂ^1, left_space(P, 1)) * apply_suffix(P, 1, S)   # ℂ^1 ← X_1 ⊗ ℂ^{D_1}, V_1 is one-dimensional
    cores[1] = core_from_basis(copy(J1'))
    cores = [c for c in cores]                                    # concrete element type
    pre_round_ranks = [dim(domain(c)) for c in cores[1:L-1]]

    rounding_error, rounding_met = 0.0, true
    isnothing(round_rank) || (@timeit to "round" (rounding_error, rounding_met) = round_cores!(cores, round_rank, round_tol))
    info = (pre_round_ranks = pre_round_ranks, sketch_widths = widths, target_met = all(met), cap_unmet = count(!, met),
            rounding_error = rounding_error, rounding_met = rounding_met)
    return cores, info
end

#==========================================================================================
 Step 7: rounding and entry points
==========================================================================================#

# Contract M: ℂ^k ← ℂ^{D_{i-1}} into the left virtual leg of the MPS core c: ℂ^{D_{i-1}} ⊗ X_i' ← ℂ^{D_i}
function absorb_left(M::AbstractTensorMap, c::AbstractTensorMap)
    N = numind(c)
    return permute(M * permute(c, ((1,), Tuple(2:N))), (Tuple(1:N-1), (N,)))
end

# Left-to-right SVD rounding of right-orthonormal cores (core 1 carries the norm) to at most max_rank = D_max, with total
# relative error tol (budget tol‖Ψ‖/√(L-1) per cut). Returns (relative rounding error, error <= tol)
function round_cores!(cores::Vector, max_rank::Int, tol::Real)
    total = LinearAlgebra.norm(cores[1])
    budget = tol * total / sqrt(max(1, length(cores) - 1))
    errors = zeros(length(cores) - 1)
    for i in 1:length(cores)-1
        U, Σ, Vh, errors[i] = svd_trunc(cores[i]; trunc = truncrank(max_rank) & truncerror(; atol = budget))
        cores[i] = U
        cores[i+1] = absorb_left(Σ * Vh, cores[i+1])
    end
    err = LinearAlgebra.norm(errors) / total
    return err, err <= tol
end

# √P(w) Z(w) of W ψ for the MPO layers Ws[i][u], without forming W ψ
function src_partition(ψ::FiniteDisorderMPS, Ws::Vector{<:Vector{<:AbstractMPOTensor}}; max_rank::Int, rtol::Real = 1e-10,
                       rank_threshold::Real = 1e-13, rng::AbstractRNG = Random.default_rng(), to::TimerOutput = TimerOutput())
    any(<=(0), ψ.ps) && throw(ArgumentError("SRC requires strictly positive disorder probabilities"))
    @timeit to "product_setup" begin
        A, la = rescale_layers(ψ.opp)
        W, lw = rescale_layers(Ws)
        P = PartitionSource(A, W, sqrt.(ψ.ps))
    end
    cores, info = run_src(P, rng; rtol = rtol, sketch_cap = max_rank, rank_threshold = rank_threshold, to = to)
    return ScalarPartition(cores, 2 * (la + lw), collect(Float64, ψ.ps)), info   # ket and bra both carry the scales
end

# √P(w) Z(w) of the evolved state Us ψ, without forming Us ψ.
# Returns (ScalarPartition(cores, log_scale, ps), info); exp(log_scale) cores[1][:, w₁, :] ⋯ cores[L][:, w_L, :] = √P(w) Z(w)
src_partition(ψ::FiniteDisorderMPS, Us::FiniteDisorderMPO; kwargs...) = src_partition(ψ, Us.opp; kwargs...)

# √P(w) Z(w) of ψ itself
src_partition(ψ::FiniteDisorderMPS; kwargs...) = src_partition(ψ, identity_layers(ψ); kwargs...)

# Compress f ⊙ (Us ψ) with SRC followed by rounding to at most maxD, without forming Us ψ.
# Returns (FiniteDisorderMPS, info), unweighted (the √p_u are divided out again) and with the absolute scale restored.
# previous_widths = info.sketch_widths of the previous call starts every cut near its previous width (warm start)
function src_compress(ψ::FiniteDisorderMPS, Us::FiniteDisorderMPO, F::FiniteDisorderFilter; maxD::Int, rtol::Real = 1e-11,
                      sketch_cap::Int = 128, rank_threshold::Real = 1e-13, round_tol::Real = 1e-10, previous_widths = nothing,
                      rng::AbstractRNG = Random.default_rng(), to::TimerOutput = TimerOutput())
    any(<=(0), ψ.ps) && throw(ArgumentError("SRC requires strictly positive disorder probabilities"))
    L = length(ψ)
    (length(Us.opp) == L == length(F.opp)) || throw(ArgumentError("ψ, Us and F should have the same length"))
    @timeit to "product_setup" begin
        A, la = rescale_layers(ψ.opp)
        W, lw = rescale_layers(Us.opp)
        f, lf = rescale_layers(F.opp)
        P = PhysicalSource(A, W, f, sqrt.(ψ.ps))
    end
    cores, info = run_src(P, rng; rtol = rtol, sketch_cap = sketch_cap, rank_threshold = rank_threshold, round_rank = maxD,
                          round_tol = round_tol, previous_widths = previous_widths, to = to)

    # Sector u of the core c: ℂ^{D_{i-1}} ⊗ ℂ^d ⊗ (ℂ^q)' ← ℂ^{D_i} is the MPS tensor ℂ^{D_{i-1}} ⊗ ℂ^d ← ℂ^{D_i}; the absolute scale is
    # spread evenly over the sites and the weight √p_u is divided out
    factor = exp((la + lw + lf + F.log_scale) / L)
    (isfinite(factor) && factor > 0) || throw(DomainError(factor, "absolute SRC scale cannot be materialized"))
    q = length(ψ.ps)
    @timeit to "materialize" opp = map(cores) do c
        map(1:q) do u
            e = Tensor([j == u ? 1.0 : 0.0 for j in 1:q], ℂ^q)   # basis ket of sector u
            @tensor cu[r s; q] := c[r s x; q] * e[x]
            cu * (factor / sqrt(ψ.ps[u]))
        end
    end
    return FiniteDisorderMPS(opp, ψ.ps), info
end
