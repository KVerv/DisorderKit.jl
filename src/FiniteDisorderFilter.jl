# Disorder normalization filter of a FiniteDisorderMPS.
# For every disorder word w = (w₁…w_L) the state defines a conditional state ψ_w with norm Z(w) = ⟨ψ_w|ψ_w⟩.
# The filter is a scalar disorder MPS f(w) ≈ |Z(w)|^(-1/2), such that f ⊙ ψ has Z(w) = 1 for every word.
# It is built compress first: √P(w) Z(w) is compressed with SRC (FiniteSRC.jl, from the separate state and evolution
# layers), and a nested QR-pivoted cross samples the cheap compressed partition. The step numbers refer to the filter
# chapter of docs/FiniteNormalizedITE:
#   The filter object       FiniteDisorderFilter, F * ψ, filter_log_amplitude
#   Conditional norms       log Z(w) of a FiniteDisorderMPS word by word, the end-to-end normalization check
#   Step 0                  the input: the compressed partition, sliced into one matrix per site and sector
#   Step 1                  families of partial words and the cross targets -½ log|Z(w)|
#   Step 2                  pivot selection by a column-pivoted QR
#   Step 3                  the nested cross: alternating prefix and suffix sweeps, cut and site blocks
#   Step 4                  reconstruction of the filter cores from the blocks (CUR)
#   Step 5                  entry points
#
# Dimensions (docs, table "Dimensions and symbols"): q number of disorder sectors (labels p = 1..q), D_{Z,i} bond of the
# compressed partition, D_{f,i} bond of the filter (the filter ranks), both capped by D_max = max_rank.
# Conventions:
#   G_i[p] : ℂ^{D_{Z,i-1}} ← ℂ^{D_{Z,i}}       matrix of site i, sector p of the compressed partition
#   prefix family C : ℂ^{|I|} ← ℂ^{D_{Z,i}}    one row per partial word on sites 1..i (like the SRC prefix sketches)
#   suffix family R : ℂ^{D_{Z,i}} ← ℂ^{|J|}    one column per partial word on sites i+1..L
#   filter core F_i[p] : ℂ^{D_{f,i-1}} ← ℂ^{D_{f,i}}
# Magnitude policy: the phase of every queried Z(w) is recorded, then -½ log|Z(w)| is used; only zero or nonfinite
# queries fail.

#==========================================================================================
 The filter object
==========================================================================================#

# Filter: opp[i][p] is a matrix ℂ^{D_{f,i-1}} ← ℂ^{D_{f,i}} at site i for disorder sector p, representing
# f(w) = exp(log_scale) * opp[1][w₁] ⋯ opp[L][w_L]
struct FiniteDisorderFilter{T<:AbstractBondTensor}
    opp::Vector{Vector{T}}
    log_scale::Float64
end

Base.getindex(F::FiniteDisorderFilter, i::Int) = F.opp[i]
Base.getindex(F::FiniteDisorderFilter, i::Int, p::Int) = F.opp[i][p]
Base.length(F::FiniteDisorderFilter) = length(F.opp)
bond_dimensions(F::FiniteDisorderFilter) = [dim(space(F[i,1], 2)) for i in 1:length(F)-1]

# Multiply every conditional state ψ_w with f(w): the bonds of ψ and F are fused, the scale is spread over the sites
function Base.:*(F::FiniteDisorderFilter, ψ::FiniteDisorderMPS)
    (length(ψ) == length(F)) || throw(ArgumentError("F should have the same length as ψ"))
    factor = exp(F.log_scale/length(ψ))
    opp = map(1:length(ψ)) do i
        map(enumerate(ψ[i])) do (p, W)
            f = F[i, p]
            isol = isomorphism(ComplexF64, fuse(space(W, 1), space(f, 1)), space(W, 1) ⊗ space(f, 1))
            isor = isomorphism(ComplexF64, fuse(space(W, 3)', space(f, 2)'), space(W, 3)' ⊗ space(f, 2)')
            @tensor Wnew[-1 -2; -4] := factor * isol[-1; 1 2] * W[1 -2; 4] * f[2; 5] * conj(isor[-4; 4 5])
        end
    end
    return FiniteDisorderMPS(opp, ψ.ps, ψ.sectors)
end

# log|f(w)| of the filter for word w: a row vector v: ℂ^1 ← ℂ^r swept through the sites, rescaled to max |entry| 1
function filter_log_amplitude(F::FiniteDisorderFilter, word::AbstractVector{Int})
    v = ones(ComplexF64, ℂ^1 ← ℂ^1)
    logv = F.log_scale
    for (i, p) in enumerate(word)
        v = v * F[i, p]
        m = LinearAlgebra.norm(v, Inf)
        m == 0 && return -Inf
        v = v / m
        logv += log(m)
    end
    return log(abs(tr(v))) + logv
end

#==========================================================================================
 Conditional norms: log Z(w) of a FiniteDisorderMPS, word by word
==========================================================================================#

# Scaled norm environment (E, log) of one word: the true environment is exp(log) E.
# Left environments E[bra; ket] contain the sites left of a bond, right environments E[ket; bra] the sites right of it.
_boundary_environment(V) = (id(ComplexF64, V), 0.0)

# Grow an environment by one site with MPS tensor A[a s; b], rescaled to max |entry| 1. A zero environment stays zero
function _advance(env::Tuple{AbstractTensorMap, Float64}, A::AbstractTensorMap; right::Bool = false)
    E, logE = env
    if right
        @tensor E′[-1; -2] := A[-1 3; 1] * conj(A[-2 3; 2]) * E[1; 2]
    else
        @tensor E′[-2; -1] := A[1 3; -1] * conj(A[2 3; -2]) * E[2; 1]
    end
    m = LinearAlgebra.norm(E′, Inf)
    (m == 0 || logE == -Inf) && return E′, -Inf
    return E′ / m, logE + log(m)
end

# log|Z(w)| of the conditional state of word w
function conditional_log_norm(ψ::FiniteDisorderMPS, word::AbstractVector{Int})
    env = _boundary_environment(space(ψ[1, 1], 1))
    for (i, p) in enumerate(word)
        env = _advance(env, ψ[i, p])
    end
    E, logE = env
    return log(abs(tr(E))) + logE
end

# Normalization residuals log|Z(w)| (+ 2 log|f(w)| if a filter is given) for a list of words; zero when normalized.
# The words are independent and evaluated on all Julia threads
function normalization_residuals(ψ::FiniteDisorderMPS, words; F::Union{Nothing, FiniteDisorderFilter} = nothing)
    words = collect(words)
    residuals = Vector{Float64}(undef, length(words))
    Threads.@threads for n in eachindex(words)
        r = conditional_log_norm(ψ, words[n])
        residuals[n] = isnothing(F) ? r : r + 2*filter_log_amplitude(F, words[n])
    end
    return residuals
end

#==========================================================================================
 Step 0: the input, the compressed partition
==========================================================================================#

# Compressed partition function: exp(log_scale) * cores[1][:, w₁, :] ⋯ cores[L][:, w_L, :] = √P(w) Z(w),
# with MPS cores c: ℂ^{D_{Z,i-1}} ⊗ (ℂ^{q_i})' ← ℂ^{D_{Z,i}} (sector u in the middle), ps[i] the probabilities of the q_i
# sectors of site i
struct ScalarPartition{T<:AbstractTensorMap}
    cores::Vector{T}
    log_scale::Float64
    ps::Vector{Vector{Float64}}
end

Base.length(Z::ScalarPartition) = length(Z.cores)
bond_dimensions(Z::ScalarPartition) = [dim(space(c, 3)) for c in Z.cores[1:end-1]]

# Sector matrices G[i][p]: ℂ^{D_{Z,i-1}} ← ℂ^{D_{Z,i}} of the partition, every site divided by its largest entry.
# Returns (G, λ) with √P(w) Z(w) = exp(λ) G[1][w₁] ⋯ G[L][w_L]
function sector_matrices(Z::ScalarPartition)
    λ = Z.log_scale
    G = map(enumerate(Z.cores)) do (i, c)
        q = length(Z.ps[i])
        map(1:q) do p
            e = Tensor([j == p ? 1.0 : 0.0 for j in 1:q], ℂ^q)   # basis ket of sector p
            @tensor g[r; q] := c[r x; q] * e[x]
        end
    end
    for (i, Gi) in enumerate(G)
        m = maximum(g -> LinearAlgebra.norm(g, Inf), Gi)
        (isfinite(m) && m > 0) || throw(DomainError(m, "unusable partition core at site $i"))
        G[i] = Gi ./ m
        λ += log(m)
    end
    return G, λ
end

#==========================================================================================
 Step 1: families of partial words and the cross targets
==========================================================================================#

# A family of partial words as one matrix M: rows ℂ^{|I|} ← ℂ^{D_{Z,i}} (prefixes, sites 1..i) or columns ℂ^{D_{Z,i}} ← ℂ^{|J|}
# (suffixes, sites i+1..L), with per word the log scale of its row/column and ½ Σ log p of its labels
struct WordFamily{T<:AbstractTensorMap}
    M::T
    logs::Vector{Float64}
    halflogp::Vector{Float64}
end

family_size(fam::WordFamily) = length(fam.logs)

# The empty word, on both boundaries
empty_family() = WordFamily(ones(ComplexF64, ℂ^1 ← ℂ^1), [0.0], [0.0])

# Rescale factors of rows with largest entry s: only finite nonzero rows are rescaled (zero rows stay zero and fail
# only when queried)
_safe_scales(s) = [(x > 0 && isfinite(x)) ? x : 1.0 for x in s]

# All one-site extensions (prefix, p) of the prefixes on site i, sectors outermost: rows of C G_i[p], stacked
function extend_prefix_family(fam::WordFamily, Gi::Vector, halflogp::Vector{Float64})
    M = reduce(catcodomain, [fam.M * g for g in Gi])
    s = _safe_scales(row_scales(M))
    M = DiagonalTensorMap(1 ./ s, ℂ^length(s)) * M
    return WordFamily(M, repeat(fam.logs, length(Gi)) .+ log.(s), reduce(vcat, [fam.halflogp .+ h for h in halflogp]))
end

# All one-site extensions (p, suffix) of the suffixes on site i, sectors outermost: columns of G_i[p] R, side by side
function extend_suffix_family(fam::WordFamily, Gi::Vector, halflogp::Vector{Float64})
    M = reduce(catdomain, [g * fam.M for g in Gi])
    s = _safe_scales(vec(maximum(abs, convert(Array, M); dims = 1)))   # largest |entry| of every column
    M = M * DiagonalTensorMap(1 ./ s, ℂ^length(s))
    return WordFamily(M, repeat(fam.logs, length(Gi)) .+ log.(s), reduce(vcat, [fam.halflogp .+ h for h in halflogp]))
end

# Keep the words idx of a prefix (rows) or suffix (columns) family
select_prefixes(fam::WordFamily, idx) = WordFamily(select_rows(idx, family_size(fam)) * fam.M, fam.logs[idx], fam.halflogp[idx])
select_suffixes(fam::WordFamily, idx) = WordFamily(fam.M * select_rows(idx, family_size(fam))', fam.logs[idx], fam.halflogp[idx])

# Phase diagnostics of all queried partition entries
mutable struct CrossDiagnostics
    examined::Int           # number of queried entries
    max_phase::Float64      # maximal |arg Z(w)|
    negative_real::Int      # entries with Re Z(w) < 0
    large_phase::Int        # entries with |arg Z(w)| > π/2
end
CrossDiagnostics() = CrossDiagnostics(0, 0.0, 0, 0)

# Cross targets -½ log|Z(w)| for every pair (prefix a, suffix b), w = C[a] ⊕ R[b], from one matrix product.
# The prior √P(w) is removed once per label through halflogp
function log_targets(λ::Float64, C::WordFamily, R::WordFamily, diag::CrossDiagnostics)
    overlap = convert(Array, C.M * R.M)
    offsets = λ .+ (C.logs .- C.halflogp) .+ transpose(R.logs .- R.halflogp)
    phase = angle.(overlap)
    diag.examined += length(overlap)
    diag.max_phase = max(diag.max_phase, maximum(abs, phase))
    diag.negative_real += count(<(0), real.(overlap))
    diag.large_phase += count(>(π/2), abs.(phase))
    all(isfinite, overlap) && all(isfinite, offsets) && !any(iszero, overlap) || throw(DomainError(overlap, "zero or nonfinite partition query"))
    return -0.5 .* (offsets .+ log.(abs.(overlap)))
end

#==========================================================================================
 Step 2: pivot selection
==========================================================================================#

# Select up to width rows of the target exp(logs) by a column-pivoted QR of its transpose (greedy volume maximization;
# MatrixAlgebraKit has no pivoted QR, so LinearAlgebra is used on the small target matrix)
function pivot_rows(logs::AbstractMatrix, width::Int)
    target = exp.(logs .- maximum(logs))
    all(isfinite, target) && !any(iszero, target) || throw(DomainError(logs, "pivot target exceeds exponential range"))
    pivots = qr(transpose(target), ColumnNorm()).p
    return pivots[1:min(width, length(pivots))]
end

#==========================================================================================
 Step 3: the nested cross
==========================================================================================#

# Nested QR-pivoted cross of the targets -½ log|Z(w)| of a compressed partition function.
# left[i+1] holds the prefixes on sites 1..i, right[i+1] the suffixes on sites i+1..L. Every family is built by
# extending its neighbour, so the families are nested. Returns the cut blocks H[i][a, b] = -½ log|Z(left_i[a] ⊕ right_i[b])|
# for i = 1..L-1, the site blocks K[i][p][a, c] = -½ log|Z(left_{i-1}[a] ⊕ p ⊕ right_i[c])| for i = 1..L and the phase
# diagnostics
function collect_nested_cross(Z::ScalarPartition; width::Int = 32, cycles::Int = 2, rng::AbstractRNG = Random.default_rng(),
                              to::TimerOutput = TimerOutput())
    G, λ = sector_matrices(Z)
    h = [0.5 .* log.(p) for p in Z.ps]
    L = length(Z)
    diag = CrossDiagnostics()
    targets(C, R) = log_targets(λ, C, R, diag)

    # Random initial suffixes, from right to left
    left = Vector{WordFamily}(undef, L)
    right = Vector{WordFamily}(undef, L+1)
    left[1], right[L+1] = empty_family(), empty_family()
    @timeit to "initial_suffixes" for i in L-1:-1:1
        candidates = extend_suffix_family(right[i+2], G[i+1], h[i+1])
        n = family_size(candidates)
        right[i+1] = select_suffixes(candidates, randperm(rng, n)[1:min(width, n)])
    end

    for _ in 1:cycles
        # Prefix sweep against the current suffixes
        @timeit to "prefix_sweep" for i in 1:L-1
            candidates = extend_prefix_family(left[i], G[i], h[i])
            left[i+1] = select_prefixes(candidates, pivot_rows(targets(candidates, right[i+1]), width))
        end
        # Suffix sweep against the new prefixes
        @timeit to "suffix_sweep" for i in L-1:-1:1
            candidates = extend_suffix_family(right[i+2], G[i+1], h[i+1])
            right[i+1] = select_suffixes(candidates, pivot_rows(transpose(targets(left[i+1], candidates)), width))
        end
    end

    @timeit to "cut_blocks" H = [targets(left[i+1], right[i+1]) for i in 1:L-1]
    @timeit to "site_blocks" K = map(1:L) do i
        n = family_size(left[i])
        logs = targets(extend_prefix_family(left[i], G[i], h[i]), right[i+1])
        [logs[(p-1)*n .+ (1:n), :] for p in eachindex(G[i])]
    end
    return H, K, diag
end

#==========================================================================================
 Step 4: reconstruction of the filter cores (CUR)
==========================================================================================#

# Filter f ≈ K₁ H₁⁺ K₂ ⋯ H_{L-1}⁺ K_L from the cross blocks. All blocks are exponentiated with one common shift σ,
# which is restored in log_scale. Each cut block is truncated to the smallest rank whose relative discarded tail is at
# most tail_tol (capped by max_rank and the numerical support s > 1e-12 s₁); discarded singular values are never
# inverted. Returns the filter, its ranks and whether every cut met the tail tolerance
function reconstruct_filter(H::Vector{<:AbstractMatrix}, K::Vector{<:Vector{<:AbstractMatrix}}; max_rank::Int = 32,
                            tail_tol::Real = 1e-8, to::TimerOutput = TimerOutput())
    L = length(K)
    σ = maximum(maximum(maximum, Ki) for Ki in K)
    (L > 1) && (σ = max(σ, maximum(maximum, H)))
    function shifted(b)
        v = exp.(b .- σ)
        all(isfinite, v) && !any(iszero, v) || throw(DomainError(b, "cross targets exceed exponential range"))
        return TensorMap(v, ℂ^size(b, 1) ← ℂ^size(b, 2))
    end

    trunc = truncrank(max_rank) & truncerror(; rtol = tail_tol) & trunctol(; rtol = 1e-12, p = Inf)
    @timeit to "svd_cuts" cuts = map(H) do b
        Hb = shifted(b)
        U, S, Vh, ϵ = svd_trunc(Hb; trunc = trunc)
        (left = inv(S) * U', right = Vh', met = ϵ <= tail_tol * LinearAlgebra.norm(Hb))   # H⁺ = Vh' S⁻¹ U'
    end

    # F_i[p] = S_{i-1}⁻¹ U_{i-1}' K̃_i[p] V_i
    @timeit to "cores" opp = map(1:L) do i
        map(K[i]) do k
            core = shifted(k)
            (i > 1) && (core = cuts[i-1].left * core)
            (i < L) && (core = core * cuts[i].right)
            isfinite(LinearAlgebra.norm(core)) || throw(DomainError(core, "filter core exceeds floating point range"))
            complex(core)
        end
    end
    ranks = [dim(domain(c.right)) for c in cuts]
    return FiniteDisorderFilter(opp, σ), ranks, all(c.met for c in cuts)
end

#==========================================================================================
 Step 5: entry points
==========================================================================================#

# Normalization filter f(w) ≈ |Z(w)|^(-1/2) from a compressed partition function. Returns the filter and a NamedTuple
# with its bond dimensions, whether every cut met tail_tol and the phase diagnostics of the cross
function normalization_filter(Z::ScalarPartition; width::Int = 32, cycles::Int = 2, max_rank::Int = 32, tail_tol::Real = 1e-8,
                              rng::AbstractRNG = Random.default_rng(), to::TimerOutput = TimerOutput())
    @timeit to "collect_nested_cross" H, K, diag = collect_nested_cross(Z; width = width, cycles = cycles, rng = rng, to = to)
    @timeit to "reconstruct_filter" F, ranks, met = reconstruct_filter(H, K; max_rank = max_rank, tail_tol = tail_tol, to = to)
    return F, (filter_ranks = ranks, filter_met = met, cross = diag)
end

# Normalization filter of the evolved state Us ψ (or of ψ itself), compress first: SRC of √P Z with rank cap max_rank
# and relative tolerance partition_rtol, then the cross of the compressed partition
function normalization_filter(ψ::FiniteDisorderMPS, Us::Union{Nothing, FiniteDisorderMPO} = nothing; max_rank::Int = 32,
                              partition_rtol::Real = 1e-10, rank_threshold::Real = 1e-13, width::Int = 32, cycles::Int = 2,
                              tail_tol::Real = 1e-8, rng::AbstractRNG = Random.default_rng(), to::TimerOutput = TimerOutput())
    @timeit to "src_partition" Z, zinfo = isnothing(Us) ?
        src_partition(ψ; max_rank = max_rank, rtol = partition_rtol, rank_threshold = rank_threshold, rng = rng, to = to) :
        src_partition(ψ, Us; max_rank = max_rank, rtol = partition_rtol, rank_threshold = rank_threshold, rng = rng, to = to)
    F, finfo = normalization_filter(Z; width = width, cycles = cycles, max_rank = max_rank, tail_tol = tail_tol, rng = rng, to = to)
    return F, (finfo..., partition_ranks = bond_dimensions(Z), partition_met = zinfo.target_met)
end
