# Disorder normalization filter of a FiniteDisorderMPS.
# For every disorder word w = (p₁…p_L) the state defines a conditional state ψ_w with norm Z(w) = ⟨ψ_w|ψ_w⟩.
# The filter is a scalar disorder MPS f(w) ≈ |Z(w)|^(-1/2) such that f ⊙ ψ has Z(w) = 1 for every word.
# It is built compress first: √P(w) Z(w) is compressed with SRC (FiniteSRC.jl, from the separate state and evolution
# layers), and a nested QR-pivoted cross samples the cheap scalar partition. The prior is removed once per label.
# Magnitude policy: the phase of every queried Z(w) is recorded, then -½ log|Z(w)| is used; only zero or
# nonfinite queries fail.

# Filter: opp[i][p] is a matrix ℂ^{r_{i-1}} ← ℂ^{r_i} at site i for disorder sector p, representing
# f(w) = exp(log_scale) * opp[1][p₁] ⋯ opp[L][p_L]
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
    return FiniteDisorderMPS(opp, ψ.ps)
end

# Dense tensors A[i][p][vL, s, vR] of a FiniteDisorderMPS
_dense(ψ::FiniteDisorderMPS) = [[convert(Array, W) for W in ψ[i]] for i in 1:length(ψ)]

# Grow a scaled conditional norm environment (env, log) by one site with tensor A[vL, s, vR].
# Left environments env[bra, ket] contain the sites left of a bond, right environments the sites right of it.
# Both the tensor and the result are rescaled by their maximal entry, the logarithms are accumulated.
function _advance(env::Tuple{Matrix{ComplexF64}, Float64}, A::Array{ComplexF64, 3}; right::Bool = false)
    return _advance(env, A, maximum(abs, A); right = right)
end

# _advance with a precomputed amplitude max|A| (only used to detect vanishing tensors), as two matrix products on reshaped tensors
function _advance(env::Tuple{Matrix{ComplexF64}, Float64}, A::Array{ComplexF64, 3}, amplitude::Float64; right::Bool = false)
    E, logE = env
    na, ns, nb = size(A)
    if amplitude == 0 || logE == -Inf
        n = right ? na : nb
        return zeros(ComplexF64, n, n), -Inf
    end
    if right
        # value[a, c] = Σ conj(A[a,s,b]) E[b,d] A[c,s,d]
        X = reshape(A, na*ns, nb) * transpose(E)
        value = conj(reshape(A, na, ns*nb)) * transpose(reshape(X, na, ns*nb))
    else
        # value[b, d] = Σ conj(A[a,s,b]) E[a,c] A[c,s,d]
        X = E * reshape(A, na, ns*nb)
        value = reshape(A, na*ns, nb)' * reshape(X, na*ns, nb)
    end
    scale = maximum(abs, value)
    scale == 0 && return value, -Inf
    value ./= scale
    # value is computed from the unscaled A, so its scale already contains the amplitude
    return value, logE + log(scale)
end

const _boundary_environment = (ones(ComplexF64, 1, 1), 0.0)

# log|Z(w)| of the conditional state of word w
function conditional_log_norm(A::Vector{<:Vector{<:Array{ComplexF64, 3}}}, word::AbstractVector{Int})
    env = _boundary_environment
    for (i, p) in enumerate(word)
        env = _advance(env, A[i][p])
    end
    return log(abs(env[1][1])) + env[2]
end
conditional_log_norm(ψ::FiniteDisorderMPS, word::AbstractVector{Int}) = conditional_log_norm(_dense(ψ), word)

# log|f(w)| of the filter for word w
function filter_log_amplitude(F::FiniteDisorderFilter, word::AbstractVector{Int})
    v = ones(ComplexF64, 1, 1)
    logv = F.log_scale
    for (i, p) in enumerate(word)
        v = v * convert(Array, F[i, p])
        scale = maximum(abs, v)
        scale == 0 && return -Inf
        v ./= scale
        logv += log(scale)
    end
    return log(abs(v[1])) + logv
end

# Normalization residuals log|Z(w)| (+ 2 log|f(w)| if a filter is given) for a list of words; zero when normalized.
# The words are independent and evaluated on all Julia threads
function normalization_residuals(ψ::FiniteDisorderMPS, words; F::Union{Nothing, FiniteDisorderFilter} = nothing)
    A = _dense(ψ)
    words = collect(words)
    residuals = Vector{Float64}(undef, length(words))
    Threads.@threads for n in eachindex(words)
        r = conditional_log_norm(A, words[n])
        residuals[n] = isnothing(F) ? r : r + 2*filter_log_amplitude(F, words[n])
    end
    return residuals
end

# Compressed partition function: exp(log_scale) * cores[1][:, w₁, :] ⋯ cores[L][:, w_L, :] = √P(w) Z(w)
struct ScalarPartition
    cores::Vector{Array{ComplexF64, 3}}
    log_scale::Float64
    ps::Vector{Float64}
end

Base.length(Z::ScalarPartition) = length(Z.cores)
bond_dimensions(Z::ScalarPartition) = [size(c, 3) for c in Z.cores[1:end-1]]

# Phase diagnostics of all queried partition entries
mutable struct CrossDiagnostics
    examined::Int           # number of queried entries
    max_phase::Float64      # maximal |arg Z(w)|
    negative_real::Int      # entries with Re Z(w) < 0
    large_phase::Int        # entries with |arg Z(w)| > π/2
end
CrossDiagnostics() = CrossDiagnostics(0, 0.0, 0, 0)

# Scalar partition queries with scaled cores. Environments are (v, log scale, ½ Σ log p of their labels),
# left environments are row vectors v cores[i][:, p, :], right environments column vectors cores[i][:, p, :] v
struct _PartitionEvaluator
    cores::Vector{Array{ComplexF64, 3}}
    scale::Float64
    halflogp::Vector{Float64}
end

function _PartitionEvaluator(Z::ScalarPartition)
    scale = Z.log_scale
    cores = similar(Z.cores)
    for (i, c) in enumerate(Z.cores)
        m = maximum(abs, c)
        (all(isfinite, c) && m > 0) || throw(DomainError(m, "unusable partition core at site $i"))
        cores[i] = c ./ m
        scale += log(m)
    end
    return _PartitionEvaluator(cores, scale, 0.5 .* log.(Z.ps))
end

const _partition_boundary = (ones(ComplexF64, 1), 0.0, 0.0)

# Extend an environment by site i in sector p. Zero environments stay zero and only fail when queried
function _update(ev::_PartitionEvaluator, env::Tuple{Vector{ComplexF64}, Float64, Float64}, i::Int, p::Int; right::Bool = false)
    v, s, h = env
    c = view(ev.cores[i], :, p, :)
    w = right ? c * v : transpose(c) * v
    m = maximum(abs, w)
    if m > 0 && isfinite(m)
        w ./= m
        s += log(m)
    end
    return (w, s, h + ev.halflogp[p])
end

# Candidate extensions (parent, sector) of a family of environments by site i, sectors outermost
function _extend(ev::_PartitionEvaluator, family::AbstractVector, i::Int; right::Bool = false)
    pairs = [(a, p) for p in eachindex(ev.halflogp) for a in eachindex(family)]
    return pairs, [_update(ev, family[a], i, p; right = right) for (a, p) in pairs]
end

# Log targets -½ log|Z(w)| for every pair (left environment, right environment)
function _join_logs(ev::_PartitionEvaluator, diag::CrossDiagnostics, left::AbstractVector, right::AbstractVector)
    overlap = transpose(reduce(hcat, first.(left))) * reduce(hcat, first.(right))
    offsets = ev.scale .+ [s - h for (_, s, h) in left] .+ transpose([s - h for (_, s, h) in right])
    phase = angle.(overlap)
    diag.examined += length(overlap)
    diag.max_phase = max(diag.max_phase, maximum(abs, phase))
    diag.negative_real += count(<(0), real.(overlap))
    diag.large_phase += count(>(π/2), abs.(phase))
    all(isfinite, overlap) && all(isfinite, offsets) && !any(iszero, overlap) || throw(DomainError(overlap, "zero or nonfinite partition query"))
    return -0.5 .* (offsets .+ log.(abs.(overlap)))
end

# Select up to width rows of the target exp(logs) by a column-pivoted QR of its transpose
function _pivot_rows(logs::AbstractMatrix, width::Int)
    target = exp.(logs .- maximum(logs))
    all(isfinite, target) && !any(iszero, target) || throw(DomainError(logs, "pivot target exceeds exponential range"))
    pivots = qr(transpose(target), ColumnNorm()).p
    return pivots[1:min(width, length(pivots))]
end

# Nested QR-pivoted cross of the targets -½ log|Z(w)| of a compressed partition function.
# Prefix family P[i+1] holds words on sites 1..i, suffix family Q[i+1] words on sites i+1..L, as (parent, sector) tables.
# Returns the cut blocks H[i][a, b] = -½ log|Z(P_i[a] ⊕ Q_i[b])| for i = 1..L-1,
# the site blocks K[i][p][a, c] = -½ log|Z(P_{i-1}[a] ⊕ p ⊕ Q_i[c])| for i = 1..L and the phase diagnostics
function collect_nested_cross(Z::ScalarPartition; width::Int = 32, cycles::Int = 2, rng::AbstractRNG = Random.default_rng(), to::TimerOutput = TimerOutput())
    ev = _PartitionEvaluator(Z)
    diag = CrossDiagnostics()
    L = length(Z)
    d = length(Z.ps)
    boundary = [_partition_boundary]
    join(l, r) = _join_logs(ev, diag, l, r)

    # Random initial suffixes, from right to left
    right = Vector{Any}(undef, L+1)
    right[L+1] = boundary
    @timeit to "initial_suffixes" for i in L-1:-1:1
        _, envs = _extend(ev, right[i+2], i+1; right = true)
        right[i+1] = envs[randperm(rng, length(envs))[1:min(width, length(envs))]]
    end

    left = Vector{Any}(undef, L+1)
    P = Vector{Vector{Tuple{Int,Int}}}(undef, L+1)
    Q = Vector{Vector{Tuple{Int,Int}}}(undef, L+1)
    for _ in 1:cycles
        # Prefix sweep against the current suffixes
        left[1] = boundary
        @timeit to "prefix_sweep" for i in 1:L-1
            pairs, envs = _extend(ev, left[i], i)
            selection = _pivot_rows(join(envs, right[i+1]), width)
            P[i+1], left[i+1] = pairs[selection], envs[selection]
        end
        # Suffix sweep against the new prefixes
        right[L+1] = boundary
        @timeit to "suffix_sweep" for i in L-1:-1:1
            pairs, envs = _extend(ev, right[i+2], i+1; right = true)
            selection = _pivot_rows(transpose(join(left[i+1], envs)), width)
            Q[i+1], right[i+1] = pairs[selection], envs[selection]
        end
    end

    # Rebuild exact environments from the frozen index tables and collect the blocks once
    @timeit to "rebuild_environments" begin
        left[1] = boundary
        for i in 1:L-1
            left[i+1] = [_update(ev, left[i][a], i, p) for (a, p) in P[i+1]]
        end
        right[L+1] = boundary
        for i in L-1:-1:1
            right[i+1] = [_update(ev, right[i+2][a], i+1, p; right = true) for (a, p) in Q[i+1]]
        end
    end
    @timeit to "cut_blocks" H = [join(left[i+1], right[i+1]) for i in 1:L-1]
    @timeit to "site_blocks" K = map(1:L) do i
        nl = length(left[i])
        _, envs = _extend(ev, left[i], i)
        logs = join(envs, right[i+1])
        [logs[(p-1)*nl .+ (1:nl), :] for p in 1:d]
    end
    return H, K, diag
end

# Smallest rank whose relative discarded Frobenius tail is at most tail_tol, capped by max_rank and the
# numerical support s > 1e-12 s₁. Tails are reverse cumulative sums. Returns the rank and whether the target is met.
function select_tail_rank(S::AbstractVector{<:Real}, max_rank::Int, tail_tol::Real)
    squares = (S ./ S[1]).^2
    remaining = [reverse(cumsum(reverse(squares))); 0.0]
    tails = sqrt.(remaining ./ remaining[1])
    support = count(>(1e-12*S[1]), S)
    available = min(max_rank, length(S), support)
    r = findfirst(<=(tail_tol), tails[2:available+1])
    rank = isnothing(r) ? available : r
    return rank, tails[rank+1] <= tail_tol
end

# Filter f ≈ K₁ H₁⁺ K₂ ⋯ H_{L-1}⁺ K_L from the cross blocks (CUR form). All blocks are exponentiated with one
# common shift, which is restored in log_scale. Cuts are truncated with select_tail_rank, discarded singular
# values are never inverted. Returns the filter, its ranks and whether every cut met the tail tolerance.
function reconstruct_filter(H::Vector{<:AbstractMatrix}, K::Vector{<:Vector{<:AbstractMatrix}}; max_rank::Int = 32, tail_tol::Real = 1e-8, to::TimerOutput = TimerOutput())
    L = length(K)
    σ = maximum(maximum(maximum, Ki) for Ki in K)
    (L > 1) && (σ = max(σ, maximum(maximum, H)))
    expshift(b) = (v = exp.(b .- σ); all(isfinite, v) && !any(iszero, v) || throw(DomainError(b, "cross targets exceed exponential range")); v)

    @timeit to "svd_cuts" gauges = map(H) do h
        F = svd(expshift(h))
        r, met = select_tail_rank(F.S, max_rank, tail_tol)
        (F.U[:, 1:r], F.S[1:r], F.V[:, 1:r], met)
    end

    @timeit to "cores" opp = map(1:L) do i
        l = (i == 1) ? ones(1, 1) : transpose(gauges[i-1][1]) ./ gauges[i-1][2]
        r = (i == L) ? ones(1, 1) : gauges[i][3]
        map(K[i]) do k
            core = l * expshift(k) * r
            all(isfinite, core) || throw(DomainError(core, "filter core exceeds floating point range"))
            TensorMap(complex(core), ℂ^size(core, 1), ℂ^size(core, 2))
        end
    end
    ranks = [length(g[2]) for g in gauges]
    return FiniteDisorderFilter(opp, σ), ranks, all(g[4] for g in gauges)
end

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
