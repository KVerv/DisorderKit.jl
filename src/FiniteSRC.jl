# Successive randomized compression (SRC, Camaño-Epperly-Tropp, Algorithm 1 with the adaptive rank of Appendix C)
# of disorder MPS that are only given implicitly by their separate layers: state A, evolution MPO W and filter f
# are never fused. Two sources are compressed:
#   partition  √P(w) Z(w), Z(w) = ‖W_w A_w‖², a scalar disorder MPS with physical index the sector u, built from
#              the four layers (A, W, W̄, Ā)
#   physical   Σ_w √P(w) |w⟩ ⊗ f(w) W_w|A_w⟩, an MPS with joint physical index x = (s, u) (s fastest), built from
#              the three layers (A, W, f)
# The √p are divided out of the physical result again and the absolute scale is restored, such that the result is
# an (unweighted, non-isometric) FiniteDisorderMPS.

# Products implement the source for the generic engine _src_run:
#   _left_shape(P, i)          the pair of left bond dimensions of the prefix sketches C[l1, l2, k] at site i
#   _physical_dim(P, i)        physical dimension of site i
#   _prefix(P, i, C, Ω)        prefixes of site i+1 from the prefixes C of site i and Gaussian vectors Ω[x, k]
#   _contract_suffix(P, i, S)  site i contracted with the compressed suffix S[r1, r2, q]: J[l1, l2, x, q]
# Sketch, projection and the first core are generic matrix products of the cut-local J, which is reused for both.
# Dense layer conventions (sector u last): A[a, t, b, u], W[h, s, t, j, u] (s output, t input), f[x, y, u].

# Dense layers of the state, the evolution MPO and the filter with the sector as last index
_stack(Ts) = cat([ComplexF64.(convert(Array, T)) for T in Ts]...; dims = numind(first(Ts))+1)
_dense_layers(ψ::FiniteDisorderMPS) = [_stack(ψ[i]) for i in 1:length(ψ)]
_dense_layers(O::FiniteDisorderMPO) = [_stack(O[i]) for i in 1:length(O)]
_dense_layers(F::FiniteDisorderFilter) = [_stack(F[i]) for i in 1:length(F)]

# Identity MPO layers W[1, s, t, 1, u] = δ_st, for the partition function of an unevolved state
function _identity_layers(ψ::FiniteDisorderMPS)
    ns, n = dim(space(ψ[1, 1], 2)), length(ψ.ps)
    W = repeat(reshape(Matrix{ComplexF64}(I, ns, ns), 1, ns, ns, 1), 1, 1, 1, 1, n)
    return [W for _ in 1:length(ψ)]
end

# Rescale every site by its maximal entry, the logarithms are accumulated
function _scaled(Xs::Vector{<:Array{ComplexF64}})
    log_scale = 0.0
    scaled = similar(Xs)
    for (i, X) in enumerate(Xs)
        m = maximum(abs, X)
        all(isfinite, X) || throw(DomainError(m, "nonfinite layer"))
        if m > 0
            X = X ./ m
            log_scale += log(m)
        end
        scaled[i] = X
    end
    return scaled, log_scale
end

# √P Z of W A from the four layers A, W, W̄, Ā. Left bond (a, h | c, l): ket A and W, bra Ā and W̄
struct _EvolvedPartitionProduct
    A::Vector{Array{ComplexF64, 4}}
    W::Vector{Array{ComplexF64, 5}}
    g::Vector{Float64}
end

Base.length(P::_EvolvedPartitionProduct) = length(P.A)
_left_shape(P::_EvolvedPartitionProduct, i::Int) = (n = size(P.A[i], 1)*size(P.W[i], 1); (n, n))
_physical_dim(P::_EvolvedPartitionProduct, i::Int) = length(P.g)

function _prefix(P::_EvolvedPartitionProduct, i::Int, C::Array{ComplexF64, 3}, Ω::Matrix{ComplexF64})
    na, nb, nh, nj, K = size(P.A[i], 1), size(P.A[i], 3), size(P.W[i], 1), size(P.W[i], 4), size(C, 3)
    e = reshape(C, na, nh, na, nh, K)
    out = zeros(ComplexF64, nb, nj, nb, nj, K)
    for (u, g) in enumerate(P.g)
        Au, Wu = P.A[i][:, :, :, u], P.W[i][:, :, :, :, u]
        @tensor X[b, j, d, m, k] := (((e[a, h, c, l, k] * Au[a, t, b]) * Wu[h, s, t, j]) * conj(Au[c, v, d])) * conj(Wu[l, s, v, m])
        out .+= X .* reshape(g .* Ω[u, :], 1, 1, 1, 1, K)
    end
    return reshape(out, nb*nj, nb*nj, K)
end

function _contract_suffix(P::_EvolvedPartitionProduct, i::Int, S::Array{ComplexF64, 3})
    na, nb, nh, nj, q = size(P.A[i], 1), size(P.A[i], 3), size(P.W[i], 1), size(P.W[i], 4), size(S, 3)
    r = reshape(S, nb, nj, nb, nj, q)
    J = zeros(ComplexF64, na, nh, na, nh, length(P.g), q)
    for (u, g) in enumerate(P.g)
        Au, Wu = P.A[i][:, :, :, u], P.W[i][:, :, :, :, u]
        @tensor X[a, h, c, l, q] := (((r[b, j, d, m, q] * Au[a, t, b]) * Wu[h, s, t, j]) * conj(Au[c, v, d])) * conj(Wu[l, s, v, m])
        J[:, :, :, :, u, :] .= g .* X
    end
    return reshape(J, na*nh, na*nh, length(P.g), q)
end

# √P f W A from the three layers A, W, f. Left bond (a | h, x), physical index (s, u) with s fastest
struct _EvolvedFilterProduct
    A::Vector{Array{ComplexF64, 4}}
    W::Vector{Array{ComplexF64, 5}}
    f::Vector{Array{ComplexF64, 3}}
    g::Vector{Float64}
end

Base.length(P::_EvolvedFilterProduct) = length(P.A)
_left_shape(P::_EvolvedFilterProduct, i::Int) = (size(P.A[i], 1), size(P.W[i], 1)*size(P.f[i], 1))
_physical_dim(P::_EvolvedFilterProduct, i::Int) = size(P.W[i], 2)*length(P.g)

function _prefix(P::_EvolvedFilterProduct, i::Int, C::Array{ComplexF64, 3}, Ω::Matrix{ComplexF64})
    na, nb, nh, ns, nj = size(P.A[i], 1), size(P.A[i], 3), size(P.W[i], 1), size(P.W[i], 2), size(P.W[i], 4)
    nx, ny, K = size(P.f[i], 1), size(P.f[i], 2), size(C, 3)
    e = reshape(C, na, nh, nx, K)
    Ωs = reshape(Ω, ns, length(P.g), K)
    out = zeros(ComplexF64, nb, nj, ny, K)
    for (u, g) in enumerate(P.g)
        Au, Wu, fu = P.A[i][:, :, :, u], P.W[i][:, :, :, :, u], P.f[i][:, :, u]
        @tensor G[b, j, x, s, k] := (e[a, h, x, k] * Au[a, t, b]) * Wu[h, s, t, j]
        Gω = dropdims(sum(G .* reshape(g .* Ωs[:, u, :], 1, 1, 1, ns, K); dims = 4); dims = 4)
        @tensor out[b, j, y, k] += Gω[b, j, x, k] * fu[x, y]
    end
    return reshape(out, nb, nj*ny, K)
end

function _contract_suffix(P::_EvolvedFilterProduct, i::Int, S::Array{ComplexF64, 3})
    na, nb, nh, ns, nj = size(P.A[i], 1), size(P.A[i], 3), size(P.W[i], 1), size(P.W[i], 2), size(P.W[i], 4)
    nx, ny, q = size(P.f[i], 1), size(P.f[i], 2), size(S, 3)
    r = reshape(S, nb, nj, ny, q)
    J = zeros(ComplexF64, na, nh, nx, ns, length(P.g), q)
    for (u, g) in enumerate(P.g)
        Au, Wu, fu = P.A[i][:, :, :, u], P.W[i][:, :, :, :, u], P.f[i][:, :, u]
        @tensor X[a, h, x, s, q] := ((r[b, j, y, q] * Au[a, t, b]) * Wu[h, s, t, j]) * fu[x, y]
        J[:, :, :, :, u, :] .= g .* X
    end
    return reshape(J, na, nh*nx, ns*length(P.g), q)
end


# Incremental sketch of one cut (C.3 with Householder QR updates, as in Finite010/011 of disorder_itebd).
# The sketch columns arrive in blocks with column log scales and are kept in the common log scale of the largest column.
# Only R and the Householder blocks (row offset, compact WY factorization of the remaining rows) are stored, Q is
# formed on demand. The inverse adjoint G = R^{-H} is updated per block while R is safely invertible.
mutable struct _IncrementalSketch
    rows::Int
    threshold::Float64
    r::Matrix{ComplexF64}
    blocks::Vector{Tuple{Int, LinearAlgebra.QRCompactWY{ComplexF64, Matrix{ComplexF64}, Matrix{ComplexF64}}}}
    nrefl::Int
    shift::Float64
    G::Union{Nothing, Matrix{ComplexF64}}
    rank::Int
    basis::Union{Nothing, Matrix{ComplexF64}}
end

_IncrementalSketch(rows::Int, threshold::Real) =
    _IncrementalSketch(rows, threshold, zeros(ComplexF64, 0, 0), [], 0, -Inf, nothing, 0, nothing)

_width(sk::_IncrementalSketch) = size(sk.r, 2)

# Append the sketch columns Y with column log scales logs
function _append!(sk::_IncrementalSketch, Y::AbstractMatrix{ComplexF64}, logs::AbstractVector{Float64})
    old, nnew = _width(sk), size(Y, 2)
    shift = max(sk.shift, maximum(logs))
    scale = old > 0 ? exp(sk.shift - shift) : 1.0
    factors = exp.(logs .- shift)
    (any(iszero, factors) || scale == 0) && throw(DomainError(shift, "Gaussian column scale exceeds numerical range"))
    sk.r .*= scale
    isnothing(sk.G) || (sk.G ./= scale)
    sk.shift = shift

    block = Y .* transpose(factors)
    for (offset, F) in sk.blocks
        lmul!(F.Q', view(block, offset+1:sk.rows, :))
    end
    k = sk.nrefl
    if k < sk.rows
        F = qr(block[k+1:end, :])
        added = min(sk.rows - k, nnew)
        push!(sk.blocks, (k, F))
        rr = F.R
    else
        added = 0
        rr = zeros(ComplexF64, 0, nnew)
    end
    R = zeros(ComplexF64, k + added, old + nnew)
    R[1:k, 1:old] .= sk.r
    R[1:k, old+1:end] .= view(block, 1:k, :)
    R[k+1:k+added, old+1:end] .= rr
    sk.r, sk.nrefl, sk.basis = R, k + added, nothing

    s = svdvals(R)
    sk.rank = (isempty(s) || s[1] == 0) ? 0 : count(>(sk.threshold*s[1]), s)
    width = old + nnew
    if sk.rank == width && width <= sk.rows
        if isnothing(sk.G) || old == 0
            sk.G = UpperTriangular(R)' \ Matrix{ComplexF64}(I, width, width)
        else
            inv = UpperTriangular(rr)' \ Matrix{ComplexF64}(I, nnew, nnew)
            lower = -inv * (view(block, 1:k, :)' * sk.G)
            sk.G = [sk.G zeros(ComplexF64, old, nnew); lower inv]
        end
        all(isfinite, sk.G) || throw(DomainError(sk.G, "nonfinite SRC inverse adjoint"))
    else
        sk.G = nothing
    end
    return sk
end

# Thin orthonormal basis Q (rows × nrefl) of the sketch, including the completion of deficient sketches
function _basis(sk::_IncrementalSketch)
    if isnothing(sk.basis)
        Q = Matrix{ComplexF64}(I, sk.rows, sk.nrefl)
        for (offset, F) in Iterators.reverse(sk.blocks)
            lmul!(F.Q, view(Q, offset+1:sk.rows, :))
        end
        sk.basis = Q
    end
    return sk.basis
end

_row_saturated(Q::Matrix) = size(Q, 1) == size(Q, 2) && LinearAlgebra.norm(Q'*Q - I) <= 1e-12*max(1, size(Q, 2))

# Numerical rank, error and norm estimates (C.2/C.3) in the common log scale of the sketch, and whether the sketch
# saturates all rows
function _estimates(sk::_IncrementalSketch; to::TimerOutput = TimerOutput())
    width = _width(sk)
    nrm = LinearAlgebra.norm(sk.r)/sqrt(width)
    if width >= sk.rows && sk.nrefl == sk.rows && _row_saturated(_basis(sk))
        return sk.rank, 0.0, nrm, true
    elseif !isnothing(sk.G)
        # Scaled column norms avoid overflow of |G|² near the inversion limit
        scales = vec(maximum(abs, sk.G; dims = 1))
        inverse_norms = (1 ./ scales) ./ sqrt.(vec(sum(abs2, sk.G ./ transpose(scales); dims = 1)))
        return sk.rank, LinearAlgebra.norm(inverse_norms)/sqrt(width), nrm, false
    end
    @timeit to "loo" err = LinearAlgebra.norm(_loo_residuals(sk.r, sk.threshold))/sqrt(width)
    return sk.rank, err, nrm, false
end

# Leave-one-out residuals of every sketch column z_j = Q r_j against the span of the others, computed from the
# triangular factor R (Q is an isometry, so ‖z_j - P_{Z₋ⱼ} z_j‖ = ‖r_j - P_{R₋ⱼ} r_j‖). Prefix reduction (Finite011):
# the columns before j live in rows 1:j-1, which are projected out completely, so only R[j:, j] against the numerical
# span of R[j:, j+1:] remains. The numerical span keeps pivoted QR diagonals > rank_threshold times the largest
# column norm of R. Falls back to the direct residuals on a nonfinite factorization.
function _loo_residuals(R::AbstractMatrix{ComplexF64}, rank_threshold::Real)
    m, k = size(R)
    reference = maximum(j -> LinearAlgebra.norm(view(R, :, j)), 1:k)
    residuals = map(1:k) do j
        j > m && return 0.0
        v = R[j:m, j]
        j == k && return LinearAlgebra.norm(v)
        F = qr(R[j:m, j+1:k], ColumnNorm())
        d = abs.(diag(F.R))
        r = count(>(rank_threshold*reference), d)
        w = F.Q' * v
        LinearAlgebra.norm(view(w, r+1:length(w)))
    end
    all(isfinite, residuals) && return residuals
    return _loo_residuals_direct(R, rank_threshold)
end

# Direct leave-one-out residuals against the numerical span of all other columns (reference implementation)
function _loo_residuals_direct(R::AbstractMatrix{ComplexF64}, rank_threshold::Real)
    k = size(R, 2)
    buffer = Matrix{ComplexF64}(undef, size(R, 1), k-1)
    map(1:k) do j
        buffer[:, 1:j-1] .= view(R, :, 1:j-1)
        buffer[:, j:k-1] .= view(R, :, j+1:k)
        F = qr!(buffer, ColumnNorm())
        d = abs.(diag(F.R))
        r = (isempty(d) || d[1] == 0) ? 0 : count(>(rank_threshold*d[1]), d)
        v = F.Q' * R[:, j]
        LinearAlgebra.norm(view(v, r+1:length(v)))
    end
end

# Left to right SVD rounding of right canonical cores (core 1 carries the norm): per-cut budget tol‖Ψ‖/√(L-1).
# Returns the relative rounding error and whether it meets tol
function _round_cores!(cores::Vector{Array{ComplexF64, 3}}, max_rank::Int, tol::Real)
    total = LinearAlgebra.norm(cores[1])
    budget = tol*total/sqrt(max(1, length(cores)-1))
    errors = Float64[]
    for i in 1:length(cores)-1
        a = cores[i]
        F = svd(reshape(a, :, size(a, 3)))
        tail = [sqrt.(reverse(cumsum(reverse(F.S.^2)))); 0.0]
        good = findfirst(<=(budget), tail[2:end])
        rank = min(max_rank, isnothing(good) ? length(F.S) : good)
        push!(errors, tail[rank+1])
        cores[i] = reshape(F.U[:, 1:rank], size(a, 1), size(a, 2), rank)
        SV = Diagonal(F.S[1:rank]) * F.Vt[1:rank, :]
        b = cores[i+1]
        cores[i+1] = reshape(SV * reshape(b, size(b, 1), :), rank, size(b, 2), size(b, 3))
    end
    error = LinearAlgebra.norm(errors)/total
    return error, error <= tol
end

# Generic adaptive SRC of the source P.
# rtol: relative error target per cut, sketch_cap: maximal sketch width (and rank) per cut,
# previous_widths: final sketch widths of the previous step (warm start ⌊4k/5⌋), or nothing for a cold start of 2,
# round_rank: maximal rank of the final SVD rounding, or nothing for no rounding.
# Returns the cores [r, x, q] (right canonical up to rounding, core 1 carries the norm) and a NamedTuple of diagnostics.
function _src_run(P, rng::AbstractRNG; rtol::Real, sketch_cap::Int, rank_threshold::Real = 1e-13, round_rank = nothing,
                  round_tol::Real = 0.0, previous_widths = nothing, to::TimerOutput = TimerOutput())
    L = length(P)
    cap = max(sketch_cap, 2)

    # Gaussian Khatri-Rao prefixes C[i][l1, l2, k] with column log scales, shared between all cuts, in capacity
    # buffers of cap+3 columns with a common active width. Cuts run from right to left, so cut i only ever needs
    # the prefixes of sites 1..i: extensions stop at the current cut and the buffers of later sites are released.
    prefixes = [zeros(ComplexF64, _left_shape(P, i)..., cap + 3) for i in 1:L]
    logs = [zeros(cap + 3) for _ in 1:L]
    width = 0
    extend!(ncols, top) = @timeit to "extend_prefixes" _extend!(ncols, top)
    function _extend!(ncols, top)
        cols = width+1:width+ncols
        C = ones(ComplexF64, 1, 1, ncols)
        ls = zeros(ncols)
        prefixes[1][:, :, cols] .= C
        logs[1][cols] .= ls
        for i in 1:top-1
            Ω = (randn(rng, _physical_dim(P, i), ncols) + im*randn(rng, _physical_dim(P, i), ncols))/sqrt(2)
            C = _prefix(P, i, C, Ω)
            all(isfinite, C) || throw(DomainError(C, "nonfinite prefix sketch"))
            scales = [maximum(abs, view(C, :, :, k)) for k in 1:ncols]
            scales[scales .== 0] .= 1.0
            C ./= reshape(scales, 1, 1, :)
            ls = ls .+ log.(scales)
            prefixes[i+1][:, :, cols] .= C
            logs[i+1][cols] .= ls
        end
        width += ncols
    end
    # Reorder (and truncate) the active columns of sites 1..top in place
    function reorder!(order, top)
        for j in 1:top
            prefixes[j][:, :, 1:length(order)] = prefixes[j][:, :, order]
            logs[j][1:length(order)] = logs[j][order]
        end
        width = length(order)
    end
    # Right sketch Y[(x, q), k] = Σ J[l, (x, q)] C[l, k] of the reshaped cut-local J
    sketch(Jm, i, cols) = transpose(Jm) * reshape(view(prefixes[i], :, :, cols), size(Jm, 1), :)

    cores = Vector{Array{ComplexF64, 3}}(undef, L)
    widths = zeros(Int, L-1)
    unmet = falses(L-1)
    S = ones(ComplexF64, 1, 1, 1)
    for i in L:-1:2
        @timeit to "contract_suffix" J = _contract_suffix(P, i, S)
        Jm = reshape(J, size(J, 1)*size(J, 2), :)
        rows = size(Jm, 2)
        k = isnothing(previous_widths) ? 2 : min(rows, max(2, (4*previous_widths[i-1]) ÷ 5))
        k = min(k, cap)
        sk = _IncrementalSketch(rows, rank_threshold)
        pending = nothing
        local met
        while true
            (width < k) && extend!(k - width, i)
            # Only new columns are sketched; failed fresh-check columns are reused as the next columns
            old = _width(sk)
            if !isnothing(pending)
                V, vlogs = pending
                take = min(k - old, size(V, 2))
                @timeit to "qr_update" _append!(sk, V[:, 1:take], vlogs[1:take])
                old += take
                pending = nothing
            end
            if old < k
                @timeit to "sketch" Y = sketch(Jm, i, old+1:k)
                all(isfinite, Y) || throw(DomainError(Y, "nonfinite right sketch"))
                @timeit to "qr_update" _append!(sk, Y, logs[i][old+1:k])
            end
            @timeit to "estimates" rank, err, nrm, saturated = _estimates(sk; to = to)
            met = saturated || err <= rtol*nrm
            if met && !saturated && rank < k
                # Deficient sketch: check with three fresh columns, which become the next columns if the check fails
                met = @timeit to "fresh_check" begin
                    old_width = width
                    extend!(3, i)
                    V = sketch(Jm, i, old_width+1:old_width+3)
                    all(isfinite, V) || throw(DomainError(V, "nonfinite validation sketch"))
                    vlogs = logs[i][old_width+1:old_width+3]
                    pending = (V, vlogs)
                    order = [1:k; old_width+1:old_width+3; k+1:old_width]
                    reorder!(order[1:min(length(order), cap)], i)
                    Vs = V .* transpose(exp.(vlogs .- maximum(vlogs)))
                    Q = _basis(sk)
                    LinearAlgebra.norm(Vs - Q*(Q'*Vs)) <= rtol*LinearAlgebra.norm(Vs)
                end
            end
            (met || k >= cap) && break
            k = min(k + 3, cap)
        end
        widths[i-1] = k
        unmet[i-1] = !met
        Q = _basis(sk)
        cores[i] = reshape(transpose(Q), size(Q, 2), _physical_dim(P, i), size(S, 3))
        @timeit to "project" S = reshape(Jm * conj(Q), size(J, 1), size(J, 2), size(Q, 2))
        # Release the prefixes of site i, later cuts never use them
        prefixes[i] = zeros(ComplexF64, 0, 0, 0)
    end
    @timeit to "first_core" cores[1] = reshape(_contract_suffix(P, 1, S), 1, _physical_dim(P, 1), size(S, 3))
    pre_round_ranks = [size(c, 3) for c in cores[1:L-1]]

    rounding_error, rounding_met = 0.0, true
    if !isnothing(round_rank)
        @timeit to "rounding" rounding_error, rounding_met = _round_cores!(cores, round_rank, round_tol)
    end
    all(c -> all(isfinite, c), cores) || throw(DomainError(cores, "nonfinite SRC output"))
    info = (pre_round_ranks = pre_round_ranks, sketch_widths = widths, cap_unmet = count(unmet), target_met = !any(unmet),
            rounding_error = rounding_error, rounding_met = rounding_met)
    return cores, info
end

function _src_partition(ψ::FiniteDisorderMPS, Ws::Vector{Array{ComplexF64, 5}}; max_rank::Int, rtol::Real = 1e-10,
                        rank_threshold::Real = 1e-13, rng::AbstractRNG = Random.default_rng(), to::TimerOutput = TimerOutput())
    any(<=(0), ψ.ps) && throw(ArgumentError("SRC requires strictly positive disorder probabilities"))
    @timeit to "product_setup" begin
        A, la = _scaled(_dense_layers(ψ))
        W, lw = _scaled(Ws)
        P = _EvolvedPartitionProduct(A, W, sqrt.(ψ.ps))
    end
    cores, info = _src_run(P, rng; rtol = rtol, sketch_cap = max_rank, rank_threshold = rank_threshold, to = to)
    # Both the ket and the bra layer carry the scales
    return ScalarPartition(cores, 2*(la + lw), collect(Float64, ψ.ps)), info
end

# Adaptive SRC of √P(w) Z(w) of the evolved state Us ψ, without forming Us ψ. max_rank caps the rank of every cut.
# Returns the ScalarPartition and a NamedTuple of diagnostics
src_partition(ψ::FiniteDisorderMPS, Us::FiniteDisorderMPO; kwargs...) = _src_partition(ψ, _dense_layers(Us); kwargs...)

# Adaptive SRC of √P(w) Z(w) of ψ itself
src_partition(ψ::FiniteDisorderMPS; kwargs...) = _src_partition(ψ, _identity_layers(ψ); kwargs...)

# Compress f ⊙ (Us ψ) with adaptive SRC followed by SVD rounding to at most maxD, without forming Us ψ or f ⊙ (Us ψ).
# rtol: relative error target per cut, sketch_cap: maximal sketch width per cut,
# previous_widths: final sketch widths of the previous step (warm start ⌊4k/5⌋), or nothing for a cold start of 2.
# Returns the compressed FiniteDisorderMPS and a NamedTuple of diagnostics.
function src_compress(ψ::FiniteDisorderMPS, Us::FiniteDisorderMPO, F::FiniteDisorderFilter; maxD::Int, rtol::Real = 1e-11,
                      sketch_cap::Int = 128, rank_threshold::Real = 1e-13, round_tol::Real = 1e-10, previous_widths = nothing,
                      rng::AbstractRNG = Random.default_rng(), to::TimerOutput = TimerOutput())
    any(<=(0), ψ.ps) && throw(ArgumentError("SRC requires strictly positive disorder probabilities"))
    L = length(ψ)
    (length(Us) == L == length(F)) || throw(ArgumentError("ψ, Us and F should have the same length"))
    @timeit to "product_setup" begin
        A, la = _scaled(_dense_layers(ψ))
        W, lw = _scaled(_dense_layers(Us))
        f, lf = _scaled(_dense_layers(F))
        P = _EvolvedFilterProduct(A, W, f, sqrt.(ψ.ps))
        log_scale = la + lw + lf + F.log_scale
    end
    cores, info = _src_run(P, rng; rtol = rtol, sketch_cap = sketch_cap, rank_threshold = rank_threshold, round_rank = maxD,
                           round_tol = round_tol, previous_widths = previous_widths, to = to)

    # Back to an unweighted FiniteDisorderMPS with the absolute scale spread over the sites
    factor = exp(log_scale/L)
    (isfinite(factor) && factor > 0) || throw(DomainError(log_scale, "absolute SRC scale cannot be materialized"))
    ns = size(W[1], 2)
    @timeit to "materialize" opp = map(1:L) do i
        c = reshape(cores[i], size(cores[i], 1), ns, length(ψ.ps), size(cores[i], 3))
        map(enumerate(ψ.ps)) do (p, prob)
            TensorMap(c[:, :, p, :] * (factor/sqrt(prob)), ℂ^size(c, 1) ⊗ ℂ^ns, ℂ^size(c, 4))
        end
    end
    return FiniteDisorderMPS(opp, ψ.ps), info
end
