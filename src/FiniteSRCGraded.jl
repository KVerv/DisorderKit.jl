# Charge-resolved SRC of the physical source f ⊙ (W ψ) on graded spaces of any sector type: abelian (U₁, Z_N),
# non-abelian (SU₂) and anyonic (Fibonacci). Same algorithm and notation as FiniteSRC.jl (Steps 1-7), with one change:
# the physical legs X_i = P' ⊗ ℂ^q carry a charge, so a Khatri-Rao test vector with Gaussian weights on all of X_i has no
# definite charge and is not a row of a graded space. Here every prefix row k has a definite charge:
#   - at every site j its test vector is supported on one sector y of the spin leg and one fusion channel c → c′ ∈ c ⊗ y
#     (Gaussian over the states of that sector, the fusion vertices of the channel and the q disorder sectors), so the
#     row charge goes from c to c′ (for abelian charges c′ = c ⊗ y is the only channel),
#   - the prefix C_{i-1} : K_i ← V_i is graded, K_i = ⊕_c (c => number of rows with charge c at cut i), and the rows of a
#     sector are kept in creation order (catcodomain appends at the end of every sector block),
#   - rows are created for a target (cut i, charge c): their charge path (spin sector and fusion channel at every site
#     1..i-1) is drawn uniformly, counted with the fusion multiplicities N_{c y}^{c′}, among the paths that end in c and
#     only pass through charges of the left bonds V_j (a counting dynamic program), so no row dies on the way.
# For a non-abelian sector a block of K, V or X is the multiplicity space of its sector (dim(V, c) multiplets); the
# blockwise QR, ranks and error estimates below act on these multiplicity spaces, as TensorKit's factorizations do.
# Rows of one sector at one cut are independent and identically distributed, so the sketch basis, the leave-one-out
# error estimate and the fresh check of Steps 4-5 are applied per sector block, each relative to its own scale (rows of
# different charges have incomparable magnitudes), and the cut is accepted when every block meets the target. For a
# single block this is the estimate of FiniteSRC.jl. Rows that are exactly zero (a charge path through a sector where
# the left network vanishes) are dropped.
#
# The Trivial (no symmetry) case keeps the dense code path of FiniteSRC.jl.

is_graded(P::Source) = sectortype(spacetype(P.A[1][1])) !== Trivial

# Charges of the fused space V (the sectors in which a map into V can be nonzero)
_charges(V) = Set(sectors(fuse(V)))

# Fusion channels (y, c′) of a row with charge a through the spin sectors y of the space `spin`, c′ ∈ a ⊗ y, with their
# multiplicities N_{a y}^{c′}
_channels(a::Sector, spin) = [((y, b), Nsymbol(a, y, b)) for y in sectors(spin) for b in a ⊗ y]

#==========================================================================================
 Step 1 primitive on graded spaces
==========================================================================================#

# J = M_i S on graded spaces: per sector u contract S with A_u, W_u, f_u (spin leg s open), weight g_u, and place it on the
# disorder leg u of X_i = P' ⊗ ℂ^q through the basis covector of u (a planar outer product with a trivial-sector leg)
function apply_suffix_graded(P::PhysicalSource, i::Int, S::AbstractTensorMap)
    q = length(P.g[i])
    T = promote_type(scalartype(S), scalartype(P.A[i][1]), scalartype(P.W[i][1]), scalartype(P.f[i][1]))
    J = zeros(T, left_space(P, i) ← domain(S) ⊗ physical_space(P, i))      # V_i ← D_i ⊗ P' ⊗ ℂ^q
    for (u, g) in enumerate(P.g[i])
        e = basis_ket(spacetype(S), q, u, T)
        X = suffix_layers(P, i, u, g, S)                                    # V_i ← D_i ⊗ P'
        @plansor J[-1 -2 -3; -4 -5 -6] += X[-1 -2 -3; -4 -5] * conj(e[-6])
    end
    return J
end

#==========================================================================================
 Step 3: charge-resolved Khatri-Rao prefix sketches
==========================================================================================#

# Prefixes shared by all cuts: C[j] holds C_{j-1}: K_j ← V_j (C_0 = 1), logs[j][c] the log scale of the rows of sector c
# (in their order within the sector block). allowed[j] are the charges of V_j reachable from the left boundary, spin[j]
# the spin leg of site j as it is contracted with the test vectors. drawn[(i, c)] are the deterministic charge paths
# (see sample_charge_paths) already drawn for target cut i and charge c
mutable struct GradedPrefixStore{TP<:PhysicalSource, R<:AbstractRNG, I<:Sector}
    P::TP
    rng::R
    C::Vector{Any}
    logs::Vector{Dict{I, Vector{Float64}}}
    allowed::Vector{Set{I}}
    spin::Vector{Any}
    drawn::Dict{Tuple{Int, I}, Set{Vector{Tuple{I, I}}}}
end

function GradedPrefixStore(P::PhysicalSource, rng::AbstractRNG)
    L = length(P)
    I = sectortype(spacetype(P.A[1][1]))
    spin = Any[space(P.W[j][1], 2) for j in 1:L]
    # Charges of V_j that a prefix can reach from the trivial left boundary: other sectors of V_j only meet zero prefixes
    # (the left half of the network never produces them), so their blocks of J are irrelevant
    allowed = Vector{Set{I}}(undef, L)
    allowed[1] = Set([one(I)])
    for j in 2:L
        reach = Set(b for a in allowed[j-1] for ((_, b), _) in _channels(a, spin[j-1]))
        allowed[j] = intersect(reach, _charges(left_space(P, j)))
    end
    return GradedPrefixStore(P, rng, Any[nothing for _ in 1:L], [Dict{I, Vector{Float64}}() for _ in 1:L], allowed, spin,
                             Dict{Tuple{Int, I}, Set{Vector{Tuple{I, I}}}}())
end

# Whether a row with this charge path is deterministic up to its scale: at every site the Gaussian test weights live on a
# one-dimensional space (one state in the spin sector, one fusion vertex, one disorder sector), e.g. Fibonacci anyons
# without disorder. Two rows with the same deterministic path are then parallel and add nothing to the sketch
function deterministic_path(store::GradedPrefixStore, path)
    a = one(first(first(path)))
    for (j, (y, b)) in enumerate(path)
        dim(store.spin[j], y) * Nsymbol(a, y, b) * length(store.P.g[j]) == 1 || return false
        a = b
    end
    return true
end

# Number of stored rows of charge c at cut i
stored_rows(store::GradedPrefixStore, i::Int, c::Sector) = length(get(store.logs[i], c, Float64[]))

# Charge paths of n rows that end in charge c at cut i (charge of C_{i-1}), uniform over the admissible paths counted with
# their fusion multiplicities: paths[k][j] = (y, c′) is the spin sector and the new row charge of row k at site j,
# j = 1..i-1. Counting from the end: N[j][a] = number of continuations from charge a at cut j (sites j..i-1) that end in c
# and only pass through allowed charges.
# Deterministic paths (deterministic_path) are drawn without replacement per (i, c): a repeated one would give a column
# parallel to an earlier one, which the leave-one-out estimate and the fresh check take as evidence that the block has
# converged although its other directions were never sampled. Once (almost) all of them are drawn, fewer than n paths are
# returned, and ensure_rows! marks the charge as exhausted: the drawn paths then span every direction the block can have
function sample_charge_paths(store::GradedPrefixStore, i::Int, c::Sector, n::Int; max_tries::Int = 20 + 10n)
    I = typeof(c)
    N = Vector{Dict{I, Float64}}(undef, i)
    N[i] = Dict(c => 1.0)
    for j in i-1:-1:1
        N[j] = Dict{I, Float64}()
        for a in store.allowed[j], ((_, b), m) in _channels(a, store.spin[j])
            w = m * get(N[j+1], b, 0.0)
            w > 0 && (N[j][a] = get(N[j], a, 0.0) + w)
        end
    end
    start = one(I)
    get(N[1], start, 0.0) > 0 || throw(ArgumentError("charge $c cannot be reached at cut $i"))
    function draw()
        a, path = start, Tuple{I, I}[]
        for j in 1:i-1
            chs = [(ch, m * N[j+1][ch[2]]) for (ch, m) in _channels(a, store.spin[j]) if get(N[j+1], ch[2], 0.0) > 0]
            ws = last.(chs)
            ch = first(chs[searchsortedfirst(cumsum(ws), rand(store.rng) * sum(ws))])
            push!(path, ch)
            a = ch[2]
        end
        return path
    end
    drawn = get!(() -> Set{Vector{Tuple{I, I}}}(), store.drawn, (i, c))
    paths = Vector{Tuple{I, I}}[]
    tries = 0
    while length(paths) < n && tries < max_tries
        tries += 1
        path = draw()
        if i > 1 && deterministic_path(store, path)
            path in drawn && continue
            push!(drawn, path)
        end
        push!(paths, path)
    end
    return paths
end

# Graded space with the given row charges, and the position of every row within its sector block (rows of a sector in
# their given order)
function row_space(S, charges::AbstractVector{<:Sector})
    counts = Dict{eltype(charges), Int}()
    pos = map(charges) do c
        counts[c] = get(counts, c, 0) + 1
    end
    return S(c => k for (c, k) in counts), pos
end

# Largest |entry| of every row of a graded C: K ← V, per sector (zero rows, also rows of a sector without block, get 0)
function graded_row_scales(C::AbstractTensorMap)
    K = codomain(C)[1]
    bs = blocksectors(C)
    return Dict(c => (c in bs && size(block(C, c), 2) > 0 ?
                      vec(maximum(abs, block(C, c); dims = 2)) : zeros(dim(K, c))) for c in sectors(K))
end

# Divide the rows of every sector block of C by s[c] (zero scales are left at 1)
function scale_rows!(C::AbstractTensorMap, s::Dict)
    bs = blocksectors(C)
    for c in keys(s)
        s[c][s[c] .== 0] .= 1.0
        c in bs && (block(C, c) ./= s[c])
    end
    return C
end

# New rows with the charge paths `paths`, on sites 1..i: every site j < i applies the Khatri-Rao step
#   C_j[k′; w] = Σ_u Σ_{s ∈ y_{k,j}} g_u ω_{k,j}[s, u] (C_{j-1} A_u W_u f_u)[k; w s]
# with complex Gaussian ω, and rescales every row. The rows are appended to the stored prefixes of every cut 1..i
function extend_prefixes!(store::GradedPrefixStore, paths::Vector, i::Int)
    P, rng = store.P, store.rng
    n = length(paths)
    S = spacetype(P.A[1][1])
    I = sectortype(S)
    charges = fill(one(I), n)
    K, pos = row_space(S, charges)
    C = ones(ComplexF64, K ← left_space(P, 1))                    # C_0 = 1, V_1 is one-dimensional and trivial
    logs = zeros(n)
    for j in 1:i
        if j > 1
            jj = j - 1                                            # the site that is applied
            new_charges = [paths[k][jj][2] for k in 1:n]
            K′, pos′ = row_space(S, new_charges)
            out = zeros(ComplexF64, K′ ← right_space(P, jj))
            for (u, g) in enumerate(P.g[jj])
                X = prefix_layers(P, jj, u, C)                     # K ← R_jj ⊗ spin'
                Xp = repartition(X, 2)                            # K ⊗ spin ← R_jj (planar)
                spin = codomain(Xp)[2]
                spin == store.spin[jj] || throw(ArgumentError("unexpected spin leg $spin at site $jj"))
                Ω = zeros(ComplexF64, K′ ← codomain(Xp))
                # Every fusion tree (c, y) → c′ (and vertex, for fusion multiplicities) gets its own Gaussian weights
                for (f1, f2) in fusiontrees(Ω)
                    c, y = f2.uncoupled
                    c′ = f2.coupled
                    blk = Ω[f1, f2]                                # [row k′, row k, state of sector y]
                    for k in 1:n
                        (charges[k] == c && paths[k][jj] == (y, c′)) || continue
                        ω = (randn(rng, dim(spin, y)) + im * randn(rng, dim(spin, y))) / sqrt(2)
                        blk[pos′[k], pos[k], :] .= g .* ω
                    end
                end
                out += Ω * Xp
            end
            C, charges, K, pos = out, new_charges, K′, pos′
            s = graded_row_scales(C)
            all(v -> all(isfinite, v), values(s)) || throw(DomainError(s, "nonfinite prefix sketch at site $j"))
            # A row whose charge path passes through a sector where the left network vanishes is exactly zero (and stays
            # zero). Such rows are dropped: counted as sketch columns they would make a block look saturated while one
            # of its columns is zero, and the cut would miss a direction of the state
            alive = [s[charges[k]][pos[k]] > 0 for k in 1:n]
            if !all(alive)
                dead_prefix_rows[] += count(!, alive)
                keep = findall(alive)
                cols = Dict(c => sort(pos[filter(k -> charges[k] == c, keep)]) for c in sectors(K))
                C = select_rows_graded(K, cols) * C
                s = Dict(c => s[c][r] for (c, r) in cols if !isempty(r))
                paths, charges, logs = paths[keep], charges[keep], logs[keep]
                n = length(keep)
                K, pos = row_space(S, charges)
                n == 0 && return store                            # every new row died: nothing to add at the later cuts
            end
            scale_rows!(C, s)
            for k in 1:n
                logs[k] += log(s[charges[k]][pos[k]])
            end
        end
        store.C[j] = isnothing(store.C[j]) ? C : catcodomain(store.C[j], C)
        for c in sectors(K)                                       # logs of the new rows, per sector, in block order
            rows = findall(==(c), charges)
            append!(get!(store.logs[j], c, Float64[]), logs[rows][sortperm(pos[rows])])
        end
    end
    return store
end

# Number of prefix rows dropped because they were exactly zero (diagnostics)
const dead_prefix_rows = Ref(0)

# Try to give cut i at least n[c] rows of every charge c. Dead rows are dropped, so a few rounds may be needed; a charge
# for which `attempts` rounds in a row add no row has a vanishing left network (its block of J is never seen by any
# prefix) or no deterministic path left that was not drawn yet (sample_charge_paths). Returns the charges that still lack rows
function ensure_rows!(store::GradedPrefixStore, i::Int, n::Dict; attempts::Int = 4)
    short = Set{keytype(n)}()
    for (c, k) in n
        failures = 0
        while (missing_rows = k - stored_rows(store, i, c)) > 0
            before = stored_rows(store, i, c)
            paths = sample_charge_paths(store, i, c, missing_rows)
            isempty(paths) || extend_prefixes!(store, paths, i)
            if stored_rows(store, i, c) == before
                failures += 1
                failures >= attempts && (push!(short, c); break)
            else
                failures = 0
            end
        end
    end
    return short
end

# Selection Π: K_sel ← K of the rows cols[c] of every sector c of K (K the stored rows of cut i)
function select_rows_graded(K, cols::Dict)
    S = typeof(K)
    Ksel = S(c => length(r) for (c, r) in cols if !isempty(r))
    Π = zeros(ComplexF64, Ksel ← K)
    for (c, r) in cols
        isempty(r) && continue
        b = block(Π, c)
        for (k, row) in enumerate(r)
            b[k, row] = 1
        end
    end
    return Π
end

# Sketch columns of cut i for the rows cols[c] of every charge: Z = (C_{i-1}[cols, :] J)': D_i ⊗ X_i ← K_sel, and their
# log scales per sector
function sketch_columns(store::GradedPrefixStore, J::AbstractTensorMap, i::Int, cols::Dict)
    Ci = select_rows_graded(codomain(store.C[i])[1], cols) * store.C[i]
    logs = Dict(c => store.logs[i][c][r] for (c, r) in cols if !isempty(r))
    return copy((Ci * J)'), logs
end

#==========================================================================================
 Steps 4-5 per sector block
==========================================================================================#

# All sketch columns of one cut with their QR factorization (blockwise), every column normalized to unit norm.
# A row follows one charge path, and with one dimensional spin sectors (bosons) a path is one occupation pattern, so the
# magnitude of a row is the amplitude of that pattern: rows of the same block can differ by many orders of magnitude
# (a path through improbable patterns), and between blocks the magnitudes are not comparable at all. Scaled by their
# true magnitudes such columns would look numerically zero and the sketch would miss directions the block needs.
# The range of the sketch does not depend on the column scales, so the columns are normalized instead, and ranks,
# error estimates and acceptance are relative within every block. The true weight of every direction is restored in
# the suffix S = J Q and judged by the rounding (Step 7).
# rows[c] = dimension of sector c of D_i ⊗ X_i, used[c] = number of columns of sector c
mutable struct GradedSketchBasis{TZ<:AbstractTensorMap, I<:Sector}
    rows::Dict{I, Int}
    threshold::Float64
    Z::TZ
    Q::TZ
    R::Any
    used::Dict{I, Int}
end

function GradedSketchBasis(V, sectors_used; threshold::Real = 1e-13)
    F = fuse(V)
    I = sectortype(F)
    E = typeof(F)()                                              # empty graded space
    Z = zeros(ComplexF64, V ← E)
    return GradedSketchBasis(Dict{I, Int}(c => dim(F, c) for c in sectors_used), Float64(threshold), Z, Z,
                             zeros(ComplexF64, E ← E), Dict{I, Int}(c => 0 for c in sectors_used))
end

sketch_width(basis::GradedSketchBasis) = sum(values(basis.used); init = 0)

# Normalize every column of every block of Z to unit norm (zero columns stay zero)
function normalize_columns!(Z::AbstractTensorMap)
    for (_, b) in blocks(Z)
        n = [LinearAlgebra.norm(col) for col in eachcol(b)]
        n[n .== 0] .= 1
        b ./= transpose(n)
    end
    return Z
end

# Append the columns Znew (one block per charge in logs) to the basis, normalized, and refactorize
function append_columns!(basis::GradedSketchBasis, Znew::AbstractTensorMap, logs::Dict)
    isempty(logs) && return basis
    for (c, l) in logs
        basis.used[c] += length(l)
    end
    basis.Z = catdomain(basis.Z, normalize_columns!(Znew))
    basis.Q, basis.R = qr_compact(basis.Z)
    return basis
end

# Singular values of every block of R (empty for a block without columns)
function block_svdvals(basis::GradedSketchBasis)
    bs = blocksectors(basis.R)
    return Dict(c => (c in bs ? svdvals(block(basis.R, c)) : Float64[]) for c in keys(basis.used))
end

# Numerical rank of every block relative to its own largest singular value
block_ranks(basis::GradedSketchBasis) =
    Dict(c => (isempty(v) || v[1] == 0 ? 0 : count(>(basis.threshold * v[1]), v)) for (c, v) in block_svdvals(basis))

# Leave-one-out residuals of the columns of a block matrix R (rank r), as loo_residuals in FiniteSRC.jl
function loo_residuals_matrix(R::AbstractMatrix, rank::Int; leverage_tol::Real = 1e-8)
    F = svd(R)
    r = min(max(rank, 1), length(F.S))
    V, σ = F.V[:, 1:r], F.S[1:r]
    return [sum(abs2, v) > 1 - leverage_tol ? 1 / LinearAlgebra.norm(v ./ σ) : 0.0 for v in eachrow(V)]
end

# Per block (as estimate_error in FiniteSRC.jl, within the block): leave-one-out error err[c], norm estimate nrm[c],
# saturated[c] (full numerical rank: the columns span all rows of the block, err = 0) and the numerical ranks. Counting
# columns is not enough: K_c >= rows_c columns can still be rank deficient when rows of the block coincide.
# Block c meets the target when saturated[c] || err[c] <= rtol nrm[c]
function estimate_error(basis::GradedSketchBasis)
    ranks = block_ranks(basis)
    I = keytype(basis.used)
    errs, nrms, sat = Dict{I, Float64}(), Dict{I, Float64}(), Dict{I, Bool}()
    for (c, K) in basis.used
        sat[c] = ranks[c] >= basis.rows[c]                        # the columns span all rows of the block
        if K == 0
            errs[c], nrms[c] = Inf, 0.0
            continue
        end
        Rc = block(basis.R, c)
        nrms[c] = LinearAlgebra.norm(Rc) / sqrt(K)
        errs[c] = sat[c] ? 0.0 : LinearAlgebra.norm(loo_residuals_matrix(Rc, ranks[c])) / sqrt(K)
    end
    return errs, nrms, sat, ranks
end

block_met(errs, nrms, sat, c, rtol) = sat[c] || errs[c] <= rtol * nrms[c]

# Fresh check of the rank-deficient blocks: new rows of those charges must lie in the span of Q, per block
# ‖V_c - Q_c Q_c' V_c‖ <= rtol ‖V_c‖. The fresh columns are appended to the basis afterwards. Returns the blocks that failed
function fresh_check(store::GradedPrefixStore, basis::GradedSketchBasis, J::AbstractTensorMap, i::Int, deficient; rtol::Real, nfresh::Int = 3)
    want = Dict(c => basis.used[c]+1:basis.used[c]+nfresh for c in deficient)
    ensure_rows!(store, i, Dict(c => last(r) for (c, r) in want))
    cols = Dict(c => first(r):min(last(r), stored_rows(store, i, c)) for (c, r) in want)
    filter!(((c, r),) -> !isempty(r), cols)
    isempty(cols) && return eltype(deficient)[]                  # no fresh rows exist: the stored rows are all there is
    V, logs = sketch_columns(store, J, i, cols)
    Vs = normalize_columns!(copy(V))
    D = Vs - basis.Q * (basis.Q' * Vs)
    failed = [c for c in keys(logs) if LinearAlgebra.norm(block(D, c)) > rtol * LinearAlgebra.norm(block(Vs, c))]
    append_columns!(basis, V, logs)
    return failed
end

# Orthonormal basis of the numerical range of every block: Q_c times the leading rank_c left singular vectors of R_c
# (at least one vector in total)
function trimmed_basis(basis::GradedSketchBasis)
    ranks = block_ranks(basis)
    all(ranks[c] == min(basis.used[c], basis.rows[c]) for c in keys(ranks)) && sum(values(ranks); init = 0) > 0 && return basis.Q
    U, = svd_compact(basis.R)
    keep = Dict(c => 1:ranks[c] for c in keys(ranks) if ranks[c] > 0)
    if isempty(keep)                                             # nothing numerically nonzero: keep one vector
        c = first(c for c in keys(basis.used) if basis.used[c] > 0)
        keep[c] = 1:1
    end
    return basis.Q * (U * select_rows_graded(domain(U)[1], keep)')
end

#==========================================================================================
 Steps 1-6 for one cut
==========================================================================================#

# One cut i on graded spaces, as compress_cut! in FiniteSRC.jl but per sector block: every active charge c (present in
# both V_i and D_i ⊗ X_i) starts with min(rows_c, 2) columns (or 4/5 of its previous width), the cut is accepted when
# every block meets rtol relative to its own norm, otherwise every unmet block grows by max(3, k_c ÷ 4) columns up to
# cap (per block).
# Returns (G_i, S_{i-1}, final widths per charge, target met)
function compress_cut!(P::PhysicalSource, store::GradedPrefixStore, S::AbstractTensorMap, i::Int; rtol::Real, cap::Int,
                       rank_threshold::Real = 1e-13, previous_width = nothing, to::TimerOutput = TimerOutput())
    @timeit to "contract_suffix" J = apply_suffix_graded(P, i, S)
    active = collect(intersect(store.allowed[i], _charges(domain(J))))
    basis = GradedSketchBasis(domain(J), active; threshold = rank_threshold)
    target = Dict(c => min(basis.rows[c], cap,
                          (isnothing(previous_width) || !haskey(previous_width, c)) ? 2 : max(2, (4 * previous_width[c]) ÷ 5))
                  for c in active)
    exhausted = Set{eltype(active)}()                            # charges for which no more live prefix rows exist
    checked = Set{eltype(active)}()                              # deficient blocks that passed the fresh check
    while true
        cols = Dict(c => basis.used[c]+1:target[c] for c in active if target[c] > basis.used[c])
        if !isempty(cols)
            @timeit to "extend_prefixes" short = ensure_rows!(store, i, Dict(c => last(r) for (c, r) in cols))
            union!(exhausted, short)
            for c in short
                target[c] = stored_rows(store, i, c)
                cols[c] = basis.used[c]+1:target[c]
                isempty(cols[c]) && delete!(cols, c)
                if target[c] == 0                                 # no prefix reaches charge c: its block is never seen
                    filter!(!=(c), active)
                    delete!(basis.used, c); delete!(basis.rows, c); delete!(target, c)
                end
            end
            if !isempty(cols)
                @timeit to "sketch" Z, logs = sketch_columns(store, J, i, cols)
                @timeit to "qr_update" append_columns!(basis, Z, logs)
            end
        end
        @timeit to "estimates" errs, nrms, sat, ranks = estimate_error(basis)
        unmet = [c for c in active if !block_met(errs, nrms, sat, c, rtol)]
        # Accepted rank-deficient blocks are double checked with fresh columns; failed ones become unmet again
        deficient = [c for c in active if block_met(errs, nrms, sat, c, rtol) && !sat[c] && 0 < ranks[c] < basis.used[c] && !(c in checked)]
        if !isempty(deficient)
            failed = @timeit to "fresh_check" fresh_check(store, basis, J, i, deficient; rtol = rtol)
            union!(checked, setdiff(deficient, failed))
            for c in deficient
                target[c] = basis.used[c]
            end
            append!(unmet, failed)
        end
        met = isempty(unmet)
        # Unmet blocks grow unless they are at the cap, have no rows left or are saturated
        # (not limited to rows_c columns: coinciding rows can leave a block with rows_c columns rank deficient)
        grow = [c for c in unmet if !sat[c] && !(c in exhausted) && basis.used[c] < cap]
        if met || isempty(grow)
            Q = trimmed_basis(basis)
            @timeit to "project" S = next_suffix(J, Q)
            return core_from_basis(Q), S, copy(basis.used), met
        end
        for c in grow
            k = basis.used[c]
            target[c] = min(k + max(3, k ÷ 4), cap)
        end
    end
end
