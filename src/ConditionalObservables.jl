# Observables of the conditional states ψ_w of a FiniteDisorderMPS (one disorder realization, the word w = (p₁…p_L) of
# sectors pᵢ of every site) and their disorder averages.
#
# Conditional (per realization) quantities of on-site operators: conditional_expectation_value,
# conditional_block_fluctuations and conditional_entanglement_entropy. Their disorder average, exact by enumeration or
# sampled: disorder_average(f, ρ) for any per-realization quantity f(word). The averages of linear quantities are also
# available exactly without enumerating the words: expectation_value and correlator (FiniteDisorderMPS.jl).
#
# On-site operators O are TensorMaps P ← P on the physical space: one operator for every site, or a vector with the
# operator Os[s] of every Hamiltonian sector s (as expectation_value). The conditional state is contracted with dense
# arrays (convert(Array, ·)), for the tensors and the operators alike, so their basis orders always match, also for
# graded (e.g. U₁) spaces.
#
# Projection (keyword charges): a conditional state of a particle number conserving H can be a superposition of
# particle numbers (ITE without U(1) symmetry, from a state that contains every N). With charges, the U(1) label of every
# dense physical basis state (e.g. 0:nmax bosons for unsymmetric tensors), ψ_w is first projected onto its most likely
# total charge N*, the particle number of its exact ground state. The environments are then resolved by charge, and the
# operators must conserve the charge. Without charges (nothing) there is one charge sector: no projection, same cost.

# Dense tensors A[a, s, b] of the conditional state of word w and the charge label of every physical basis state
function _conditional_tensors(ρ::FiniteDisorderMPS, word::AbstractVector{Int}, charges)
    length(word) == length(ρ) || throw(ArgumentError("word of length $(length(word)) for a chain of length $(length(ρ))"))
    all(i -> 1 <= word[i] <= length(ρ.ps[i]), eachindex(word)) || throw(ArgumentError("word $word outside the sectors of ρ"))
    As = [ComplexF64.(convert(Array, ρ[i, p])) for (i, p) in enumerate(word)]
    d = size(As[1], 2)
    q = isnothing(charges) ? zeros(Int, d) : collect(Int, charges)
    all(size(A, 2) == length(q) for A in As) || throw(ArgumentError("charges should have one entry per physical basis state"))
    return As, q
end

# Charge resolved Gram environments of the conditional state of word w: lefts[i][c] of sites 1..i-1 with total charge c
# (G_L[b̄, b] = Σ conj(X_b̄) X_b) and rights[i][c] of sites i..L (G_R[a, ā] = Σ Y_a conj(Y_ā)), i = 1..L+1. All charge
# blocks of a site are divided by one common factor, which cancels in every ratio. Also returns the most likely total
# charge N* (0 without charges)
function _conditional_environments(As, q)
    L = length(As)
    rescale!(E) = (m = maximum(e -> LinearAlgebra.norm(e, Inf), values(E)); m > 0 && foreach(c -> E[c] ./= m, keys(E)); E)
    lefts = Vector{Dict{Int, Matrix{ComplexF64}}}(undef, L+1)
    rights = Vector{Dict{Int, Matrix{ComplexF64}}}(undef, L+1)
    lefts[1] = Dict(0 => ones(ComplexF64, 1, 1))
    for i in 1:L
        E = Dict{Int, Matrix{ComplexF64}}()
        for (c, G) in lefts[i], (s, n) in enumerate(q)
            A = As[i][:, s, :]
            acc = get!(() -> zeros(ComplexF64, size(A, 2), size(A, 2)), E, c + n)
            acc .+= A' * G * A
        end
        lefts[i+1] = rescale!(E)
    end
    rights[L+1] = Dict(0 => ones(ComplexF64, 1, 1))
    for i in L:-1:1
        E = Dict{Int, Matrix{ComplexF64}}()
        for (c, G) in rights[i+1], (s, n) in enumerate(q)
            A = As[i][:, s, :]
            acc = get!(() -> zeros(ComplexF64, size(A, 1), size(A, 1)), E, c + n)
            acc .+= A * G * A'
        end
        rights[i] = rescale!(E)
    end
    Nstar = argmax(N -> real(tr(lefts[end][N])), collect(keys(lefts[end])))
    return lefts, rights, Nstar
end

# Dense operator of every site for word w: O a single TensorMap, or Os[s] per Hamiltonian sector s. With charges the
# operators must conserve the charge
function _site_operators(ρ::FiniteDisorderMPS, word, O, q)
    Os = O isa AbstractVector ? O : fill(O, num_hamiltonian_sectors(ρ))
    ops = [ComplexF64.(convert(Array, Os[ρ.sectors[i][p]])) for (i, p) in enumerate(word)]
    for M in ops
        size(M) == (length(q), length(q)) || throw(ArgumentError("operator of size $(size(M)) on a physical space of dimension $(length(q))"))
        all(iszero(M[s, t]) || q[s] == q[t] for s in axes(M, 1), t in axes(M, 2)) ||
            throw(ArgumentError("with charges the operator must conserve the charge"))
    end
    return ops
end

# Apply the dense operator M to the physical leg of A: (MA)[a, s, b] = Σ_t M[s, t] A[a, t, b]
_apply(M, A) = (@tensor B[a, s, b] := M[s, t] * A[a, t, b]; B)

# Σ_c tr(G[c] R[N* - c]): close the charge resolved accumulators G (sites 1..i) with the right environment R (sites i+1..L)
_close(G, R, Nstar) = sum((tr(Gc * R[Nstar - c]) for (c, Gc) in G if haskey(R, Nstar - c)); init = zero(ComplexF64))

# Real part for a Hermitian operator (numerical noise in the imaginary part), complex otherwise
_hermitian_real(x, ops) = all(M -> M ≈ M', ops) ? real(x) : x

# ⟨O_i⟩_w of the conditional state of word w on every site i = 1..L (projected onto the most likely total charge with
# charges); ψ_w does not have to be normalized
function conditional_expectation_value(ρ::FiniteDisorderMPS, word::AbstractVector{Int}, O; charges = nothing)
    As, q = _conditional_tensors(ρ, word, charges)
    lefts, rights, Nstar = _conditional_environments(As, q)
    ops = _site_operators(ρ, word, O, q)
    vals = map(eachindex(As)) do i
        OA = _apply(ops[i], As[i])
        num, den = zero(ComplexF64), zero(ComplexF64)
        for (c, G) in lefts[i], (s, n) in enumerate(q)
            haskey(rights[i+1], Nstar - c - n) || continue
            R = rights[i+1][Nstar - c - n]
            A = As[i][:, s, :]
            num += tr(A' * G * OA[:, s, :] * R)
            den += tr(A' * G * A * R)
        end
        num / den
    end
    return _hermitian_real(vals, ops)
end

# Fluctuations Var_w(O_A) = ⟨O_A²⟩_w - ⟨O_A⟩_w² of O_A = Σ_{i∈A} O_i (e.g. the particle number N_A for O = n) of the
# conditional state of word w for every block A = a:b in blocks (projected onto the most likely total charge with
# charges). The blocks are grouped by their left edge a: one sweep from a carries the moments 0, 1 and 2 of O_A,
#   G0′ = Σ_s A_s† G0 A_s,   G1′ = Σ_s A_s† (G1 A_s + G0 (OA)_s),   G2′ = Σ_s A_s† (G2 A_s + 2 G1 (OA)_s + G0 (O²A)_s),
# keyed by the total charge of sites 1..i, and closes them with the right environment at every right edge b of the group,
# so blocks with a common left edge cost one sweep. O must be Hermitian
function conditional_block_fluctuations(ρ::FiniteDisorderMPS, word::AbstractVector{Int}, O,
                                        blocks::AbstractVector{<:UnitRange{<:Integer}}; charges = nothing)
    L = length(ρ)
    all(b -> 1 <= first(b) <= last(b) <= L, blocks) || throw(ArgumentError("blocks should be nonempty ranges within 1:$L"))
    As, q = _conditional_tensors(ρ, word, charges)
    lefts, rights, Nstar = _conditional_environments(As, q)
    ops = _site_operators(ρ, word, O, q)
    all(M -> M ≈ M', ops) || throw(ArgumentError("the operator must be Hermitian"))
    F = zeros(Float64, length(blocks))
    for a in unique(first.(blocks))
        group = findall(b -> first(b) == a, blocks)
        G0 = Dict(c => copy(G) for (c, G) in lefts[a])
        G1 = Dict(c => zero(G) for (c, G) in lefts[a])
        G2 = Dict(c => zero(G) for (c, G) in lefts[a])
        for i in a:maximum(last(blocks[k]) for k in group)
            OA, O2A = _apply(ops[i], As[i]), _apply(ops[i]^2, As[i])
            nb = size(As[i], 3)
            new() = Dict{Int, Matrix{ComplexF64}}()
            G0′, G1′, G2′ = new(), new(), new()
            for c in keys(G0), (s, n) in enumerate(q)
                A, A1, A2 = As[i][:, s, :], OA[:, s, :], O2A[:, s, :]
                zero_acc() = zeros(ComplexF64, nb, nb)
                get!(zero_acc, G0′, c + n) .+= A' * G0[c] * A
                get!(zero_acc, G1′, c + n) .+= A' * (G1[c] * A .+ G0[c] * A1)
                get!(zero_acc, G2′, c + n) .+= A' * (G2[c] * A .+ 2 .* (G1[c] * A1) .+ G0[c] * A2)
            end
            m = maximum(G -> LinearAlgebra.norm(G, Inf), values(G0′))
            m > 0 || throw(DomainError(m, "conditional state of the word vanishes"))
            G0 = Dict(c => G ./ m for (c, G) in G0′)
            G1 = Dict(c => G ./ m for (c, G) in G1′)
            G2 = Dict(c => G ./ m for (c, G) in G2′)
            for k in group
                last(blocks[k]) == i || continue
                Z = real(_close(G0, rights[i+1], Nstar))
                m1, m2 = real(_close(G1, rights[i+1], Nstar)) / Z, real(_close(G2, rights[i+1], Nstar)) / Z
                F[k] = max(m2 - m1^2, 0.0)
            end
        end
    end
    return F
end

# Entanglement entropy S_w(k) of the conditional state of word w on every bond k = 1..L-1 (projected onto the most
# likely total charge with charges). The Schmidt spectrum on bond k is the union over the charge c left of the cut of
# the spectra of the Gram matrices rights[k+1][N* - c] * lefts[k+1][c], normalized to one, so ψ_w does not have to be
# normalized. Without the projection, the particle number fluctuations of a state that mixes N no longer entangle the
# two halves, which lowers S most in the bulk
function conditional_entanglement_entropy(ρ::FiniteDisorderMPS, word::AbstractVector{Int}; charges = nothing)
    As, q = _conditional_tensors(ρ, word, charges)
    lefts, rights, Nstar = _conditional_environments(As, q)
    return map(1:length(As)-1) do k
        λ = Float64[]
        for (c, G) in lefts[k+1]
            haskey(rights[k+1], Nstar - c) || continue
            append!(λ, max.(real.(eigvals(rights[k+1][Nstar - c] * G)), 0.0))
        end
        λ ./= sum(λ)
        -sum(x -> x > 0 ? x * log(x) : 0.0, λ)
    end
end

# Disorder word with the sector of site i drawn from ps[i]
sample_word(rng::AbstractRNG, ps::Vector{<:Vector{<:Real}}) = [sample(rng, eachindex(p), Weights(p)) for p in ps]

# Disorder average 𝔼_w[f(w)] of a per-realization quantity f(word) (a number or an array, e.g. a conditional observable
# above) over the words w = (p₁…p_L) with pᵢ ~ ρ.ps[i] independent. Returns (𝔼_w[f], its standard error, Var_w[f]),
# elementwise for an array. If there are at most n_samples words they are all enumerated with their weights P(w) (exact,
# zero standard error), otherwise n_samples words are sampled (sample_word with rng)
function disorder_average(f, ρ::FiniteDisorderMPS; n_samples::Int = 1000, rng::AbstractRNG = Random.default_rng())
    ps = ρ.ps
    if prod(float.(length.(ps))) <= n_samples
        m, m2 = nothing, nothing
        for word in ProgressBar(Iterators.product(eachindex.(ps)...))
            w = prod(ps[i][p] for (i, p) in enumerate(word))
            iszero(w) && continue
            x = f(collect(word))
            m = isnothing(m) ? w .* x : m .+ w .* x
            m2 = isnothing(m2) ? w .* abs2.(x) : m2 .+ w .* abs2.(x)
        end
        return m, zero.(real.(m2)), max.(m2 .- abs2.(m), 0.0)
    end
    xs = [f(sample_word(rng, ps)) for _ in ProgressBar(1:n_samples)]
    m = sum(xs) ./ n_samples
    v = sum(x -> abs2.(x .- m), xs) ./ (n_samples - 1)
    return m, sqrt.(v ./ n_samples), v
end

# Effective Luttinger parameter from the block fluctuations F[k] of the particle number in the blocks[k] (sites a:b) of
# an open chain of L sites. Bosonization with hard walls at 0 and L+1 (L′ = L+1) gives, for the block between the cuts
# x₁ = a-1 and x₂ = b,
#   F = K/(2π²) X + c,   X = ln[d(2x₁) d(2x₂) d(x₂-x₁)² / d(x₁+x₂)²],   d(x) = (2L′/π) sin(πx/(2L′)),
# which reduces to F = (K/π²) ln l + c far from the walls. K and c from a linear least-squares fit of F against X over the
# blocks with at least lmin sites whose edges are at least edge_min sites from the walls (single sites and cuts next to a
# wall carry large lattice corrections). Returns (K, c, X, fit) with fit the indices of the fitted blocks; K = c = NaN if
# fewer than 2 blocks qualify. In a gapped phase F saturates and K → 0
function luttinger_parameter(F::AbstractVector{<:Real}, blocks::AbstractVector{<:UnitRange{<:Integer}}, L::Int;
                             lmin::Int = 2, edge_min::Int = 2)
    length(F) == length(blocks) || throw(ArgumentError("F and blocks should have the same length"))
    Lp = L + 1
    d(x) = (2Lp / π) * sin(π * x / (2Lp))
    X = [log(d(2(first(b) - 1)) * d(2last(b)) * d(length(b))^2 / d(first(b) - 1 + last(b))^2) for b in blocks]
    fit = findall(k -> length(blocks[k]) >= lmin && first(blocks[k]) - 1 >= edge_min && L - last(blocks[k]) >= edge_min &&
                       isfinite(X[k]) && isfinite(F[k]), eachindex(blocks))
    length(unique(X[fit])) >= 2 || return NaN, NaN, X, fit
    slope, c = hcat(X[fit], ones(length(fit))) \ F[fit]
    return 2π^2 * slope, c, X, fit
end
