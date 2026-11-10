# Nearest-neighbour disorder Hamiltonians, one MPO per disorder sector p, in Jordan form
#   W_p = [1  L_p  D_p;
#          0  A_p  R_p;
#          0  0    1  ]
# with P the physical space and V the space of the bond terms:
#   L_p : P ← P ⊗ V        operator on the left site of a bond term
#   R_p : V ⊗ P ← P        operator on the right site of a bond term (carries the coupling)
#   A_p : V ⊗ P ← P ⊗ V    passes a bond term through a site (zero for nearest-neighbour terms)
#   D_p : P ← P            on-site term
# The bond term of sites (i, i+1) is Σ_v L[v] ⊗ R[v].
struct DisorderMPOHam{T <: Number, S <: IndexSpace}
    As::Vector{<:AbstractTensorMap{T, S, 2, 2}}
    Rs::Vector{<:AbstractTensorMap{T, S, 2, 1}}
    Ls::Vector{<:AbstractTensorMap{T, S, 1, 2}}
    Ds::Vector{<:AbstractTensorMap{T, S, 1, 1}}
    function DisorderMPOHam(As::Vector{<:AbstractTensorMap{T, S, 2, 2}}, Rs::Vector{<:AbstractTensorMap{T, S, 2, 1}},
                            Ls::Vector{<:AbstractTensorMap{T, S, 1, 2}}, Ds::Vector{<:AbstractTensorMap{T, S, 1, 1}}) where {T, S}
        length(As) == length(Rs) == length(Ls) == length(Ds) || throw(ArgumentError("All input vectors must have the same length"))
        return new{T, S}(As, Rs, Ls, Ds)
    end
end

# Random transverse-field Ising chain H_w = -Σ h_{w_i} X_i - Σ J_{w_i} Z_i Z_{i+1} (- ϵ Σ Z_i), one sector per pair
# (h, J) with h running fastest
function random_transverse_field_ising(Js::Vector{Float64}, hs::Vector{Float64}; ϵ::Real = 0.0)
    X = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)
    A = zeros(ComplexF64, ℂ^1 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^1)
    R = TensorMap(ComplexF64[1 0; 0 -1], ℂ^1 ⊗ ℂ^2 ← ℂ^2)
    L = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2 ⊗ ℂ^1)

    sectors = vec(collect(Iterators.product(hs, Js)))
    return DisorderMPOHam(fill(A, length(sectors)), [-J * R for (h, J) in sectors], fill(L, length(sectors)),
                          [-h * X - ϵ * Z for (h, J) in sectors])
end

# Random Heisenberg chain H_w = -Σ J_{w_i} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}), one sector per J
function random_heisenberg(Js::Vector{Float64})
    X = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
    Y = TensorMap(ComplexF64[0 -im; im 0], ℂ^2 ← ℂ^2)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)
    A = zeros(ComplexF64, ℂ^3 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^3)

    # Split the bond term X⊗X + Y⊗Y + Z⊗Z into L[v] ⊗ R[v] (rank 3)
    U, S, V = svd_trunc(permute(X ⊗ X + Y ⊗ Y + Z ⊗ Z, ((1, 3), (2, 4))); trunc = trunctol(; atol = 1e-12))
    L = permute(U * S, ((1,), (2, 3)))
    R = permute(V, ((1, 2), (3,)))
    D = zeros(ComplexF64, ℂ^2 ← ℂ^2)

    return DisorderMPOHam(fill(A, length(Js)), [-J * R for J in Js], fill(L, length(Js)), fill(D, length(Js)))
end

# Random Bose-Hubbard chain with at most nmax bosons per site,
#   H_w = -Σ t_{w_i} (b†_i b_{i+1} + b_i b†_{i+1}) + Σ [U_{w_i}/2 n_i(n_i - 1) - μ_{w_i} n_i],
# one sector per triple (μ, U, t) with μ running fastest.
# symmetry = U1Irrep conserves the particle number: |n⟩ carries the U₁ charge n - filling, so a state with total charge 0
# (trivial boundary bonds) has exactly N = filling L bosons. The bond term space is then V = U₁(-1 => 1, 1 => 1).
function random_bose_hubbard(ts::Vector{Float64}, Us::Vector{Float64}, μs::Vector{Float64}; nmax::Int = 3,
                             symmetry::Type{<:Sector} = Trivial, filling::Integer = 0)
    d = nmax + 1
    b = zeros(ComplexF64, d, d)
    for k in 1:nmax
        b[k, k+1] = sqrt(k)                                   # b |k⟩ = √k |k-1⟩ (index k+1 holds |k⟩)
    end
    n = Diagonal(ComplexF64.(0:nmax))
    Ld = cat(b', b; dims = 3)                                 # L[v] = (b†, b)
    Rd = permutedims(cat(b, b'; dims = 3), (3, 1, 2))         # R[v] = (b, b†)
    if symmetry === Trivial
        iszero(filling) || throw(ArgumentError("a fixed filling requires symmetry = U1Irrep"))
        P, V = ℂ^d, ℂ^2
        pP, pV = 1:d, 1:2
    elseif symmetry === U1Irrep
        P = U1Space(k - filling => 1 for k in 0:nmax)
        V = U1Space(1 => 1, -1 => 1)
        pP = basis_order(P, [U1Irrep(k - filling) for k in 0:nmax])
        pV = basis_order(V, [U1Irrep(1), U1Irrep(-1)])        # L[1] = b† raises the charge, L[2] = b lowers it
    else
        throw(ArgumentError("symmetry $symmetry not implemented"))
    end
    A = zeros(ComplexF64, V ⊗ P ← P ⊗ V)
    L = TensorMap(Ld[pP, pP, pV], P ← P ⊗ V)
    R = TensorMap(Rd[pV, pP, pP], V ⊗ P ← P)

    sectors = vec(collect(Iterators.product(μs, Us, ts)))
    Ds = [TensorMap(Matrix(U / 2 * n * (n - I) - μ * n)[pP, pP], P ← P) for (μ, U, _) in sectors]
    return DisorderMPOHam(fill(A, length(sectors)), [-t * R for (μ, U, t) in sectors], fill(L, length(sectors)), Ds)
end

# Bose-Hubbard chain with correlated hopping and potential disorder,
#   H_w = -Σ t_{wᵢ} (b†ᵢbᵢ₊₁ + h.c.) + U/2 Σ nᵢ(nᵢ-1) - Σ μ_{wᵢ} nᵢ,   t_{wᵢ} ∈ {t(1-Ω), t(1+Ω)}, μ_{wᵢ} ∈ {μ-Δ, μ+Δ},
# where the label wᵢ of site i sets both μᵢ and the hopping of bond (i, i+1). Hopping and potential are correlated by
# r ∈ (-1, 1): P = (1+r)/4 if both are high or both low, (1-r)/4 otherwise. Identical values (Δ = 0 or Ω = 0) are merged
# into one with the marginal probability: 4 sectors for Δ, Ω > 0, 2 if one of them is 0, 1 if both are.
# Returns (Hs, ps); the sectors run over Iterators.product(μs, [U], ts), μ fastest (column-major order of P):
# for Δ, Ω > 0: 1 = (μ-Δ, t(1-Ω)), 2 = (μ+Δ, t(1-Ω)), 3 = (μ-Δ, t(1+Ω)), 4 = (μ+Δ, t(1+Ω)).
# symmetry and filling are passed to random_bose_hubbard (U1Irrep: particle number conserving tensors at the given filling).
function correlated_bose_hubbard(t::Real, U::Real, μ::Real, Δ::Real, Ω::Real, r::Real; nmax::Int = 3,
                                 symmetry::Type{<:Sector} = Trivial, filling::Integer = 0)
    -1 < r < 1 || throw(ArgumentError("r = $r must lie in (-1, 1): SRC requires strictly positive disorder probabilities"))
    Δ >= 0 || throw(ArgumentError("Δ = $Δ must be ≥ 0"))
    0 <= Ω <= 1 || throw(ArgumentError("Ω = $Ω must lie in [0, 1], so that the hopping t(1-Ω) is not negative"))
    P = [(1+r)/4 (1-r)/4; (1-r)/4 (1+r)/4]           # P[a, b]: μ_a ∈ (μ-Δ, μ+Δ), t_b ∈ (t(1-Ω), t(1+Ω))
    μs = iszero(Δ) ? [float(μ)] : [μ-Δ, μ+Δ]
    iszero(Δ) && (P = sum(P; dims = 1))
    ts = iszero(Ω) ? [float(t)] : [t*(1-Ω), t*(1+Ω)]
    iszero(Ω) && (P = sum(P; dims = 2))
    Hs = random_bose_hubbard(Float64.(ts), [Float64(U)], Float64.(μs); nmax, symmetry, filling)
    return Hs, vec(P)
end

# Jordan MPO tensor W = [1 L D; 0 A R; 0 0 1] as a SparseBlockTensorMap (1 ⊕ V ⊕ 1) ⊗ P ← P ⊗ (1 ⊕ V ⊕ 1), 1 the unit space
function jordan_mpo_tensor(A::AbstractTensorMap, R::AbstractTensorMap, L::AbstractTensorMap, D::AbstractTensorMap)
    P = space(D, 1)
    V = domain(L)[2]
    S = spacetype(D)
    virtual = BlockTensorKit.boxplus(oneunit(S), V, oneunit(S))
    W = SparseBlockTensorMap{AbstractTensorMap{ComplexF64, S, 2, 2}}(undef, virtual ⊗ BlockTensorKit.boxplus(P),
                                                                     BlockTensorKit.boxplus(P) ⊗ virtual)
    W[1, 1, 1, 1] = BraidingTensor{ComplexF64, S}(P, oneunit(S))
    W[1, 1, 1, 2] = BlockTensorKit.insertrightunit(L, 0)
    W[1, 1, 1, 3] = MPSKit.add_util_leg(D)
    W[2, 1, 1, 2] = A
    W[2, 1, 1, 3] = BlockTensorKit.insertrightunit(R, 3)
    W[3, 1, 1, 3] = BraidingTensor{ComplexF64, S}(P, oneunit(S))
    return W
end

# Imaginary time evolution MPO exp(-dt H_p) of every sector (TaylorCluster of order N)
function time_evolution_MPO(Hs::DisorderMPOHam, dt::Real; N::Int = 1)
    Us = Vector{typeof(Hs.As[1])}(undef, length(Hs.Rs))   # dense MPO tensors
    for p in eachindex(Hs.Rs)
        W = jordan_mpo_tensor(Hs.As[p], Hs.Rs[p], Hs.Ls[p], Hs.Ds[p])
        H = MPOHamiltonian(PeriodicArray([MPSKit.JordanMPOTensor(W)]))
        Us[p] = MPSKit.make_time_mpo(H, dt, TaylorCluster(; N = N, compression = true, extension = true); imaginary_evolution = true)[1]
    end
    return InfiniteDisorderMPO(Us)
end
