
# Hopping matrix of the RTFIM H = -∑ Jᵢ ZᵢZᵢ₊₁ - ∑ hᵢ Xᵢ on an open chain of length L = length(hs)
# Jordan-Wigner (Xᵢ = 1 - 2cᵢ†cᵢ) gives H = ½ Ψ† H_BdG Ψ with Nambu spinor Ψ = (c₁..c_L, c₁†..c_L†),
# which BenchmarkFreeFermions treats as a hopping problem H = -∑ Tᵢⱼ Ψᵢ† Ψⱼ on 2L sites at μ = 0, i.e. T = -H_BdG
function RTFIM_hopping(hs::Vector{<:Real}, Js::Vector{<:Real})
    L = length(hs)
    length(Js) == L-1 || throw(ArgumentError("Need length(Js) == length(hs) - 1 for an open chain"))

    A = zeros(Float64, L, L) # normal part
    B = zeros(Float64, L, L) # pairing part
    for i in 1:L
        A[i, i] = 2 * hs[i]
    end
    for i in 1:L-1
        A[i, i+1] = -Js[i]
        A[i+1, i] = -Js[i]
        B[i, i+1] = -Js[i]
        B[i+1, i] = Js[i]
    end
    H_BdG = [A B; -B -A]
    return -H_BdG
end

# Energy, transverse magnetisation ⟨Xᵢ⟩ and spin-spin correlator ⟨Z_{i0} Z_{i0+r}⟩ (r = 1..L-i0)
# of the RTFIM for a single configuration (hs, Js) at inverse temperature β
function RTFIM_free_fermion(hs::Vector{<:Real}, Js::Vector{<:Real}; β::Real=Inf, i0::Int=1)
    L = length(hs)
    T = RTFIM_hopping(hs, Js)

    # ½ for the Nambu double counting
    ϵ = SingleParticleSpectrum(T)
    E = Energy(ϵ, β, 0.0) / 2

    # Nambu Green function G[a, b] = ⟨Ψₐ Ψ_b†⟩
    ξ, V = EigenModes(T)
    G = GreenFunction(ξ, V, β)
    cc_dag(i, j) = G[i, j]          # ⟨cᵢ cⱼ†⟩
    cc(i, j) = G[i, L+j]            # ⟨cᵢ cⱼ⟩
    c_dagc_dag(i, j) = G[L+i, j]    # ⟨cᵢ† cⱼ†⟩
    c_dagc(i, j) = G[L+i, L+j]      # ⟨cᵢ† cⱼ⟩

    Xs = [real(1 - 2 * c_dagc(i, i)) for i in 1:L]

    # Majorana correlator ⟨Bₖ Aₗ⟩ with Aₖ = cₖ† + cₖ, Bₖ = cₖ† - cₖ
    BA(k, l) = c_dagc_dag(k, l) + c_dagc(k, l) - cc_dag(k, l) - cc(k, l)
    # Zᵢ Zⱼ = ∏_{k=i}^{j-1} Bₖ Aₖ₊₁, evaluated with Wick's theorem as a determinant
    ZZs = map(1:L-i0) do r
        M = [BA(k, l+1) for k in i0:i0+r-1, l in i0:i0+r-1]
        real(det(M))
    end

    # Entanglement entropy of sites 1..k on every bond k = 1..L-1, from the eigenvalues λ of the Nambu Green function
    # restricted to the block (modes come in pairs λ, 1-λ). The JW string of a left block stays inside it, so this
    # equals the spin entropy (for β < ∞ it is the von Neumann entropy of the mixed reduced state)
    Ss = map(1:L-1) do k
        block = [1:k; L+1:L+k]
        λ = clamp.(eigvals(Hermitian(G[block, block])), 1e-14, 1 - 1e-14)
        -sum(λ .* log.(λ))
    end

    return E, Xs, ZZs, Ss
end

# Draw a disorder configuration on a chain of length L: every site k gets a disorder sector pₖ ~ ps,
# with sectors ordered as in random_transverse_field_ising, and Jₖ couples sites k and k+1
function sample_RTFIM_configuration(L::Int, Js::Vector{<:Real}, hs::Vector{<:Real}; ps::Vector{<:Real}=fill(1/(length(Js)*length(hs)), length(Js)*length(hs)))
    sectors = vec(collect(Iterators.product(hs, Js)))
    length(ps) == length(sectors) || throw(ArgumentError("ps should have length length(hs)*length(Js)"))

    pks = sample(1:length(sectors), Weights(ps), L)
    hconf = [sectors[p][1] for p in pks]
    Jconf = [sectors[p][2] for p in pks[1:L-1]]
    return hconf, Jconf
end

# All distinct disorder configurations of a chain of length L as (weight, hconf, Jconf).
# Sites 1..L-1 run over every sector, the last site only over the fields (its coupling is unused),
# weighted by the marginal of ps over the couplings. Zero-weight configurations are skipped.
function enumerate_RTFIM_configurations(L::Int, Js::Vector{<:Real}, hs::Vector{<:Real}; ps::Vector{<:Real}=fill(1/(length(Js)*length(hs)), length(Js)*length(hs)))
    sectors = vec(collect(Iterators.product(hs, Js)))
    length(ps) == length(sectors) || throw(ArgumentError("ps should have length length(hs)*length(Js)"))

    # sectors are ordered with h running fastest, so sector p has field index mod1(p, length(hs))
    ph = [sum(ps[p] for p in eachindex(sectors) if mod1(p, length(hs)) == a) for a in eachindex(hs)]

    confs = Iterators.product(ntuple(_ -> eachindex(sectors), L-1)..., eachindex(hs))
    configurations = map(vec(collect(confs))) do c
        pks = c[1:L-1]
        weight = prod(ps[p] for p in pks; init=1.0) * ph[c[L]]
        hconf = [[sectors[p][1] for p in pks]; hs[c[L]]]
        Jconf = [sectors[p][2] for p in pks]
        (weight, hconf, Jconf)
    end
    return filter(c -> !iszero(c[1]), configurations)
end

# Disorder average of the RTFIM on a chain of length L. If the number of distinct configurations is at most
# n_samples, every configuration is enumerated and the average is exact (zero errors); otherwise n_samples
# configurations are sampled
function sample_RTFIM(L::Int, Js::Vector{<:Real}, hs::Vector{<:Real}; ps::Vector{<:Real}=fill(1/(length(Js)*length(hs)), length(Js)*length(hs)), n_samples::Int=1000, β::Real=Inf, i0::Int=L÷4)
    n_conf = float(length(ps))^(L-1) * length(hs)
    if n_conf <= n_samples
        E_avg = 0.0
        X_avg = zeros(Float64, L)
        ZZ_avg = zeros(Float64, L-i0)
        S_avg = zeros(Float64, L-1)
        for (w, hconf, Jconf) in ProgressBar(enumerate_RTFIM_configurations(L, Js, hs; ps=ps))
            E, X, ZZ, S = RTFIM_free_fermion(hconf, Jconf; β=β, i0=i0)
            E_avg += w * E / L
            X_avg .+= w .* X
            ZZ_avg .+= w .* ZZ
            S_avg .+= w .* S
        end
        return (E = E_avg, E_err = 0.0,
                Xs = X_avg, Xs_err = zeros(Float64, L),
                ZZs = ZZ_avg, ZZs_err = zeros(Float64, L-i0),
                Ss = S_avg, Ss_err = zeros(Float64, L-1),
                n_samples = Int(n_conf), exact = true)
    end

    Es = zeros(Float64, n_samples)
    Xs = zeros(Float64, n_samples, L)
    ZZs = zeros(Float64, n_samples, L-i0)
    Ss = zeros(Float64, n_samples, L-1)

    for n in ProgressBar(1:n_samples)
        hconf, Jconf = sample_RTFIM_configuration(L, Js, hs; ps=ps)
        E, X, ZZ, S = RTFIM_free_fermion(hconf, Jconf; β=β, i0=i0)
        Es[n] = E / L
        Xs[n, :] = X
        ZZs[n, :] = ZZ
        Ss[n, :] = S
    end

    err(x) = std(x; dims=1) ./ sqrt(n_samples)
    return (E = mean(Es), E_err = std(Es)/sqrt(n_samples),
            Xs = vec(mean(Xs; dims=1)), Xs_err = vec(err(Xs)),
            ZZs = vec(mean(ZZs; dims=1)), ZZs_err = vec(err(ZZs)),
            Ss = vec(mean(Ss; dims=1)), Ss_err = vec(err(Ss)),
            n_samples = n_samples, exact = false)
end
