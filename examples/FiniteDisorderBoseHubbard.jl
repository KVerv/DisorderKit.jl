# Disordered Bose-Hubbard chain on an open chain of L sites, solved with disorder-normalized imaginary time evolution:
#   H_w = -Σ t_{wᵢ} (b†ᵢbᵢ₊₁ + h.c.) + U/2 Σ nᵢ(nᵢ-1) - μ Σ nᵢ,   t_{wᵢ} ∈ {t, t/3} with probability 1/2 each,
# where the hopping of bond (i, i+1) is set by the label of site i. For a fixed t/U the ground state is computed for a
# range of μ/U, and the disorder-averaged density ρ, compressibility κ = dρ/dμ, Luttinger parameter K and superfluid
# stiffness ρ_s are plotted.
#
# On an open chain a phase twist can be gauged away, so ρ_s is obtained through the Luttinger relations ρ_s = uK/π and
# κ = K/(πu), i.e. ρ_s = K²/(π²κ). K follows from the bipartite particle-number fluctuations of the subsystem A = 1..ℓ,
#   F(ℓ) = ⟨N_A²⟩ - ⟨N_A⟩² ≈ K/(2π²) ln[(2L/π) sin(πℓ/L)] + c   (open boundaries),
# evaluated for every disorder word and then averaged. K < 3/2 signals the Bose glass (Giamarchi-Schulz).
#
# H conserves the particle number, so imaginary time evolution only reaches the ground state from a start that contains
# its particle number: every μ starts from the uniform superposition of all occupations (not from the previous ground
# state, whose particle number is fixed). Near the edges of a Mott plateau the gap between neighbouring particle numbers
# is small, so the evolution needs a long imaginary time (τ ~ 100 here).
using TensorKit, DisorderKit, LinearAlgebra, Random, StatsBase, CairoMakie, Printf

# Model
U = 1.0
t = 0.16                                          # hopping of the strong bonds (fixed t/U), the weak bonds have t/3
nmax = 4                                         # at most nmax bosons per site
L = 6
μs = collect(0.0:0.5:1.)
ps = [0.5, 0.5]

# Normalized ITE
Δτ = 0.05
maxD = 8
maxiter = 500
convtol = 1e-8
n_samples = 10^3                                # words are enumerated exactly if q^L ≤ n_samples
κ_min = 0.05                                     # ρ_s is shown only where κ > κ_min

d = nmax + 1
n_op = TensorMap(ComplexF64.(diagm(0:nmax)), ℂ^d ← ℂ^d)
n2_op = n_op * n_op

# Uniform superposition of 0..nmax bosons on every site and in every sector: every conditional state is normalized
function uniform_state(ps, L, d)
    W = TensorMap(fill(ComplexF64(1/sqrt(d)), d), ℂ^1 ⊗ ℂ^d ← ℂ^1)
    return DisorderKit.FiniteDisorderMPS([[W for _ in ps] for _ in 1:L], ps)
end

# Left transfer of an environment E[bra; ket] by the site tensor A[a s; b] with the operator X applied to the ket
transfer(E, A, X) = @tensor E′[-2; -1] := A[1 3; -1] * X[4; 3] * conj(A[2 4; -2]) * E[2; 1]

# Number statistics of the conditional state of word w: ⟨N⟩ and F(ℓ) = ⟨N_A²⟩ - ⟨N_A⟩² for A = 1..ℓ, ℓ = 1..L-1.
# One left sweep carries the norm (E0), one inserted n (E1) and two inserted n's (E2), all with a common scale
function number_statistics(ψ, w)
    L = length(w)
    E0 = id(ComplexF64, space(ψ[1, 1], 1))
    E1, E2 = zero(E0), zero(E0)
    lefts = Vector{Any}(undef, L)
    for (i, p) in enumerate(w)
        A, I = ψ[i, p], id(ComplexF64, space(ψ[i, p], 2))
        E0, E1, E2 = transfer(E0, A, I), transfer(E1, A, I) + transfer(E0, A, n_op),
                     transfer(E2, A, I) + 2 * transfer(E1, A, n_op) + transfer(E0, A, n2_op)
        m = LinearAlgebra.norm(E0, Inf)
        E0, E1, E2 = E0 / m, E1 / m, E2 / m
        lefts[i] = (E0, E1, E2)
    end
    # Right norm environments R[ket; bra] of the sites ℓ+1..L
    rights = Vector{Any}(undef, L)
    rights[L] = DisorderKit._boundary_environment(domain(ψ[L, 1])[1])
    for i in L:-1:2
        rights[i-1] = DisorderKit._advance(rights[i], ψ[i, w[i]]; right = true)
    end
    moments(ℓ) = (Z = tr(rights[ℓ][1] * lefts[ℓ][1]); (tr(rights[ℓ][1] * lefts[ℓ][2]) / Z, tr(rights[ℓ][1] * lefts[ℓ][3]) / Z))
    F = map(1:L-1) do ℓ
        N, N2 = moments(ℓ)
        real(N2 - N^2)
    end
    return real(first(moments(L))), F
end

# Disorder-averaged density and number fluctuations: all words if there are at most n_samples, else sampled words
function disorder_average(ψ; rng = Xoshiro(1))
    q = length(ψ.ps)
    words = float(q)^L <= n_samples ? [collect(w) for w in vec(collect(Iterators.product(fill(1:q, L)...)))] :
                                       [sample(rng, 1:q, Weights(ψ.ps), L) for _ in 1:n_samples]
    weights = float(q)^L <= n_samples ? [prod(ψ.ps[w]) for w in words] : fill(1/length(words), length(words))
    ρ, F = 0.0, zeros(L-1)
    for (w, P) in zip(words, weights)
        N, Fw = number_statistics(ψ, w)
        ρ += P * N / L
        F .+= P .* Fw
    end
    return ρ, F
end

# Luttinger parameter from F(ℓ) = K/(2π²) ln[(2L/π) sin(πℓ/L)] + c over the bulk bonds
function luttinger_parameter(F)
    edge = max(2, L ÷ 4)
    ℓs = edge:L-edge
    x = log.((2L/π) .* sin.(π .* ℓs ./ L))
    a, _ = [x ones(length(x))] \ F[ℓs]
    return 2π^2 * a
end

# Every μ starts from the uniform state, which contains every particle number
ρs, Ks, Es, converged = Float64[], Float64[], Float64[], Bool[]
for μ in μs
    Hs = DisorderKit.random_bose_hubbard([t, t/3], [U], [μ]; nmax)
    alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD, order = 2, convtol, maxiter, verbosity = 2)
    ψ, _, info = DisorderKit.groundstate(uniform_state(ps, L, d), Hs, Δτ, alg)
    ρ, F = disorder_average(ψ)
    push!(ρs, ρ)
    push!(Ks, luttinger_parameter(F))
    push!(Es, info.Es[end])
    push!(converged, info.ϵsconv[end] <= convtol)
    converged[end] || @warn "μ = $μ not converged after $maxiter steps (ϵ_conv = $(info.ϵsconv[end]))"
    @printf("μ = %5.2f: ρ = %.4f, K = %.3f, E/L = %.5f, %d steps, max residual %.1e\n",
            μ, ρ, Ks[end], Es[end], length(info.Es), info.residual_max[end])
end

# Compressibility κ = dρ/dμ (central differences, one-sided at the ends) and stiffness ρ_s = K²/(π²κ)
κs = similar(ρs)
for k in eachindex(μs)
    lo, hi = max(k - 1, 1), min(k + 1, length(μs))
    κs[k] = (ρs[hi] - ρs[lo]) / (μs[hi] - μs[lo])
end
ρ_ss = [κ > κ_min ? K^2 / (π^2 * κ) : NaN for (K, κ) in zip(Ks, κs)]

println("\n    μ/U        ρ        κ        K      ρ_s   converged")
for k in eachindex(μs)
    @printf("%7.2f  %7.4f  %7.3f  %7.3f  %7.3f   %s\n", μs[k], ρs[k], κs[k], Ks[k], ρ_ss[k], converged[k])
end

# Figure
set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor = :white, fontsize = 30, size = (1800, 1300))
title = L"$L=%$L$, $t/U=%$t$, $t' = t/3$"
ax1 = Axis(fig[1, 1]; xlabel = L"μ/U", ylabel = L"ρ", title)
ax2 = Axis(fig[1, 2]; xlabel = L"μ/U", ylabel = L"κ = dρ/dμ")
ax3 = Axis(fig[2, 1]; xlabel = L"μ/U", ylabel = L"ρ_s = K^2/(π^2 κ)")
ax4 = Axis(fig[2, 2]; xlabel = L"μ/U", ylabel = L"K")
scatterlines!(ax1, μs, ρs; markersize = 12)
hlines!(ax1, [1.0]; color = :gray, linestyle = :dash)
scatterlines!(ax2, μs, κs; markersize = 12)
scatterlines!(ax3, μs, ρ_ss; markersize = 12)
scatterlines!(ax4, μs, Ks; markersize = 12)
hlines!(ax4, [1.5]; color = :gray, linestyle = :dash, label = L"K = 3/2")
axislegend(ax4; position = :rt)

# mkpath("plots")
# file = joinpath("plots", "FiniteDisorderBoseHubbard_L$(L)_t$(t)_bimodal.png")
# save(file, fig)
# println("Saved $file")
fig
