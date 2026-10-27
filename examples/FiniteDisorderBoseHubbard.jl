# Disordered Bose-Hubbard chain on an open chain of L sites, solved with disorder-normalized imaginary time evolution:
#   H_w = -Σ t_{wᵢ} (b†ᵢbᵢ₊₁ + h.c.) + U/2 Σ nᵢ(nᵢ-1) - μ Σ nᵢ,   t_{wᵢ} ∈ {t, t/3} with probability 1/2 each,
# where the hopping of bond (i, i+1) is set by the label of site i. For a fixed t/U the ground state is computed for a
# range of μ/U, and the disorder-averaged density ρ and compressibility κ = dρ/dμ are plotted together.
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
μs = collect(0.0:0.05:.7)
ps = [0.5, 0.5]

# Normalized ITE
Δτ = 0.05
maxD = 8
maxiter = 200
convtol = 1e-8
n_samples = 10^3                                # words are enumerated exactly if q^L ≤ n_samples

d = nmax + 1
n_op = TensorMap(ComplexF64.(diagm(0:nmax)), ℂ^d ← ℂ^d)

# Uniform superposition of 0..nmax bosons on every site and in every sector: every conditional state is normalized
function uniform_state(ps, L, d)
    W = TensorMap(fill(ComplexF64(1/sqrt(d)), d), ℂ^1 ⊗ ℂ^d ← ℂ^1)
    return DisorderKit.FiniteDisorderMPS([[W for _ in ps] for _ in 1:L], ps)
end

# Left transfer of an environment E[bra; ket] by the site tensor A[a s; b] with the operator X applied to the ket
transfer(E, A, X) = @tensor E′[-2; -1] := A[1 3; -1] * X[4; 3] * conj(A[2 4; -2]) * E[2; 1]

# ⟨N⟩ of the conditional state of word w: one left sweep carries the norm (E0) and one inserted n (E1), with a common scale
function particle_number(ψ, w)
    E0 = id(ComplexF64, space(ψ[1, 1], 1))
    E1 = zero(E0)
    for (i, p) in enumerate(w)
        A, I = ψ[i, p], id(ComplexF64, space(ψ[i, p], 2))
        E0, E1 = transfer(E0, A, I), transfer(E1, A, I) + transfer(E0, A, n_op)
        m = LinearAlgebra.norm(E0, Inf)
        E0, E1 = E0 / m, E1 / m
    end
    R = first(DisorderKit._boundary_environment(domain(ψ[length(w), 1])[1]))
    return real(tr(R * E1) / tr(R * E0))
end

# Disorder-averaged density: all words if there are at most n_samples, else sampled words
function disorder_average(ψ; rng = Xoshiro(1))
    q = length(ψ.ps)
    words = float(q)^L <= n_samples ? [collect(w) for w in vec(collect(Iterators.product(fill(1:q, L)...)))] :
                                       [sample(rng, 1:q, Weights(ψ.ps), L) for _ in 1:n_samples]
    weights = float(q)^L <= n_samples ? [prod(ψ.ps[w]) for w in words] : fill(1/length(words), length(words))
    return sum(P * particle_number(ψ, w) / L for (w, P) in zip(words, weights))
end

# Every μ starts from the uniform state, which contains every particle number
ρs, Es, converged = Float64[], Float64[], Bool[]
for μ in μs
    Hs = DisorderKit.random_bose_hubbard([t, t/3], [U], [μ]; nmax)
    alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD, order = 2, convtol, maxiter, verbosity = 2, cross_width = 50)
    ψ, _, info = DisorderKit.groundstate(uniform_state(ps, L, d), Hs, Δτ, alg)
    push!(ρs, disorder_average(ψ))
    push!(Es, info.Es[end])
    push!(converged, info.ϵsconv[end] <= convtol)
    converged[end] || @warn "μ = $μ not converged after $maxiter steps (ϵ_conv = $(info.ϵsconv[end]))"
    @printf("μ = %5.2f: ρ = %.4f, E/L = %.5f, %d steps, max residual %.1e\n",
            μ, ρs[end], Es[end], length(info.Es), info.residual_max[end])
end

# Compressibility κ = dρ/dμ (central differences, one-sided at the ends)
κs = similar(ρs)
for k in eachindex(μs)
    lo, hi = max(k - 1, 1), min(k + 1, length(μs))
    κs[k] = (ρs[hi] - ρs[lo]) / (μs[hi] - μs[lo])
end

println("\n    μ/U        ρ        κ   converged")
for k in eachindex(μs)
    @printf("%7.2f  %7.4f  %7.3f   %s\n", μs[k], ρs[k], κs[k], converged[k])
end

# Figure: ρ on the left axis, κ on the right axis, sharing μ/U
set_theme!(theme_latexfonts())
cρ, cκ = Makie.wong_colors()[1], Makie.wong_colors()[2]
fig = Figure(backgroundcolor = :white, fontsize = 30, size = (1200, 800))
title = L"$L=%$L$, $t/U=%$t$, $t' = t/3$"
axρ = Axis(fig[1, 1]; xlabel = L"μ/U", ylabel = L"ρ", title, ylabelcolor = cρ, yticklabelcolor = cρ)
axκ = Axis(fig[1, 1]; ylabel = L"κ = dρ/dμ", yaxisposition = :right, ylabelcolor = cκ, yticklabelcolor = cκ,
           ygridvisible = false)
hidespines!(axκ)
hidexdecorations!(axκ)
linkxaxes!(axρ, axκ)
hlines!(axρ, [1.0]; color = :gray, linestyle = :dash)
lρ = scatterlines!(axρ, μs, ρs; color = cρ, markersize = 14, linewidth = 3)
lκ = scatterlines!(axκ, μs, κs; color = cκ, marker = :rect, markersize = 14, linewidth = 3)
ylims!(axκ, 0, nothing)
axislegend(axρ, [lρ, lκ], [L"ρ", L"κ"]; position = :lt)

# mkpath("plots")
# file = joinpath("plots", "FiniteDisorderBoseHubbard_L$(L)_t$(t)_bimodal.png")
# save(file, fig)
# println("Saved $file")
fig
