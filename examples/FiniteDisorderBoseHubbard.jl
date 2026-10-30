# Disordered Bose-Hubbard chain on an open chain of L sites, solved with disorder-normalized imaginary time evolution:
#   H_w = -Σ t_{wᵢ} (b†ᵢbᵢ₊₁ + h.c.) + U/2 Σ nᵢ(nᵢ-1) - μ Σ nᵢ,   t_{wᵢ} ∈ {t, t/3} with probability 1/2 each,
# where the hopping of bond (i, i+1) is set by the label of site i. For a fixed t/U the ground state is computed for a
# range of μ/U, and the disorder-averaged density ρ, compressibility κ = dρ/dμ and condensate weight |⟨aᵢ⟩|² are
# plotted together.
#
# H conserves N, so an exact ground state with a fixed particle number has ⟨aᵢ⟩ = 0. ⟨aᵢ⟩ is nonzero only when the state
# mixes neighbouring particle numbers, which the evolution from the uniform state keeps where they are nearly degenerate
# (superfluid) and the bond cap maxD favours. |⟨a⟩|² is thus a superfluid indicator (≈ 0 on a Mott plateau), not a
# converged ground-state value. Every conditional state has its own global phase, so the squared modulus is averaged:
#   |⟨a⟩|² = Σ_w P(w) (1/L) Σᵢ |⟨aᵢ⟩_w|².
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
μs = collect(0.0:0.05:1.)
ps = [0.5, 0.5]

# Normalized ITE
Δτ = 0.05
maxD = 16
maxiter = 400
convtol = 1e-8
n_samples = 10^3                            # words are enumerated exactly if q^L ≤ n_samples

d = nmax + 1
n_op = TensorMap(ComplexF64.(diagm(0:nmax)), ℂ^d ← ℂ^d)
a_op = TensorMap(ComplexF64.(diagm(1 => sqrt.(1:nmax))), ℂ^d ← ℂ^d)   # a|n⟩ = √n |n-1⟩

# Uniform superposition of 0..nmax bosons on every site and in every sector: every conditional state is normalized
function uniform_state(Hs, ps, L, d)
    W = TensorMap(fill(ComplexF64(1/sqrt(d)), d), ℂ^1 ⊗ ℂ^d ← ℂ^1)
    site_ps, sectors = DisorderKit.site_sectors(Hs, ps, L)
    return DisorderKit.FiniteDisorderMPS([[W for _ in p] for p in site_ps], site_ps, sectors)
end

# Left transfer of an environment E[bra; ket] by the site tensor A[a s; b] with the operator X applied to the ket
transfer(E, A, X) = @tensor E′[-2; -1] := A[1 3; -1] * X[4; 3] * conj(A[2 4; -2]) * E[2; 1]

# ⟨N⟩ and ⟨aᵢ⟩ of the conditional state of word w. One left sweep carries the norm (E0) and one inserted n (E1), with a
# common scale, and stores the norm environment before every site; ⟨aᵢ⟩ then follows from a right sweep
function word_observables(ψ, w)
    L = length(w)
    E0 = id(ComplexF64, space(ψ[1, 1], 1))
    E1 = zero(E0)
    lefts = Vector{Any}(undef, L)
    for (i, p) in enumerate(w)
        lefts[i] = E0
        A, I = ψ[i, p], id(ComplexF64, space(ψ[i, p], 2))
        E0, E1 = transfer(E0, A, I), transfer(E1, A, I) + transfer(E0, A, n_op)
        m = LinearAlgebra.norm(E0, Inf)
        E0, E1 = E0 / m, E1 / m
    end
    R = first(DisorderKit._boundary_environment(domain(ψ[L, 1])[1]))
    N = real(tr(R * E1) / tr(R * E0))
    as = Vector{ComplexF64}(undef, L)
    for i in L:-1:1
        A, I = ψ[i, w[i]], id(ComplexF64, space(ψ[i, w[i]], 2))
        as[i] = tr(R * transfer(lefts[i], A, a_op)) / tr(R * transfer(lefts[i], A, I))
        R = first(DisorderKit._advance((R, 0.0), A; right = true))
    end
    return N, as
end

# Disorder-averaged density and |⟨a⟩|²: all words if there are at most n_samples, else sampled words
function disorder_average(ψ; rng = Xoshiro(1))
    exact = prod(float.(length.(ψ.ps))) <= n_samples
    words = DisorderKit.assessment_words(ψ.ps, n_samples; rng)
    weights = exact ? [prod(ψ.ps[i][u] for (i, u) in enumerate(w)) for w in words] : fill(1/length(words), length(words))
    ρ, φ = 0.0, 0.0
    for (w, P) in zip(words, weights)
        N, as = word_observables(ψ, w)
        ρ += P * N / L
        φ += P * sum(abs2, as) / L
    end
    return ρ, φ
end

# Every μ starts from the uniform state, which contains every particle number
ρs, φs, Es, converged = Float64[], Float64[], Float64[], Bool[]
for μ in μs
    Hs = DisorderKit.random_bose_hubbard([t, t/3], [U], [μ]; nmax)
    alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD, order = 2, convtol, maxiter, verbosity = 2, cross_width = 50)
    ψ, _, info = DisorderKit.groundstate(uniform_state(Hs, ps, L, d), Hs, Δτ, alg)
    ρ, φ = disorder_average(ψ)
    push!(ρs, ρ)
    push!(φs, φ)
    push!(Es, info.Es[end])
    push!(converged, info.ϵsconv[end] <= convtol)
    converged[end] || @warn "μ = $μ not converged after $maxiter steps (ϵ_conv = $(info.ϵsconv[end]))"
    @printf("μ = %5.2f: ρ = %.4f, |⟨a⟩|² = %.4f, E/L = %.5f, %d steps, max residual %.1e\n",
            μ, ρ, φ, Es[end], length(info.Es), info.residual_max[end])
end

# Compressibility κ = dρ/dμ (central differences, one-sided at the ends)
κs = similar(ρs)
for k in eachindex(μs)
    lo, hi = max(k - 1, 1), min(k + 1, length(μs))
    κs[k] = (ρs[hi] - ρs[lo]) / (μs[hi] - μs[lo])
end

println("\n    μ/U        ρ        κ   |⟨a⟩|²   converged")
for k in eachindex(μs)
    @printf("%7.2f  %7.4f  %7.3f  %7.4f   %s\n", μs[k], ρs[k], κs[k], φs[k], converged[k])
end

# Figure: ρ and |⟨a⟩|² on the left axis, κ on the right axis, sharing μ/U
set_theme!(theme_latexfonts())
cρ, cκ, cφ = Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3]
fig = Figure(backgroundcolor = :white, fontsize = 30, size = (1200, 800))
title = L"$L=%$L$, $t/U=%$t$, $t' = t/3$"
axρ = Axis(fig[1, 1]; xlabel = L"μ/U", ylabel = L"ρ,\ |\langle a \rangle|^2", title)
axκ = Axis(fig[1, 1]; ylabel = L"κ = dρ/dμ", yaxisposition = :right, ylabelcolor = cκ, yticklabelcolor = cκ,
           ygridvisible = false)
hidespines!(axκ)
hidexdecorations!(axκ)
linkxaxes!(axρ, axκ)
hlines!(axρ, [1.0]; color = :gray, linestyle = :dash)
lρ = scatterlines!(axρ, μs, ρs; color = cρ, markersize = 14, linewidth = 3)
lφ = scatterlines!(axρ, μs, φs; color = cφ, marker = :diamond, markersize = 16, linewidth = 3)
lκ = scatterlines!(axκ, μs, κs; color = cκ, marker = :rect, markersize = 14, linewidth = 3)
ylims!(axρ, 0, nothing)
ylims!(axκ, 0, nothing)
axislegend(axρ, [lρ, lφ, lκ], [L"ρ", L"|\langle a \rangle|^2", L"κ"]; position = :lt)

# mkpath("plots")
# file = joinpath("plots", "FiniteDisorderBoseHubbard_L$(L)_t$(t)_bimodal.png")
# save(file, fig)
# println("Saved $file")
fig
