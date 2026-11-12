# Disordered Bose-Hubbard chain at filling 1 on an open chain of L sites, with U(1)-symmetric TensorMaps:
#   H_w = -t Σ (b†ᵢbᵢ₊₁ + h.c.) + U/2 Σ nᵢ(nᵢ-1) - Σ μ_{wᵢ} nᵢ,   μ_{wᵢ} uniform on q equally spaced values in [-Δ, Δ],
# solved with disorder-normalized imaginary time evolution (Groundstate_FiniteNormalizedITE) for one set of parameters.
#
# The particle number is conserved exactly: |n⟩ carries the U₁ charge n - 1 (random_bose_hubbard with symmetry = U1Irrep,
# filling = 1), so a state with trivial boundary bonds (total charge 0) has exactly N = L bosons for every disorder word.
# The evolution starts from the Mott product state |1 1 … 1⟩ and never leaves the N = L sector. This avoids the problem of
# the unsymmetric runs (FiniteDisorderBoseHubbard.jl), where the evolution from a superposition of all particle numbers
# only selects N through μ. At fixed N a uniform shift of μ only adds a constant, so only the disorder Δ matters.
using TensorKit, DisorderKit, LinearAlgebra, Printf, CairoMakie, Random

# Model
L = 10
U = 4.0
t = 0.5                                          # hopping (t/U)
Δ = 0.25                                          # μᵢ uniform in [-Δ, Δ] (keep 2Δ < U: at 2Δ = U moving a boson
                                                 # from μ = -Δ to a neighbouring μ = +Δ costs nothing, the gap closes
                                                 # and the imaginary time evolution becomes very slow) ...
q = 2                                            # ... discretized to q equally likely values
nmax = 4                                         # at most nmax bosons per site
μs = collect(range(-Δ, Δ; length = q))
ps = fill(1/q, q)

# Normalized ITE
Δτ = 0.05                                         # with order = 2 the time step error is far below the cap error
maxD = [50 => 16, 50 => 32, 100 => 64]                                       # one bond basis carries the conditional states of all words: at
                                                 # maxD = 16 the cap dominates the error (L = 4: 3e-5 in E/L, 5e-6 at 32)
maxiter = 200
convtol = 1e-7

Hs = DisorderKit.random_bose_hubbard([t], [U], μs; nmax, symmetry = U1Irrep, filling = 1)
P = space(Hs.Ds[1], 1)                           # Rep[U₁](-1 => 1, 0 => 1, …, nmax-1 => 1): charge n - 1
S = spacetype(P)
pP = DisorderKit.basis_order(P, [U1Irrep(n - 1) for n in 0:nmax])   # TensorKit basis order of |0⟩, …, |nmax⟩
n_op = TensorMap(ComplexF64.(diagm(0:nmax))[pP, pP], P ← P)

# Mott product state |1⟩ on every site and in every disorder sector: charge 0 everywhere, trivial bonds
function mott_state(Hs, ps, L)
    v = zeros(ComplexF64, nmax + 1)
    v[2] = 1                                     # |n = 1⟩
    W = TensorMap(reshape(v[pP], 1, nmax + 1, 1), oneunit(S) ⊗ P ← oneunit(S))
    site_ps, sectors = DisorderKit.site_sectors(Hs, ps, L)
    return DisorderKit.FiniteDisorderMPS([[W for _ in p] for p in site_ps], site_ps, sectors)
end

alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD, order = 2, convtol, maxiter, verbosity = 1)
ψ, _, info = DisorderKit.groundstate(mott_state(Hs, ps, L), Hs, Δτ, alg)

converged = info.ϵsconv[end] <= convtol
converged || @warn "not converged after $maxiter steps (ϵ_conv = $(info.ϵsconv[end]))"
@printf("\nL = %d, t/U = %.3f, Δ/U = %.3f (q = %d), filling 1\n", L, t/U, Δ/U, q)
@printf("E/L = %.8f after %d steps (ϵ_conv = %.1e), max normalization residual %.1e\n",
        info.Es[end], length(info.Es), info.ϵsconv[end], info.residual_max[end])
println("bond spaces (charge => dimension):")
for i in 1:L-1
    println("  bond $i: ", space(ψ[i, 1], 3)')
end

# Disorder-averaged density ⟨nᵢ⟩ (sums to L exactly) and the density conditioned on the local potential μᵢ = μ_u,
# averaged over the sites: Σ_w P(w) ⟨nᵢ⟩_w δ(wᵢ = u) / P(u)
ns = DisorderKit.expectation_value(ψ, n_op)
println("\n⟨nᵢ⟩ = ", round.(ns; digits = 4), ",  Σᵢ⟨nᵢ⟩ = ", round(sum(ns); digits = 8))
zero_op = zero(n_op)
for (u, μ) in enumerate(μs)
    Os = [s == u ? n_op : zero_op for s in eachindex(μs)]
    nμ = sum(DisorderKit.expectation_value(ψ, Os, i) / ψ.ps[i][u] for i in 1:L) / L
    @printf("⟨n | μ = %+.3f⟩ = %.4f\n", μ, nμ)
end

# Entanglement entropy S_w(k) of the conditional state ψ_w on every bond k, for one typical disorder word w, and the
# distribution of the middle-bond entropy S_w(L/2) over sampled words. Every ψ_w is in the N = L sector, so no charge
# projection is needed
rng = Random.MersenneTwister(1234)
n_samples = 1000
k_mid = L ÷ 2
word_typ = DisorderKit.sample_word(rng, ψ.ps)
S_typ = DisorderKit.conditional_entanglement_entropy(ψ, word_typ)
S_mid = [DisorderKit.conditional_entanglement_entropy(ψ, DisorderKit.sample_word(rng, ψ.ps))[k_mid] for _ in 1:n_samples]
S_typ_avg = sum(S_typ) / length(S_typ)
S_mid_avg = sum(S_mid) / n_samples
S_mid_std = sqrt(sum(abs2, S_mid .- S_mid_avg) / (n_samples - 1))
println("\ntypical word μ_w = ", μs[word_typ])
println("S_w(k) = ", round.(S_typ; digits = 4), ",  bond average = ", round(S_typ_avg; digits = 4))
@printf("S_w(L/2) over %d samples: mean %.4f, std %.4f\n", n_samples, S_mid_avg, S_mid_std)

# Figure: the evolution (energy, convergence, normalization, ranks) and the final state (density, charge structure)
τs = Δτ .* (1:length(info.Es))
E_final = info.Es[end]
set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor = :white, fontsize = 24, size = (1600, 2400))
Label(fig[0, 1:2], L"$L=%$L$, $t/U=%$t$, $Δ/U=%$Δ$ ($q=%$q$), filling 1, $D_{\max}=%$maxD$, $Δτ=%$Δτ$"; fontsize = 28)

ax1 = Axis(fig[1, 1]; xlabel = L"τ", ylabel = L"E/L")
scatterlines!(ax1, τs, info.Es; markersize = 6)
ax1b = Axis(fig[1, 1]; ylabel = L"|E - E_{\mathrm{final}}|/L", yaxisposition = :right, yscale = log10,
            ylabelcolor = Makie.wong_colors()[2], yticklabelcolor = Makie.wong_colors()[2], ygridvisible = false)
hidespines!(ax1b); hidexdecorations!(ax1b); linkxaxes!(ax1, ax1b)
lines!(ax1b, τs[1:end-1], abs.(info.Es[1:end-1] .- E_final) .+ 1e-16; color = Makie.wong_colors()[2])

ax2 = Axis(fig[1, 2]; xlabel = L"τ", ylabel = L"ϵ_{\mathrm{conv}} = |ΔE|/Δτ", yscale = log10)
scatterlines!(ax2, τs, info.ϵsconv .+ 1e-16; markersize = 6)
hlines!(ax2, [convtol]; color = :black, linestyle = :dash, label = L"\mathrm{convtol}")
axislegend(ax2; position = :rt)

ax3 = Axis(fig[2, 1]; xlabel = L"τ", ylabel = L"$|\log Z(w)|$", yscale = log10)
scatterlines!(ax3, τs, info.residual_rms .+ 1e-16; markersize = 6, label = L"\mathrm{rms}")
scatterlines!(ax3, τs, info.residual_max .+ 1e-16; markersize = 6, label = L"\mathrm{max}")
hlines!(ax3, [alg.post_warn]; color = :black, linestyle = :dash, label = L"\mathrm{warning}")
axislegend(ax3; position = :rb)

ax4 = Axis(fig[2, 2]; xlabel = L"τ", ylabel = L"\mathrm{bond\ dimension\ (max\ over\ bonds)}", yscale = log2)
scatterlines!(ax4, τs, maximum.(info.bond_dimensions); markersize = 6, label = L"ψ")
scatterlines!(ax4, τs, maximum.(info.src_ranks); markersize = 6, label = L"\mathrm{SRC\ (before\ rounding)}")
scatterlines!(ax4, τs, maximum.(info.filter_ranks); markersize = 6, label = L"\mathrm{filter}")
scatterlines!(ax4, τs, maximum.(info.partition_ranks); markersize = 6, label = L"\mathrm{partition}")
lines!(ax4, τs, info.caps; color = :black, linestyle = :dash, label = L"\mathrm{cap}")
axislegend(ax4; position = :rb)

ax5 = Axis(fig[3, 1]; xlabel = L"i", ylabel = L"\langle n_i \rangle", xticks = 1:L)
hlines!(ax5, [1.0]; color = :gray, linestyle = :dash)
scatterlines!(ax5, 1:L, ns; markersize = 14, label = L"\overline{\langle n_i \rangle}")
for (u, μ) in enumerate(μs)
    Os = [s == u ? n_op : zero_op for s in eachindex(μs)]
    nμs = [DisorderKit.expectation_value(ψ, Os, i) / ψ.ps[i][u] for i in 1:L]
    scatterlines!(ax5, 1:L, nμs; markersize = 10, linestyle = :dot, label = L"\langle n_i | μ_i = %$(round(μ; digits = 3)) \rangle")
end
axislegend(ax5; position = :rc)

# Final bond dimension per U₁ charge (charge c = number of bosons left of the bond minus the number of sites)
ax6 = Axis(fig[3, 2]; xlabel = L"\mathrm{bond}\ k", ylabel = L"\dim V_c", xticks = 1:L-1)
bonds = [space(ψ[i, 1], 3)' for i in 1:L-1]
charges = sort(unique(c.charge for V in bonds for c in sectors(V)))
for c in charges
    dims = [hassector(V, U1Irrep(c)) ? dim(V, U1Irrep(c)) : 0 for V in bonds]
    scatterlines!(ax6, 1:L-1, dims; markersize = 12, label = L"c = %$c")
end
lines!(ax6, 1:L-1, dim.(bonds); color = :black, linestyle = :dash, label = L"\mathrm{total}")
axislegend(ax6; position = :rt)

# Entanglement entropy of one typical sample on every bond, and the distribution of the middle-bond entropy
word_str = join([μ < 0 ? "-" : "+" for μ in μs[word_typ]])
ax7 = Axis(fig[4, 1]; xlabel = L"\mathrm{bond}\ k", ylabel = L"S_w(k)", xticks = 1:L-1,
           title = L"\mathrm{typical\ sample}\ \mathrm{sign}(μ_w) = %$word_str")
scatterlines!(ax7, 1:L-1, S_typ; markersize = 12, label = L"S_w(k)")
hlines!(ax7, [S_typ_avg]; color = :black, linestyle = :dash, label = L"\mathrm{bond\ average}")
axislegend(ax7; position = :rt)

ax8 = Axis(fig[4, 2]; xlabel = L"S_w(L/2)", ylabel = L"P(S_w(L/2))",
           title = L"%$(n_samples)\ \mathrm{samples}")
hist!(ax8, S_mid; bins = 40, normalization = :pdf)
vlines!(ax8, [S_mid_avg]; color = :black, linestyle = :dash, label = L"\mathrm{mean}")
vlines!(ax8, [S_typ[k_mid]]; color = Makie.wong_colors()[2], label = L"\mathrm{typical\ sample}")
axislegend(ax8; position = :rt)

fig
