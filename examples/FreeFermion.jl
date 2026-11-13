# Exact free-fermion solution of the random transverse-field Ising chain (DisorderKit.sample_RTFIM): every disorder
# realization H_w = -Σ Jᵢ ZᵢZᵢ₊₁ - Σ hᵢ Xᵢ maps to free Majorana fermions, so its ground-state energy, ⟨Xᵢ⟩, ⟨ZᵢZⱼ⟩ and
# entanglement entropy S(k) cost O(L³). The disorder averages are exact (all realizations enumerated) for small L and
# sampled otherwise. This is the reference for the normalized ITE in examples/RTFIM.jl.
#
# Shown: the disorder-averaged entanglement entropy S̄(L/2) against L at the critical point δ = 0 and off criticality, and
# the Calabrese-Cardy fit S̄(k) = c_eff/6 ln[(2L/π) sin(πk/L)] + s₀ of the largest chain, with c_eff → ln(2)/2 at the
# infinite-randomness fixed point (c = 1/2 for the clean chain).
using DisorderKit, CairoMakie, Printf

W = 0.8                                          # disorder strength: h ∈ e^δ {1, e^{-2W}}, J ∈ {1, e^{-2W}}
Js = [1.0, exp(-2W)]
ps = fill(0.25, 4)
δs = [0.0, 0.5]
Ls = [8, 16, 32, 64]
n_samples = 2000

results = Dict(δ => [DisorderKit.sample_RTFIM(L, Js, exp(δ) .* [1.0, exp(-2W)]; ps, n_samples) for L in Ls] for δ in δs)
for δ in δs, (L, r) in zip(Ls, results[δ])
    @printf("δ = %.2f, L = %3d: E/L = %.6f ± %.1e,  S̄(L/2) = %.4f ± %.1e  (%s)\n", δ, L, r.E, r.E_err, r.Ss[L÷2],
            r.Ss_err[L÷2], r.exact ? "exact" : "$(r.n_samples) samples")
end

# Calabrese-Cardy fit at δ = 0 for the largest L, away from the edges
L = last(Ls)
r = last(results[0.0])
ks = 1:L-1
X = log.((2L / π) .* sin.(π .* ks ./ L)) ./ 6
kfit = L÷8:L-L÷8
c_eff, s0 = hcat(X[kfit], ones(length(kfit))) \ r.Ss[kfit]
@printf("c_eff = %.3f (IRFP: ln(2)/2 = %.3f, clean: 0.5)\n", c_eff, log(2) / 2)

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor = :white, fontsize = 24, size = (1600, 650))
ax1 = Axis(fig[1, 1]; xlabel = L"L", ylabel = L"\overline{S(L/2)}", xscale = log2, title = L"W = %$W")
for δ in δs
    S = [res.Ss[L÷2] for (L, res) in zip(Ls, results[δ])]
    err = [res.Ss_err[L÷2] for (L, res) in zip(Ls, results[δ])]
    errorbars!(ax1, Ls, S, err; whiskerwidth = 8)
    scatterlines!(ax1, Ls, S; markersize = 14, label = L"δ = %$δ")
end
axislegend(ax1; position = :lt)
ax2 = Axis(fig[1, 2]; xlabel = L"\frac{1}{6}\ln[(2L/π)\sin(πk/L)]", ylabel = L"\overline{S(k)}",
           title = L"δ = 0,\ L = %$L,\ c_\mathrm{eff} = %$(round(c_eff; digits = 3))")
scatter!(ax2, X, r.Ss; markersize = 10, label = L"\mathrm{FF}")
lines!(ax2, X, c_eff .* X .+ s0; color = :black, linestyle = :dash, label = L"\mathrm{fit}")
axislegend(ax2; position = :lt)

fig
