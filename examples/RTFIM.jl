# Random transverse-field Ising chain on an open chain of L sites,
#   H_w = -Σ J_{wᵢ} Zᵢ Zᵢ₊₁ - Σ h_{wᵢ} Xᵢ,
# solved with disorder-normalized imaginary time evolution (Groundstate_FiniteNormalizedITE) and compared with the exact
# free-fermion solution of every disorder realization (DisorderKit.sample_RTFIM, see examples/FreeFermion.jl).
#
# The bond and field values follow the binary distribution of the infinite-randomness literature: hᵢ ∈ e^δ {1, e^{-2W}},
# Jᵢ ∈ {1, e^{-2W}} with equal probabilities, so δ = 0 is the critical point and W sets the disorder strength.
using TensorKit, DisorderKit, CairoMakie, TimerOutputs, Printf

# Model
J₀ = 1.0
δ = 0.0
W = 0.8
hs = exp(δ) * [1.0, exp(-2W)]
Js = J₀ * [1.0, exp(-2W)]
ps = fill(0.25, 4)                               # P(h, J) for the pairs (h, J), h running fastest
Hs = DisorderKit.random_transverse_field_ising(Js, hs)

# Normalized ITE
L = 6
Δτ = 0.02
maxiter = 700
maxD = [100 => 8, 200 => 16]                     # cap schedule (number of steps => cap), the last cap is kept
ψ0 = DisorderKit.plus_state(Hs, ps, L)           # |+x⟩ on every site, every disorder word normalized

Z = TensorMap([1.0 0.0; 0.0 -1.0], ℂ^2, ℂ^2)
X = TensorMap([0.0 1.0; 1.0 0.0], ℂ^2, ℂ^2)

# Energy density and average |⟨Zᵢ⟩| after every time step
finalize(ρ, Hs) = (DisorderKit.energy_density(ρ, Hs), sum(abs, DisorderKit.expectation_value(ρ, Z)) / length(ρ))
to = TimerOutput()
alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD, order = 2, convtol = 1e-8, maxiter, verbosity = 1, timer_output = to,
                                                  finalizer = DisorderKit.Finalizer(Tuple{Float64, Float64}, finalize))
ψ, data, info = DisorderKit.groundstate(ψ0, Hs, Δτ, alg)
Es = getindex.(data, 1)
τs = Δτ .* (1:length(Es))

# Observables of the final state: exact disorder averages of on-site operators (no sampling)
E = DisorderKit.energy_density(ψ, Hs)
Zs = DisorderKit.expectation_value(ψ, Z)        # ⟨Zᵢ⟩ (zero by symmetry, up to the evolution)
Xs = DisorderKit.expectation_value(ψ, X)        # ⟨Xᵢ⟩
pairs = [((L - r) ÷ 2, (L - r) ÷ 2 + r) for r in 1:L÷2]   # centred pairs (i, i + r)
Cs = [DisorderKit.correlator(ψ, Z, Z, i, j)[end] for (i, j) in pairs]

# Entanglement entropy of every realization, averaged over the disorder (all 4^L words enumerated when ≤ n_samples)
n_samples = 10000
Ss, Ss_err, _ = DisorderKit.disorder_average(w -> DisorderKit.conditional_entanglement_entropy(ψ, w), ψ; n_samples)

# Free-fermion reference with the same disorder distribution and pairs
ff = DisorderKit.sample_RTFIM(L, Js, hs; ps, n_samples, β = Inf, pairs)
@printf("E/L = %.10f (ITE)  %.10f (free fermions),  |ΔE| = %.1e after %d steps\n", E, ff.E, abs(E - ff.E), length(Es))
@printf("max |ΔS(k)| = %.1e,  max |Δ⟨Xᵢ⟩| = %.1e,  max |ΔC(r)| = %.1e\n", maximum(abs, Ss .- ff.Ss),
        maximum(abs, Xs .- ff.Xs), maximum(abs, Cs .- ff.ZZs))

# Figure: convergence of the evolution (left column) and the final state against the free fermions (right column)
set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor = :white, fontsize = 24, size = (1600, 1500))
Label(fig[0, 1:2], L"$L=%$L$, $δ=%$δ$, $W=%$W$, $Δτ=%$Δτ$"; fontsize = 28)

ax1 = Axis(fig[1, 1]; xlabel = L"τ", ylabel = L"|E - E_\mathrm{FF}|/L", yscale = log10)
scatterlines!(ax1, τs, abs.(Es .- ff.E) .+ 1e-16; markersize = 6)
ax2 = Axis(fig[2, 1]; xlabel = L"τ", ylabel = L"|\log Z(w)|", yscale = log10)
scatterlines!(ax2, τs, info.residual_rms .+ 1e-16; markersize = 6, label = L"\mathrm{rms}")
scatterlines!(ax2, τs, info.residual_max .+ 1e-16; markersize = 6, label = L"\mathrm{max}")
axislegend(ax2; position = :rt)
ax3 = Axis(fig[3, 1]; xlabel = L"τ", ylabel = L"\mathrm{bond\ dimension}")
scatterlines!(ax3, τs, maximum.(info.bond_dimensions); markersize = 6, label = L"ψ")
scatterlines!(ax3, τs, maximum.(info.src_ranks); markersize = 6, label = L"\mathrm{SRC\ (before\ rounding)}")
scatterlines!(ax3, τs, maximum.(info.filter_ranks); markersize = 6, label = L"\mathrm{filter}")
lines!(ax3, τs, info.caps; color = :black, linestyle = :dash, label = L"\mathrm{cap}")
axislegend(ax3; position = :rb)

ax4 = Axis(fig[1, 2]; xlabel = L"i", ylabel = L"\langle X_i \rangle", xticks = 1:L)
scatterlines!(ax4, 1:L, Xs; markersize = 14, label = L"\mathrm{ITE}")
scatterlines!(ax4, 1:L, ff.Xs; markersize = 10, linestyle = :dash, label = L"\mathrm{FF}")
axislegend(ax4; position = :rt)
ax5 = Axis(fig[2, 2]; xlabel = L"r", ylabel = L"\langle Z_{(L-r)/2} Z_{(L+r)/2} \rangle", xticks = 1:L÷2)
scatterlines!(ax5, 1:length(Cs), Cs; markersize = 14, label = L"\mathrm{ITE}")
scatterlines!(ax5, 1:length(Cs), ff.ZZs; markersize = 10, linestyle = :dash, label = L"\mathrm{FF}")
axislegend(ax5; position = :rt)
ax6 = Axis(fig[3, 2]; xlabel = L"\mathrm{bond}\ k", ylabel = L"\overline{S(k)}", xticks = 1:L-1)
errorbars!(ax6, 1:L-1, Ss, Ss_err; whiskerwidth = 8)
scatterlines!(ax6, 1:L-1, Ss; markersize = 14, label = L"\mathrm{ITE}")
scatterlines!(ax6, 1:L-1, ff.Ss; markersize = 10, linestyle = :dash, label = L"\mathrm{FF}")
axislegend(ax6; position = :cb)

fig
