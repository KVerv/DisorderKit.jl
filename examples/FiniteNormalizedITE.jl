using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, CairoMakie, TimerOutputs


# Define model
J₀ = 1.0
δ = 0.0
W = 0.8
hs = exp(δ) * [1., exp(-2W)]
Js = J₀ * [1., exp(-2W)]
ps = [0.25, 0.25, 0.25, 0.25]
# hs = [1.]
# Js = [1.]
# ps = [1.]

Hs = DisorderKit.random_transverse_field_ising(Js, hs)


Δτ = 0.02 # Step size for imaginary time evolution
maxiter = 700 # Maximum number of time steps
L = 6 # System size
Dmax = [100 => 8, 200 => 16] # Cap schedule (number of steps => cap of partition, filter and state), the last cap is kept
ψ0 = DisorderKit.plus_state(Hs, ps, L) # |+x⟩ on every site, every disorder word normalized

Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
X = TensorMap([0. 1.; 1. 0.], ℂ^2, ℂ^2)

# Energy density and average |⟨Zᵢ⟩| after every time step
function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    M = sum(abs.(real.(DisorderKit.expectation_value(ρ, Z))))/length(ρ)
    return (E, M)
end

myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64}, my_finalize!)
to = TimerOutput()
alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD = Dmax, order= 2, filter_tail_tol = 1e-8, convtol = 1e-8, maxiter = maxiter, verbosity = 2, timer_output = to, finalizer = myFinalizer)
ψgs, data, info = DisorderKit.groundstate(ψ0, Hs, Δτ, alg)
# show(to)

Es = getindex.(data, 1)
Ms_τ = getindex.(data, 2)
τs = Δτ:Δτ:length(Es)*Δτ

# Observables of the final state
E = DisorderKit.energy_density(ψgs, Hs)
Ms = real.(DisorderKit.expectation_value(ψgs, Z)) # ⟨Zᵢ⟩ on every site
Mxs = real.(DisorderKit.expectation_value(ψgs, X)) # ⟨Xᵢ⟩ on every site
M = sum(abs.(Ms))/L

# Bulk spin-spin correlation function ⟨Z_i Z_{i+r}⟩, centred on the chain: i = (L-r)÷2
pairs = [((L-r)÷2, (L-r)÷2 + r) for r in 1:round(Int, L/2)]
Cs = [DisorderKit.correlator(ψgs, Z, Z, i, j)[end] for (i, j) in pairs]
rs = 1:length(Cs)

# Raw energy density after an additional normalization filter (not divided by the global norm)
E_filtered, N_filtered, _ = DisorderKit.filter_normalized_energy_density(ψgs, Hs)

@show (E, E_filtered, N_filtered, M)

# Free fermion reference (exact enumeration of the disorder configurations for small L)
n_samples = 10000
data_FF = DisorderKit.sample_RTFIM(L, Js, hs; ps=ps, n_samples=n_samples, β=Inf, pairs=pairs)
EFF = data_FF.E
@show abs(E-EFF), abs(E_filtered-EFF)

# Disorder averaged entanglement entropy on every bond (enumerated exactly when 4^L ≤ n_samples)
Ss, Ss_err = DisorderKit.average_entanglement_entropy(ψgs; n_samples=10000)
@show maximum(abs.(Ss .- data_FF.Ss))

# Calabrese-Cardy fit for an open chain: S(k) = c/6 log[(2L/π) sin(πk/L)] + s₀
ks = 1:L-1
chordd = log.((2L/π) .* sin.(π .* ks ./ L)) ./ 6
kfit = 3:L-3 # drop bonds near the edges where boundary corrections dominate
c_fit, s0_fit = hcat(chordd[kfit], ones(length(kfit))) \ data_FF.Ss[kfit]
@show c_fit # compare with c = 1/2 (clean Ising) and c_eff = ln(2)/2 (IRFP)

intersection_point = 4

# set_theme!(theme_latexfonts())
# fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 4000))
# ax1 = Axis(fig[1, 1],
#         xlabel = L"τ",
#         ylabel = L"$E - E_{FF}$",
#         yscale = log10
#         )
# ax2 = Axis(fig[1, 2],
#         xlabel = L"τ",
#         ylabel = L"$ϵ_{conv}$",
#         yscale = log10
#         )
# ax3 = Axis(fig[2, 1],
#         xlabel = L"τ",
#         ylabel = L"$|\log Z(w)|$",
#         yscale = log10
#         )
# ax4 = Axis(fig[2, 2],
#         xlabel = L"τ",
#         ylabel = L"\mathrm{bond\ dimension}",
#         )
# ax5 = Axis(fig[3, 1],
#         xlabel = L"i",
#         ylabel = L"$\langle O_i \rangle$",
#         )
# ax6 = Axis(fig[3, 2],
#         xlabel = L"r",
#         ylabel = L"$C(r)$",
#         xscale = log10,
#         yscale = log10
#         )
# ax7 = Axis(fig[4, 1],
#         xlabel = L"k",
#         ylabel = L"$S(k)$",
#         )

scatterlines!(ax1, τs, abs.(Es .- EFF) .+ 1e-16, label=L"$Δτ=%$Δτ$", markersize=10)
scatterlines!(ax2, τs, info.ϵsconv, markersize=10)
scatterlines!(ax3, τs, info.residual_rms .+ 1e-16, label=L"\mathrm{rms}", markersize=10)
scatterlines!(ax3, τs, info.residual_max .+ 1e-16, label=L"\mathrm{max}", markersize=10)
scatterlines!(ax4, τs, maximum.(info.bond_dimensions), label=L"$\psi$", markersize=10)
scatterlines!(ax4, τs, maximum.(info.src_ranks), label=L"$\mathrm{SRC\ (before\ rounding)}$", markersize=10)
scatterlines!(ax4, τs, maximum.(info.filter_ranks), label=L"$\mathrm{filter}$", markersize=10)
scatterlines!(ax4, τs, maximum.(info.partition_ranks), label=L"$\mathrm{partition}$", markersize=10)
lines!(ax4, τs, info.caps, label=L"$\mathrm{cap}$", color=:black, linestyle=:dash)
scatterlines!(ax5, 1:L, Ms, label=L"$\langle Z_i \rangle$", markersize=20)
scatterlines!(ax5, 1:L, Mxs, label=L"$\langle X_i \rangle$", markersize=20)
scatterlines!(ax5, 1:L, data_FF.Xs, label=L"$\langle X_i \rangle_{FF}$", linestyle=:dash, markersize=20)
scatterlines!(ax6, rs, Cs, label=L"$\langle Z_{(L-r)/2} Z_{(L+r)/2} \rangle$", markersize=20)
scatterlines!(ax6, rs, data_FF.ZZs, label=L"$\mathrm{FF}$", linestyle=:dash, markersize=20)
# lines!(ax6, rs, abs(data_FF.ZZs[intersection_point]) .* (rs ./ intersection_point) .^ (-0.25), label=L"$r^{-1/4}\ \mathrm{(clean)}$", color=:black, linestyle=:dot, linewidth=3)
# lines!(ax6, rs, abs(data_FF.ZZs[intersection_point]) .* (rs ./ intersection_point) .^ (-0.38), label=L"$r^{-0.38}\ \mathrm{(IRFP)}$", color=:gray, linestyle=:dashdot, linewidth=3)
errorbars!(ax7, 1:L-1, Ss, Ss_err, whiskerwidth=10)
scatterlines!(ax7, 1:L-1, Ss, label=L"\mathrm{MPS}", markersize=20)
errorbars!(ax7, 1:L-1, data_FF.Ss, data_FF.Ss_err, whiskerwidth=10)
scatterlines!(ax7, 1:L-1, data_FF.Ss, label=L"\mathrm{FF}", linestyle=:dash, markersize=20)
# lines!(ax7, ks, c_fit .* chordd .+ s0_fit, label=L"$\mathrm{CC},\ c=%$(round(c_fit, digits=3))$", color=:black, linestyle=:dot, linewidth=3)

axislegend(ax1, position=:rt)
axislegend(ax3, position=:rt)
axislegend(ax4, position=:rb)
axislegend(ax5, position=:rt)
axislegend(ax6, position=:rt)
axislegend(ax7, position=:cb)

fig
