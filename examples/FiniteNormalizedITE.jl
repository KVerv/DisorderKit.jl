using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, CairoMakie, TimerOutputs


# Define model
J₀ = 1.0
δ = 0.0
W = 0.1
hs = exp(δ) * [1., exp(-2W)]
Js = J₀ * [1., exp(-2W)]
ps = [0.25, 0.25, 0.25, 0.25]

Hs = DisorderKit.random_transverse_field_ising(Js, hs)


Δτ = 0.02 # Step size for imaginary time evolution
maxiter = 100 # Maximum number of time steps
L = 20 # System size
Dmax = [100 => 8, 200 => 16, 300 => 32] # Cap schedule (number of steps => cap of partition, filter and state), the last cap is kept
ψ0 = DisorderKit.plus_state(ps, L) # |+x⟩ on every site, every disorder word normalized

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

# Spin-spin correlation function from site i0
i0 = 1
j0 = L
Cs = DisorderKit.correlator(ψgs, Z, Z, i0, j0) # ⟨Z_{i0} Z_{i0+r}⟩, r = 1..j0-i0
rs = 1:length(Cs)

# Raw energy density after an additional normalization filter (not divided by the global norm)
E_filtered, N_filtered, _ = DisorderKit.filter_normalized_energy_density(ψgs, Hs)

@show (E, E_filtered, N_filtered, M)

# Free fermion reference (exact enumeration of the disorder configurations for small L)
n_samples = 100000
data_FF = DisorderKit.sample_RTFIM(L, Js, hs; ps=ps, n_samples=n_samples, β=Inf, i0=i0)
EFF = data_FF.E
@show abs(E-EFF), abs(E_filtered-EFF)

# Disorder averaged entanglement entropy on every bond (enumerated exactly when 4^L ≤ n_samples)
Ss, Ss_err = DisorderKit.average_entanglement_entropy(ψgs; n_samples=10000)
@show maximum(abs.(Ss .- data_FF.Ss))

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 4000))
ax1 = Axis(fig[1, 1],
        xlabel = L"τ",
        ylabel = L"$E - E_{FF}$",
        yscale = log10
        )
ax2 = Axis(fig[1, 2],
        xlabel = L"τ",
        ylabel = L"$ϵ_{conv}$",
        yscale = log10
        )
ax3 = Axis(fig[2, 1],
        xlabel = L"τ",
        ylabel = L"$|\log Z(w)|$",
        yscale = log10
        )
ax4 = Axis(fig[2, 2],
        xlabel = L"τ",
        ylabel = L"\mathrm{bond\ dimension}",
        )
ax5 = Axis(fig[3, 1],
        xlabel = L"i",
        ylabel = L"$\langle O_i \rangle$",
        )
ax6 = Axis(fig[3, 2],
        xlabel = L"r",
        ylabel = L"$C(r)$",
        xscale = log10,
        yscale = log10
        )
ax7 = Axis(fig[4, 1],
        xlabel = L"k",
        ylabel = L"$S(k)$",
        )

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
scatterlines!(ax6, rs, Cs, label=L"$\langle Z_{%$i0} Z_{%$i0+r} \rangle$", markersize=20)
scatterlines!(ax6, rs, data_FF.ZZs, label=L"$\mathrm{FF}$", linestyle=:dash, markersize=20)
lines!(ax6, rs, abs(data_FF.ZZs[1]) .* rs .^ (-0.25), label=L"$r^{-1/4}\ \mathrm{(clean)}$", color=:black, linestyle=:dot, linewidth=3)
lines!(ax6, rs, abs(data_FF.ZZs[1]) .* rs .^ (-0.38), label=L"$r^{-0.38}\ \mathrm{(IRFP)}$", color=:gray, linestyle=:dashdot, linewidth=3)
errorbars!(ax7, 1:L-1, Ss, Ss_err, whiskerwidth=10)
scatterlines!(ax7, 1:L-1, Ss, label=L"\mathrm{MPS}", markersize=20)
errorbars!(ax7, 1:L-1, data_FF.Ss, data_FF.Ss_err, whiskerwidth=10)
scatterlines!(ax7, 1:L-1, data_FF.Ss, label=L"\mathrm{FF}", linestyle=:dash, markersize=20)

axislegend(ax1, position=:rt)
axislegend(ax3, position=:rt)
axislegend(ax4, position=:rb)
axislegend(ax5, position=:rt)
axislegend(ax6, position=:rt)
axislegend(ax7, position=:rt)

fig
