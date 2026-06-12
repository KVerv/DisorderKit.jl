using Revise, TensorKit, MPSKit, MPSKitModels, KrylovKit
using OptimKit, TensorKitManifolds, Zygote, StatsBase
using DisorderKit, TimerOutputs, CairoMakie, LsqFit, ProgressBars
using DelimitedFiles


chain = InfiniteChain(1)
D = 4
H = transverse_field_ising(chain; J=1.0, g=1.0)
physical_space = ℂ^2
virtual_space_inf = ℂ^D
ψ₀_inf = InfiniteMPS([physical_space], [virtual_space_inf])
# ψ₀_inf = InfiniteMPS([physical_space, physical_space], [virtual_space_inf, virtual_space_inf])
ψ_inf, envs_inf, delta_inf = find_groundstate(ψ₀_inf, H; verbosity = 3, tol = 1e-6)

E₀ = real(MPSKit.expectation_value(ψ_inf, H))

Z = TensorMap([1. 0.; 0. -1.], ℂ^1 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^1)
rs = 1:2:200
Cs = real.(MPSKit.correlator(ψ_inf, Z, Z, 0, rs))

Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2 )
M = real(MPSKit.local_expectation_value1(ψ_inf, 1, Z))
fig2 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax21 = Axis(fig2[1, 1], 
        xlabel = L"\ln r",
        ylabel = L"$\ln (C_r-M^2)$",
        # xscale = log10,
        # yscale = log10
        )

ξ₀ = DisorderKit.correlation_length(ψ_inf)
scatter!(ax21, log.(rs), log.(Cs), markersize=20)

lines!(ax21, log.(rs), -0.25 *(log.(rs).-log.(rs[1])).+log.(Cs[1]), color=:black, linewidth=2)
fig2