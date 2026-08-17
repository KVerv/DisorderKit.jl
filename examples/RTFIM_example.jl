using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit
using BlockTensorKit, MatrixAlgebraKit, OptimKit

function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    ξ = DisorderKit.average_correlation_length(ρ)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
    M = real.(DisorderKit.expectation_value(ρ, Z))
    return (E, M, ξ)
end


# Define model
N = 2
a = 0.7
b = 1.3

Js = Vector(a:(b-a)/(N-1):b)
hs = Vector(a:(b-a)/(N-1):b)
ps = ones(N^2)./N^2

# Js = [1.]
# hs = [1.]
# ps = [1.]

Hs = DisorderKit.random_transverse_field_ising(Js, hs)

Δβ = 0.02
β₀ = Δβ
β₁ = 20
βspan = β₀:Δβ:β₁

myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64, Float64}, my_finalize!)
alg_ρ_trunc = MatrixAlgebraKit.truncrank(4) 
alg_evo = DisorderKit.FiniteTemperature_iDTEBD(alg_ρ_trunc; momenttol = 1e-6, verbosity = 2, timer_output = TimerOutput(), finalizer = myFinalizer)

Us = DisorderKit.time_evolution_MPO(Hs, Δβ/2)
ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix(Us.opp, ps)
# ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix(ps, ℂ^2, ℂ^1, ℂ^1)
# ρ₀ = ρ₀ * Us

ρ, ϵs, data = evolve_densitymatrix(ρ₀, Hs, βspan, alg_evo)

Es = getindex.(data, 1)
Ms = getindex.(data, 2)
ξs = getindex.(data, 3)

# set_theme!(theme_latexfonts())
# fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
# ax1 = Axis(fig[1, 1], 
#         xlabel = L"β",
#         ylabel = L"$E$",
#         # xscale = log10,
#         # yscale = log10
#         )
# ax2 = Axis(fig[1, 2], 
#         xlabel = L"β",
#         ylabel = L"$ξ$",
#         # xscale = log10,
#         # yscale = log10
#         )
# ax3 = Axis(fig[2, 1], 
#         xlabel = L"β",
#         ylabel = L"$|1-N_2|$",
#         # xscale = log10,
#         # yscale = log10
#         )
# ax4 = Axis(fig[2, 2], 
#         xlabel = L"(\ln β)^2",
#         ylabel = L"$ξ$",
#         # xscale = log10,
#         # yscale = log10
#         )

index1 = findfirst(βspan .> 1.0)
scatterlines!(ax1, βspan, Es, label=L"$E$", markersize=20)
scatterlines!(ax3, βspan, ϵs, label=L"$ϵ$", markersize=20)
# scatterlines!(ax3, βspan, Ms, label=L"$M$", markersize=20)
scatterlines!(ax2, βspan[index1:end], ξs[index1:end], label=L"$ξ$", markersize=20)
# scatterlines!(ax4, (log.(βspan[index1:end])).^2, ξs[index1:end], label=L"$ξ$", markersize=20)
scatterlines!(ax4, log.(βspan[index1:end]).^2, ξs[index1:end], label=L"$ξ$", markersize=20)

fig

# dlbs = diff(log.(βspan[index1:end]).^2)
# dbs = diff(βspan[index1:end])
# dys = diff((ξs[index1:end]))
# dd = dys./dlbs