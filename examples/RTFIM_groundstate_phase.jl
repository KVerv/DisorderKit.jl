using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit
using BlockTensorKit, MatrixAlgebraKit, OptimKit

D = 6 # State Bonddimension
D_R = 2 # Renormalization operator Bonddimension
Δτ = 0.05 # Step size for imaginary time evolution
Ws = -0.5:0.1:0.5 # Disorder imbalance parameter
maxiter = 500 # Maximum number of iterations for the groundstate algorithm

a = 0.7
b = 1.3
w1 = 0.5
w2 = 1-w1
ps = [w1*w1, w1*w2, w2*w1, w2*w2]

# Construct Finalizer for computing observables after each iteration of the groundstate algorithm
function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    ξ = DisorderKit.average_correlation_length(ρ)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
    M = real.(DisorderKit.expectation_value(ρ, Z))
    S, _ = DisorderKit.average_entanglement_entropy(ρ)
    return (E, ξ, M, S)
end

myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64, Float64, Float64}, my_finalize!)

# Construct initial random state
A = spzeros(ComplexF64, BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(ℂ^2) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...), BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...) ⊗ BlockTensorKit.boxplus(ℂ^1))
Aᵢ = rand(ComplexF64, ℂ^1 ⊗ ℂ^2 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1 ⊗ ℂ^1) 
for i in eachindex(ps)
        A[1, 1, i, 1, i, 1] = Aᵢ # Same tensor for each disorder sector
end

ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix([A], ps)
ρ₀ = DisorderKit.gauge(ρ₀)


# Es = zeros(length(Ws), maxiter)
# ξs = zeros(length(Ws), maxiter)
# Ms = zeros(length(Ws), maxiter)
# Ss = zeros(length(Ws), maxiter)
# ϵsconv = zeros(length(Ws), maxiter)
# ϵsent = zeros(length(Ws), maxiter)
# ϵsz = zeros(length(Ws), maxiter)
# ρ_trunc_err = zeros(length(Ws), maxiter)
# R_trunc_err = zeros(length(Ws), maxiter)
# ϵsA1 = zeros(length(Ws), maxiter)
# ϵsA2 = zeros(length(Ws), maxiter)
Es = zeros(length(Ws))
ξs = zeros(length(Ws))
Ms = zeros(length(Ws))
Ss = zeros(length(Ws))
ϵsconv = zeros(length(Ws))
ϵsent = zeros(length(Ws))
ϵsz = zeros(length(Ws))
ρ_trunc_err = zeros(length(Ws))
R_trunc_err = zeros(length(Ws))
ϵsA1 = zeros(length(Ws))
ϵsA2 = zeros(length(Ws))

δs = Ws
τs = Δτ:Δτ:maxiter*Δτ

for (ix, W) in enumerate(Ws)
        hs = [a, b]*exp(W)
        Js = [a, b]


        Hs = DisorderKit.random_transverse_field_ising(Js, hs)

        # Generate algorithm object with corresponding parameters
        alg_evo = DisorderKit.Groundstate_iDTEBD(MatrixAlgebraKit.truncrank(D), MatrixAlgebraKit.truncrank(D_R); convtol = 1e-6, maxiter = maxiter, verbosity = 2, timer_output = TimerOutput(), finalizer = myFinalizer)

        # Compute the groundstate
        @timeit alg_evo.timer_output "compute_groundstate" ρs, data, info = DisorderKit.groundstate(ρ₀, Hs, Δτ, alg_evo)

        # Es[ix, :] = getindex.(data, 1)
        # ξs[ix, :] = getindex.(data, 2)
        # Ms[ix, :] = getindex.(data, 3)
        # Ss[ix, :] = getindex.(data, 4)

        # ϵsconv[ix, :] = info.ϵsconv
        # ϵsent[ix, :] = info.ϵsent
        # ϵsz[ix, :] = info.ϵsz
        # ρ_trunc_err[ix, :] = info.ρ_trunc_err
        # R_trunc_err[ix, :] = info.R_trunc_err
        # ϵsA1[ix, :] = info.ϵsA1
        # ϵsA2[ix, :] = info.ϵsA2

        Es[ix] = getindex.(data, 1)[end]
        ξs[ix] = getindex.(data, 2)[end]
        Ms[ix] = getindex.(data, 3)[end]
        Ss[ix] = getindex.(data, 4)[end]
end

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax1 = Axis(fig[1, 1], 
        xlabel = L"δ",
        ylabel = L"$E$",
        # xscale = log10,
        # yscale = log10
        )
ax2 = Axis(fig[1, 2], 
        xlabel = L"δ",
        ylabel = L"$S$",
        # xscale = log10,
        # yscale = log10
        )
ax3 = Axis(fig[2, 1], 
        xlabel = L"δ",
        ylabel = L"$M$",
        # xscale = log10,
        # yscale = log10
        )
ax4 = Axis(fig[2, 2], 
        xlabel = L"δ",
        ylabel = L"$ξ$",
        # xscale = log10,
        # yscale = log10
        )

colors = Makie.wong_colors()

# plotindices = 40:40:length(τs)
# for i in plotindices
#     scatterlines!(ax1, δs, Es[:, i], label=L"$D=%$D$", markersize=20)
#     scatterlines!(ax2, δs, Ss[:, i], label=L"$S$", markersize=20)
#     scatterlines!(ax3, δs, Ms[:, i], label=L"$ϵ_{acc}$", markersize=20)
#     scatterlines!(ax4, δs, ξs[:, i], label=L"$ϵ_{z}$", markersize=20, marker=:utriangle)
# end

scatterlines!(ax1, δs, Es, label=L"$D=%$D$", markersize=20)
scatterlines!(ax2, δs, Ss, label=L"$S$", markersize=20)
scatterlines!(ax3, δs, Ms, label=L"$ϵ_{acc}$", markersize=20)
scatterlines!(ax4, δs, ξs, label=L"$ϵ_{z}$", markersize=20, marker=:utriangle)



axislegend(ax1, position=:rt)
fig
   
# set_theme!(theme_latexfonts())
# fig2 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
# ax21 = Axis(fig2[1, 1], 
#         xlabel = L"δ",
#         ylabel = L"$ϵ_z$",
#         # xscale = log10,
#         yscale = log10
#         )
# ax22 = Axis(fig2[1, 2], 
#         xlabel = L"δ",
#         ylabel = L"$ϵ_R$",
#         # xscale = log10,
#         yscale = log10
#         )
# ax23 = Axis(fig2[2, 1], 
#         xlabel = L"δ",
#         ylabel = L"$ϵ_{A1}$",
#         # xscale = log10,
#         yscale = log10
#         )
# ax24 = Axis(fig2[2, 2], 
#         xlabel = L"δ",
#         ylabel = L"$ϵ_{z}$",
#         # xscale = log10,
#         yscale = log10
#         )

# colors = Makie.wong_colors()
# scatterlines!(ax21, δs, ϵsz, label=L"$ϵ_{ρ}$", markersize=20)
# scatterlines!(ax22, δs, R_trunc_err, label=L"$ϵ_{R}$", markersize=20)
# scatterlines!(ax23, δs, ϵsA1, label=L"$ϵ_{A1}$", markersize=20)
# scatterlines!(ax24, δs, ϵsA2, label=L"$ϵ_{A2}$", markersize=20, marker=:utriangle)

# axislegend(ax1, position=:rt)
# fig2

# fig3 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
# ax31 = Axis(fig3[1, 1], 
#         xlabel = L"δ",
#         ylabel = L"$τ$",
#         # xscale = log10,
#         # yscale = log10
#         )


# contourf!(ax31, δs, τs, abs.(Ss), levels=0:0.1:1)


# # axislegend(ax1, position=:rt)
# fig3