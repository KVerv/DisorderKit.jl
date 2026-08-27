using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit
using BlockTensorKit, MatrixAlgebraKit, OptimKit

D = 4 # State Bonddimension
D_R = 2 # Renormalization operator Bonddimension
Δτ = 0.05 # Step size for imaginary time evolution
maxiter = 200 # Maximum number of iterations for the groundstate algorithm

Ws = 0.:0.5:1.0 # Disorder strength
δs = -0.1:0.1:0.1 # Disorder imbalance parameter
J₀ = 1.0
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



Es = zeros(length(Ws), length(gs))
ξs = zeros(length(Ws), length(gs))
Ms = zeros(length(Ws), length(gs))
Ss = zeros(length(Ws), length(gs))
ϵsconv = zeros(length(Ws), length(gs))
ϵsent = zeros(length(Ws), length(gs))
ϵsz = zeros(length(Ws), length(gs))
ρ_trunc_err = zeros(length(Ws), length(gs))
R_trunc_err = zeros(length(Ws), length(gs))
ϵsA1 = zeros(length(Ws), length(gs))
ϵsA2 = zeros(length(Ws), length(gs))

τs = Δτ:Δτ:maxiter*Δτ

for (ix, W) in enumerate(Ws)
        for (jx, δ) in enumerate(δs)
                hs = exp(δ) * J₀ * [1., exp(-2W)]
                Js =  J₀ * [1., exp(-2*W)]

                Hs = DisorderKit.random_transverse_field_ising(Js, hs)

                # Generate algorithm object with corresponding parameters
                alg_evo = DisorderKit.Groundstate_iDTEBD(MatrixAlgebraKit.truncrank(D), MatrixAlgebraKit.truncrank(D_R); convtol = 1e-6, maxiter = maxiter, verbosity = 2, timer_output = TimerOutput(), finalizer = myFinalizer)

                # Compute the groundstate
                @timeit alg_evo.timer_output "compute_groundstate" ρs, data, info = DisorderKit.groundstate(ρ₀, Hs, Δτ, alg_evo)


                Es[ix, jx] = getindex.(data, 1)[end]
                ξs[ix, jx] = getindex.(data, 2)[end]
                Ms[ix, jx] = getindex.(data, 3)[end]
                Ss[ix, jx] = getindex.(data, 4)[end]
        end
end

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax1 = Axis(fig[1, 1], 
        xlabel = L"$W^2$",
        ylabel = L"$δ$",
        # xscale = log10,
        # yscale = log10
        )
ax2 = Axis(fig[1, 2], 
        xlabel = L"$W^2$",
        ylabel = L"$δ$",
        # xscale = log10,
        # yscale = log10
        )
ax3 = Axis(fig[2, 1], 
        xlabel = L"$W^2$",
        ylabel = L"$δ$",
        # xscale = log10,
        # yscale = log10
        )
ax4 = Axis(fig[2, 2], 
        xlabel = L"$W^2$",
        ylabel = L"$δ$",
        # xscale = log10,
        # yscale = log10
        )

colors = Makie.wong_colors()

centers_x = Ws
centers_y = δs

heatmap!(ax1, centers_x, centers_y, Es, colormap=:viridis)
scatter!(ax1, [(x, y) for x in centers_x for y in centers_y], color=:white, strokecolor=:black, strokewidth=1)
heatmap!(ax2, centers_x, centers_y, Ss, colormap=:viridis)
scatter!(ax2, [(x, y) for x in centers_x for y in centers_y], color=:white, strokecolor=:black, strokewidth=1)
heatmap!(ax3, centers_x, centers_y, Ms, colormap=:viridis)
scatter!(ax3, [(x, y) for x in centers_x for y in centers_y], color=:white, strokecolor=:black, strokewidth=1)
heatmap!(ax3, centers_x, centers_y, ξs, colormap=:viridis)
scatter!(ax3, [(x, y) for x in centers_x for y in centers_y], color=:white, strokecolor=:black, strokewidth=1)



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