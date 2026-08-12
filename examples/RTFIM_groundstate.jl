using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit
using BlockTensorKit, MatrixAlgebraKit, OptimKit

# Define model
# ζs = [0.58579, 3.41421]
# hs = exp.(-ζs)
# Js = exp.(-ζs)
# w1 = 0.853527437561
# w2 = 1-w1
# ps = [w1*w1, w1*w2, w2*w1, w2*w2]

hs = [0.7, 1.3]
Js = [0.7, 1.3]
w1 = 0.5
w2 = 1-w1
ps = [w1*w1, w1*w2, w2*w1, w2*w2]

Hs = DisorderKit.random_transverse_field_ising(Js, hs)


Δτ = 0.05 # Step size for imaginary time evolution
maxiter = 1000 # Maximum number of iterations for the groundstate algorithm
D = 4 # State Bonddimension
D_R = 2 # Renormalization operator Bonddimension

# Construct Finalizer for computing observables after each iteration of the groundstate algorithm
function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    ξ = DisorderKit.average_correlation_length(ρ)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
    M = real.(DisorderKit.expectation_value(ρ, Z))
    rmax = round(Int, 5*ξ)
    Cs = real.(DisorderKit.two_point_correlator(ρ, Z, Z, rmax+1))
    return (E, M, ξ, Cs)
end

myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64, Float64, Vector{Float64}}, my_finalize!)

# Generate algorithm object with corresponding parameters
alg_evo = DisorderKit.Groundstate_iDTEBD(MatrixAlgebraKit.truncrank(D), MatrixAlgebraKit.truncrank(D_R); convtol = 1e-9, maxiter = maxiter, verbosity = 2, timer_output = TimerOutput(), finalizer = myFinalizer)


# Construct initial random state
A = spzeros(ComplexF64, BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(ℂ^2) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...), BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...) ⊗ BlockTensorKit.boxplus(ℂ^1))
Aᵢ = rand(ComplexF64, ℂ^1 ⊗ ℂ^2 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1 ⊗ ℂ^1) 
for i in eachindex(ps)
        A[1, 1, i, 1, i, 1] = Aᵢ # Same tensor for each disorder sector
end

ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix([A], ps)
ρ₀ = DisorderKit.gauge(ρ₀)

# Compute the groundstate
@timeit alg_evo.timer_output "compute_groundstate" ρs, data, info = DisorderKit.groundstate(ρ₀, Hs, Δτ, alg_evo)

τs = Δτ:Δτ:maxiter*Δτ
Es = getindex.(data, 1)
Ms = getindex.(data, 2)
ξs = getindex.(data, 3)
Cs = getindex.(data, 4)

E = Es[end]
ξ = ξs[end]

@show (E, ξ)

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax1 = Axis(fig[1, 1], 
        xlabel = L"τ",
        ylabel = L"$E$",
        # xscale = log10,
        # yscale = log10
        )
ax2 = Axis(fig[1, 2], 
        xlabel = L"τ",
        ylabel = L"$ϵ_{conv}$",
        # xscale = log10,
        yscale = log10
        )
ax3 = Axis(fig[2, 1], 
        xlabel = L"τ",
        ylabel = L"$ϵ_σ$",
        # xscale = log10,
        yscale = log10
        )
ax4 = Axis(fig[2, 2], 
        xlabel = L"τ",
        ylabel = L"$ϵ_{z}$",
        # xscale = log10,
        yscale = log10
        )

colors = Makie.wong_colors()
scatterlines!(ax1, τs, Es, label=L"$Δτ=%$Δτ$", markersize=20)
scatterlines!(ax2, τs, info.ϵsconv, label=L"$ϵ_{conv}$", markersize=20)
scatterlines!(ax3, τs, (abs.(info.ϵsent)), label=L"$ϵ_{acc}$", markersize=20)
scatterlines!(ax4, τs, (info.ϵsz), label=L"$ϵ_{acc}$", markersize=20, marker=:utriangle)

axislegend(ax1, position=:rt)
fig
   
set_theme!(theme_latexfonts())
fig2 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax21 = Axis(fig2[1, 1], 
        xlabel = L"τ",
        ylabel = L"$ϵ_ρ$",
        # xscale = log10,
        yscale = log10
        )
ax22 = Axis(fig2[1, 2], 
        xlabel = L"τ",
        ylabel = L"$ϵ_R$",
        # xscale = log10,
        yscale = log10
        )
ax23 = Axis(fig2[2, 1], 
        xlabel = L"τ",
        ylabel = L"$ϵ_{A1}$",
        # xscale = log10,
        yscale = log10
        )
ax24 = Axis(fig2[2, 2], 
        xlabel = L"τ",
        ylabel = L"$ϵ_{z}$",
        # xscale = log10,
        yscale = log10
        )

colors = Makie.wong_colors()
scatterlines!(ax21, τs[2:end], info.ρ_trunc_err[2:end], label=L"$Δτ=%$Δτ$", markersize=20)
scatterlines!(ax22, τs, info.R_trunc_err, label=L"$ϵ_{conv}$", markersize=20)
scatterlines!(ax23, τs[2:end], info.ϵsA1[2:end], label=L"$ϵ_{A1}$", markersize=20)
scatterlines!(ax24, τs[2:end], info.ϵsA2[2:end], label=L"$ϵ_{acc}$", markersize=20, marker=:utriangle)

axislegend(ax1, position=:rt)
fig2
# fig2 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
# ax21 = Axis(fig2[1, 1], 
#         xlabel = L"\ln r",
#         ylabel = L"$\ln (C_r)$",
#         # xscale = log10,
#         # yscale = log10
# )

# for (i, ys) in enumerate(Cs)
#         ξ = ξs[i]
#         rs = 1:length(ys)
#         # rs *= 1/ξ
#         yss = ys./ys[1]#.*rs.^0.38
#         M = Ms[i]#.- M^2

#         if i % length(Cs) == 220
#         # if i % 100 == 0
#             scatter!(ax21, log.(rs), log.(yss), label=L"$D=%$D$", markersize=20)
#         # scatter!(ax21, (rs), (yss), label=L"$D=%$D$", markersize=20)
#                 @show ξ
#         end
#         # if i == length(Cs)
#         if i == 220
#                 lines!(ax21, log.(rs), -0.25 *(log.(rs).-log.(rs[2])).+log.(yss[2]), color=:black, linewidth=2)
#                 lines!(ax21, log.(rs), -0.38 *(log.(rs).-log.(rs[5])).+log.(yss[5]), color=:red, linewidth=2)
#                 # lines!(ax21, log.(rs), -0.25 *(log.(rs).-log.(rs[2])).+log.(yss[2]), color=:black, linewidth=2)
#                 # lines!(ax21, log.(rs), -0.38 *(log.(rs).-log.(rs[2])).+log.(yss[2]), color=:red, linewidth=2)
           
#                 # # lines!(ax21, log.(rs), -5/6 *(log.(rs).-log.(rs[3])).+log.(yss[3]), color=:blue, linewidth=2)

#         end

#         # if i == 800
#         #         scatter!(ax21, log.(rs), log.(yss), label=L"$D=%$D$", markersize=20)
#         #         lines!(ax21, log.(rs), -0.38 *(log.(rs).-log.(rs[5])).+log.(yss[5]), color=:red, linewidth=2)
#         # end

# end
# # rs = 1:length(Cs2[end])
# # scatter!(ax21, log.(rs), log.(Cs2[end]), label=L"$D=%$D$", markersize=20)
# # rs = 1:length(Cs4[180])
# # scatter!(ax21, log.(rs), log.(Cs4[180]), label=L"$D=%$D$", markersize=20)
# fig2
