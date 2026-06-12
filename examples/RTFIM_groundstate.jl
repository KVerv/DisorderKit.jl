using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit
using BlockTensorKit, MatrixAlgebraKit, OptimKit

function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    ξ = DisorderKit.average_correlation_length(ρ)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
    M = real.(DisorderKit.expectation_value(ρ, Z))
    rmax = round(Int, 5*ξ)
    Cs = real.(DisorderKit.two_point_correlator(ρ, Z, Z, rmax+1))
    T = real(DisorderKit.left_environment(ρ)[1])
    Ds = norm(DisorderKit.PartitionFunction(ρ; χ=2).D)
    Rs = norm(DisorderKit.PartitionFunction(ρ; χ=2).R)
    Ls = norm(DisorderKit.PartitionFunction(ρ; χ=2).L)
    return (E, M, ξ, T, Cs, Ds, Rs, Ls)
end


# Define model
N = 2
a = 0.7
b = 1.3

Js = Vector(a:(b-a)/(N-1):b)
hs = Vector(a:(b-a)/(N-1):b)
# hs .-= 0.5
ps = ones(N^2)./N^2

# hs = [0.5, 3]
# Js = [0.5, 3]
# ps = [16/25, 4/25, 4/25, 1/25]

# Js = [1.]
# hs = [1.]
# ps = [1.]

Hs = DisorderKit.random_transverse_field_ising(Js, hs)

Δβ = 0.05
D = 8

myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64, Float64, Float64, Vector{Float64}, Float64, Float64, Float64}, my_finalize!)
alg_evo = DisorderKit.Groundstate_iDTEBD(MatrixAlgebraKit.truncrank(D); convtol = 1e-9, maxiter = 100, verbosity = 2, timer_output = TimerOutput(), finalizer = myFinalizer)

Us = DisorderKit.time_evolution_MPO(Hs, Δβ/2)
# A = spzeros(ComplexF64, BlockTensorKit.boxplus(ℂ^4) ⊗ BlockTensorKit.boxplus(ℂ^2) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...), BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...) ⊗ BlockTensorKit.boxplus(ℂ^4))
# for i in eachindex(ps)
#         A[1, 1, i, 1, i, 1] = TensorMap(W.data, ℂ^4 ⊗ ℂ^2 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1 ⊗ ℂ^4)
# end

A = spzeros(ComplexF64, BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(ℂ^2) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...), BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...) ⊗ BlockTensorKit.boxplus(ℂ^1))
# Aᵢ = rand(ComplexF64, ℂ^1 ⊗ ℂ^2 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1 ⊗ ℂ^1)
for i in eachindex(ps)
        A[1, 1, i, 1, i, 1] = Aᵢ
end
ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix([A], ps)
# ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix(Us.opp, ps)
# ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix(ps, ℂ^2, ℂ^1, ℂ^1)
ρ₀ = DisorderKit.gauge(ρ₀)
# ρ₀ = ρ₀ * Us

ρs, ϵsconv, ϵsacc, data = DisorderKit.evolve_densitymatrix(ρ₀, Hs, Δβ, alg_evo)
Z = DisorderKit.PartitionFunction(ρs; χ=1)

βs = Δβ:Δβ:(length(ϵsconv)*Δβ)
Es = getindex.(data, 1)
Ms = getindex.(data, 2)
ξs = getindex.(data, 3)
Ts = getindex.(data, 4)
Cs = getindex.(data, 5)
Ds = getindex.(data, 6)
Rs = getindex.(data, 7)
Ls = getindex.(data, 8)

E = Es[end]
ξ = ξs[end]


# @show (E₀, ξ₀)
@show (E, ξ)

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
#         ylabel = L"$ϵ_{conv}$",
#         # xscale = log10,
#         yscale = log10
#         )
# ax3 = Axis(fig[2, 1], 
#         xlabel = L"β",
#         ylabel = L"$|D|$",
#         # xscale = log10,
#         yscale = log10
#         )
# ax4 = Axis(fig[2, 2], 
#         xlabel = L"β",
#         ylabel = L"$ϵ_{acc}$",
#         # xscale = log10,
#         yscale = log10
#         )

scatterlines!(ax1, log.(βs), Es, label=L"$Δτ=%$Δβ$", markersize=20)
scatterlines!(ax2, log.(βs[2:end]), ϵsconv[2:end], label=L"$ϵ_{conv}$", markersize=20)
# scatterlines!(ax3, log.(βs), (Ms), label=L"$M$", markersize=20)
scatterlines!(ax4, log.(βs[2:end]), (ϵsacc[2:end]), label=L"$ϵ_{acc}$", markersize=20)
scatterlines!(ax3,log.(βs[2:end]), (Ds[2:end]), label=L"$|D|$", markersize=20)
scatterlines!(ax3,log.(βs[2:end]), (Rs[2:end]), label=L"$|R|$", markersize=20, marker = :utriangle)
scatterlines!(ax3,log.(βs[2:end]), (Rs[2:end]), label=L"$|R|$", markersize=20, marker = :dtriangle)

# lines!(ax4, log.(βs[2:end]),  2.5*(log.(βs[2:end]).-log.(βs[40])).+log.(ϵsacc[40]), color=:black, linewidth=2)
# lines!(ax3, log.(βs[2:end]),  0.8*(log.(βs[2:end]).-log.(βs[40])).+log.(Ds[40]), color=:black, linewidth=2)

axislegend(ax1, position=:rt)
fig
        
fig2 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax21 = Axis(fig2[1, 1], 
        xlabel = L"\ln r",
        ylabel = L"$\ln (C_r-M^2)$",
        # xscale = log10,
        # yscale = log10
)

for (i, ys) in enumerate(Cs)
        rs = 1:length(ys)
        M = Ms[i]
        yss = ys #.- M^2
        ξ = ξs[i]

        if i % length(Cs) == 0
        # if i % 20 == 0
            scatter!(ax21, log.(rs), log.(yss), label=L"$D=%$D$", markersize=20)
        end
        if i == length(Cs)
                lines!(ax21, log.(rs), -0.25 *(log.(rs).-log.(rs[3])).+log.(yss[3]), color=:black, linewidth=2)
                lines!(ax21, log.(rs), -0.38 *(log.(rs).-log.(rs[3])).+log.(yss[3]), color=:red, linewidth=2)
                # lines!(ax21, log.(rs), -5/6 *(log.(rs).-log.(rs[3])).+log.(yss[3]), color=:blue, linewidth=2)

        end
end

fig2

# fig3 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
# ax31 = Axis(fig3[1, 1], 
#         xlabel = L"E",
#         ylabel = L"$ϵ_{acc}$",
#         # xscale = log10,
#         # yscale = log10
# )
# scatterlines!(ax31, Es[2:end], ξs[2:end], label=L"$ϵ_{acc}$", markersize=20)

# fig3

# save("GSD4.pdf",fig)