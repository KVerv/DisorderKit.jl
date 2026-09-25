using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit
using BlockTensorKit, MatrixAlgebraKit, OptimKit
using StatsBase

function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    ξ = DisorderKit.average_correlation_length(ρ)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
    M = real.(DisorderKit.expectation_value(ρ, Z))
    return (E, M, ξ)
end


function scan_phasespace(δs, W, Δτ::Float64, alg, Dmax::Int)
    Es = zeros(Float64, length(δs))
    Ms = zeros(Float64, length(δs))
    ξs = zeros(Float64, length(δs))
    ϵs = zeros(Float64, length(δs))

    # ρprev = DisorderKit.InfiniteDisorderDensityMatrix(ps, ℂ^2, ℂ^1, ℂ^2)
    for (iδ, δ) in enumerate(δs)
        # Define model
        J₀ = 1.0
        hs = exp(δ) * [1., exp(-2W)]
        Js = J₀*[1., exp(-2W)]
        w1 = 0.5
        w2 = 1-w1
        ps = [w1*w1, w1*w2, w2*w1, w2*w2]

        Hs = DisorderKit.random_transverse_field_ising(Js, hs)
        D_disorder = length(Js)*length(hs)
        ps = ones(D_disorder)./D_disorder

        ψ = DisorderKit.InfiniteDisorderMPS(ps, D_disorder, 2, 1)


        ψgs, data, info = DisorderKit.groundstate(ψ, Hs, Δτ, alg, maxD = Dmax)


        Es[iδ] = getindex.(data, 1)[end]
        Ms[iδ] = getindex.(data, 2)[end]
        ξs[iδ] = getindex.(data, 3)[end]
        ϵs[iδ] = info.ϵsconv[end]
    end
    return Es, Ms, ξs, ϵs
end

Δτ = 0.05
maxiter = 1000
Dmax = 2
W = 0.25
δs = -0.1:0.05:0.3

myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64, Float64}, my_finalize!)
alg = DisorderKit.Groundstate_iDTEBD(1e-6, 1e-7, maxiter, 5, TimerOutput(), myFinalizer)

Es, Ms, ξs, ϵs = scan_phasespace(δs, W, Δτ, alg, Dmax)
Ms = abs.(Ms)

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax1 = Axis(fig[1, 1], 
        xlabel = L"$δ$",
        ylabel = L"$E₀$",
        # xscale = log10,
        # yscale = log10
        )
ax2 = Axis(fig[1, 2], 
        xlabel = L"$δ$",
        ylabel = L"$M$",
        # xscale = log10,
        # yscale = log10
        )
ax3 = Axis(fig[2, 1], 
        xlabel = L"$δ$",
        ylabel = L"$ξ$",
        # xscale = log10,
        # yscale = log10
        )

ax4 = Axis(fig[2, 2], 
    xlabel = L"$δ$",
    ylabel = L"$ϵ$",
    # xscale = log10,
    yscale = log10
)
ax1.title = L"$E$"
ax2.title = L"$M$"
ax3.title = L"$ξ$"
ax4.title = L"$ϵ$"

scatterlines!(ax1, δs, Es)
scatterlines!(ax2, δs, Ms)
scatterlines!(ax3, δs, ξs)
scatterlines!(ax4, δs, abs.(ϵs.+1e-16))
fig

# ξWs = []
# Ws = []
# push!(ξWs, ξs)
# push!(Ws, W)


# set_theme!(theme_latexfonts())
# fig2 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
# ax21 = Axis(fig2[1, 1], 
#         xlabel = L"$δ/W^2$",
#         ylabel = L"$ξW^α$",
#         # xscale = log10,
#         # yscale = log10
#         )

# ϕ = 0.72
# iδ0 = 3
# for (iW,W) in enumerate(Ws[1:end])
#     # ys = ξWs[iW]./ξWs[iW][iδ0]
#     # xs = δs/W^ϕ
#     ys = ξWs[iW][iδ0]
#     xs = W
#     scatter!(ax21, xs, ys, markersize=20, color=:blue)
# end

# fig2