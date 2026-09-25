using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit
using BlockTensorKit, MatrixAlgebraKit, OptimKit
using StatsBase

function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    ξ = DisorderKit.average_correlation_length(ρ)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
    M = real.(DisorderKit.expectation_value(ρ, Z))
    rmax = round(Int, ξ)
    return (E, M, ξ)
end


function scan_phasespace(hs, Ws, Δτ::Float64, alg, Dmax::Int)
    Es = zeros(Float64, length(hs), length(Ws))
    Ms = zeros(Float64, length(hs), length(Ws))
    ξs = zeros(Float64, length(hs), length(Ws))
    ϵs = zeros(Float64, length(hs), length(Ws))

    # ρprev = DisorderKit.InfiniteDisorderDensityMatrix(ps, ℂ^2, ℂ^1, ℂ^2)
    for (iW, W) in enumerate(Ws)
        for (ih, h) in enumerate(hs)
            # Define model
            J₀ = 1.0
            hsi = [h * (1 - W), h * (1 + W)]
            Js = [J₀]

            Hs = DisorderKit.random_transverse_field_ising(Js, hsi)
            D_disorder = length(Js)*length(hsi)
            ps = ones(D_disorder)./D_disorder

            ψ = DisorderKit.InfiniteDisorderMPS(ps, D_disorder, 2, 1)


            ψgs, data, info = DisorderKit.groundstate(ψ, Hs, Δτ, alg, maxD = Dmax)


            Es[ih, iW] = getindex.(data, 1)[end]
            Ms[ih, iW] = getindex.(data, 2)[end]
            ξs[ih, iW] = getindex.(data, 3)[end]
            ϵs[ih, iW] = info.ϵsconv[end]

            # ρprev = deepcopy(ρs)
        end
    end
    return Es, Ms, ξs, ϵs
end

Δτ = 0.05
maxiter = 1000
Dmax = 4
hs = 0.8:0.1:1.2
Ws = 0.0:0.1:0.5

myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64, Float64}, my_finalize!)
alg = DisorderKit.Groundstate_iDTEBD(1e-6, 1e-7, maxiter, 5, TimerOutput(), myFinalizer)

Es, Ms, ξs, ϵs = scan_phasespace(hs, Ws, Δτ, alg, Dmax)
Ms = abs.(Ms)

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax1 = Axis(fig[1, 1], 
        xlabel = L"$h$",
        ylabel = L"$W/h$",
        # xscale = log10,
        # yscale = log10
        )
ax2 = Axis(fig[1, 2], 
        xlabel = L"h",
        ylabel = L"$W/h$",
        # xscale = log10,
        # yscale = log10
        )
ax3 = Axis(fig[2, 1], 
        xlabel = L"h",
        ylabel = L"$W/h$",
        # xscale = log10,
        # yscale = log10
        )

ax4 = Axis(fig[2, 2], 
    xlabel = L"$h$",
    ylabel = L"$W/h$",
    # xscale = log10,
    # yscale = log10
)
ax1.title = L"$E$"
ax2.title = L"$M$"
ax3.title = L"$ξ$"
ax4.title = L"$ϵ$"

heatmap!(ax1, hs, Ws, Es)
heatmap!(ax2, hs, Ws, Ms)
heatmap!(ax3, hs, Ws, ξs)
heatmap!(ax4, hs, Ws, ϵs)
fig
