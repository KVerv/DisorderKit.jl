using Revise, TensorKit, MPSKit, MPSKitModels, KrylovKit
using OptimKit, TensorKitManifolds, Zygote, StatsBase
using DisorderKit, TimerOutputs, CairoMakie, LsqFit, ProgressBars
using DelimitedFiles

const AbstractMPOTensor = AbstractTensorMap{T, S, 2, 2} where {T, S}

function sweep(D::Int, a::Float64, b::Float64)
    ws = -0.2:0.1:1.2
    # ws = 1.:-0.1:-1.
    Es = Float64[]
    δs = Float64[]
    Ss = Float64[]
    ξs = Float64[]
    ξts = Float64[]
    ess = Vector{Float64}[]

    D_dis = 4
    ps = ones(D_dis)./D_dis
    V_phys = Z2Space(0 => 1, 1 => 1)
    V_virt = Z2Space(0 => D, 1 => D)
    ρ0 = InfiniteDisorderMPS(ps, D_dis, V_phys, V_virt; T=ComplexF64)
    for w in ws
        hs = [a, b]
        V = var(log.(hs);corrected = false)
        hs = [a, b]*exp(2*w*V)
        Js = [a, b]
        δ = (mean(log.(hs)) - mean(log.(Js)))/(var(log.(hs);corrected = false) + var(log.(Js);corrected = false))
        # δ = (mean(log.(hs)) - mean(log.(Js)))
        push!(δs, δ)   


        Hs = DisorderKit.random_transverse_field_isingZ2(Js, hs)

        # ρ0 = InfiniteDisorderMPS(ps, D_dis, V_phys, V_virt; T=ComplexF64)

        ρgs, gradhist = DisorderKit.groundstate!(ρ0, Hs; gradtol=1e-7, verbosity=5, maxiter=5000)

        ρ0 = ρgs
        E = DisorderKit.energy_density(ρgs, Hs)
        push!(Es, E)

        @show DisorderKit.average_entanglement_entropy(ρgs)
        S, es = DisorderKit.average_entanglement_entropy(ρgs)
        push!(Ss, S)

        push!(ess, es)

        ξ = DisorderKit.average_correlation_length(ρgs)

        push!(ξs, ξ)

        # ξt = DisorderKit.typical_correlation_length(ρgs; L=50, Nsamples=10)
        # push!(ξts, ξt)
    end

    return Es, δs, Ss, ξs, ξts, ess
end
D = 13

Es, δs, Ss, ξs, ξts, ess = sweep(D, 0.7, 1.3)


# set_theme!(theme_latexfonts())
# fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
# ax1 = Axis(fig[1, 1], 
#         xlabel = L"δ",
#         ylabel = L"$E$",
#         # xscale = log10,
#         # yscale = log10
#         )
# ax2 = Axis(fig[2, 1], 
#         xlabel = L"δ",
#         ylabel = L"$ξ_t$",
#         # xscale = log10,
#         # yscale = log10
#         )
# ax3 = Axis(fig[1, 2], 
#         xlabel = L"δ",
#         ylabel = L"$ξ$",
#         # xscale = log10,
#         # yscale = log10
#         )
# ax4 = Axis(fig[2, 2], 
# xlabel = L"δ",
# ylabel = L"$S$",
# # xscale = log10,
# # yscale = log10
# )


scatterlines!(ax1, δs, Es, label=L"$E$", markersize=20)
# scatterlines!(ax2, δs, ξts, label=L"$M$", markersize=20)
scatterlines!(ax3, δs, ξs, label=L"$S$", markersize=20)
scatterlines!(ax4, δs, Ss, label=L"$S$", markersize=20)
# hlines!(ax4, log(2*D), color=:black, linestyle=:dash, linewidth=2)

fig





# save("C_MPS55.pdf",fig)