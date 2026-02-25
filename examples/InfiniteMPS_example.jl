using Revise, TensorKit, MPSKit, MPSKitModels, KrylovKit
using OptimKit, TensorKitManifolds, Zygote, StatsBase
using DisorderKit, TimerOutputs, CairoMakie, LsqFit, ProgressBars
using DelimitedFiles

const AbstractMPOTensor = AbstractTensorMap{T, S, 2, 2} where {T, S}

#Critical at D=30, w=0.35

# Define model
a = 0.5
b = 1.0
w = 0.22
w0 = 0

hs = [a, b]
V = var(log.(hs);corrected = false)
hs = [a, b]*exp(2*w*V)
Js = [a, b]
δ = (mean(log.(hs)) - mean(log.(Js)))/(var(log.(hs);corrected = false) + var(log.(Js);corrected = false))

Ds = [13]
# Ds = [15, 30, 60]


# alphas, fs, dfs1, dfs2 = optimtest(fg, ρ, fg(ρ)[2]; alpha=-0.1:0.01:0.1, retract=DisorderKit.retract, inner=DisorderKit.inner)

# set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax1 = Axis(fig[1, 1], 
        xlabel = L"\ln r/ξ",
        ylabel = L"$\ln (C_r-M^2)$",
        # xscale = log10,
        # yscale = log10
        )
ax2 = Axis(fig[2, 1], 
        xlabel = L"\ln r/ξ",
        ylabel = L"$\ln ((C_r-M^2)\cdot r^{0.38})$",
        # xscale = log10,
        # yscale = log10
        )

fig2 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax21 =  Axis(fig2[1, 1], limits = (nothing, nothing, 0, 1),
        xlabel = L"r",
        ylabel = L"$α$",
        # xscale = log10,
        # yscale = log10
        )

function simulate_Ds(Ds, Js, hs)
    D_dis = length(hs)*length(Js)
    ps = ones(D_dis)./D_dis

    Hs = DisorderKit.random_transverse_field_ising(Js, hs)

    Es = []
    Ms = []
    Ss = []
    αs = []
    ξss = []
    δeffs = []
    Veffs = []
    exths = []
    extJs = []
    # ρ = DisorderKit.InfiniteDisorderMPS(ps, D_dis, 2, Ds[1]; T=ComplexF64)
    for (i, D) in enumerate(Ds)
        heffs = []
        Jeffs = []
        # if i > 1
        #     ρ = DisorderKit.expand(ρ, round(Int64, D/Ds[i-1]))
        # end
        ρ = DisorderKit.InfiniteDisorderMPSC(ps, D_dis, 2, D; T=ComplexF64)
        ρgs, gradhist = DisorderKit.groundstate!(ρ, Hs; gradtol=1e-6, verbosity=5, maxiter=5000)

        E = DisorderKit.energy_density(ρgs, Hs)
        push!(Es, E)
        # S, _ = DisorderKit.average_entanglement_entropy(ρgs)
        # push!(Ss, S)

        ξ = DisorderKit.average_correlation_length(ρgs)
        ξ = round(Int, ξ)
        push!(ξss, ξ)
        r = min(1000, 2*ξ)
        # r = 1000
        rs = 1:r
        Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
        Cs = DisorderKit.correlator(ρgs, Z, Z, 0, r)
        M = DisorderKit.expectation_value(ρgs, Z)
        
        push!(Ms, M)

        Csc = abs.(Cs .- real(M.^2))

        α = -diff(log.(Csc/Csc[1]))./diff(log.(rs/ξ))
        push!(αs, α)

        # ξt = DisorderKit.typical_correlation_length(ρgs; L=50, Nsamples=50)
        # push!(ξts, ξt)

        ϕ = (1+sqrt(5))/2
        scatter!(ax1, log.(rs/ξ),log.(Csc/Csc[1]), label=L"$D=%$D$, $\xi=%$ξ$",markersize = 20)
        lines!(ax1, log.(rs/ξ), -(2-ϕ)*(log.(rs/ξ).-log.(rs[1]/ξ)), color=:black, linestyle = :dot, linewidth=2, label=L"$r^{-0.38}$")
        scatter!(ax2, log.(rs/ξ),log.(Csc/Csc[1]).*rs.^(2-ϕ), label=L"$D=%$D$, $\xi=%$ξ$",markersize = 20)

        colors = [:blue, :red, :green, :yellow, :orange, :cyan]
        # vlines!(ax1, log.([ξt/ξ, 1]), color=colors[i], linestyle=:dash, linewidth=2)
        # vlines!(ax2, log.([ξt/ξ, 1]), color=colors[i], linestyle=:dash, linewidth=2)

        # scatter!(ax21, log.(rs[1:end-1]/ξ),α, markersize = 20)
        # hlines!(ax21, 0.38, color=colors[i], linestyle=:dash, linewidth=2)
        # hlines!(ax21, 0.25, color=colors[i], linestyle=:dash, linewidth=2)

         _, vr = DisorderKit.right_environment(ρgs)    
        vl = zeros(ComplexF64, space(ρgs.opp[1],3)',space(ρgs.opp[1],3)')

        X = TensorMap([0. 1.; 1. 0.], ℂ^2, ℂ^2)

        couplings = collect(Iterators.product(hs, Js))
        for (p, W) in enumerate(ρgs.opp)
            @tensor vlO1[-1; -2] := W[1 3; -2] * X[2; 3] * conj(W[1 2; -1]) 
            heff = real.(tr(vlO1 * vr)*couplings[p][1])

            heffs = push!(heffs, heff)
            for (q, V) in enumerate(ρgs.opp)
                @tensor ECB = W[1 2; 4] * Z[3; 2] * conj(W[1 3; 6]) * V[4 7; 9] * Z[8; 7] * conj(V[6 8; 10]) * vr[9;10]
                Jeff = real.(ECB*couplings[p][2])
                push!(Jeffs, Jeff)
            end
        end
        Veff = var(log.(heffs);corrected = false)+var(log.(Jeffs);corrected = false)
        δeff = (mean(log.(heffs)) - mean(log.(Jeffs)))/Veff
        push!(Veffs, Veff)
        push!(δeffs, δeff)
        push!(exths, (minimum(heffs), maximum(heffs)))
        push!(extJs, (minimum(Jeffs), maximum(Jeffs)))

    end
    return Es, Ms, Ss, αs, ξss, δeffs, Veffs, exths, extJs
end

Es, Ms, Ss, αs, ξs, δeffs, Veffs, exths, extJs =simulate_Ds(Ds, Js, hs)


fig[1, 2] = Legend(fig, ax1, framevisible = false)
fig

# fig2 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
# ax21 = Axis(fig2[1, 1], 
#         xlabel = L"δ_{eff}",
#         ylabel = L"$V_{eff}$",
#         # xscale = log10,
#         # yscale = log10
#         )

# scatter!(ax21, δeffs, Veffs, markersize=20)
# fig2


# fig4 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
# ax41 =  Axis(fig4[1, 1],
#         xlabel = L"ln(ξ)",
#         ylabel = L"$S$",
#         # xscale = log10,
#         # yscale = log10
#         )
# scatter!(ax41, log.(ξs), Ss, markersize = 20)

# minfit = 1
# maxfit = 4
# p0 = [1.,1.]
# linmodel(t,p) = p[1].+p[2] .*t
# linfit = curve_fit(linmodel, log.(ξs[minfit:maxfit]), Ss[minfit:maxfit], p0)
# c = linfit.param[2]*6/log(2)
# @show c
# lines!(ax41, log.(ξs[minfit:maxfit]), linmodel(log.(ξs[minfit:maxfit]),linfit.param), color=:black, linestyle = :dot, linewidth=2, label=L"$$")
# fig4

# fig3 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
# ax31 =  Axis(fig3[1, 1], limits = (nothing, nothing, nothing, 3),
#         xlabel = L"r",
#         ylabel = L"$α$",
#         # xscale = log10,
#         # yscale = log10
#         )
# ax32 =  Axis(fig3[2, 1],
# xlabel = L"r",
# ylabel = L"$α$",
# # xscale = log10,
# # yscale = log10
# )

# for (i, α) in enumerate(deepcopy(αs[1:end]))
#     ξ = ξs[i]
#     ξ = round(Int, ξ)
#     r = min(1000, 2*ξ)
#     # r = round(Int, 2*ξ)
#     rs = 1:r
#     a = 0
#     scatter!(ax31, log.(rs[1:end-1]), α, markersize = 20)
#     hlines!(ax31, 0.38, linestyle=:dash, linewidth=2)
#     hlines!(ax31, 0.25, linestyle=:dash, linewidth=2)
#     j = 5
#     # scatter!(ax32, log.(rs[1:end-1]), α/α[1], markersize = 20)
#     # hlines!(ax32, [ξs[1]/ξ], linestyle=:dash, linewidth=2)
#     # lines!(ax32, log.(rs[1:end-1]), -0.09*(log.(rs[1:end-1]).-log.(rs[j])).+α[j]./(αs[1][j]), color=:black, linestyle = :dot, linewidth=2, label=L"$r^{-0.38}$")
#     # scatter!(ax32, (rs[1:end-2]), diff(α)./diff((rs[1:end-1])), markersize = 20)
# end

# fig3




# save("C_MPS3.pdf",fig)