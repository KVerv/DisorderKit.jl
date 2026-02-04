using Revise, TensorKit, MPSKit, MPSKitModels, KrylovKit
using OptimKit, TensorKitManifolds, Zygote, StatsBase
using DisorderKit, TimerOutputs, CairoMakie, LsqFit, ProgressBars
using DelimitedFiles

const AbstractMPOTensor = AbstractTensorMap{T, S, 2, 2} where {T, S}

# Define model
a = 0.5
b = 1.0
w = 0.68
hs = [a, b]
V = var(log.(hs);corrected = false)
hs = [a, b]*exp(w*V)
Js = [a, b]
δ = (mean(log.(hs)) - mean(log.(Js)))/(var(log.(hs);corrected = false) + var(log.(Js);corrected = false))

Ds = [10]


# alphas, fs, dfs1, dfs2 = optimtest(fg, ρ, fg(ρ)[2]; alpha=-0.1:0.01:0.1, retract=DisorderKit.retract, inner=DisorderKit.inner)

set_theme!(theme_latexfonts())
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


function simulate_Ds(Ds, Js, hs)
    D_dis = length(hs)*length(Js)
    ps = ones(D_dis)./D_dis

    Hs = DisorderKit.random_transverse_field_ising(Js, hs)

    Es = []
    Ms = []
    Ss = []
    for (i, D) in enumerate(Ds)
        ρ = InfiniteDisorderMPS(ps, D_dis, 2, D; T=ComplexF64)

        ρgs, gradhist = groundstate!(ρ, Hs; gradtol=1e-6, verbosity=5, maxiter=5000)

        E = DisorderKit.energy_density(ρgs, Hs)
        push!(Es, E)
        S = DisorderKit.average_entanglement_entropy(ρgs)
        push!(Ss, S)

        ξ = DisorderKit.average_correlation_length(ρgs)
        ξ = round(Int, ξ)
        r = min(1000, 2*ξ)
        rs = 1:r
        Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
        Cs = DisorderKit.correlator(ρgs, Z, Z, 0, r)
        M = DisorderKit.expectation_value(ρgs, Z)
        push!(Ms, M)

        Csc = abs.(Cs .- real(M.^2))
        scatter!(ax1, log.(rs/ξ),log.(Csc/Csc[1]), label=L"$D=%$D$, $\xi=%$ξ$",markersize = 20)
        lines!(ax1, log.(rs/ξ), -0.38*(log.(rs/ξ).-log.(rs[1]/ξ)), color=:black, linestyle = :dot, linewidth=2, label=L"$r^{-0.38}$")
        scatter!(ax2, log.(rs/ξ),log.(Csc/Csc[1]).*rs.^(0.38), label=L"$D=%$D$, $\xi=%$ξ$",markersize = 20)

        colors = [:blue, :red, :green, :yellow]
        ξt = DisorderKit.typical_correlation_length(ρgs; L=50, Nsamples=10)
        vlines!(ax1, log.([ξt/ξ, 1]), color=colors[i], linestyle=:dash, linewidth=2)
        vlines!(ax2, log.([ξt/ξ, 1]), color=colors[i], linestyle=:dash, linewidth=2)
    end
    return Es, Ms, Ss
end

Es, Ms, Ss =simulate_Ds(Ds, Js, hs)

fig[1, 2] = Legend(fig, ax1, framevisible = false)
fig



# save("C_MPS3.pdf",fig)