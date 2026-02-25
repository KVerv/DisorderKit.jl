using Revise, TensorKit, MPSKit, MPSKitModels, KrylovKit
using OptimKit, TensorKitManifolds, Zygote, StatsBase
using DisorderKit, TimerOutputs, CairoMakie, LsqFit, ProgressBars
using DelimitedFiles

const AbstractMPOTensor = AbstractTensorMap{T, S, 2, 2} where {T, S}


function sweep(D::Int, a::Float64, b::Float64)
    # ws = 1.5:-0.1:-0.5
    ws = 0.0:0.05:0.5
    Es = Float64[]
    Ms = Float64[]
    δs = Float64[]
    Ss = Float64[]
    αs = Vector{Float64}[]
    ξs = Float64[]
    ξts = Float64[]
    Veffs = Float64[]
    δeffs = Float64[]
    hmaxs = Float64[]
    Jmaxs = Float64[]
    hmins = Float64[]
    Jmins = Float64[]

    D_dis = 4
    ps = ones(D_dis)./D_dis
    ρ0 = DisorderKit.InfiniteDisorderMPSC(ps, D_dis, 2, D; T=ComplexF64)
    for w in ws
        hs = [a, b]
        V = var(log.(hs);corrected = false)
        hs = [a, b]*exp(w*2*V)
        Js = [a, b]
        δ = (mean(log.(hs)) - mean(log.(Js)))/(var(log.(hs);corrected = false) + var(log.(Js);corrected = false))
        push!(δs, δ)   


        Hs = DisorderKit.random_transverse_field_ising(Js, hs)

        ρgs, gradhist = DisorderKit.groundstate!(ρ0, Hs; gradtol=1e-6, verbosity=5, maxiter=5000)

        ρ0 = ρgs
        E = DisorderKit.energy_density(ρgs, Hs)
        push!(Es, E)

        @show DisorderKit.average_entanglement_entropy(ρgs)
        S, es = DisorderKit.average_entanglement_entropy(ρgs)
        push!(Ss, S)

        ξ = DisorderKit.average_correlation_length(ρgs)

        push!(ξs, ξ)

        Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)

        M = DisorderKit.expectation_value(ρgs, Z)
        
        push!(Ms, abs.(M))
        # ξt = DisorderKit.typical_correlation_length(ρgs; L=50, Nsamples=10)
        # push!(ξts, ξt)

        heffs = Float64[]
        Jeffs = Float64[]

         _, vr = DisorderKit.right_environment(ρgs)    
        vl = zeros(ComplexF64, space(ρgs.opp[1],3)',space(ρgs.opp[1],3)')

        X = TensorMap([0. 1.; 1. 0.], ℂ^2, ℂ^2)
        @show space(vl)
        couplings = collect(Iterators.product(hs, Js))
        for (p, W) in enumerate(ρgs.opp)
            @tensor vlO1[-1; -2] := W[1 3; -2] * X[2; 3] * conj(W[1 2; -1]) 
            heff = real(tr(vlO1 * vr)*couplings[p][1])

            heffs = push!(heffs, heff)
            for (q, V) in enumerate(ρgs.opp)
                @tensor ECB = W[1 2; 4] * Z[3; 2] * conj(W[1 3; 6]) * V[4 7; 9] * Z[8; 7] * conj(V[6 8; 10]) * vr[9;10]
                Jeff = real(ECB*couplings[p][2])
                push!(Jeffs, Jeff)
            end
        end

        Veff = var(log.(heffs);corrected = false)+var(log.(Jeffs);corrected = false)
        δeff = (mean(log.(heffs)) - mean(log.(Jeffs)))/Veff
        push!(Veffs, Veff)
        push!(δeffs, δeff)
        push!(hmaxs, maximum(heffs)/maximum(hs))
        push!(Jmaxs, maximum(Jeffs)/maximum(hs))
        push!(hmins, minimum(heffs)/maximum(hs))
        push!(Jmins, minimum(Jeffs)/maximum(hs))

    end

    return Es, Ms, δs, Ss, ξs, Veffs, δeffs, hmaxs, Jmaxs, hmins, Jmins
end
D = 19
a = 0.5
b = 1.0
Es, Ms, δs, Ss, ξs, Veffs, δeffs, hmaxs, Jmaxs, hmins, Jmins = sweep(D, a, b)

hs = [a, b]
V = var(log.(hs);corrected = false)


set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax1 = Axis(fig[1, 1], 
        xlabel = L"δ",
        ylabel = L"$E$",
        # xscale = log10,
        # yscale = log10
        )
ax2 = Axis(fig[2, 1], 
        xlabel = L"δ",
        ylabel = L"$M$",
        # xscale = log10,
        # yscale = log10
        )
ax3 = Axis(fig[1, 2], 
        xlabel = L"δ",
        ylabel = L"$ξ$",
        # xscale = log10,
        # yscale = log10
        )
ax4 = Axis(fig[2, 2], 
xlabel = L"δ",
ylabel = L"$S$",
# xscale = log10,
# yscale = log10
)

β = (3-sqrt(5))/2
scatterlines!(ax1, δs, Es, label=L"$E$", markersize=20)
scatterlines!(ax2, δs, Ms, label=L"$M$", markersize=20)
scatterlines!(ax3, δs, ξs, label=L"$S$", markersize=20)
scatterlines!(ax4, δs, Ss, label=L"$b=%$b$", markersize=20)

fig

fig2 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax21 = Axis(fig2[1, 1], 
        xlabel = L"δ_{eff}",
        ylabel = L"$V_{eff}$",
        # xscale = log10,
        # yscale = log10
        )
ax22 = Axis(fig2[2, 1], 
xlabel = L"δ_{eff}",
ylabel = L"$δ$",
# xscale = log10,
# yscale = log10
)

scatter!(ax21, δeffs, Veffs, markersize=20)
scatter!(ax22, δeffs, δs, markersize=20)
fig2

# fig3 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
# ax31 = Axis(fig3[1, 1], 
#         xlabel = L"δ_{eff}",
#         ylabel = L"coupling",
#         # xscale = log10,
#         # yscale = log10
#         )

scatterlines!(ax31, δs, hmaxs, markersize=20, label=L"$h_{max}, D=%$D$")
scatterlines!(ax31, δs, Jmaxs, markersize=20, label=L"$J_{max}$")
scatterlines!(ax31, δs, hmins, markersize=20, label=L"$h_{min}$")
scatterlines!(ax31, δs, Jmins, markersize=20, label=L"$J_{min}$")

fig3[1, 3] = Legend(fig3, ax31, framevisible = false)
fig3
# save("C_MPS55.pdf",fig)