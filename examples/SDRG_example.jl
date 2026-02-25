using Revise, TensorKit, MPSKit, MPSKitModels, KrylovKit
using OptimKit, TensorKitManifolds, Zygote, StatsBase
using DisorderKit, TimerOutputs, CairoMakie, LsqFit, ProgressBars
using DelimitedFiles

# Define model
a = 0.5
b = 1.0
w = 0.0

hs = [a, b]
V = var(log.(hs);corrected = false)
hs = [a, b]*exp(w*V)
Js = [a, b]
# hs = a:0.01:b
# V = var(log.(hs);corrected = false)
# hs = hs*exp(w*V)
# Js = a:0.01:b
δ = (mean(log.(hs)) - mean(log.(Js)))/(var(log.(hs);corrected = false) + var(log.(Js);corrected = false))

function random_J()
    return rand(Js)
end

function random_h()
    return rand(hs)
end

L = 10000
nsamples = 100
Γs, Ms, Jdist, hdist = DisorderKit.ising_sdrg(L, nsamples, random_J, random_h)

Γsplot = Float64[]
Msplot = Float64[]
dΓ = maximum(Γs)/25
bins = 0:dΓ:maximum(Γs)
for Γ in bins
    indices = findall(x -> Γ <= x < Γ+dΓ, Γs)
    mean_M = mean(Ms[indices])
    push!(Γsplot, Γ)
    push!(Msplot, mean_M)
end

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax1 = Axis(fig[1, 1], 
        xlabel = L"Γ",
        ylabel = L"$M$",
        # xscale = log10,
        # yscale = log10
        )
ax2 = Axis(fig[2, 1], 
        xlabel = L"Γ",
        ylabel = L"$M$",
        # xscale = log10,
        # yscale = log10
        )

j = 5
for i in 1:5:nsamples
    scatterlines!(ax1, log.(Γs[:,i]), log.(Ms[:,i]), markersize=20)
    lines!(ax1, log.(Γsplot), -0.38*(log.(Γsplot).-log.(Γsplot[j])).+log.(Msplot[j]), color=:black, linestyle = :dot, linewidth=2, label=L"$r^{-0.38}$")

end
scatterlines!(ax2, log.(Γsplot), log.(Msplot), markersize=20)

lines!(ax2, log.(Γsplot), -0.38*(log.(Γsplot).-log.(Γsplot[j])).+log.(Msplot[j]), color=:black, linestyle = :dot, linewidth=2, label=L"$r^{-0.38}$")
vlines!(ax2, log.(Γsplot[j]), color=:red, linestyle=:dash, linewidth=2, label=L"$\Gamma=%$(Γsplot[j])$")


fig
# fig2 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
# ax21 = Axis(fig2[1, 1], 
#         xlabel = L"Γ",
#         ylabel = L"$M$",
#         # xscale = log10,
#         # yscale = log10
#         )
# ax22 = Axis(fig2[2, 1], 
# xlabel = L"Γ",
# ylabel = L"$M$",
# # xscale = log10,
# # yscale = log10
# )



# Jhist = Float64[]
# Jcounts = Int64[]
# hhist = Float64[]
# hcounts = Int64[]

# for (J, n) in Jdist[2]
#     push!(Jhist, J)
#     push!(Jcounts, n)
# end

# barplot!(ax21,log.(b*exp(w*V)./Jhist), Jcounts, color = :red, strokecolor = :black, strokewidth = 5)

# for (h, n) in hdist[1]
#     push!(hhist, h)
#     push!(hcounts, n)
# end

# barplot!(ax22,log.(b*exp(w*V)./hhist), hcounts, color = :blue, strokecolor = :black, strokewidth = 5)

# fig2