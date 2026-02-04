using Revise, TensorKit, MPSKit, MPSKitModels, KrylovKit
using OptimKit, TensorKitManifolds, Zygote
using DisorderKit, TimerOutputs, CairoMakie, LsqFit, ProgressBars
const AbstractMPOTensor = AbstractTensorMap{T, S, 2, 2} where {T, S}
const AbstractMPSTensor = AbstractTensorMap{T, S, 2, 1} where {T, S}

# Define model
N = 2
a = 0.7
b = 1.3
hs = Vector(a:(b-a)/(N-1):b)
Js = hs
# hs = [a]
# Js = [b]
D = 4
L = 4
D_dis = length(hs)*length(Js)

# Make DisorderMPS
# D = 5
# As = [[TensorMap(rand, ComplexF64, ℂ^D⊗ℂ^2,ℂ^D) for i in 1:D_dis] for j in 1:L]
# # As = push!(As,[TensorMap(rand, ComplexF64, ℂ^D⊗ℂ^2,ℂ^D) for i in 1:2])
# ρ = FiniteDisorderMPS(As)
ρ = FiniteDisorderMPS(L, D_dis, 2, D; T=ComplexF64)
ρ = DisorderKit.left_gauge(ρ)

function hams_Ising(Js::Vector{Float64}, hs::Vector{Float64})
    X, Z, Id = zeros(ComplexF64, ℂ^2, ℂ^2), zeros(ComplexF64, ℂ^2, ℂ^2), zeros(ComplexF64, ℂ^2, ℂ^2)
    X[1, 2], X[2, 1] = 1, 1
    Z[1, 1], Z[2, 2] = 1, -1
    Id[1, 1], Id[2, 2] = 1, 1
    Hs = AbstractMPOTensor[]
    for (i, (h, J)) in enumerate(Iterators.product(hs, Js))
        H = zeros(ComplexF64, ℂ^3⊗ℂ^2, ℂ^2⊗ℂ^3)
        H[1,:, :, 1] = Id.data
        H[2,:, :, 1] = zeros(ComplexF64, 2, 2)
        H[3,:, :, 1] = zeros(ComplexF64, 2, 2)
        H[1,:, :, 2] = Z.data
        H[2,:, :, 2] = zeros(ComplexF64, 2, 2)
        H[3,:, :, 2] = zeros(ComplexF64, 2, 2)
        H[1,:, :, 3] = -h*X.data
        H[2,:, :, 3] = -J*Z.data
        H[3,:, :, 3] = Id.data
        push!(Hs,convert(TensorMap,H))
    end
    return Hs
end

Hs = hams_Ising(Js, hs)

# alg = StiefelOptim(2, 1e-4, 5)
fg = DisorderKit.target_func(Hs)
E, g = fg(ρ)
# Stiefel.retract(ρ[3][1], g[3][1], 0.1)
# ρs = groundstate!(ρ, Hs, alg)
# E = measure(ρs, Hs)

function compute_Es(Ls::Vector{Int}, Ds::Vector{Int}, Hs::Vector{<:AbstractMPOTensor})
    Es = Vector{Float64}[]
    ρs = []
    alg = StiefelOptim(2, 1e-4, 5)
    hists = []
    for (iD, D) in ProgressBar(enumerate(Ds))
        ELs = Float64[]
        ρLs = []
        for (iL,L) in ProgressBar(enumerate(Ls))
            # As = [[TensorMap(rand, ComplexF64, ℂ^D⊗ℂ^2,ℂ^D) for i in 1:D_dis] for j in 1:L]
            # ρ = FiniteDisorderMPS(As)
            ρ = FiniteDisorderMPS(L, D_dis, 2, D; T=ComplexF64)
            ρ = DisorderKit.left_gauge(ρ)
            ρL, hist = groundstate!(ρ, Hs, alg)
            EL = measure(ρL, Hs)
            push!(ELs, EL)
            push!(ρLs, ρL)
            push!(hists, hist)
        end
        push!(Es, ELs)
        push!(ρs, ρLs)
    end
    return Es, ρs, hists
end

Ds = [4]
Ls = [4, 8]
Es, ρs, hists = compute_Es(Ls, Ds, Hs)

# 194its
X, Z, Id = zeros(ComplexF64, ℂ^2, ℂ^2), zeros(ComplexF64, ℂ^2, ℂ^2), zeros(ComplexF64, ℂ^2, ℂ^2)
X[1, 2], X[2, 1] = 1, 1
Z[1, 1], Z[2, 2] = 1, -1
Id[1, 1], Id[2, 2] = 1, 1


# set_theme!(theme_latexfonts())
# fig = Figure(backgroundcolor=:white, fontsize=40, size=(1000, 1000))
# ax1 = Axis(fig[1, 1], 
#         xlabel = L"t",
#         ylabel = L"$||G||$",
#         # xscale = log10,
#         yscale = log10
#         )
# ax2 = Axis(fig[2, 1], 
# xlabel = L"t",
# ylabel = L"$f$",
# # xscale = log10,
# # yscale = log10
# )
# scatter!(ax1,1:length(hist00),hist00, label=L"00", markersize = 20)
# scatter!(ax1,1:length(hist12),hist12, label=L"12", markersize = 20)
# scatter!(ax1,1:length(hist14),hist14, label=L"14", markersize = 20)
# scatter!(ax1,1:length(hist16),hist16, label=L"16", markersize = 20)

# scatter!(ax2,10:length(fhist00),fhist00[10:end], label=L"00", markersize = 20)
# scatter!(ax2,10:length(fhist12),fhist12[10:end], label=L"12", markersize = 20)
# scatter!(ax2,10:length(fhist14),fhist14[10:end], label=L"14", markersize = 20)
# scatter!(ax2,10:length(fhist16),fhist16[10:end], label=L"16", markersize = 20)
# axislegend(ax1, position=:rt)
# fig


# set_theme!(theme_latexfonts())
# fig = Figure(backgroundcolor=:white, fontsize=40, size=(1000, 1000))
# ax1 = Axis(fig[1, 1], 
#         xlabel = L"r",
#         ylabel = L"$G$",
#         # xscale = log10,
#         # yscale = log10
#         )

# for (iD,D) in enumerate(Ds)
#     for (iL,L) in enumerate(Ls)
#         rmax = convert(Int,L-3)
#         colors = [:red, :red, :blue]
#         markers = [:circle, :diamond, :star4]
#         G =  [measure(ρs[iD][iL], Z, Z, 1, i) for i in 4:rmax]
#         # G =  [measure(ρs[iD][iL], Z, Z, i) for i in 1:rmax]
#         EZ = measure(ρs[iD][iL], Z)^2
#         # G .-= EZ
#         # G ./= 20
#         @show G, EZ
#         scatter!(ax1,log.(4:rmax),log.(abs.(G)), label=L"$L = %$L$ $D = %$D$", color = colors[iD], marker = markers[iD], markersize = 20)
#     end
#     xs = log.(2:Ls[end]/2).-log(2)
#     ys = -0.25*xs
#     # lines!(ax1, xs, ys, linestyle = :dash, color = :black, label = L"$G \sim \frac{-1}{4}\ln{r}$")
# end
# axislegend(ax1, position=:lb)
# fig