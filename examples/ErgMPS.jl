using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit
using BlockTensorKit, MatrixAlgebraKit, OptimKit


# Define model
J₀ = 1.0
δ = 0.0
W = 0.1
hs = exp(δ) * [1., exp(-2W)]
Js = J₀*[1., exp(-2W)]
w1 = 0.5
w2 = 1-w1
ps = [w1*w1, w1*w2, w2*w1, w2*w2]

# Js = [1., 1.]
# hs = [1., 1.]
# ps = [0.25, 0.25, 0.25, 0.25]

Hs = DisorderKit.random_transverse_field_ising(Js, hs)


Δτ = 0.02 # Step size for imaginary time evolution
maxiter = 2000 # Maximum number of iterations for the groundstate algorithm
D = 1 # State Bonddimension
Dmax = 4
D_dis = length(ps) # Disorder Bonddimension
D_phys = 2 # Physical dimension
ψ = DisorderKit.InfiniteDisorderMPS(ps, D_dis, D_phys, D)

# U = DisorderKit.time_evolution_MPO(Hs, Δτ; N=2)
# ψ = U*ψ

function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    ξ = DisorderKit.average_correlation_length(ρ)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
    M = real.(DisorderKit.expectation_value(ρ, Z))
    rmax = round(Int, ξ)
    # Cs = real.(DisorderKit.correlator(ρ, Z, Z, 0, rmax))
    Cs = zeros(Float64, rmax)
    return (E, M, ξ, Cs)
end

myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64, Float64, Vector{Float64}}, my_finalize!)

ψgs, data, info = DisorderKit.groundstate(ψ, Hs, Δτ, DisorderKit.Groundstate_iDTEBD(1e-6, 1e-7, maxiter, 5, TimerOutput(), myFinalizer), maxD = Dmax)

Es = getindex.(data, 1)
Ms = getindex.(data, 2)
ξs = getindex.(data, 3)
Cs = getindex.(data, 4)
τs = Δτ:Δτ:length(Es)*Δτ

E = Es[end]
ξ = ξs[end]
M = Ms[end]

@show (E, ξ)

# set_theme!(theme_latexfonts())
# fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
# ax1 = Axis(fig[1, 1], 
#         xlabel = L"τ",
#         ylabel = L"$E$",
#         # xscale = log10,
#         # yscale = log10
#         )
# ax2 = Axis(fig[1, 2], 
#         xlabel = L"τ",
#         ylabel = L"$ϵ_{conv}$",
#         # xscale = log10,
#         yscale = log10
#         )
# ax3 = Axis(fig[2, 1], 
#         xlabel = L"τ",
#         ylabel = L"$M$",
#         # xscale = log10,
#         # yscale = log10
#         )
# ax4 = Axis(fig[2, 2], 
#         xlabel = L"τ",
#         ylabel = L"$ξ$",
#         # xscale = log10,
#         # yscale = log10
#         )

colors = Makie.wong_colors()
scatterlines!(ax1, τs, Es, label=L"$Δτ=%$Δτ$", markersize=20)
scatterlines!(ax2, τs[1:length(τs)-1], info.ϵsconv[1:length(τs)-1], label=L"$ϵ_{conv}$", markersize=20)
scatterlines!(ax3, τs, Ms, label=L"$ϵ_{acc}$", markersize=20)
scatterlines!(ax4, τs, ξs, label=L"$ϵ_{acc}$", markersize=20)
# lines!(ax1, τs, fill(EF, length(τs)), color=:red, linestyle=:dash, label=L"$E_{gs}$")

axislegend(ax1, position=:rt)
fig
# t = U*ψ

# λ, _ = DisorderKit.environments(t)

# t = DisorderKit.rescale(t, 1/sqrt(λ))
# λ, _ = DisorderKit.environments(t)

# DisorderKit.fidelity(ψ, t)
# DisorderKit.fidelity(t, t)
# DisorderKit.fidelity(ψ, ψ)

# ψ_opt, gradhist = DisorderKit.approximate(t, ψ, 4; maxiter=maxiter, gradtol = 1e-6)
# DisorderKit.fidelity(t, ψ_opt)

# fig3 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
# ax31 = Axis(fig3[1, 1], 
#         xlabel = L"\ln r",
#         ylabel = L"$\ln (C_r)$",
#         # xscale = log10,
#         # yscale = log10
# )
# ax32 = Axis(fig3[1, 2], 
#         xlabel = L"\ln r",
#         ylabel = L"$η$",
#         # xscale = log10,
#         # yscale = log10
# )

# rmax = round(Int, ξ)
# Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
# ys = real.(DisorderKit.correlator(ψgs, Z, Z, 0, rmax))
# rs = 1:length(ys)
# # yss = ys./ys[1]#.*rs.^0.38
# yss = ys .- M^2
# # yss ./= yss[1]
# # yss .*= rs.^0.4

# η = -diff(log.(abs.(yss)))./diff(log.(rs))
# rs *= 1/ξ
# # yss .*= rs.^0.38
# # yss .*= ξ^0.38

# scatter!(ax31, log.(rs), log.(yss), label=L"$D=%$D$", markersize=20)

# @show ξ
# @show η[1:3]
# scatter!(ax32, log.(rs)[1:end-1], η, label=L"$D=%$D$", markersize=20)



# lines!(ax31, log.(rs), -0.25 *(log.(rs).-log.(rs[1])).+log.(yss[1]), color=:black, linewidth=2)
# lines!(ax31, log.(rs), -0.38 *(log.(rs).-log.(rs[1])).+log.(yss[1]), color=:red, linewidth=2)
# # lines!(ax31, log.(rs), -5/6 *(log.(rs).-log.(rs[1])).+log.(yss[1]), color=:red, linewidth=2)



# fig3