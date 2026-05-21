using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit
using BlockTensorKit, MatrixAlgebraKit, OptimKit

function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    ξ = DisorderKit.average_correlation_length(ρ)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
    M = real.(DisorderKit.expectation_value(ρ, Z))
    T = real(DisorderKit.left_environment(ρ)[1])
    return (E, M, ξ, T)
end


# Define model
N = 2
a = 0.7
b = 1.3

Js = Vector(a:(b-a)/(N-1):b)
hs = Vector(a:(b-a)/(N-1):b)
ps = ones(N^2)./N^2

# Js = [1.]
# hs = [1.]
# ps = [1.]

Hs = DisorderKit.random_transverse_field_ising(Js, hs)

Δβ = 0.08

myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64, Float64, Float64}, my_finalize!)
alg_evo = DisorderKit.Groundstate_iDTEBD(MatrixAlgebraKit.truncrank(8); convtol = 1e-8, maxiter = 100, verbosity = 2, timer_output = TimerOutput(), finalizer = myFinalizer)

Us = DisorderKit.time_evolution_MPO(Hs, Δβ/2)
# ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix(Us.opp, ps)
ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix(ps, ℂ^2, ℂ^1, ℂ^2)
ρ₀ = DisorderKit.gauge!(ρ₀)
# ρ₀ = ρ₀ * Us

ρs, ϵsconv, ϵsacc, data = evolve_densitymatrix(ρ₀, Hs, Δβ, alg_evo)

βs = Δβ:Δβ:(length(ϵsconv)*Δβ)
Es = getindex.(data, 1)
Ms = getindex.(data, 2)
ξs = getindex.(data, 3)
Ts = getindex.(data, 4)

E = Es[end]
ξ = ξs[end]


@show (E₀, ξ₀)
@show (E, ξ)

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax1 = Axis(fig[1, 1], 
        xlabel = L"β",
        ylabel = L"$E$",
        # xscale = log10,
        # yscale = log10
        )
ax2 = Axis(fig[1, 2], 
        xlabel = L"β",
        ylabel = L"$ϵ_{conv}$",
        # xscale = log10,
        yscale = log10
        )
ax3 = Axis(fig[2, 1], 
        xlabel = L"β",
        ylabel = L"$M$",
        # xscale = log10,
        # yscale = log10
        )
ax4 = Axis(fig[2, 2], 
        xlabel = L"β",
        ylabel = L"$ϵ_{acc}$",
        # xscale = log10,
        yscale = log10
        )

scatterlines!(ax1, log.(βs), Es, label=L"$E$", markersize=20)
scatterlines!(ax2, βs, ϵsconv, label=L"$ϵ_{conv}$", markersize=20)
scatterlines!(ax3, log.(βs), Ms, label=L"$M$", markersize=20)
scatterlines!(ax4, βs, ϵsacc, label=L"$ϵ_{acc}$", markersize=20)
fig

rmax = round(Int, 5*ξ)
rs = 1:rmax
Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
Cs = real.(DisorderKit.correlationfunc(ρs, Z, rmax))
M = Ms[end]
ys = Cs #.- M^2

fig2 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax21 = Axis(fig2[1, 1], 
        xlabel = L"\ln r",
        ylabel = L"$\ln (C_r-M^2)$",
        # xscale = log10,
        # yscale = log10
        )

scatter!(ax21, log.(rs), log.(ys), markersize=20)

lines!(ax21, log.(rs), -0.25 *(log.(rs).-log.(rs[1])).+log.(ys[1]), color=:black, linewidth=2)
lines!(ax21, log.(rs), -0.38 *(log.(rs).-log.(rs[2])).+log.(ys[2]), color=:black, linewidth=2)
# lines!(ax21, log.(rs), -5/6 *(log.(rs).-log.(rs[2])).+log.(ys[2]), color=:black, linewidth=2)
fig2
