using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit
using BlockTensorKit, MatrixAlgebraKit, OptimKit
using StatsBase

function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    ξ = DisorderKit.average_correlation_length(ρ)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
    M = real.(DisorderKit.expectation_value(ρ, Z))
    T = real(DisorderKit.left_environment(ρ)[1])
    return (E, M, ξ, T)
end

function scan_phasespace(δs, Δβ::Float64, alg_evo)
    a = 0.7
    b = 1.3
    Es = zeros(Float64, length(δs))
    Ms = zeros(Float64, length(δs))
    ξs = zeros(Float64, length(δs))
    ϵs = zeros(Float64, length(δs))

    # ρprev = DisorderKit.InfiniteDisorderDensityMatrix(ps, ℂ^2, ℂ^1, ℂ^2)
    for (p, δ) in enumerate(δs)
        # Js = [a, b]
        # hs = [a, b]*exp(δ)
        Js = [1.]
        hs = [1.]*exp(δ)

        D_disorder = length(Js)*length(hs)
        ps = ones(D_disorder)./D_disorder

        Hs = DisorderKit.random_transverse_field_ising(Js, hs)
        Us = DisorderKit.time_evolution_MPO(Hs, Δβ/2)
        # ρ₀ = deepcopy(ρprev)
        ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix(ps, ℂ^2, ℂ^1, ℂ^2)
        ρ₀ = DisorderKit.gauge!(ρ₀)


        ρs, ϵsconv, ϵsacc, data = evolve_densitymatrix(ρ₀, Hs, Δβ, alg_evo)

        Es[p] = getindex.(data, 1)[end]
        Ms[p] = getindex.(data, 2)[end]
        ξs[p] = getindex.(data, 3)[end]
        ϵs[p] = ϵsacc[end]

        # ρprev = deepcopy(ρs)
    end

    return Es, Ms, ξs, ϵs
end

Δβ = 0.05
maxiter = 100
δs = -0.2:0.1:0.2

myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64, Float64, Float64}, my_finalize!)
alg_evo = DisorderKit.Groundstate_iDTEBD(MatrixAlgebraKit.truncrank(4); convtol = 1e-8, maxiter = maxiter, verbosity = 2, timer_output = TimerOutput(), finalizer = myFinalizer)

Es, Ms, ξs, ϵs = scan_phasespace(δs, Δβ, alg_evo)
Ms = abs.(Ms)

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax1 = Axis(fig[1, 1], 
        xlabel = L"δ",
        ylabel = L"$E$",
        # xscale = log10,
        # yscale = log10
        )
ax2 = Axis(fig[1, 2], 
        xlabel = L"δ",
        ylabel = L"$M$",
        # xscale = log10,
        # yscale = log10
        )
ax3 = Axis(fig[2, 1], 
        xlabel = L"δ",
        ylabel = L"$ξ$",
        # xscale = log10,
        # yscale = log10
        )

ax4 = Axis(fig[2, 2], 
    xlabel = L"δ",
    ylabel = L"$ϵ$",
    # xscale = log10,
    yscale = log10
)


scatterlines!(ax1, δs, Es, label=L"$E$", markersize=20)
scatterlines!(ax2, δs, Ms, label=L"$M$", markersize=20)
scatterlines!(ax3, δs, ξs, label=L"$ξ$", markersize=20)
scatterlines!(ax4, δs, ϵs, label=L"$ϵ$", markersize=20)
fig
