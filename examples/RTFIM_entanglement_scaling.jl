using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit
using BlockTensorKit, MatrixAlgebraKit, OptimKit

Ds = [2, 3, 4] # State Bonddimension
D_R = 2 # Renormalization operator Bonddimension
Δτ = 0.05 # Step size for imaginary time evolution
maxiter = 100 # Maximum number of iterations for the groundstate algorithm


w1 = 0.5
w2 = 1-w1
ps = [w1*w1, w1*w2, w2*w1, w2*w2]

hs = [0.7, 1.3]
Js = [0.7, 1.3]

Hs = DisorderKit.random_transverse_field_ising(Js, hs)

# Construct Finalizer for computing observables after each iteration of the groundstate algorithm
function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    ξ = DisorderKit.average_correlation_length(ρ)
    S, _ = DisorderKit.average_entanglement_entropy(ρ)
    return (E, ξ, M, S)
end

myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64, Float64}, my_finalize!)

Es = zeros(length(Ds))
ξs = zeros(length(Ds))
Ss = zeros(length(Ds))
ϵsconv = zeros(length(Ds))
ϵsent = zeros(length(Ds))
ϵsz = zeros(length(Ds))
ρ_trunc_err = zeros(length(Ds))
R_trunc_err = zeros(length(Ds))
ϵsA1 = zeros(length(Ds))
ϵsA2 = zeros(length(Ds))

Aᵢ = rand(ComplexF64, ℂ^1 ⊗ ℂ^2 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1 ⊗ ℂ^1) 
for (ix, D) in enumerate(Ds)

    # Construct initial random state
    A = spzeros(ComplexF64, BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(ℂ^2) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...), BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...) ⊗ BlockTensorKit.boxplus(ℂ^1))
    for i in eachindex(ps)
            A[1, 1, i, 1, i, 1] = Aᵢ # Same tensor for each disorder sector
    end

    ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix([A], ps)
    ρ₀ = DisorderKit.gauge(ρ₀)

    # Generate algorithm object with corresponding parameters
    alg_evo = DisorderKit.Groundstate_iDTEBD(MatrixAlgebraKit.truncrank(D), MatrixAlgebraKit.truncrank(D_R); convtol = 1e-9, maxiter = maxiter, verbosity = 2, timer_output = TimerOutput(), finalizer = myFinalizer)

    # Compute the groundstate
    @timeit alg_evo.timer_output "compute_groundstate" ρs, data, info = DisorderKit.groundstate(ρ₀, Hs, Δτ, alg_evo)

    Es[ix] = getindex.(data, 1)[end]
    ξs[ix] = getindex.(data, 2)[end]
    Ms[ix] = getindex.(data, 3)[end]
    Ss[ix] = getindex.(data, 4)[end]

    ϵsconv[ix] = info.ϵsconv[end]
    ϵsent[ix] = info.ϵsent[end]
    ϵsz[ix] = info.ϵsz[end]
    ρ_trunc_err[ix] = info.ρ_trunc_err[end]
    R_trunc_err[ix] = info.R_trunc_err[end]
    ϵsA1[ix] = info.ϵsA1[end]
    ϵsA2[ix] = info.ϵsA2[end]
end

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
        ylabel = L"$S$",
        # xscale = log10,
        # yscale = log10
        )
ax3 = Axis(fig[2, 1], 
        xlabel = L"δ",
        ylabel = L"$M$",
        # xscale = log10,
        # yscale = log10
        )
ax4 = Axis(fig[2, 2], 
        xlabel = L"δ",
        ylabel = L"$ξ$",
        # xscale = log10,
        # yscale = log10
        )

colors = Makie.wong_colors()
scatterlines!(ax1, δs, Es, label=L"$D=%$D$", markersize=20)
scatterlines!(ax2, δs, Ss, label=L"$S$", markersize=20)
scatterlines!(ax3, δs, Ms, label=L"$ϵ_{acc}$", markersize=20)
scatterlines!(ax4, δs, ξs, label=L"$ϵ_{z}$", markersize=20, marker=:utriangle)

axislegend(ax1, position=:rt)
fig
   