using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit
using BlockTensorKit, MatrixAlgebraKit, OptimKit

# Define model
# ζs = [0.58579, 3.41421]
# hs = exp.(-ζs)
# Js = exp.(-ζs)
# w1 = 0.853527437561
# w2 = 1-w1
# ps = [w1*w1, w1*w2, w2*w1, w2*w2]

hs = [0.7, 1.3]
Js = [0.7, 1.3]
w1 = 0.5
w2 = 1-w1
ps = [w1*w1, w1*w2, w2*w1, w2*w2]

# hs = [1.0]
# Js = [1.0]
# w1 = 0.5
# w2 = 1-w1
# ps = [1.]

Hs = DisorderKit.random_transverse_field_ising(Js, hs)


Δτ = 0.05 # Step size for imaginary time evolution
maxiter = 500 # Maximum number of iterations for the groundstate algorithm
D = 6 # State Bonddimension
D_R = 2 # Renormalization operator Bonddimension


# function RG_scale(ρ)
#         N_dis = length(ρ.ps)
#         X = TensorMap([0. 1.; 1. 0.], ℂ^2, ℂ^2)
#         Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
#         λ, l, r = DisorderKit.environments(ρ)

#         @tensor EX[-1; -2] :=  l[6; 1] * ρ[1][1 2 -1; 7 8 11] * X[4; 2] * conj(ρ[1][6 4 -2; 7 8 10]) * r[11; 10]
#         EX /= λ
#         #FIXME Include long range interactions
#         @tensor vL[-1 -2; -3 -4] := l[1; 4] * ρ[1][4 5 -2; 2 3 -3] * Z[6; 5] * conj(ρ[1][1 6 -4; 2 3 -1])
#         @tensor vR[-1 -2; -3 -4] := r[1; 4] * ρ[1][-1 5 -2; 2 3 1] * Z[6; 5] * conj(ρ[1][-3 6 -4; 2 3 4])

#         @tensor EZZ[-1 -2; -3 -4] := vL[1 -1; 2 -3] * vR[2 -2; 1 -4]
#         EZZ /= λ^2

#         qh = [(EX.data[i,i].data[1]) for i in 1:N_dis]
#         qJ = [(EZZ.data[i,j,i,j].data[1]) for i in 1:N_dis, j in 1:N_dis]
#         mJ = real(sum(qJ)/length(qJ))
#         mh = real(sum(qh)/length(qh))
#         VJ = real(sum((qJ .- mJ).^2)/length(qJ))
#         Vh = real(sum((qh .- mh).^2)/length(qh))
#         return mh, mJ, Vh, VJ
# end

# Construct Finalizer for computing observables after each iteration of the groundstate algorithm
function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    ξ = DisorderKit.average_correlation_length(ρ)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
    M = real.(DisorderKit.expectation_value(ρ, Z))
    rmax = round(Int, 5*ξ)
    Cs = real.(DisorderKit.two_point_correlator(ρ, Z, Z, rmax+1))
    S = DisorderKit.average_entanglement_entropy(ρ)[1]
    return (E, M, ξ, Cs, S)
end

myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64, Float64, Vector{Float64}, Float64}, my_finalize!)

# Generate algorithm object with corresponding parameters
alg_evo = DisorderKit.Groundstate_iDTEBD(MatrixAlgebraKit.truncrank(D), MatrixAlgebraKit.truncrank(D_R); convtol = 1e-9, maxiter = maxiter, verbosity = 2, timer_output = TimerOutput(), finalizer = myFinalizer)

# Construct initial random state
A = spzeros(ComplexF64, BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(ℂ^2) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...), BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...) ⊗ BlockTensorKit.boxplus(ℂ^1))
# Aᵢ = rand(ComplexF64, ℂ^1 ⊗ ℂ^2 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1 ⊗ ℂ^1) 
for i in eachindex(ps)
        A[1, 1, i, 1, i, 1] = Aᵢ # Same tensor for each disorder sector
end

ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix([A], ps)
ρ₀ = DisorderKit.gauge(ρ₀)

# Compute the groundstate
@timeit alg_evo.timer_output "compute_groundstate" ρs, data, info = DisorderKit.groundstate(ρ₀, Hs, Δτ, alg_evo)

Es = getindex.(data, 1)
Ms = getindex.(data, 2)
ξs = getindex.(data, 3)
Cs = getindex.(data, 4)
Ss = getindex.(data, 5)
τs = Δτ:Δτ:length(Es)*Δτ

E = Es[end]
ξ = ξs[end]

@show (E, ξ)

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax1 = Axis(fig[1, 1], 
        xlabel = L"τ",
        ylabel = L"$E$",
        xscale = log10,
        # yscale = log10
        )
ax2 = Axis(fig[1, 2], 
        xlabel = L"τ",
        ylabel = L"$ϵ_{conv}$",
        xscale = log10,
        yscale = log10
        )
ax3 = Axis(fig[2, 1], 
        xlabel = L"τ",
        ylabel = L"$M$",
        # xscale = log10,
        # yscale = log10
        )
ax4 = Axis(fig[2, 2], 
        xlabel = L"τ",
        ylabel = L"$ξ$",
        # xscale = log10,
        # yscale = log10
        )

colors = Makie.wong_colors()
scatterlines!(ax1, τs, Es, label=L"$Δτ=%$Δτ$", markersize=20)
scatterlines!(ax2, τs[1:length(τs)-1], info.ϵsconv[1:length(τs)-1], label=L"$ϵ_{conv}$", markersize=20)
scatterlines!(ax2, τs, (info.ρ_trunc_err[1:length(τs)])/Δτ, label=L"$ϵ_{ρ}$", markersize=20)
scatterlines!(ax3, τs, Ms, label=L"$ϵ_{acc}$", markersize=20)
scatterlines!(ax4, τs, ξs, label=L"$ϵ_{acc}$", markersize=20)

axislegend(ax1, position=:rt)
fig
   
set_theme!(theme_latexfonts())
fig2 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax21 = Axis(fig2[1, 1], 
        xlabel = L"τ",
        ylabel = L"$ϵ_ρ$",
        xscale = log10,
        yscale = log10
        )
ax22 = Axis(fig2[1, 2], 
        xlabel = L"τ",
        ylabel = L"$ϵ_R$",
        xscale = log10,
        yscale = log10
        )
ax23 = Axis(fig2[2, 1], 
        xlabel = L"τ",
        ylabel = L"$ϵ_{A1}$",
        xscale = log10,
        yscale = log10
        )
ax24 = Axis(fig2[2, 2], 
        xlabel = L"τ",
        ylabel = L"$ϵ_{z}$",
        xscale = log10,
        yscale = log10
        )

colors = Makie.wong_colors()
scatterlines!(ax21, τs[2:end], info.ρ_trunc_err[2:end]/Δτ, label=L"$Δτ=%$Δτ$", markersize=20)
scatterlines!(ax22, τs, info.R_trunc_err, label=L"$ϵ_{conv}$", markersize=20)
scatterlines!(ax23, τs[2:end], info.ϵsA1[2:end], label=L"$ϵ_{A1}$", markersize=20)
scatterlines!(ax24, τs[2:end], info.ϵsA2[2:end], label=L"$ϵ_{acc}$", markersize=20, marker=:utriangle)

axislegend(ax1, position=:rt)
fig2



fig3 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax31 = Axis(fig3[1, 1], 
        xlabel = L"\ln r",
        ylabel = L"$\ln (C_r)$",
        # xscale = log10,
        # yscale = log10
)

for (i, ys) in enumerate(Cs)
        ξ = ξs[i]
        rs = 1:length(ys)
        # rs *= 1/ξ
        yss = ys./ys[1]#.*rs.^0.38
        j = 200

        # if i % length(Cs) == j
        if i % 50 == 0
            scatter!(ax31, log.(rs), log.(yss), label=L"$D=%$D$", markersize=20)
        # scatter!(ax21, (rs), (yss), label=L"$D=%$D$", markersize=20)
                @show ξ
        end
        # if i == length(Cs)
        if i == j
                lines!(ax31, log.(rs), -0.25 *(log.(rs).-log.(rs[1])).+log.(yss[1]), color=:black, linewidth=2)
                lines!(ax31, log.(rs), -0.38 *(log.(rs).-log.(rs[1])).+log.(yss[1]), color=:red, linewidth=2)
        end

end
fig3


minfit = 30
maxfit = 130
p0q = [1., 1.]
linmodel(t, p) = p[1] .+ p[2] * t
xs = ξs[minfit:maxfit]
ys = Ss[minfit:maxfit]
linfit = curve_fit(linmodel, log.(xs), ys, p0q)


set_theme!(theme_latexfonts())
fig4 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax41 = Axis(fig4[1, 1], 
        xlabel = L"log(ξ)",
        ylabel = L"$log(S)$",
        # xscale = log10,
        # yscale = log10
        )
scatter!(ax41, log.(ξs[10:end]), Ss[10:end], label=L"$D=%$D$", markersize=20)
lines!(ax41, log.(xs), linmodel(log.(xs), linfit.param), color=:black, linewidth=2)

fig4
