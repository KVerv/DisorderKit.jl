using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, CairoMakie, TimerOutputs
using BlockTensorKit, MatrixAlgebraKit, OptimKit


# Define model
J₀ = 1.0
δ = 0.0
W = 0.1
hs = exp(δ) * [1., exp(-2W)]
Js = J₀ * [1., exp(-2W)]
ps = [0.25, 0.25, 0.25, 0.25]

Hs = DisorderKit.random_transverse_field_ising(Js, hs)


Δτ = 0.02 # Step size for imaginary time evolution
maxiter = 100 # Maximum number of time steps
approx_maxiter = 100 # Maximum number of LBFGS iterations in the approximation step
L = 6 # System size
D = 8 # Initial state Bonddimension
Dmax = 8 # Maximal state Bonddimension
D_dis = length(ps) # Disorder Bonddimension
D_phys = 2 # Physical dimension
ψ = DisorderKit.FiniteDisorderMPS(ps, L, D_dis, D_phys, D; zero_variance = false)

Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
X = TensorMap([0. 1.; 1. 0.], ℂ^2, ℂ^2)

# Energy density and average |⟨Zᵢ⟩| after every time step
function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    M = sum(abs.(real.(DisorderKit.expectation_value(ρ, Z))))/length(ρ)
    return (E, M)
end

myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64}, my_finalize!)
alg = DisorderKit.Groundstate_FiniteDTEBD(; trunc_tol = 1e-8, convtol = 1e-8, maxiter = maxiter, approx_maxiter = approx_maxiter, verbosity = 2, timer_output = TimerOutput(), finalizer = myFinalizer)
ψ, gradhist = DisorderKit.groundstate!(ψ, Hs; gradtol = 1e-10, verbosity = 5, maxiter = 500)
ψgs, data, info = DisorderKit.groundstate(ψ, Hs, Δτ, alg; maxD = Dmax)


Es = getindex.(data, 1)
Ms_τ = getindex.(data, 2)
τs = Δτ:Δτ:length(Es)*Δτ

# Observables of the final state
E = DisorderKit.energy_density(ψgs, Hs)
Ms = real.(DisorderKit.expectation_value(ψgs, Z)) # ⟨Zᵢ⟩ on every site
Mxs = real.(DisorderKit.expectation_value(ψgs, X)) # ⟨Xᵢ⟩ on every site
M = sum(abs.(Ms))/L

# Spin-spin correlation function from site i0
i0 = 1
j0 = L
Cs = DisorderKit.correlator(ψgs, Z, Z, i0, j0) # ⟨Z_{i0} Z_{i0+r}⟩, r = 1..j0-i0
Ccs = Cs .- Ms[i0] .* Ms[i0+1:j0] # connected part (with disorder-averaged ⟨Zᵢ⟩)
rs = 1:length(Cs)

@show (E, M)

# Free fermion reference
n_samples = 2048
data_FF = DisorderKit.sample_RTFIM(L, Js, hs; ps=ps, n_samples=n_samples, β=Inf, i0=i0)
EFF = data_FF.E
@show abs(E-EFF)

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 3000))
ax1 = Axis(fig[1, 1],
        xlabel = L"τ",
        ylabel = L"$E$",
        )
ax2 = Axis(fig[1, 2],
        xlabel = L"τ",
        ylabel = L"$ϵ_{conv}$",
        yscale = log10
        )
ax3 = Axis(fig[2, 1],
        xlabel = L"τ",
        ylabel = L"$ϵ_{approx}$",
        yscale = log10
        )
ax4 = Axis(fig[2, 2],
        xlabel = L"i",
        ylabel = L"$\langle O_i \rangle$",
        )
ax5 = Axis(fig[3, 1],
        xlabel = L"r",
        ylabel = L"$C(r)$",
        )
ax6 = Axis(fig[3, 2],
        xlabel = L"r",
        ylabel = L"$|C_c(r)|$",
        yscale = log10
        )

scatterlines!(ax1, τs, Es, label=L"$Δτ=%$Δτ$", markersize=20)
hlines!(ax1, [EFF], color=:red, linestyle=:dash, label=L"$E_{FF}$")
scatterlines!(ax2, τs, info.ϵsconv, markersize=20)
scatterlines!(ax3, τs, abs.(info.ρ_trunc_err) .+ 1e-16, markersize=20)
scatterlines!(ax4, 1:L, Ms, label=L"$\langle Z_i \rangle$", markersize=20)
scatterlines!(ax4, 1:L, Mxs, label=L"$\langle X_i \rangle$", markersize=20)
scatterlines!(ax4, 1:L, data_FF.Xs, label=L"$\langle X_i \rangle_{FF}$", linestyle=:dash, markersize=20)
scatterlines!(ax5, rs, Cs, label=L"$\langle Z_{%$i0} Z_{%$i0+r} \rangle$", markersize=20)
scatterlines!(ax5, rs, data_FF.ZZs, label=L"$\mathrm{FF}$", linestyle=:dash, markersize=20)
scatterlines!(ax6, rs, abs.(Ccs) .+ 1e-16, label=L"$E=%$(round(E, digits=6))$", markersize=20)

axislegend(ax1, position=:rt)
axislegend(ax4, position=:rt)
axislegend(ax5, position=:rt)
axislegend(ax6, position=:rt)

fig
