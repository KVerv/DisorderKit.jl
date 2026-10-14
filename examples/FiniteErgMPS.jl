using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, CairoMakie
using BlockTensorKit, MatrixAlgebraKit, OptimKit


# Define model
J₀ = 1.0
δ = 0.0
W = 0.1
hs = exp(δ) * [1., exp(-2W)]
Js = J₀ * [1., exp(-2W)]
ps = [0.25, 0.25, 0.25, 0.25]

Hs = DisorderKit.random_transverse_field_ising(Js, hs)


maxiter = 1000 # Maximum number of iterations for the groundstate algorithm
L = 6 # System size
D = 8 # State Bonddimension
D_dis = length(ps) # Disorder Bonddimension
D_phys = 2 # Physical dimension
ψ = DisorderKit.FiniteDisorderMPS(ps, L, D_dis, D_phys, D; zero_variance = false)

ψgs, gradhist = DisorderKit.groundstate!(ψ, Hs; gradtol = 1e-6, verbosity = 5, maxiter = maxiter)

# Observables
Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
X = TensorMap([0. 1.; 1. 0.], ℂ^2, ℂ^2)

E = DisorderKit.energy_density(ψgs, Hs)
Ms = real.(DisorderKit.expectation_value(ψgs, Z)) # ⟨Zᵢ⟩ on every site
Mxs = real.(DisorderKit.expectation_value(ψgs, X)) # ⟨Xᵢ⟩ on every site
M = sum(abs.(Ms))/L

# Spin-spin correlation function from a reference site away from the boundary
i0 = 1
j0 = L
Cs = DisorderKit.correlator(ψgs, Z, Z, i0, j0) # ⟨Z_{i0} Z_{i0+r}⟩, r = 1..j0-i0
Ccs = Cs .- Ms[i0] .* Ms[i0+1:j0] # connected part (with disorder-averaged ⟨Zᵢ⟩)
rs = 1:length(Cs)

@show (E, M)

# Free-fermion disorder average for comparison
n_samples = 100000
data = DisorderKit.sample_RTFIM(L, Js, hs; ps=ps, n_samples=n_samples, β=Inf, i0=i0)
@show abs(E - data.E)

# Fit window for the log-log fit of the sampled connected correlator
r_fit_min = 2
r_fit_max = 5

# Linear fit log|C_c(r)| = a + b log r on the fit window
fit_rs = r_fit_min:r_fit_max
A_fit = [ones(length(fit_rs)) log.(fit_rs)]
a_fit, b_fit = A_fit \ log.(abs.(data.ZZs[fit_rs]))
@show b_fit

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax1 = Axis(fig[1, 1],
        xlabel = L"\mathrm{iteration}",
        ylabel = L"$\Vert \nabla E \Vert$",
        yscale = log10
        )
ax2 = Axis(fig[1, 2],
        xlabel = L"i",
        ylabel = L"$\langle O_i \rangle$",
        )
ax3 = Axis(fig[2, 1],
        xlabel = L"r",
        ylabel = L"$C(r)$",
        )
ax4 = Axis(fig[2, 2],
        xlabel = L"r",
        ylabel = L"$|C_c(r)|$",
        xscale = log10,
        yscale = log10
        )

scatterlines!(ax1, 1:size(gradhist, 1), gradhist[:, 2], markersize=10) # gradhist columns: [E, ‖∇E‖]
# MPS: filled circles with lines, sampling: open squares with error bars
cols = Makie.wong_colors()
sampling_style = (marker=:rect, markersize=20, color=:transparent, strokewidth=2)

scatterlines!(ax2, 1:L, Ms, label=L"$\langle Z_i \rangle$ MPS", markersize=20, color=cols[1])
scatterlines!(ax2, 1:L, Mxs, label=L"$\langle X_i \rangle$ MPS", markersize=20, color=cols[2])
errorbars!(ax2, 1:L, data.Xs, data.Xs_err, color=cols[3], whiskerwidth=10)
scatter!(ax2, 1:L, data.Xs; label=L"$\langle X_i \rangle$ sampling", strokecolor=cols[3], sampling_style...)

scatterlines!(ax3, rs, Cs, label=L"$\langle Z_{%$i0} Z_{%$i0+r} \rangle$ MPS", markersize=20, color=cols[1])
errorbars!(ax3, rs, data.ZZs, data.ZZs_err, color=cols[3], whiskerwidth=10)
scatter!(ax3, rs, data.ZZs; label=L"$\langle Z_{%$i0} Z_{%$i0+r} \rangle$ sampling", strokecolor=cols[3], sampling_style...)

# Sampling has ⟨Zᵢ⟩ = 0 by Z₂ symmetry, so ZZ is already connected
scatterlines!(ax4, rs, abs.(Ccs) .+ 1e-16, label=L"MPS: $E=%$(round(E, digits=6))$", markersize=20, color=cols[1])
scatter!(ax4, rs, abs.(data.ZZs) .+ 1e-16;
        label=L"sampling: $E=%$(round(data.E, digits=6)) \pm %$(round(data.E_err, sigdigits=2))$",
        strokecolor=cols[3], sampling_style...)
lines!(ax4, fit_rs, exp(a_fit) .* fit_rs .^ b_fit;
        label=L"fit: $|C_c| \propto r^{%$(round(b_fit, digits=3))}$",
        color=cols[3], linestyle=:dash, linewidth=4)
vlines!(ax4, [r_fit_min, r_fit_max], color=(:gray, 0.5), linestyle=:dot)

axislegend(ax2, position=:rt)
axislegend(ax3, position=:rt)
axislegend(ax4, position=:rt)

fig