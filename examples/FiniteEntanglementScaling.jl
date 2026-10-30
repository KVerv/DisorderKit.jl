using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, CairoMakie, TimerOutputs, LinearAlgebra
using Distributions: TDist, quantile

# Entanglement scaling of the RTFIM groundstate with FiniteNormalizedITE.
# At criticality the half-chain entropy of an open chain grows as S = (c_eff/6) log L + const,
# with c_eff = 1/2 for the clean Ising chain and c_eff = ln2/2 at the infinite-randomness fixed point.
# Both the MPS and the free fermion entropies are disorder averages 𝔼_w[S_w] over sampled (or enumerated) words.
# The system sizes are small, so the fit is rough.

# Define model
J₀ = 1.0
δ = 0.0
W = 0.5
hs = exp(δ) * [1., exp(-2W)]
Js = J₀ * [1., exp(-2W)]
ps = [0.25, 0.25, 0.25, 0.25]

Hs = DisorderKit.random_transverse_field_ising(Js, hs)


Δτ = 0.02 # Step size for imaginary time evolution
maxiter = 20 # Maximum number of time steps
Ls = [4] # System sizes
Dmax = 2 # Maximal state Bonddimension
n_samples = 10000 # Disorder samples for the MPS and free fermion entropies
Ls_FF = [16, 32, 64] # Larger system sizes, free fermions only

Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)

# Energy density and average |⟨Zᵢ⟩| after every time step
function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    M = sum(abs.(real.(DisorderKit.expectation_value(ρ, Z))))/length(ρ)
    return (E, M)
end

myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64}, my_finalize!)

Ss = zeros(length(Ls)) # Average entanglement entropy on the central bond
Ss_err = zeros(length(Ls))
Sprofiles = Vector{Vector{Float64}}(undef, length(Ls)) # Average entanglement entropy on every bond
Sprofiles_err = Vector{Vector{Float64}}(undef, length(Ls))
S_FF = zeros(length(Ls)) # Free fermion disorder average of S on the central bond
S_FF_err = zeros(length(Ls))
Sprofiles_FF = Vector{Vector{Float64}}(undef, length(Ls))
Sprofiles_FF_err = Vector{Vector{Float64}}(undef, length(Ls))
for (ix, L) in enumerate(Ls)
    ψ0 = DisorderKit.plus_state(Hs, ps, L) # |+x⟩ on every site, every disorder word normalized
    alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD = Dmax, order= 2, convtol = 1e-6, maxiter = maxiter, verbosity = 1, timer_output = TimerOutput(), finalizer = myFinalizer)
    ψgs, data, info = DisorderKit.groundstate(ψ0, Hs, Δτ, alg)

    Sprofiles[ix], Sprofiles_err[ix] = DisorderKit.average_entanglement_entropy(ψgs; n_samples=n_samples)
    Ss[ix], Ss_err[ix] = Sprofiles[ix][L ÷ 2], Sprofiles_err[ix][L ÷ 2]

    data_FF = DisorderKit.sample_RTFIM(L, Js, hs; ps=ps, n_samples=n_samples)
    S_FF[ix], S_FF_err[ix] = data_FF.Ss[L ÷ 2], data_FF.Ss_err[L ÷ 2]
    Sprofiles_FF[ix], Sprofiles_FF_err[ix] = data_FF.Ss, data_FF.Ss_err
    println("L = $L: E = $(getindex(data[end], 1)), S(L/2) = $(Ss[ix]) ± $(Ss_err[ix]), S_FF(L/2) = $(S_FF[ix]) ± $(S_FF_err[ix]), steps = $(length(data))")
end

# Free fermion reference at larger system sizes
S_FF_large = zeros(length(Ls_FF))
S_FF_large_err = zeros(length(Ls_FF))
for (ix, L) in enumerate(Ls_FF)
    data_FF = DisorderKit.sample_RTFIM(L, Js, hs; ps=ps, n_samples=n_samples)
    S_FF_large[ix], S_FF_large_err[ix] = data_FF.Ss[L ÷ 2], data_FF.Ss_err[L ÷ 2]
end

# Linear fits S = a + b log L, c_eff = 6b, with a confidence interval Δc on c_eff at the given level.
# With errors σs the fit is weighted by 1/σ² (unweighted if any σ is zero, i.e. exact enumeration). The parameter
# covariance is scaled by the reduced χ² of the residuals and the interval uses a Student t with n-2 degrees of freedom,
# so it also reflects how well the data follows a straight line. It is undefined (NaN) for only two points.
function log_fit(Ls, Ss; σs = nothing, level = 0.95)
    X = [ones(length(Ls)) log.(Ls)]
    w = (isnothing(σs) || any(iszero, σs)) ? ones(length(Ls)) : 1 ./ σs.^2
    Xw, yw = X .* sqrt.(w), Ss .* sqrt.(w)
    a, b = Xw \ yw
    dof = length(Ls) - 2
    dof > 0 || return a, b, 6b, NaN
    χ²_red = sum(abs2, yw - Xw * [a, b]) / dof
    cov = χ²_red * inv(Xw' * Xw)
    Δc = 6 * quantile(TDist(dof), (1 + level) / 2) * sqrt(cov[2, 2])
    return a, b, 6b, Δc
end
a, b, c_eff, Δc = log_fit(Ls, Ss; σs = Ss_err)
a_FF, b_FF, c_eff_FF, Δc_FF = log_fit(Ls, S_FF; σs = S_FF_err)
a_FFl, b_FFl, c_eff_FFl, Δc_FFl = log_fit(Ls_FF, S_FF_large; σs = S_FF_large_err)
println("c_eff (95% CI): MPS = $c_eff ± $Δc, FF (MPS sizes) = $c_eff_FF ± $Δc_FF, FF (all sizes) = $c_eff_FFl ± $Δc_FFl")

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(2400, 1000))
ax1 = Axis(fig[1, 1],
        xlabel = L"\log L",
        ylabel = L"$S(L/2)$",
        )
ax2 = Axis(fig[1, 2],
        xlabel = L"k",
        ylabel = L"$S(k)$",
        )

logLs = log.(Ls)
logLs_FF = log.(Ls_FF)
logL_fine = range(minimum(logLs), maximum(logLs), 100)
logL_fine_FF = range(minimum(logLs_FF), maximum(logLs_FF), 100)
errorbars!(ax1, logLs, Ss, Ss_err, whiskerwidth=10)
scatter!(ax1, logLs, Ss, markersize=25, label=L"\mathrm{MPS}")
lines!(ax1, logL_fine, a .+ b .* logL_fine, label=L"$\mathrm{MPS\ fit:}\ c_{eff} = %$(round(c_eff, digits=3)) \pm %$(round(Δc, digits=3))$")
errorbars!(ax1, logLs_FF, S_FF_large, S_FF_large_err, whiskerwidth=10)
scatter!(ax1, logLs_FF, S_FF_large, marker=:diamond, markersize=25, label=L"\mathrm{FF}")
lines!(ax1, logL_fine_FF, a_FFl .+ b_FFl .* logL_fine_FF, label=L"$\mathrm{FF\ fit\ (all\ sizes):}\ c_{eff} = %$(round(c_eff_FFl, digits=3)) \pm %$(round(Δc_FFl, digits=3))$")
lines!(ax1, logL_fine_FF, S_FF_large[1] .+ (logL_fine_FF .- logLs_FF[1]) ./ 12, linestyle=:dash, label=L"$c = 1/2$")
lines!(ax1, logL_fine_FF, S_FF_large[1] .+ log(2) .* (logL_fine_FF .- logLs_FF[1]) ./ 12, linestyle=:dash, label=L"$c_{eff} = \ln 2/2$")
for (ix, L) in enumerate(Ls)
    color = Makie.wong_colors()[mod1(ix, 7)]
    scatterlines!(ax2, 1:L-1, Sprofiles[ix], color=color, markersize=15, label=L"$L = %$L$")
    errorbars!(ax2, 1:L-1, Sprofiles[ix], Sprofiles_err[ix], color=color, whiskerwidth=10)
    lines!(ax2, 1:L-1, Sprofiles_FF[ix], color=color, linestyle=:dash)
    errorbars!(ax2, 1:L-1, Sprofiles_FF[ix], Sprofiles_FF_err[ix], color=color, whiskerwidth=10)
end

axislegend(ax1, position=:rb)
axislegend(ax2, position=:rt)

fig
