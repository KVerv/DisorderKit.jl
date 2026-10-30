# Effective central charge of the critical random transverse-field Ising chain (δ = 0) from the disorder-averaged
# entanglement entropies of Groundstate_FiniteNormalizedITE ground states on open chains of several lengths L.
# Model (as in hpc/RTFIM_NormalizedITE.jl): h ∈ e^δ {1, e^{-2W}}, J ∈ {1, e^{-2W}}, p_u = 1/4.
#
# c_eff is extracted in two ways:
#   left panel   size scaling of the half-chain entropy S(L/2) and of the bond average S̄ = mean_k S(k),
#                both S = (c_eff/6) log L + const on an open chain
#   right panel  Calabrese-Cardy fit of the profiles, S(k) = (c_eff/6) log[(2L/π) sin(πk/L)] + s₀, jointly over all L
# Expected: c = 1/2 for the clean chain, c_eff = ln2/2 ≈ 0.347 at the infinite-randomness fixed point; at these small
# sizes the value lies in between. The exact free fermion disorder average at the same sizes is shown dashed.
using TensorKit, DisorderKit, LinearAlgebra, Random, Printf, CairoMakie
using Distributions: TDist, quantile, mean

# Model
δ = 0.0
W = 0.5
hs = exp(δ) .* [1.0, exp(-2W)]
Js = [1.0, exp(-2W)]
ps = fill(0.25, 4)
Hs = DisorderKit.random_transverse_field_ising(Js, hs)

# Normalized ITE
Ls = [32, 40, 48]                    # system sizes
Δτ = 0.02
maxD = [100 => 8, 100 => 16, 100 => 24]            # stage schedule (steps => bond cap), the last cap is kept
maxiter = 300
convtol = 1e-8
n_samples = 5000                        # disorder words for the MPS and free fermion averages
edge_skip = 1                           # bonds k ≤ edge_skip and k ≥ L - edge_skip are left out of the Calabrese-Cardy fit

# Weighted linear fit y = a + b x, c_eff = 6b, with a confidence interval Δc on c_eff at the given level.
# With errors σs the fit is weighted by 1/σ² (unweighted if any σ is zero, i.e. exact enumeration). The parameter
# covariance is scaled by the reduced χ² of the residuals and the interval uses a Student t with n-2 degrees of freedom,
# so it also reflects how well the data follows a straight line. It is undefined (NaN) for only two points.
function linear_fit(x, y; σs = nothing, level = 0.95)
    X = [ones(length(x)) x]
    w = (isnothing(σs) || any(iszero, σs)) ? ones(length(x)) : 1 ./ σs.^2
    Xw, yw = X .* sqrt.(w), y .* sqrt.(w)
    a, b = Xw \ yw
    dof = length(x) - 2
    dof > 0 || return a, b, 6b, NaN
    χ²_red = sum(abs2, yw - Xw * [a, b]) / dof
    cov = χ²_red * inv(Xw' * Xw)
    Δc = 6 * quantile(TDist(dof), (1 + level) / 2) * sqrt(cov[2, 2])
    return a, b, 6b, Δc
end

# Calabrese-Cardy chord coordinate of bond k on an open chain of length L
chord(k, L) = log(2L / π * sin(π * k / L))

# Ground states and disorder-averaged entropy profiles S(k), k = 1..L-1
Random.seed!(1)                         # free fermion configurations are sampled from the global RNG
profiles, profiles_err = Vector{Vector{Float64}}(), Vector{Vector{Float64}}()
profiles_FF, profiles_FF_err = Vector{Vector{Float64}}(), Vector{Vector{Float64}}()
nsteps, converged = Int[], Bool[]
for L in Ls
    alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD, order = 2, convtol, maxiter, verbosity = 1)
    walltime = @elapsed ψ, _, info = DisorderKit.groundstate(DisorderKit.plus_state(Hs, ps, L), Hs, Δτ, alg)
    push!(nsteps, length(info.Es))
    push!(converged, info.ϵsconv[end] <= convtol)
    converged[end] || @warn "L = $L not converged after $maxiter steps (ϵ_conv = $(info.ϵsconv[end]))"

    S, S_err, _ = DisorderKit.average_entanglement_entropy(ψ; n_samples, rng = Xoshiro(1))
    push!(profiles, S)
    push!(profiles_err, S_err)
    data_FF = DisorderKit.sample_RTFIM(L, Js, hs; ps, n_samples)
    push!(profiles_FF, data_FF.Ss)
    push!(profiles_FF_err, data_FF.Ss_err)
    @printf("L = %2d: %d steps in %.0f s, S(L/2) = %.4f, S_FF(L/2) = %.4f\n", L, nsteps[end], walltime, S[L ÷ 2], data_FF.Ss[L ÷ 2])
end

# Half-chain entropy and bond average. The error of S̄ is the mean of the per-bond errors, an upper bound since the
# bonds are correlated
half(Ss) = [S[L ÷ 2] for (S, L) in zip(Ss, Ls)]
S_half, S_half_err, S_half_FF, S_half_FF_err = half(profiles), half(profiles_err), half(profiles_FF), half(profiles_FF_err)
S_bar, S_bar_err = mean.(profiles), mean.(profiles_err)
S_bar_FF, S_bar_FF_err = mean.(profiles_FF), mean.(profiles_FF_err)

# Size scaling fits
logLs = log.(Ls)
fit_half = linear_fit(logLs, S_half; σs = S_half_err)
fit_bar = linear_fit(logLs, S_bar; σs = S_bar_err)
fit_half_FF = linear_fit(logLs, S_half_FF; σs = S_half_FF_err)
fit_bar_FF = linear_fit(logLs, S_bar_FF; σs = S_bar_FF_err)

# Joint Calabrese-Cardy fit over all L and the bulk bonds
function cardy_fit(Ss, Ss_err)
    x, y, σ = Float64[], Float64[], Float64[]
    for (L, S, S_err) in zip(Ls, Ss, Ss_err), k in edge_skip+1:L-1-edge_skip
        push!(x, chord(k, L))
        push!(y, S[k])
        push!(σ, S_err[k])
    end
    return linear_fit(x, y; σs = σ)
end
fit_CC = cardy_fit(profiles, profiles_err)
fit_CC_FF = cardy_fit(profiles_FF, profiles_FF_err)

println("\n  L  steps  conv    S(L/2)   S_FF(L/2)        S̄      S̄_FF")
for (ix, L) in enumerate(Ls)
    @printf("%3d  %5d  %4s  %8.4f  %10.4f  %8.4f  %8.4f\n", L, nsteps[ix], converged[ix], S_half[ix], S_half_FF[ix], S_bar[ix], S_bar_FF[ix])
end
println("\nc_eff (95% CI)          MPS               free fermions")
for (name, f, f_FF) in (("S(L/2) vs log L", fit_half, fit_half_FF), ("S̄ vs log L", fit_bar, fit_bar_FF),
                        ("Calabrese-Cardy", fit_CC, fit_CC_FF))
    @printf("%-18s  %.3f ± %.3f     %.3f ± %.3f\n", name, f[3], f[4], f_FF[3], f_FF[4])
end

# Figure
fmt(f) = L"%$(round(f[3]; digits = 3)) \pm %$(round(f[4]; digits = 3))"
set_theme!(theme_latexfonts())
colors = Makie.wong_colors()
fig = Figure(backgroundcolor = :white, fontsize = 30, size = (2000, 1050))
ax1 = Axis(fig[1, 1]; xlabel = L"\log L", ylabel = L"S", title = L"size scaling, $\delta = %$δ$, $W = %$W$")
ax2 = Axis(fig[1, 2]; xlabel = L"\textrm{bond}\ k", ylabel = L"\overline{S}(k)",
           title = L"Calabrese-Cardy: $c_{eff} = $ %$(fmt(fit_CC)), FF (dashed) %$(fmt(fit_CC_FF))")

# Left: S(L/2) and S̄ with their fits, free fermions dashed, guide lines for c = 1/2 and ln2/2
logL_fine = range(minimum(logLs), maximum(logLs), 100)
for (S, S_err, S_FF, f, f_FF, color, marker, name) in ((S_half, S_half_err, S_half_FF, fit_half, fit_half_FF, colors[1], :circle, "S(L/2)"),
                                                       (S_bar, S_bar_err, S_bar_FF, fit_bar, fit_bar_FF, colors[2], :rect, "\\overline{S}"))
    errorbars!(ax1, logLs, S, S_err; color, whiskerwidth = 10)
    scatter!(ax1, logLs, S; color, marker, markersize = 22)
    lines!(ax1, logL_fine, f[1] .+ f[2] .* logL_fine; color, linewidth = 3, label = L"$%$name$: $c_{eff} = $ %$(fmt(f))")
    scatter!(ax1, logLs, S_FF; color = :white, strokecolor = color, strokewidth = 2, marker, markersize = 22)
    lines!(ax1, logL_fine, f_FF[1] .+ f_FF[2] .* logL_fine; color, linewidth = 3, linestyle = :dash,
           label = L"$%$name$ FF: $c_{eff} = $ %$(fmt(f_FF))")
end
lines!(ax1, logL_fine, S_half_FF[1] .+ (logL_fine .- logLs[1]) ./ 12; color = :gray, linestyle = :dot, label = L"$c = 1/2$")
lines!(ax1, logL_fine, S_half_FF[1] .+ log(2) .* (logL_fine .- logLs[1]) ./ 12; color = :black, linestyle = :dot,
       label = L"$c_{eff} = \ln 2/2$")
Legend(fig[2, 1], ax1; nbanks = 3, framevisible = false, labelsize = 22)

# Right: profiles with the joint Calabrese-Cardy fit drawn for every L, free fermion profiles dashed
for (ix, L) in enumerate(Ls)
    color = colors[mod1(ix, 7)]
    k_fine = range(1, L - 1, 200)
    errorbars!(ax2, 1:L-1, profiles[ix], profiles_err[ix]; color, whiskerwidth = 8)
    scatter!(ax2, 1:L-1, profiles[ix]; color, markersize = 16, label = L"$L = %$L$")
    lines!(ax2, k_fine, fit_CC[1] .+ fit_CC[2] .* chord.(k_fine, L); color, linewidth = 3)
    lines!(ax2, 1:L-1, profiles_FF[ix]; color, linewidth = 2, linestyle = :dash)
end
Legend(fig[2, 2], ax2; orientation = :horizontal, framevisible = false, labelsize = 22, valign = :top)

# mkpath("plots")
# file = joinpath("plots", "RTFIM_central_charge_W$(W).png")
# save(file, fig)
# println("Saved $file")
fig
