# Observables and plot of a hpc/sweep_BoseHubbard_NormalizedITE.sh run: from the stored states it computes the
# disorder-averaged density ρ, compressibility κ = dρ/dμ, condensate weight |⟨a⟩|² and entanglement entropy profile
# S(k) on every bond k = 1..L-1 with its disorder variance Var[S](k) and the chain average of that variance. It plots ρ,
# κ and |⟨a⟩|² against μ/U, S(k) and Var[S](k) for a few μ/U in two panels in the middle, and κ with the chain-averaged
# Var[S] against μ/U on the right.
#
# Usage (from the repo root, normally through hpc/submit_plot_BoseHubbard_NormalizedITE.sh, which sets the environment):
#   T=0.16 U=1.0 NMAX=4 DTAU=0.05 MAXD=16 DATA_DIR=$VSC_DATA/data N_SAMPLES=10000 S_MUS="0.3 0.6 1.0" \
#       julia --project=. hpc/plot_BoseHubbard_NormalizedITE.jl <L>
#
# Environment (all required; defaults in hpc/submit_plot_BoseHubbard_NormalizedITE.sh): T, U, NMAX, DTAU, MAXD,
#   DATA_DIR (the same as for the sweep, so the run tag matches), N_SAMPLES (disorder words are enumerated exactly if
#   q^L ≤ N_SAMPLES, else sampled), S_MUS (μ/U values of the entropy profiles; the nearest grid points are used)
# Output: the table on stdout, <DATA_DIR>/BoseHubbard_NormalizedITE/<run tag>/L<L>/observables.csv (with the chain-
#   averaged variance in varS_avg, the profile in the columns S_1..S_{L-1} and its variance in varS_1..varS_{L-1}) and plots/BoseHubbard_NormalizedITE_<run tag>_L<L>.png
#
# H conserves N, so an exact ground state with a fixed particle number has ⟨aᵢ⟩ = 0. ⟨aᵢ⟩ is nonzero only when the state
# mixes neighbouring particle numbers, which the evolution from the uniform state keeps where they are nearly degenerate
# (superfluid) and the bond cap maxD favours. |⟨a⟩|² is thus a superfluid indicator (≈ 0 on a Mott plateau), not a
# converged ground-state value. Every conditional state has its own global phase, so the squared modulus is averaged:
#   |⟨a⟩|² = Σ_w P(w) (1/L) Σᵢ |⟨aᵢ⟩_w|².

length(ARGS) == 1 || error("usage: julia --project=. hpc/plot_BoseHubbard_NormalizedITE.jl <L>")
L = parse(Int, ARGS[1])

# Same run tag as BoseHubbard_NormalizedITE.jl
param(k) = haskey(ENV, k) ? ENV[k] : error("environment variable $k not set; run parameters are set in hpc/submit_plot_BoseHubbard_NormalizedITE.sh")
t_str = param("T")
U_str = param("U")
nmax = parse(Int, param("NMAX"))
dtau_str = param("DTAU")
maxD = parse(Int, param("MAXD"))
run_tag = "t$(t_str)_U$(U_str)_nmax$(nmax)_dtau$(dtau_str)_D$(maxD)"
data_root = param("DATA_DIR")
folder = joinpath(data_root, "BoseHubbard_NormalizedITE", run_tag, "L$(L)")
n_samples = parse(Int, param("N_SAMPLES"))
S_mus = parse.(Float64, split(param("S_MUS")))

using TensorKit, DisorderKit, LinearAlgebra, Random, StatsBase, CairoMakie, Printf, JLD2

d = nmax + 1
n_op = TensorMap(ComplexF64.(diagm(0:nmax)), ℂ^d ← ℂ^d)
a_op = TensorMap(ComplexF64.(diagm(1 => sqrt.(1:nmax))), ℂ^d ← ℂ^d)   # a|n⟩ = √n |n-1⟩

# Disorder-averaged density and |⟨a⟩|²: all words if there are at most n_samples, else sampled words
function disorder_average_observables(ψ)
    ρs = DisorderKit.expectation_value(ψ, n_op)
    φs = DisorderKit.expectation_value(ψ, a_op)
    return sum(ρs)/length(ρs), sum(φs)/length(φs)
end

# Load every μ of the run, sorted by μ
isdir(folder) || error("no results in $folder")
files = filter(f -> occursin(r"^mu-?[\d.]+\.jld2$", f), readdir(folder))
isempty(files) && error("no results in $folder")
results = map(files) do f
    data = load(joinpath(folder, f))
    ρ, φ = disorder_average_observables(data["ψ"])
    # Disorder-averaged entanglement entropy on every bond, its standard error (zero if the words are enumerated) and
    # its disorder variance
    S, S_err, S_var = DisorderKit.average_entanglement_entropy(data["ψ"]; n_samples, rng = Xoshiro(1))
    p = data["params"]
    (; μ = p.μ, U = p.U, t = p.t, ρ, φ, S, S_err, S_var, nsteps = data["nsteps"], converged = data["converged"],
     ϵconv = data["info"].ϵsconv[end])
end
sort!(results; by = r -> r.μ)
μs = [r.μ / r.U for r in results]
ρs = [r.ρ for r in results]
φs = [r.φ for r in results]
converged = [r.converged for r in results]
t, U = results[1].t, results[1].U

# Compressibility κ = dρ/dμ (central differences, one-sided at the ends)
κs = similar(ρs)
for k in eachindex(μs)
    lo, hi = max(k - 1, 1), min(k + 1, length(μs))
    κs[k] = hi == lo ? NaN : (ρs[hi] - ρs[lo]) / (U * (μs[hi] - μs[lo]))
end

# Disorder variance of S averaged over all bonds of the chain
varSs = [sum(r.S_var) / length(r.S_var) for r in results]

println("Run tag $run_tag, L = $L ($folder)")
kmid = max(L ÷ 2, 1)   # central bond
println("    μ/U        ρ        κ   |⟨a⟩|²  S(L/2)  ⟨Var S⟩   steps   ϵ_conv   converged")
for (k, r) in enumerate(results)
    @printf("%7.3f  %7.4f  %7.3f  %7.4f  %6.4f  %7.4f  %6d  %7.1e   %s\n", μs[k], ρs[k], κs[k], φs[k], r.S[kmid],
            varSs[k], r.nsteps, r.ϵconv, r.converged)
end
all(converged) || println("$(count(!, converged)) of $(length(converged)) points not converged (open markers in the plot)")

csv = joinpath(folder, "observables.csv")
open(csv, "w") do io
    println(io, "mu_over_U,rho,kappa,a2,varS_avg,nsteps,eps_conv,converged,", join(("S_$b" for b in 1:L-1), ","), ",",
            join(("varS_$b" for b in 1:L-1), ","))
    for (k, r) in enumerate(results)
        println(io, "$(μs[k]),$(ρs[k]),$(κs[k]),$(φs[k]),$(varSs[k]),$(r.nsteps),$(r.ϵconv),$(r.converged),",
                join(r.S, ","), ",", join(r.S_var, ","))
    end
end
println("Saved $csv")

# Figure: ρ and |⟨a⟩|² on the left axis, κ on the right axis, sharing μ/U. Unconverged points have open markers
set_theme!(theme_latexfonts())
cρ, cκ, cφ = Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3]
fig = Figure(backgroundcolor = :white, fontsize = 30, size = (3300, 1100))
title = L"$L=%$L$, $t/U=%$(t/U)$, $t' = t/3$, $D=%$maxD$"
axρ = Axis(fig[1:2, 1]; xlabel = L"μ/U", ylabel = L"ρ,\ |\langle a \rangle|^2", title)
axκ = Axis(fig[1:2, 1]; ylabel = L"κ = dρ/dμ", yaxisposition = :right, ylabelcolor = cκ, yticklabelcolor = cκ,
           ygridvisible = false)
hidespines!(axκ)
hidexdecorations!(axκ)
linkxaxes!(axρ, axκ)
hlines!(axρ, [1.0]; color = :gray, linestyle = :dash)

markerfill(c) = [conv ? c : :white for conv in converged]
lines!(axρ, μs, ρs; color = cρ, linewidth = 3)
lines!(axρ, μs, φs; color = cφ, linewidth = 3)
lines!(axκ, μs, κs; color = cκ, linewidth = 3)
scatter!(axρ, μs, ρs; color = markerfill(cρ), strokecolor = cρ, strokewidth = 2, markersize = 14)
scatter!(axρ, μs, φs; color = markerfill(cφ), strokecolor = cφ, strokewidth = 2, marker = :diamond, markersize = 16)
scatter!(axκ, μs, κs; color = markerfill(cκ), strokecolor = cκ, strokewidth = 2, marker = :rect, markersize = 14)
ylims!(axρ, 0, nothing)
ylims!(axκ, 0, nothing)
entry(c, m, ms) = [LineElement(color = c, linewidth = 3), MarkerElement(color = c, marker = m, markersize = ms)]
axislegend(axρ, [entry(cρ, :circle, 14), entry(cφ, :diamond, 16), entry(cκ, :rect, 14)],
           [L"ρ", L"|\langle a \rangle|^2", L"κ"]; position = :lt)

# Right panels: entropy profile S(k) and its disorder variance for the grid points nearest to the requested μ/U
profile_ks = Int[]
for m in S_mus
    k = argmin(abs.(μs .- m))
    isapprox(μs[k], m; atol = 1e-8) || println("No μ/U = $m in the run, the entropy profile uses the nearest μ/U = $(μs[k])")
    k in profile_ks || push!(profile_ks, k)
end
bonds = 1:L-1
axS = Axis(fig[1, 2]; ylabel = L"\overline{S}(k)", title = "entanglement entropy", xticks = bonds)
axV = Axis(fig[2, 2]; xlabel = L"\textrm{bond}\ k", ylabel = L"\mathrm{Var}[S](k)", xticks = bonds)
hidexdecorations!(axS; grid = false)
linkxaxes!(axS, axV)
cSs = Makie.wong_colors()[4:end]
for (j, k) in enumerate(profile_ks)
    r, c = results[k], cSs[mod1(j, length(cSs))]
    fill = r.converged ? c : :white
    lines!(axS, bonds, r.S; color = c, linewidth = 3, label = L"μ/U = %$(round(μs[k]; digits = 3))")
    any(>(0), r.S_err) && errorbars!(axS, bonds, r.S, r.S_err; color = c, whiskerwidth = 10)
    scatter!(axS, bonds, r.S; color = fill, strokecolor = c, strokewidth = 2, markersize = 14)
    lines!(axV, bonds, r.S_var; color = c, linewidth = 3)
    scatter!(axV, bonds, r.S_var; color = fill, strokecolor = c, strokewidth = 2, markersize = 14)
end
ylims!(axS, 0, nothing)
ylims!(axV, 0, nothing)
axislegend(axS; position = :cb)

# Right panel: κ on the left axis and the chain-averaged Var[S] on the right axis, sharing μ/U
cV = Makie.wong_colors()[7]
axκ2 = Axis(fig[1:2, 3]; xlabel = L"μ/U", ylabel = L"κ = dρ/dμ", ylabelcolor = cκ, yticklabelcolor = cκ,
            title = "compressibility vs. entropy fluctuations")
axV2 = Axis(fig[1:2, 3]; ylabel = L"\overline{\mathrm{Var}[S]}", yaxisposition = :right, ylabelcolor = cV,
            yticklabelcolor = cV, ygridvisible = false)
hidespines!(axV2)
hidexdecorations!(axV2)
linkxaxes!(axκ2, axV2)
lines!(axκ2, μs, κs; color = cκ, linewidth = 3)
lines!(axV2, μs, varSs; color = cV, linewidth = 3)
scatter!(axκ2, μs, κs; color = markerfill(cκ), strokecolor = cκ, strokewidth = 2, marker = :rect, markersize = 14)
scatter!(axV2, μs, varSs; color = markerfill(cV), strokecolor = cV, strokewidth = 2, marker = :utriangle,
         markersize = 16)
ylims!(axκ2, 0, nothing)
ylims!(axV2, 0, nothing)
axislegend(axκ2, [entry(cκ, :rect, 14), entry(cV, :utriangle, 16)], [L"κ", L"\overline{\mathrm{Var}[S]}"];
           position = :lt)

mkpath("plots")
file = joinpath("plots", "BoseHubbard_NormalizedITE_$(run_tag)_L$(L).png")
save(file, fig)
println("Saved $file")
