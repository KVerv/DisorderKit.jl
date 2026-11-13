# ρ, κ and K against μ for the random-hopping Bose-Hubbard chain,
#   H_w = -Σ t_{wᵢ} (b†ᵢbᵢ₊₁ + h.c.) + U/2 Σ nᵢ(nᵢ-1) - μ Σ nᵢ,   t_{wᵢ} ∈ {t, t/3} with probability 1/2 each,
# swept over μ by the condor dashboard (model BHRand). It is the correlated model of
# hpc/BoseHubbardCorrelated_NormalizedITE.jl with Δ = 0, Ω = 1/2, r = 0: the hoppings t̄(1 ± Ω) = {1.5 t̄, 0.5 t̄} = {t, t/3}
# with the mean hopping t̄ = 2t/3, which names the t<t̄> folders. The jobs and the observables are those of that model
# (hpc/condor/sweep_BHCorr.sh, hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl); this script reads the per-point
# files t<t̄>/mu<μ>_obs.csv and plots, in one figure against μ/U,
#   ρ   the disorder-averaged density                         (left axis)
#   K   the effective Luttinger parameter (block fluctuations) (left axis)
#   κ   the compressibility dρ/dμ, central differences between neighbouring μ (one-sided at the ends)  (right axis)
# Unconverged points have open markers. Several t̄ columns (if present) get one line style each. No state is loaded.
#
# Usage (from the repo root or the dashboard's data root, with the same run parameters as the sweep):
#   U=1.0 R=0.0 DELTA=0.0 OMEGA=0.5 NMAX=4 DTAU=0.05 MAXD=16 DATA_DIR=data \
#       julia --project=examples hpc/plot_BoseHubbardRandomHopping_NormalizedITE.jl <L>
# Output: the table on stdout, <DATA_DIR>/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/observables_mu.csv and
#   plots/BoseHubbardCorrelated_NormalizedITE_<run tag>_L<L>_mu.png

length(ARGS) == 1 || error("usage: julia --project=examples hpc/plot_BoseHubbardRandomHopping_NormalizedITE.jl <L>")
L = parse(Int, ARGS[1])

# Same run tag as hpc/BoseHubbardCorrelated_NormalizedITE.jl
param(k) = haskey(ENV, k) ? ENV[k] : error("environment variable $k not set (U, R, DELTA, OMEGA, NMAX, DTAU, MAXD, DATA_DIR as for the sweep)")
U_str, r_str, Δ_str, Ω_str = param("U"), param("R"), param("DELTA"), param("OMEGA")
nmax = parse(Int, param("NMAX"))
dtau_str = param("DTAU")
maxD = parse(Int, param("MAXD"))
run_tag = "U$(U_str)_r$(r_str)_Delta$(Δ_str)_Omega$(Ω_str)_nmax$(nmax)_dtau$(dtau_str)_D$(maxD)"
folder = joinpath(param("DATA_DIR"), "BoseHubbardCorrelated_NormalizedITE", run_tag, "L$(L)")
U, r, Δ, Ω = parse.(Float64, (U_str, r_str, Δ_str, Ω_str))
(r, Δ, Ω) == (0.0, 0.0, 0.5) ||
    @warn "R = $r_str, DELTA = $Δ_str, OMEGA = $Ω_str: not the random-hopping preset (0, 0, 0.5); the hoppings are t̄(1 ± $Ω_str) and the potentials μ ± $Δ_str"

using CairoMakie, Printf

# Header and first row of a CSV as column => value (strings)
function read_row(file)
    lines = filter(!isempty, readlines(file))
    length(lines) >= 2 || error("no data row")
    header, row = split(lines[1], ','), split(lines[2], ',')
    length(header) == length(row) || error("header and row differ in length")
    return Dict(String(h) => String(v) for (h, v) in zip(header, row))
end

isdir(folder) || error("no results in $folder")
tdirs = sort(filter(s -> occursin(r"^t-?[\d.]+$", s) && isdir(joinpath(folder, s)), readdir(folder));
             by = s -> parse(Float64, s[2:end]))

# One curve set per t̄: the points with observables, sorted by μ; unreadable files are skipped with a warning
columns = map(tdirs) do td
    files = filter(f -> occursin(r"^mu-?[\d.]+_obs\.csv$", f), readdir(joinpath(folder, td)))
    pts = filter(!isnothing, map(files) do f
        file = joinpath(folder, td, f)
        try
            x = read_row(file)
            num(k) = haskey(x, k) ? parse(Float64, x[k]) : NaN
            (; μ = num("mu_over_U"), ρ = num("rho"), K = num("K"), nsteps = parse(Int, x["nsteps"]), ϵconv = num("eps_conv"),
             converged = parse(Bool, x["converged"]))
        catch e
            @warn "skipping $file: $(sprint(showerror, e))"
            nothing
        end
    end)
    sort!(pts; by = p -> p.μ)
    μs = [p.μ for p in pts]
    ρs = [p.ρ for p in pts]
    # κ = dρ/dμ (μ in units of U): central differences, one-sided at the ends, NaN for a single point
    κs = map(eachindex(μs)) do k
        lo, hi = max(k - 1, 1), min(k + 1, length(μs))
        hi == lo ? NaN : (ρs[hi] - ρs[lo]) / (U * (μs[hi] - μs[lo]))
    end
    (; t̄ = parse(Float64, td[2:end]), μs, ρs, κs, Ks = [p.K for p in pts], pts)
end
filter!(c -> !isempty(c.μs), columns)
isempty(columns) && error("no observables (*_obs.csv) in $folder: gather the data first")

println("Run tag $run_tag, L = $L ($folder)")
println("     t̄    t=1.5t̄     μ/U        ρ         κ         K   steps   ϵ_conv  converged")
for c in columns, (k, p) in enumerate(c.pts)
    @printf("%7.4f  %7.4f  %7.3f  %7.4f  %8.4f  %8.4f  %6d  %7.1e   %s\n", c.t̄, 1.5c.t̄, c.μs[k], c.ρs[k], c.κs[k], c.Ks[k],
            p.nsteps, p.ϵconv, p.converged)
end
nunconv = sum(c -> count(p -> !p.converged, c.pts), columns)
nunconv > 0 && println("$nunconv point(s) not converged (open markers in the plot)")

csv = joinpath(folder, "observables_mu.csv")
open(csv, "w") do io
    println(io, "tbar_over_U,mu_over_U,rho,kappa,K,nsteps,eps_conv,converged")
    for c in columns, (k, p) in enumerate(c.pts)
        println(io, "$(c.t̄),$(c.μs[k]),$(c.ρs[k]),$(c.κs[k]),$(c.Ks[k]),$(p.nsteps),$(p.ϵconv),$(p.converged)")
    end
end
println("Saved $csv")

# Figure: ρ and K on the left axis, κ on the right axis, sharing μ/U; one line style per t̄
set_theme!(theme_latexfonts())
cρ, cκ, cK = Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3]
styles = [:solid, :dash, :dot, :dashdot]
hops = join((@sprintf("%.4g", 1.5c.t̄ / U) for c in columns), ", ")
fig = Figure(backgroundcolor = :white, fontsize = 26, size = (1300, 800))
title = L"$L=%$L$, hopping $t$ or $t/3$ ($p = 1/2$), $t/U = %$hops$, $D=%$maxD$"
axL = Axis(fig[1, 1]; xlabel = L"μ/U", ylabel = L"ρ,\ K", title)
axκ = Axis(fig[1, 1]; ylabel = L"κ = dρ/dμ", yaxisposition = :right, ylabelcolor = cκ, yticklabelcolor = cκ,
           ygridvisible = false)
hidespines!(axκ)
hidexdecorations!(axκ)
linkxaxes!(axL, axκ)
hlines!(axL, [1.0]; color = :gray, linestyle = :dash)
for (j, c) in enumerate(columns)
    ls = styles[mod1(j, length(styles))]
    fill(col) = [p.converged ? col : :white for p in c.pts]
    lines!(axL, c.μs, c.ρs; color = cρ, linewidth = 3, linestyle = ls)
    lines!(axL, c.μs, c.Ks; color = cK, linewidth = 3, linestyle = ls)
    lines!(axκ, c.μs, c.κs; color = cκ, linewidth = 3, linestyle = ls)
    scatter!(axL, c.μs, c.ρs; color = fill(cρ), strokecolor = cρ, strokewidth = 2, markersize = 14)
    scatter!(axL, c.μs, c.Ks; color = fill(cK), strokecolor = cK, strokewidth = 2, marker = :diamond, markersize = 16)
    scatter!(axκ, c.μs, c.κs; color = fill(cκ), strokecolor = cκ, strokewidth = 2, marker = :rect, markersize = 14)
end
# K can come out negative on short or unconverged chains: the left axis then extends below 0 so those points show
lowest = minimum(x -> isfinite(x) ? x : 0.0, (v for c in columns for v in vcat(c.ρs, c.Ks)); init = 0.0)
ylims!(axL, lowest < 0 ? 1.05lowest : 0, nothing)
ylims!(axκ, min(0, minimum(x -> isfinite(x) ? x : 0.0, (v for c in columns for v in c.κs); init = 0.0)), nothing)
entry(col, m, ms) = [LineElement(color = col, linewidth = 3), MarkerElement(color = col, marker = m, markersize = ms)]
elements, labels = [entry(cρ, :circle, 14), entry(cK, :diamond, 16), entry(cκ, :rect, 14)], Any[L"ρ", L"K", L"κ"]
if length(columns) > 1
    for (j, c) in enumerate(columns)
        push!(elements, [LineElement(color = :black, linewidth = 3, linestyle = styles[mod1(j, length(styles))])])
        tU = @sprintf("%.4g", 1.5c.t̄ / U)
        push!(labels, L"t/U = %$tU")
    end
end
axislegend(axL, elements, labels; position = :lt)

mkpath("plots")
file = joinpath("plots", "BoseHubbardCorrelated_NormalizedITE_$(run_tag)_L$(L)_mu.png")
save(file, fig)
println("Saved $file")
