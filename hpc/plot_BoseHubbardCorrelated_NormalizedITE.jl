# Phase diagram of a hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh run from the observables computed by
# hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl (submitted with hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh):
# it reads the per-point files t<t>/mu<μ>_obs.csv, computes the compressibility κ = dρ/dμ (along μ at fixed t) and plots,
# in one figure over (t/U, μ/U), κ, the coefficient of variation of the bond-averaged entropy
# CV = S_bstd/S_avg = 𝔼_w[std_k S_w(k)] / 𝔼_w[mean_k S_w(k)], the quasicondensate density |⟨a⟩|², the density ρ,
# the bond-averaged entropy std S_bstd = 𝔼_w[std_k S_w(k)] and the MI/SF/BG phase classification.
# t and μ are the mean values, the disorder values are t(1±Ω) and μ±Δ. No state is loaded, so the script is fast and
# can run on the login node.
#
# Usage (from the repo root, with the same run parameters as the sweep, so the run tag matches):
#   U=1.0 R=0.0 DELTA=0.2 OMEGA=0.5 NMAX=4 DTAU=0.05 MAXD=16 KAPPA_TOL=0.05 SBSTD_TOL=0.05 A2_TOL=0.05 DATA_DIR=$VSC_DATA/data julia --project=. hpc/plot_BoseHubbardCorrelated_NormalizedITE.jl <L>
#
# Environment (all required): U, R, DELTA, OMEGA, NMAX, DTAU, MAXD, DATA_DIR, and the phase classification thresholds
#   KAPPA_TOL, SBSTD_TOL, A2_TOL: with κ low ⇔ κ ≤ KAPPA_TOL, S_bstd (bulk bonds, see N_EDGE in the observables script)
#   low ⇔ S_bstd < SBSTD_TOL and |⟨a⟩|² low ⇔ |⟨a⟩|² ≤ A2_TOL, a point is a
#     Mott insulator (MI): κ low,  S_bstd low,  |⟨a⟩|² low
#     superfluid (SF):     κ high, S_bstd low,  |⟨a⟩|² high
#     Bose glass (BG):     κ high, S_bstd high, |⟨a⟩|² low
#   and unclassified otherwise (also when S_bstd is NaN, observables of an older format).
# Output: the table on stdout, <DATA_DIR>/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/observables.csv (all points)
#   and the figure plots/BoseHubbardCorrelated_NormalizedITE_<run tag>_L<L>.png. The grid is set by the t<t>/mu<μ>.jld2 and
#   mu<μ>_obs.csv names; grid points without (readable) observables are set to 0 and marked with a cross.

length(ARGS) == 1 || error("usage: julia --project=. hpc/plot_BoseHubbardCorrelated_NormalizedITE.jl <L>")
L = parse(Int, ARGS[1])

# Same run tag as BoseHubbardCorrelated_NormalizedITE.jl
param(k) = haskey(ENV, k) ? ENV[k] : error("environment variable $k not set (U, R, DELTA, OMEGA, NMAX, DTAU, MAXD, DATA_DIR as for the sweep)")
U_str = param("U")
r_str = param("R")
Δ_str = param("DELTA")
Ω_str = param("OMEGA")
nmax = parse(Int, param("NMAX"))
dtau_str = param("DTAU")
maxD = parse(Int, param("MAXD"))
run_tag = "U$(U_str)_r$(r_str)_Delta$(Δ_str)_Omega$(Ω_str)_nmax$(nmax)_dtau$(dtau_str)_D$(maxD)"
data_root = param("DATA_DIR")
κtol = parse(Float64, param("KAPPA_TOL"))
sbstdtol = parse(Float64, param("SBSTD_TOL"))
a2tol = parse(Float64, param("A2_TOL"))
folder = joinpath(data_root, "BoseHubbardCorrelated_NormalizedITE", run_tag, "L$(L)")

using CairoMakie, Printf

# Header and first row of a CSV as column => value (strings)
function read_row(file)
    lines = filter(!isempty, readlines(file))
    length(lines) >= 2 || error("no data row")
    header, row = split(lines[1], ','), split(lines[2], ',')
    length(header) == length(row) || error("header and row differ in length")
    return Dict(String(h) => String(v) for (h, v) in zip(header, row))
end

# (t/U, μ/U) grid from the folder and file names, so that points without observables (job failed, still running or
# never ran) are part of it too
U, r = parse(Float64, U_str), parse(Float64, r_str)
isdir(folder) || error("no results in $folder")
tdirs = filter(s -> occursin(r"^t-?[\d.]+$", s) && isdir(joinpath(folder, s)), readdir(folder))
tvalue(td) = parse(Float64, td[2:end]) / U
μvalue(f) = parse(Float64, match(r"^mu(-?[\d.]+?)(_obs\.csv|\.jld2)$", f)[1]) / U
pointfiles = [(td, f) for td in tdirs for f in readdir(joinpath(folder, td)) if occursin(r"^mu-?[\d.]+?(_obs\.csv|\.jld2)$", f)]
isempty(pointfiles) && error("no results in $folder")
ts = sort(unique(tvalue(td) for (td, _) in pointfiles))
μs = sort(unique(μvalue(f) for (_, f) in pointfiles))

# Observables of every point; a file that cannot be read is skipped with a warning
obsfiles = [joinpath(folder, td, f) for (td, f) in pointfiles if endswith(f, "_obs.csv")]
results = map(obsfiles) do file
    try
        x = read_row(file)
        num(k) = parse(Float64, x[k])
        (; t = num("t_over_U"), μ = num("mu_over_U"), ρ = num("rho"), φ = num("a2"), varS = num("varS_avg"),
         Savg = num("S_avg"), Sinhom = num("S_inhom"),
         Sbstd = haskey(x, "S_bstd") ? num("S_bstd") : NaN,   # NaN for observables of an older format
         n_samples = parse(Int, x["n_samples"]), exact = parse(Bool, x["exact"]), nsteps = parse(Int, x["nsteps"]),
         ϵconv = num("eps_conv"), converged = parse(Bool, x["converged"]))
    catch e
        @warn "skipping $file: $(sprint(showerror, e))"
        nothing
    end
end
results = filter(!isnothing, results)

# Observables on the grid, 0 where there is no result (marked with a cross in the plot)
grid() = zeros(length(ts), length(μs))
ρs, φs, varSs, Savgs, Sinhoms, Sbstds, κs = grid(), grid(), grid(), grid(), grid(), grid(), grid()
converged = fill(true, length(ts), length(μs))
hasdata = fill(false, length(ts), length(μs))
index(x) = (argmin(abs.(ts .- x.t)), argmin(abs.(μs .- x.μ)))
for x in results
    i, j = index(x)
    ρs[i, j], φs[i, j], varSs[i, j], converged[i, j], hasdata[i, j] = x.ρ, x.φ, x.varS, x.converged, true
    Savgs[i, j], Sinhoms[i, j], Sbstds[i, j] = x.Savg, x.Sinhom, x.Sbstd
end
# Relative bond-to-bond fluctuation of the entropy, NaN (white) where S_avg = 0 and 0 where there is no result
ratios = [hasdata[i, j] ? (iszero(Savgs[i, j]) ? NaN : Sbstds[i, j] / Savgs[i, j]) : 0.0 for i in eachindex(ts), j in eachindex(μs)]

# Compressibility κ = dρ/dμ along μ at fixed t: central differences over the nearest neighbours with a result
# (one-sided at the ends and next to missing points), 0 where there is no result or no neighbour
for i in eachindex(ts), j in eachindex(μs)
    hasdata[i, j] || continue
    lo = j > 1 && hasdata[i, j-1] ? j - 1 : j
    hi = j < length(μs) && hasdata[i, j+1] ? j + 1 : j
    κs[i, j] = hi == lo ? 0.0 : (ρs[i, hi] - ρs[i, lo]) / (U * (μs[hi] - μs[lo]))
end

# Phase classification from κ, S_bstd and |⟨a⟩|²: 1 = MI, 2 = SF, 3 = BG, 4 = unclassified, 0 = no result
const phasenames = ("MI", "SF", "BG", "unclassified")
function classify(κ, sbstd, a2)
    isnan(sbstd) && return 4   # observables of an older format
    compressible, largestd, condensed = κ > κtol, sbstd >= sbstdtol, a2 > a2tol
    !compressible && !largestd && !condensed && return 1
    compressible && !largestd && condensed && return 2
    compressible && largestd && !condensed && return 3
    return 4
end
phases = [hasdata[i, j] ? classify(κs[i, j], Sbstds[i, j], φs[i, j]) : 0 for i in eachindex(ts), j in eachindex(μs)]

println("Run tag $run_tag, L = $L ($folder)")
println("    t/U      μ/U        ρ        κ   |⟨a⟩|²  ⟨Var S⟩      ⟨S⟩  Var_k S   std_k S   ratio  samples   steps   ϵ_conv   converged  phase")
sort!(results; by = x -> (x.t, x.μ))
for x in results
    i, j = index(x)
    @printf("%7.4f  %7.4f  %7.4f  %7.3f  %7.4f  %7.4f  %7.4f  %7.4f  %8.4f  %6.3f  %7s  %6d  %7.1e   %-9s  %s\n", ts[i], μs[j], ρs[i, j],
            κs[i, j], φs[i, j], varSs[i, j], Savgs[i, j], Sinhoms[i, j], Sbstds[i, j], ratios[i, j], x.exact ? "exact" : string(x.n_samples), x.nsteps, x.ϵconv, x.converged,
            phasenames[phases[i, j]])
end
println("Phases with κ_tol = $κtol, S_bstd_tol = $sbstdtol, |⟨a⟩|²_tol = $a2tol: ", join(("$(n) $(count(==(k), phases))" for (k, n) in enumerate(phasenames)), ", "))
nconv = count(!, (x.converged for x in results))
nconv > 0 && println("$nconv of $(length(results)) points not converged (open circles in the plot)")
nsamp = unique(x.n_samples for x in results if !x.exact)
length(nsamp) > 1 && println("sampled points use different numbers of words: $(sort(nsamp))")
missingpts = [(ts[i], μs[j]) for i in eachindex(ts), j in eachindex(μs) if !hasdata[i, j]]
if !isempty(missingpts)
    println("$(length(missingpts)) grid points without observables (0, marked × in the plot), (t/U, μ/U):")
    foreach(p -> @printf("  %7.4f  %7.4f\n", p...), missingpts)
end

csv = joinpath(folder, "observables.csv")
open(csv, "w") do io
    println(io, "t_over_U,mu_over_U,rho,kappa,a2,varS_avg,S_avg,S_inhom,S_bstd,S_bstd_over_S_avg,n_samples,exact,nsteps,eps_conv,converged,phase")
    for x in results
        i, j = index(x)
        println(io, "$(ts[i]),$(μs[j]),$(ρs[i, j]),$(κs[i, j]),$(φs[i, j]),$(varSs[i, j]),$(Savgs[i, j]),$(Sinhoms[i, j]),",
                "$(Sbstds[i, j]),$(ratios[i, j]),$(x.n_samples),$(x.exact),",
                "$(x.nsteps),$(x.ϵconv),$(x.converged),$(phasenames[phases[i, j]])")
    end
end
println("Saved $csv")

# Figure: 2×3 panels over (t/U, μ/U): κ, CV = S_bstd/S_avg, |⟨a⟩|² (top), ρ, S_bstd and the phase classification
# (bottom); unconverged points are marked with open circles, points without a result with crosses
set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor = :white, fontsize = 30, size = (3200, 1700))
Label(fig[0, 1:6], L"$L=%$L$, $r=%$r$, $t(1\pm%$Ω_str)$, $μ\pm%$Δ_str$, $D=%$maxD$, $κ_\mathrm{tol}=%$κtol$, $S_{\mathrm{bstd},\mathrm{tol}}=%$sbstdtol$, $|\langle a\rangle|^2_\mathrm{tol}=%$a2tol$";
      fontsize = 36)
unconv = [(ts[i], μs[j]) for i in eachindex(ts), j in eachindex(μs) if hasdata[i, j] && !converged[i, j]]
function markpoints!(ax, circlecolor)
    isempty(unconv) || scatter!(ax, unconv; color = :transparent, strokecolor = circlecolor, strokewidth = 2, markersize = 12)
    isempty(missingpts) || scatter!(ax, missingpts; marker = :xcross, color = :white, strokecolor = :black,
                                    strokewidth = 1.5, markersize = 22)
end
panels = [(κs, L"κ = dρ/dμ", :viridis, (1, 1)),
          (ratios, L"\mathrm{CV} = \mathbb{E}_w[\mathrm{std}_{k\in\mathrm{bulk}} S_w(k)] / \overline{S}_\mathrm{bulk}", :inferno, (1, 2)),
          (φs, L"|\langle a\rangle|^2", :plasma, (1, 3)),
          (ρs, L"ρ", :cividis, (2, 1)),
          (Sbstds, L"\mathbb{E}_w[\mathrm{std}_{k\in\mathrm{bulk}} S_w(k)]", :magma, (2, 2))]
for (M, label, cmap, (row, col)) in panels
    ax = Axis(fig[row, 2col-1]; xlabel = L"t/U", ylabel = L"μ/U", title = label)
    finite = filter(isfinite, M[hasdata])
    colorrange = isempty(finite) || minimum(finite) == maximum(finite) ? (0, 1) : extrema(finite)
    hm = heatmap!(ax, ts, μs, M; colormap = cmap, colorrange, nan_color = :white)
    markpoints!(ax, :white)
    Colorbar(fig[row, 2col], hm)
end

# Phase classification: MI / SF / BG / unclassified per point, white where there is no result
phasecolors = [:white, :royalblue, :orange, :forestgreen, :gray70]   # no result, MI, SF, BG, unclassified
axp = Axis(fig[2, 5]; xlabel = L"t/U", ylabel = L"μ/U", title = L"\mathrm{phase}")
heatmap!(axp, ts, μs, phases; colormap = cgrad(phasecolors, 5; categorical = true), colorrange = (-0.5, 4.5))
markpoints!(axp, :black)
Legend(fig[2, 6], [PolyElement(color = c, strokecolor = :black, strokewidth = 1) for c in phasecolors[2:end]],
       ["MI", "SF", "BG", "uncl."])

mkpath("plots")
file = joinpath("plots", "BoseHubbardCorrelated_NormalizedITE_$(run_tag)_L$(L).png")
save(file, fig)
println("Saved $file")
