# Load all EMPS phase-line results from a data folder and plot them.
# Usage (from repo root): julia --project=. hpc/plot_EMPS.jl [data_folder]
# PLOT_MODE selects how the δ-scan figure is split up:
#   all  (default) one figure with every (W, Dmax) curve
#   fixW           one figure per W, one curve per Dmax
#   fixD           one figure per Dmax, one curve per W
using JLD2, CairoMakie

data_root = get(ENV, "DATA_DIR", joinpath(get(ENV, "VSC_DATA", "."), "data"))
folder = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(data_root, "EMPS_phase_line", "dtau0.05_maxiter1000")
mode = get(ENV, "PLOT_MODE", "all")
mode in ("all", "fixW", "fixD") || error("PLOT_MODE must be all, fixW or fixD, got $mode")

files = filter(f -> occursin(r"^W.*_Dmax\d+\.jld2$", f), readdir(folder))
isempty(files) && error("No result files found in $folder")
# Skip the stored states ψs, so DisorderKit is not needed to load the results
load_results(file) = jldopen(file) do f
    haskey(f, "μs") || error("$file has no μs: it was made before the δ = μ/W² convention")
    Dict(k => f[k] for k in keys(f) if k != "ψs")
end
results = [load_results(joinpath(folder, f)) for f in files]
sort!(results, by = r -> (r["Dmax"], r["W"]))

Ws = sort(unique(r["W"] for r in results))
Dmaxs = sort(unique(r["Dmax"] for r in results))
colors = Makie.wong_colors()
markers = [:circle, :rect, :utriangle, :diamond, :star5, :cross]
linestyles = [:solid, :dash, :dot, :dashdot]
wcolor(W) = colors[mod1(findfirst(==(W), Ws), length(colors))]
dmarker(D) = markers[mod1(findfirst(==(D), Dmaxs), length(markers))]
dstyle(D) = linestyles[mod1(findfirst(==(D), Dmaxs), length(linestyles))]

dcolor(D) = colors[mod1(findfirst(==(D), Dmaxs), length(colors))]

set_theme!(theme_latexfonts())

# δ scan of E, M, ξ and S for the given results; style(r) returns (color, marker, linestyle, label)
function phase_line_figure(rs, style; title = "")
    fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
    ax1 = Axis(fig[1, 1], xlabel = L"$δ$", ylabel = L"$E₀$", title = L"$E$")
    ax2 = Axis(fig[1, 2], xlabel = L"$δ$", ylabel = L"$M$", title = L"$M$")
    ax3 = Axis(fig[2, 1], xlabel = L"$δ$", ylabel = L"$ξ$", title = L"$ξ$")
    ax4 = Axis(fig[2, 2], xlabel = L"$δ$", ylabel = L"$S$", title = L"$S$")
    isempty(title) || Label(fig[0, 1:2], title, fontsize = 50)

    for r in rs
        color, marker, linestyle, label = style(r)
        kw = (; color, marker, linestyle, label, markersize = 20, linewidth = 3)
        δs = r["δs"]
        scatterlines!(ax1, δs, r["Es"]; kw...)
        scatterlines!(ax2, δs, abs.(r["Ms"]); kw...)
        scatterlines!(ax3, δs, r["ξs"]; kw...)
        scatterlines!(ax4, δs, r["Ss"]; kw...)
    end
    ylims!(ax3, nothing, 50)
    Legend(fig[1:2, 3], ax1)
    return fig
end

# name => figure for the chosen PLOT_MODE
phase_figs = if mode == "all"
    ["phase_line" => phase_line_figure(results,
        r -> (wcolor(r["W"]), dmarker(r["Dmax"]), dstyle(r["Dmax"]), "W = $(r["W"]), D = $(r["Dmax"])"))]
elseif mode == "fixW"
    ["phase_line_W$(W)" => phase_line_figure(filter(r -> r["W"] == W, results),
        r -> (dcolor(r["Dmax"]), dmarker(r["Dmax"]), dstyle(r["Dmax"]), "D = $(r["Dmax"])"); title = "W = $W")
     for W in Ws]
else
    ["phase_line_Dmax$(D)" => phase_line_figure(filter(r -> r["Dmax"] == D, results),
        r -> (wcolor(r["W"]), :circle, :solid, "W = $(r["W"])"); title = "D = $D")
     for D in Dmaxs]
end

# S vs ξ over all W and δ, one series per Dmax, with a fit S = a ln ξ + b
fig2 = Figure(backgroundcolor=:white, fontsize=40, size=(2000, 1500))
ax21 = Axis(fig2[1, 1], xlabel = L"$ξ$", ylabel = L"$S$", xscale = log10)
for D in Dmaxs
    rs = filter(r -> r["Dmax"] == D, results)
    ξ = reduce(vcat, [r["ξs"] for r in rs])
    S = reduce(vcat, [r["Ss"] for r in rs])
    ok = isfinite.(ξ) .& isfinite.(S) .& (ξ .> 0)
    ξ, S = ξ[ok], S[ok]
    isempty(ξ) && continue
    scatter!(ax21, ξ, S; color = dcolor(D), marker = dmarker(D), markersize = 20, label = "D = $D")
    length(unique(ξ)) ≥ 2 || continue
    a, b = [log.(ξ) ones(length(ξ))] \ S
    ξfit = exp.(range(log(minimum(ξ)), log(maximum(ξ)), length = 100))
    lines!(ax21, ξfit, a .* log.(ξfit) .+ b; color = dcolor(D), linestyle = dstyle(D), linewidth = 3,
           label = "D = $D: S = $(round(a, sigdigits = 3)) ln ξ + $(round(b, sigdigits = 3))")
end
axislegend(ax21, position = :lt)

# Organized storage, mirroring the data layout: <root>/EMPS_phase_line/<data folder name>/*.png
plot_root = get(ENV, "PLOT_DIR", joinpath(get(ENV, "VSC_DATA", "."), "plots"))
plot_folder = joinpath(plot_root, "EMPS_phase_line", basename(normpath(folder)))
mkpath(plot_folder)
for (name, f) in phase_figs
    file = joinpath(plot_folder, "$name.png")
    save(file, f)
    println("Saved $file")
end
file2 = joinpath(plot_folder, "S_vs_xi.png")
save(file2, fig2)
println("Saved $file2")
