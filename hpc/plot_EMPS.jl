# Load all EMPS phase-line results from a data folder and plot them.
# Usage (from repo root): julia --project=. hpc/plot_EMPS.jl [data_folder] [iδ0]
using JLD2, CairoMakie

data_root = get(ENV, "DATA_DIR", joinpath(get(ENV, "VSC_DATA", "."), "data"))
folder = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(data_root, "EMPS_phase_line", "dtau0.05_maxiter1000")
iδ0 = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 3 # δ index used for ξ vs W

files = filter(f -> occursin(r"^W.*_Dmax\d+\.jld2$", f), readdir(folder))
isempty(files) && error("No result files found in $folder")
results = [load(joinpath(folder, f)) for f in files]
sort!(results, by = r -> (r["Dmax"], r["W"]))

Ws = sort(unique(r["W"] for r in results))
Dmaxs = sort(unique(r["Dmax"] for r in results))
colors = Makie.wong_colors()
markers = [:circle, :rect, :utriangle, :diamond, :star5, :cross]
linestyles = [:solid, :dash, :dot, :dashdot]
wcolor(W) = colors[mod1(findfirst(==(W), Ws), length(colors))]
dmarker(D) = markers[mod1(findfirst(==(D), Dmaxs), length(markers))]
dstyle(D) = linestyles[mod1(findfirst(==(D), Dmaxs), length(linestyles))]

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax1 = Axis(fig[1, 1], xlabel = L"$δ$", ylabel = L"$E₀$", title = L"$E$")
ax2 = Axis(fig[1, 2], xlabel = L"$δ$", ylabel = L"$M$", title = L"$M$")
ax3 = Axis(fig[2, 1], xlabel = L"$δ$", ylabel = L"$ξ$", title = L"$ξ$")
ax4 = Axis(fig[2, 2], xlabel = L"$δ$", ylabel = L"$ϵ$", title = L"$ϵ$", yscale = log10)

for r in results
    δs, W, D = r["δs"], r["W"], r["Dmax"]
    kw = (color = wcolor(W), marker = dmarker(D), linestyle = dstyle(D), markersize = 20, linewidth = 3,
          label = "W = $W, D = $D")
    scatterlines!(ax1, δs, r["Es"]; kw...)
    scatterlines!(ax2, δs, abs.(r["Ms"]); kw...)
    scatterlines!(ax3, δs, r["ξs"]; kw...)
    scatterlines!(ax4, δs, abs.(r["ϵs"] .+ 1e-16); kw...)
end
Legend(fig[1:2, 3], ax1)

# ξ vs W at fixed δ, one series per Dmax
fig2 = Figure(backgroundcolor=:white, fontsize=40, size=(2000, 1500))
δ0 = results[1]["δs"][iδ0]
ax21 = Axis(fig2[1, 1], xlabel = L"$W$", ylabel = L"$ξ$", title = "δ = $δ0")
for D in Dmaxs
    rs = filter(r -> r["Dmax"] == D, results)
    scatterlines!(ax21, [r["W"] for r in rs], [r["ξs"][iδ0] for r in rs];
                  marker = dmarker(D), linestyle = dstyle(D), markersize = 20, linewidth = 3, label = "D = $D")
end
axislegend(ax21, position = :lt)

mkpath("plots")
tag = basename(normpath(folder))
save("plots/EMPS_phase_line_$(tag).png", fig)
save("plots/EMPS_phase_line_$(tag)_xi_vs_W.png", fig2)
println("Saved plots/EMPS_phase_line_$(tag).png and plots/EMPS_phase_line_$(tag)_xi_vs_W.png")
