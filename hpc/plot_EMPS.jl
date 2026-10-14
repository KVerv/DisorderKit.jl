# Load all EMPS phase-line results from a data folder and plot them, one curve per Dmax.
# Usage (from repo root): julia --project=. hpc/plot_EMPS.jl [data_folder]
# MFIT_SIDE selects which side of δ = 0 enters the log|M| vs log|δ| fits:
#   neg  (default) δ < 0, the ordered side
#   pos            δ > 0
#   both           every δ ≠ 0
using JLD2, CairoMakie

data_root = get(ENV, "DATA_DIR", joinpath(get(ENV, "VSC_DATA", "."), "data"))
folder = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(data_root, "RTFIM_phase_line", "dtau0.05_maxiter1000")
mfit_side = get(ENV, "MFIT_SIDE", "neg")
mfit_side in ("neg", "pos", "both") || error("MFIT_SIDE must be neg, pos or both, got $mfit_side")

files = filter(f -> occursin(r"^Dmax\d+\.jld2$", f), readdir(folder))
isempty(files) && error("No result files found in $folder")
# Skip the stored states ψs, so DisorderKit is not needed to load the results
load_results(file) = jldopen(file) do f
    haskey(f, "h0s") || error("$file has no h0s: it was made before the h0 scan with δ = ln h0/2")
    Dict(k => f[k] for k in keys(f) if k != "ψs")
end
results = [load_results(joinpath(folder, f)) for f in files]
sort!(results, by = r -> r["Dmax"])

Dmaxs = sort(unique(r["Dmax"] for r in results))
colors = Makie.wong_colors()
markers = [:circle, :rect, :utriangle, :diamond, :star5, :cross]
linestyles = [:solid, :dash, :dot, :dashdot]
dcolor(D) = colors[mod1(findfirst(==(D), Dmaxs), length(colors))]
dmarker(D) = markers[mod1(findfirst(==(D), Dmaxs), length(markers))]
dstyle(D) = linestyles[mod1(findfirst(==(D), Dmaxs), length(linestyles))]
style(r) = (dcolor(r["Dmax"]), dmarker(r["Dmax"]), dstyle(r["Dmax"]), "D = $(r["Dmax"])")

set_theme!(theme_latexfonts())

# Scan of E, M, ξ and S against r[x] (δ or h0) for the given results; style(r) returns (color, marker, linestyle, label)
function phase_line_figure(rs, style; x = "δs", xlabel = L"$δ$")
    fig = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
    ax1 = Axis(fig[1, 1]; xlabel, ylabel = L"$E₀$", title = L"$E$")
    ax2 = Axis(fig[1, 2]; xlabel, ylabel = L"$M$", title = L"$M$")
    ax3 = Axis(fig[2, 1]; xlabel, ylabel = L"$ξ$", title = L"$ξ$")
    ax4 = Axis(fig[2, 2]; xlabel, ylabel = L"$S$", title = L"$S$")

    for r in rs
        color, marker, linestyle, label = style(r)
        kw = (; color, marker, linestyle, label, markersize = 20, linewidth = 3)
        xs = r[x]
        scatterlines!(ax1, xs, r["Es"]; kw...)
        scatterlines!(ax2, xs, abs.(r["Ms"]); kw...)
        scatterlines!(ax3, xs, r["ξs"]; kw...)
        scatterlines!(ax4, xs, r["Ss"]; kw...)
    end
    Legend(fig[1:2, 3], ax1)
    return fig
end

# Finite-size-scaling collapse of |M| with ξ as the length scale L:
# |M| = ξ^{-β/ν} f(δ ξ^{1/ν}), with the RTFIM infinite-randomness exponents ν = 2, β = (3 - √5)/2 by default
ν = parse(Float64, get(ENV, "NU", "2"))
β = parse(Float64, get(ENV, "BETA", string((3 - √5) / 2)))
function collapse_figure(rs, style)
    fig = Figure(backgroundcolor=:white, fontsize=40, size=(2500, 1500))
    ax = Axis(fig[1, 1], xlabel = L"$δ\,ξ^{1/ν}$", ylabel = L"$|M|\,ξ^{β/ν}$",
              title = L"$ν = %$(round(ν, sigdigits = 3))$, $β = %$(round(β, sigdigits = 3))$")
    for r in rs
        color, marker, _, label = style(r)
        δ, ξ, M = r["δs"], r["ξs"], abs.(r["Ms"])
        ok = isfinite.(ξ) .& isfinite.(M) .& (ξ .> 0) .& (M .> 0)
        any(ok) || continue
        scatter!(ax, δ[ok] .* ξ[ok] .^ (1 / ν), M[ok] .* ξ[ok] .^ (β / ν); color, marker, label, markersize = 20)
    end
    Legend(fig[1, 2], ax)
    return fig
end

# log|M| vs log|δ| on the MFIT_SIDE side of δ = 0, with a fit log|M| = a log|δ| + b per series (a ≈ β)
function M_scaling_figure(rs, style)
    fig = Figure(backgroundcolor=:white, fontsize=40, size=(2500, 1500))
    ax = Axis(fig[1, 1], xlabel = L"$|δ|$", ylabel = L"$|M|$", xscale = log10, yscale = log10,
              title = "δ side: $mfit_side")
    for r in rs
        color, marker, linestyle, label = style(r)
        δ, M = r["δs"], abs.(r["Ms"])
        side = mfit_side == "neg" ? δ .< 0 : mfit_side == "pos" ? δ .> 0 : δ .!= 0
        ok = side .& isfinite.(δ) .& isfinite.(M) .& (M .> 0)
        any(ok) || continue
        x, y = abs.(δ[ok]), M[ok]
        scatter!(ax, x, y; color, marker, label, markersize = 20)
        length(unique(x)) ≥ 2 || continue
        a, b = [log.(x) ones(length(x))] \ log.(y)
        xfit = exp.(range(log(minimum(x)), log(maximum(x)), length = 100))
        lines!(ax, xfit, exp(b) .* xfit .^ a; color, linestyle, linewidth = 3,
               label = "$label: a = $(round(a, sigdigits = 3))")
    end
    Legend(fig[1, 2], ax)
    return fig
end

figs = ["phase_line" => phase_line_figure(results, style),
        "phase_line_h0" => phase_line_figure(results, style; x = "h0s", xlabel = L"$h_0$"),
        "M_collapse" => collapse_figure(results, style),
        "M_scaling" => M_scaling_figure(results, style)]

# S vs ξ over all δ, one series per Dmax, with a fit S = a ln ξ + b
fig2 = Figure(backgroundcolor=:white, fontsize=40, size=(2000, 1500))
ax21 = Axis(fig2[1, 1], xlabel = L"$ξ$", ylabel = L"$S$", xscale = log10)
for r in results
    D = r["Dmax"]
    ξ, S = r["ξs"], r["Ss"]
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

# Organized storage, mirroring the data layout: <root>/RTFIM_phase_line/<data folder name>/*.png
plot_root = get(ENV, "PLOT_DIR", joinpath(get(ENV, "VSC_DATA", "."), "plots"))
plot_folder = joinpath(plot_root, "RTFIM_phase_line", basename(normpath(folder)))
mkpath(plot_folder)
for (name, f) in figs
    file = joinpath(plot_folder, "$name.png")
    save(file, f)
    println("Saved $file")
end
file2 = joinpath(plot_folder, "S_vs_xi.png")
save(file2, fig2)
println("Saved $file2")
