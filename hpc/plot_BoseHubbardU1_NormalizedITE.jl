# Heatmap PNG of one observable of a hpc/condor/sweep_BHU1.sh sweep (U(1) Bose-Hubbard, fixed filling), over the
# sweep's two axes and restricted to a chosen window. It reads the per-point files <X><x>/<Y><y>_obs.csv written by
# hpc/observables_BoseHubbardU1_NormalizedITE.jl; no state is loaded. Normally run by the dashboard's "Heatmap PNG"
# button, on the laptop's gathered data.
#
# Usage (from the folder that holds data/, normally the repo root):
#   TAG=t-Delta_U1.0_Omega0.0_r0.0_f1_nmax4_dtau0.05_D32 XAXIS=t YAXIS=Delta DATA_DIR=data OBS_COL=S_bar \
#       XMIN=0.05 XMAX=0.2 julia --project=examples hpc/plot_BoseHubbardU1_NormalizedITE.jl <L>
#
# Environment: TAG, XAXIS, YAXIS, DATA_DIR, OBS_COL (a column of the _obs.csv files, e.g. S_bar, S_bstd, E, eps_conv,
#   residual_max) are required. Optional: XMIN, XMAX, YMIN, YMAX (the window, inclusive; unset or empty = unbounded),
#   LOG=1 (logarithmic colour scale, non-positive values count as missing).
# The colour range is the range of the plotted values. Grid points inside the window that have no value (no
# observables file, or the observable not computed) are grey.
# Output: plots/BoseHubbardU1_NormalizedITE_<TAG>_L<L>_<OBS_COL>.png

length(ARGS) == 1 || error("usage: julia --project=examples hpc/plot_BoseHubbardU1_NormalizedITE.jl <L>")
L = parse(Int, ARGS[1])

param(k) = haskey(ENV, k) && !isempty(ENV[k]) ? ENV[k] : error("environment variable $k not set")
tag, xaxis, yaxis, data_root, col = param("TAG"), param("XAXIS"), param("YAXIS"), param("DATA_DIR"), param("OBS_COL")
bound(k, default) = (s = strip(get(ENV, k, "")); isempty(s) ? default : parse(Float64, s))
xmin, xmax = bound("XMIN", -Inf), bound("XMAX", Inf)
ymin, ymax = bound("YMIN", -Inf), bound("YMAX", Inf)
logscale = get(ENV, "LOG", "0") == "1"
xmin <= xmax || error("XMIN = $xmin > XMAX = $xmax")
ymin <= ymax || error("YMIN = $ymin > YMAX = $ymax")

folder = joinpath(data_root, "BoseHubbardU1_NormalizedITE", tag, "L$L")
isdir(folder) || error("no data in $folder (gather first)")

# Value of the column `col` in an observables CSV (header and one row), NaN if missing or not a number
function read_value(file)
    lines = filter(!isempty, readlines(file))
    length(lines) >= 2 || return NaN
    header, row = split(lines[1], ','), split(lines[2], ',')
    k = findfirst(==(col), header)
    (isnothing(k) || k > length(row)) && return NaN
    v = tryparse(Float64, row[k])
    return isnothing(v) ? NaN : v
end

# Every grid point (state or observables file) with its value
value_re(name) = Regex("^" * name * "(-?[\\d.]+)\$")
file_re(name) = Regex("^" * name * "(-?[\\d.]+)(_obs\\.csv|\\.jld2)\$")
vals = Dict{Tuple{Float64, Float64}, Float64}()
for xd in readdir(folder)
    mx = match(value_re(xaxis), xd)
    (isnothing(mx) || !isdir(joinpath(folder, xd))) && continue
    x = parse(Float64, mx[1])
    for f in readdir(joinpath(folder, xd))
        my = match(file_re(yaxis), f)
        isnothing(my) && continue
        y = parse(Float64, my[1])
        (xmin <= x <= xmax && ymin <= y <= ymax) || continue
        v = my[2] == "_obs.csv" ? read_value(joinpath(folder, xd, f)) : NaN
        vals[(x, y)] = isnan(get(vals, (x, y), NaN)) ? v : vals[(x, y)]
    end
end
isempty(vals) && error("no grid point of $folder in the window $xaxis ∈ [$xmin, $xmax], $yaxis ∈ [$ymin, $ymax]")
logscale && foreach(k -> vals[k] > 0 || (vals[k] = NaN), collect(keys(vals)))
nvalues = count(!isnan, values(vals))
nvalues > 0 || error("no point in the window has a value of $col (compute the observables and gather first)")

xs = sort(unique(first.(keys(vals))))
ys = sort(unique(last.(keys(vals))))
M = [get(vals, (x, y), NaN) for x in xs, y in ys]

# Cell edges halfway between the grid values (uneven steps allowed), the outer cells as wide as their neighbour;
# a single value gets a cell of width 10 % of it (0.1 at 0)
function edges(v)
    if length(v) == 1
        h = v[1] == 0 ? 0.05 : 0.05 * abs(v[1])
        return [v[1] - h, v[1] + h]
    end
    mid = (v[1:end-1] .+ v[2:end]) ./ 2
    return [2v[1] - mid[1]; mid; 2v[end] - mid[end]]
end
xe, ye = edges(xs), edges(ys)

using CairoMakie

axislabel = Dict("t" => L"t", "U" => L"U", "Delta" => L"Δ", "Omega" => L"Ω", "r" => L"r")
obslabel = Dict("S_bar" => L"\bar{S} = \mathbb{E}_w\left[\overline{S_w(k)}_{k\in\mathrm{bulk}}\right]",
                "S_bstd" => L"σ_S = \mathbb{E}_w\left[\mathrm{std}_{k\in\mathrm{bulk}}\, S_w(k)\right]",
                "E" => L"E/L", "eps_conv" => L"ϵ_\mathrm{conv}", "residual_max" => L"\max_w |\log Z(w)|")

finite = filter(!isnan, M)
lo, hi = extrema(finite)
colorrange = lo == hi ? (logscale ? (lo / 2, 2hi) : (lo - 0.5, hi + 0.5)) : (lo, hi)

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor = :white, fontsize = 24, size = (1100, 850))
ax = Axis(fig[1, 1]; xlabel = get(axislabel, xaxis, xaxis), ylabel = get(axislabel, yaxis, yaxis),
          title = "L = $L, $tag", titlesize = 18, titlefont = :regular)
hm = heatmap!(ax, xe, ye, M; colormap = :viridis, colorrange, nan_color = :gray80,
              colorscale = logscale ? log10 : identity)
Colorbar(fig[1, 2], hm; label = get(obslabel, col, col))
limits!(ax, xe[1], xe[end], ye[1], ye[end])

mkpath("plots")
file = joinpath("plots", "BoseHubbardU1_NormalizedITE_$(tag)_L$(L)_$(col).png")
save(file, fig)
println("$col over $(length(xs)) × $(length(ys)) points ($nvalues with a value), $xaxis ∈ [$(xs[1]), $(xs[end])], ",
        "$yaxis ∈ [$(ys[1]), $(ys[end])], colour range $(colorrange)")
println("Saved $file")
