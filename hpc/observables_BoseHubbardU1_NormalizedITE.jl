# Observables of points of a hpc/BoseHubbardU1_NormalizedITE.jl sweep (U(1) tensors, fixed filling), written to a
# one-row CSV next to every state.
#
# Observables are entries of the registry OBSERVABLES below. Every entry has
#   name       selects it (environment OBS, comma separated; default: every applicable one)
#   columns    the CSV columns it writes
#   applies    ctx -> Bool, whether it makes sense for this sweep/point (e.g. needs enough bulk bonds); entries that do
#              not apply are skipped and their columns left empty
#   compute    (ψ, data, ctx) -> NamedTuple of its columns
# ctx has the stored params, the axes, the bulk bonds and sites (N_EDGE sites dropped at each end), N_SAMPLES and an
# rng. Quantities that need several points (a derivative along an axis, a phase boundary) are not per point: the
# dashboard derives them from the per-point values.
#
# Registered:
#   diagnostics  E (final energy density), eps_conv, residual_max (normalization residual of the last step), converged,
#                nsteps; read from the stored ITE info, always computed
#   entropy      S_bar = E_w[(1/|B|) Σ_{k ∈ B} S_w(k)]    disorder average of the bond-averaged entanglement entropy
#                S_bstd = E_w[std_{k ∈ B} S_w(k)]          disorder average of its bond-to-bond standard deviation
#                S_bar_err, S_bstd_err                    standard errors (0 if the words are enumerated)
#                over the bulk bonds B = N_EDGE+1..L-N_EDGE-1 (DisorderKit.bond_averaged_entanglement_entropy). The
#                particle number is fixed, so no projection onto a particle number is needed. Words are enumerated
#                exactly if ∏ᵢ qᵢ ≤ N_SAMPLES, else N_SAMPLES words are sampled
#
# Usage (from the repo root, normally inside the sweep job or through hpc/condor/obs_BHU1.sh):
#   julia --project=. hpc/observables_BoseHubbardU1_NormalizedITE.jl <L> <x1> <y1> [<x2> <y2> ...]   compute and store
#       (every point is tried, a failing one does not stop the others, the exit code is 1 if any failed)
#   julia hpc/observables_BoseHubbardU1_NormalizedITE.jl --status <L> <x> <y>   print the tag, the state file, the
#       observables file and the status: nostate, uptodate or compute
#
# Environment (required): TAG, XAXIS, YAXIS, DATA_DIR (as for the ITE, so the paths match), N_SAMPLES, N_EDGE
# Optional: OBS (observables to compute), OBS_VERSION (version stamp of this script, `git hash-object` of it)
#
# Output: <state file without .jld2>_obs.csv, columns xaxis, x, yaxis, y, the columns of the computed observables,
#   observables (names computed, ';' separated), n_edge, n_samples, exact, obs_version
#
# Status compute if there is no readable observables file, a requested observable is missing from it, its n_edge or
# obs_version differs, the state is newer than it, or its words were sampled and N_SAMPLES is now larger. Columns of
# observables that are not requested now are kept from the old file if that file is still valid.

const MODEL = "BoseHubbardU1_NormalizedITE"

# ---- Observable registry
struct Observable
    name::String
    columns::Vector{String}
    applies::Function
    compute::Function
end

lastor(v) = isempty(v) ? NaN : last(v)

function compute_diagnostics(ψ, data, ctx)
    info = data["info"]
    return (; E = lastor(info.Es), eps_conv = lastor(info.ϵsconv), residual_max = lastor(info.residual_max),
            converged = get(data, "converged", false), nsteps = get(data, "nsteps", -1))
end

function compute_entropy(ψ, data, ctx)
    S_bar, S_bar_err, S_bstd, S_bstd_err =
        DisorderKit.bond_averaged_entanglement_entropy(ψ; n_samples = ctx.n_samples, rng = ctx.rng(), bonds = ctx.bulk_bonds)
    return (; S_bar, S_bar_err, S_bstd, S_bstd_err)
end

const OBSERVABLES = Observable[
    Observable("diagnostics", ["E", "eps_conv", "residual_max", "converged", "nsteps"], ctx -> true, compute_diagnostics),
    Observable("entropy", ["S_bar", "S_bar_err", "S_bstd", "S_bstd_err"], ctx -> length(ctx.bulk_bonds) >= 2, compute_entropy),
]
function observable(name)
    i = findfirst(o -> o.name == name, OBSERVABLES)
    isnothing(i) && error("unknown observable $name; registered: $(join((o.name for o in OBSERVABLES), ", "))")
    return i
end

# ---- Parameters, paths and status (before loading any package, so --status is fast)
status_only = !isempty(ARGS) && ARGS[1] == "--status"
args = status_only ? ARGS[2:end] : ARGS
usage = "usage: julia --project=. hpc/observables_$MODEL.jl [--status] <L> <x> <y> [<x2> <y2> ...] (--status: one point)"
(length(args) >= 3 && isodd(length(args)) && !(status_only && length(args) > 3)) || error(usage)
L = parse(Int, args[1])
points = [(args[k], args[k+1]) for k in 2:2:length(args)]   # (x, y) as strings, used verbatim in the file names

param(k) = haskey(ENV, k) ? ENV[k] : error("environment variable $k not set; parameters are set in hpc/condor/sweep_BHU1.sh")
tag, xaxis, yaxis = param("TAG"), param("XAXIS"), param("YAXIS")
data_root = param("DATA_DIR")
n_samples = parse(Int, param("N_SAMPLES"))
n_edge = parse(Int, param("N_EDGE"))
obs_version = get(ENV, "OBS_VERSION", "")
requested = let s = strip(get(ENV, "OBS", ""))
    isempty(s) ? [o.name for o in OBSERVABLES] : String.(strip.(split(s, ',')))
end
foreach(observable, requested)                               # unknown names fail early
"diagnostics" in requested || pushfirst!(requested, "diagnostics")
n_edge >= 0 || error("N_EDGE = $n_edge must be ≥ 0")
bulk_bonds = n_edge+1:L-n_edge-1
bulk_sites = n_edge+1:L-n_edge

function point_files(x_str, y_str)
    folder = joinpath(data_root, MODEL, tag, "L$(L)", "$(xaxis)$(x_str)")
    return joinpath(folder, "$(yaxis)$(y_str).jld2"), joinpath(folder, "$(yaxis)$(y_str)_obs.csv")
end

# Header and first row of a CSV as column => value (strings), nothing if the file is missing or malformed
function read_row(file)
    isfile(file) || return nothing
    lines = filter(!isempty, readlines(file))
    length(lines) >= 2 || return nothing
    header, row = split(lines[1], ','), split(lines[2], ',')
    length(header) == length(row) || return nothing
    return Dict(String(h) => String(v) for (h, v) in zip(header, row))
end

stored_observables(row) = isnothing(row) ? String[] : String.(filter(!isempty, split(get(row, "observables", ""), ';')))

# Whether the stored row can be kept: same n_edge and version, not older than the state, enough samples
function row_valid(row, statefile, obsfile)
    isnothing(row) && return false
    tryparse(Int, get(row, "n_edge", "")) == n_edge || return false
    isempty(obs_version) || get(row, "obs_version", "") == obs_version || return false
    mtime(statefile) > mtime(obsfile) && return false
    stored, exact = tryparse(Int, get(row, "n_samples", "")), tryparse(Bool, get(row, "exact", ""))
    (isnothing(stored) || isnothing(exact)) && return false
    return exact || n_samples <= stored
end

function status(statefile, obsfile)
    isfile(statefile) || return "nostate"
    row = read_row(obsfile)
    row_valid(row, statefile, obsfile) || return "compute"
    # Observables that did not apply are recorded as computed (with empty columns), so only missing ones count
    return all(in(stored_observables(row)), requested) ? "uptodate" : "compute"
end

if status_only
    statefile, obsfile = point_files(points[1]...)
    println(tag)
    println(statefile)
    println(obsfile)
    println(status(statefile, obsfile))
    exit()
end

# ---- Observables
using TensorKit, DisorderKit, LinearAlgebra, Random, JLD2, Printf, Dates

function compute_point(x_str, y_str)
    statefile, obsfile = point_files(x_str, y_str)
    st = status(statefile, obsfile)
    st == "nostate" && error("no state $statefile")
    if st == "uptodate"
        println("Up to date ($(join(requested, ", ")), N_SAMPLES = $n_samples, N_EDGE = $n_edge): $obsfile")
        return
    end
    println("U(1) Bose-Hubbard observables $(join(requested, ", ")): L = $L, $xaxis = $x_str, $yaxis = $y_str, tag $tag, ",
            "N_SAMPLES = $n_samples, N_EDGE = $n_edge (bulk bonds $bulk_bonds)")
    println("State: $statefile")
    data = load(statefile)
    ψ = data["ψ"]
    exact = prod(float.(length.(ψ.ps))) <= n_samples
    ctx = (; params = data["params"], axes = (xaxis, yaxis), bulk_bonds, bulk_sites, n_samples, rng = () -> Xoshiro(1))

    # Start from the stored columns if that file is still valid (observables not requested now are kept)
    old = read_row(obsfile)
    keep = row_valid(old, statefile, obsfile)
    values = Dict{String, String}()
    done = String[]
    if keep
        for name in stored_observables(old)
            name in requested && continue
            i = findfirst(o -> o.name == name, OBSERVABLES)
            isnothing(i) && continue
            for c in OBSERVABLES[i].columns
                values[c] = get(old, c, "")
            end
            push!(done, name)
        end
    end
    walltime = @elapsed for name in requested
        o = OBSERVABLES[observable(name)]
        if o.applies(ctx)
            result = o.compute(ψ, data, ctx)
            for c in o.columns
                values[c] = string(result[Symbol(c)])
            end
        else
            println("  $name does not apply to this point, skipped")
            for c in o.columns
                values[c] = ""
            end
        end
        push!(done, name)
    end

    columns = ["xaxis", "x", "yaxis", "y"]
    row = Any[xaxis, x_str, yaxis, y_str]
    for o in OBSERVABLES, c in o.columns
        haskey(values, c) || continue
        push!(columns, c)
        push!(row, values[c])
    end
    append!(columns, ["observables", "n_edge", "n_samples", "exact", "obs_version"])
    append!(row, [join(sort(unique(done)), ';'), n_edge, n_samples, exact, obs_version])

    # Write to a temporary file and move it in place, so an interrupted job never leaves a partial result
    tmp = obsfile * ".tmp"
    open(tmp, "w") do io
        println(io, join(columns, ','))
        println(io, join(string.(row), ','))
    end
    mv(tmp, obsfile; force = true)
    shown = join(("$c = $(values[c])" for o in OBSERVABLES if o.name in requested for c in o.columns if !isempty(get(values, c, ""))), ", ")
    println(shown, " ($(exact ? "exact enumeration" : "$n_samples sampled words"), $(round(walltime; digits = 1)) s)")
    println("Saved $obsfile")
end

failed = Tuple{String, String}[]
for (k, (x_str, y_str)) in enumerate(points)
    println("== $(Dates.format(now(), "HH:MM:SS")) point $k of $(length(points)): $xaxis = $x_str, $yaxis = $y_str")
    flush(stdout)
    try
        compute_point(x_str, y_str)
    catch e
        push!(failed, (x_str, y_str))
        println(stderr, "ERROR for point $xaxis = $x_str, $yaxis = $y_str:")
        showerror(stderr, e, catch_backtrace())
        println(stderr)
    end
    flush(stdout); flush(stderr)
end
if !isempty(failed)
    println(stderr, "$(length(failed)) of $(length(points)) point(s) failed: ", join(("$xaxis = $x, $yaxis = $y" for (x, y) in failed), "; "))
    exit(1)
end
