# Status of the Bose-Hubbard sweeps under data/ as JSON, for the dashboard (hpc/condor/dashboard/server.py).
# One Julia call reads every stored state of a sweep, instead of one `--status` call per point.
#
# Usage (from the repo root):
#   julia --project=. hpc/condor/sweep_status.jl [--model M] --list          all sweeps: [{tag, L, axes, n_states, n_obs, mtime}]
#   julia --project=. hpc/condor/sweep_status.jl [--model M] <tag> <L>       one sweep:
#     {tag, L, axes: [x, y], points: [{x, y, state, error, converged, eps, nsteps, maxiter, convtol, resumed_from, seed, mtime,
#      obs, n_edge, obs_version, values}], planned: [[x, y], ...]}
# Models (M):
#   BHCorr (default)  data/BoseHubbardCorrelated_NormalizedITE/<tag>/L<L>/t<t>/mu<μ>.jld2, axes t and mu
#   BHU1              data/BoseHubbardU1_NormalizedITE/<tag>/L<L>/<X><x>/<Y><y>.jld2, axes X and Y named by the tag
#                     (it starts with "<X>-<Y>_", see the dashboard)
# x and y are the value strings of the file names (%.4f). `state` is true if the state file exists; `error` is set if it
# cannot be read. `converged` is the stored flag (final ϵ_conv ≤ CONVTOL of that run). `obs` is true if the
# observables file (<state>_obs.csv) exists, `n_edge` its N_EDGE, `obs_version` the version stamp of the observables
# script and `values` every numeric column of it (for the dashboard's heatmaps). `planned` are the points of every
# points file the condor scripts wrote for this sweep (logs/.../<tag>/L<L>/condor[/obs]/points_*.txt), so submitted
# points without a state are part of the grid too.

args = copy(ARGS)
model = "BHCorr"
if length(args) >= 2 && args[1] == "--model"
    model = args[2]
    args = args[3:end]
end
const MODELDIR = Dict("BHCorr" => "BoseHubbardCorrelated_NormalizedITE", "BHU1" => "BoseHubbardU1_NormalizedITE")
haskey(MODELDIR, model) || error("unknown model $model, known: $(join(keys(MODELDIR), ", "))")
const ROOT = joinpath("data", MODELDIR[model])
const LOGROOT = joinpath("logs", MODELDIR[model])

# Axis names of a sweep
function axes_of(tag)
    model == "BHCorr" && return ("t", "mu")
    m = match(r"^([A-Za-z]+)-([A-Za-z]+)_", tag)
    isnothing(m) && error("tag $tag does not start with <X>-<Y>_")
    return (String(m[1]), String(m[2]))
end

# Minimal JSON writer (no JSON package in the project)
json(x::AbstractString) = "\"" * escape_string(x) * "\""
json(x::Bool) = x ? "true" : "false"
json(x::Integer) = string(x)
json(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
json(::Nothing) = "null"
json(x::Union{AbstractVector, Tuple}) = "[" * join(map(json, x), ",") * "]"
json(x::AbstractDict) = "{" * join(("$(json(string(k))):$(json(v))" for (k, v) in x), ",") * "}"
json(x::NamedTuple) = "{" * join(("$(json(string(k))):$(json(v))" for (k, v) in pairs(x)), ",") * "}"

dirs(path, re) = isdir(path) ? sort(filter(s -> occursin(re, s) && isdir(joinpath(path, s)), readdir(path))) : String[]
value_re(name) = Regex("^" * name * "(-?[\\d.]+)\$")
file_re(name) = Regex("^" * name * "(-?[\\d.]+)\\.jld2\$")

# State files of a sweep folder: (x, y, file)
function statefiles(folder, (xname, yname))
    out = Tuple{String, String, String}[]
    for xd in dirs(folder, value_re(xname)), f in readdir(joinpath(folder, xd))
        m = match(file_re(yname), f)
        isnothing(m) || push!(out, (match(value_re(xname), xd)[1], String(m[1]), joinpath(folder, xd, f)))
    end
    return out
end

function list_sweeps()
    out = []
    for tag in dirs(ROOT, r"."), Ldir in dirs(joinpath(ROOT, tag), r"^L\d+$")
        ax = try axes_of(tag) catch; continue end
        folder = joinpath(ROOT, tag, Ldir)
        states = statefiles(folder, ax)
        obs = filter(s -> isfile(replace(s[3], r"\.jld2$" => "_obs.csv")), states)
        files = [s[3] for s in states]
        push!(out, (; tag, L = parse(Int, Ldir[2:end]), axes = ax, n_states = length(states), n_obs = length(obs),
                    mtime = isempty(files) ? mtime(folder) : maximum(mtime, files)))
    end
    return out
end

# n_edge, obs_version and the numeric columns of an observables file
function read_obs(file)
    lines = filter(!isempty, readlines(file))
    length(lines) >= 2 || return nothing, nothing, Dict{String, Float64}()
    header, row = split(lines[1], ','), split(lines[2], ',')
    col(name) = (k = findfirst(==(name), header); isnothing(k) || k > length(row) ? nothing : String(row[k]))
    n_edge = col("n_edge")
    values = Dict{String, Float64}()
    for (h, v) in zip(header, row)
        x = tryparse(Float64, v)
        isnothing(x) || (values[String(h)] = x)
    end
    return isnothing(n_edge) ? nothing : tryparse(Int, n_edge), col("obs_version"), values
end

function sweep_status(tag, L)
    ax = axes_of(tag)
    folder = joinpath(ROOT, tag, "L$L")
    points = []
    files = statefiles(folder, ax)
    !isempty(files) && @eval using JLD2
    for (x, y, file) in files
        p = Dict{String, Any}("x" => x, "y" => y, "state" => true, "mtime" => mtime(file), "error" => nothing)
        try
            Base.invokelatest() do
                jldopen(file, "r") do f
                    ϵs = f["info"].ϵsconv
                    params = f["params"]
                    p["eps"] = isempty(ϵs) ? nothing : last(ϵs)
                    p["nsteps"] = haskey(f, "nsteps") ? f["nsteps"] : length(ϵs)
                    p["maxiter"] = params.maxiter
                    p["convtol"] = params.convtol
                    p["resumed_from"] = haskey(params, :resumed_from) ? params.resumed_from : 0
                    p["seed"] = haskey(params, :seed) ? params.seed : nothing
                    p["converged"] = haskey(f, "converged") ? f["converged"] : (!isempty(ϵs) && last(ϵs) <= params.convtol)
                end
            end
        catch e
            p["error"] = sprint(showerror, e)
        end
        obsfile = replace(file, r"\.jld2$" => "_obs.csv")
        p["obs"] = isfile(obsfile)
        p["n_edge"], p["obs_version"], p["values"] = p["obs"] ? read_obs(obsfile) : (nothing, nothing, Dict{String, Float64}())
        push!(points, p)
    end
    planned = Set{Tuple{String, String}}()
    for dir in (joinpath(LOGROOT, tag, "L$L", "condor"), joinpath(LOGROOT, tag, "L$L", "condor", "obs"))
        isdir(dir) || continue
        for f in readdir(dir; join = true)
            occursin(r"points_\d+_\d+\.txt$", f) || continue
            for line in eachline(f)
                parts = split(line)
                length(parts) == 2 && push!(planned, (parts[1], parts[2]))
            end
        end
    end
    return (; tag, L, axes = ax, points, planned = [collect(p) for p in sort!(collect(planned))])
end

if length(args) == 1 && args[1] == "--list"
    println(json(list_sweeps()))
elseif length(args) == 2
    println(json(sweep_status(args[1], parse(Int, args[2]))))
else
    error("usage: julia --project=. hpc/condor/sweep_status.jl [--model BHCorr|BHU1] --list | <tag> <L>")
end
