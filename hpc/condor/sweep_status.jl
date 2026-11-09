# Status of the correlated Bose-Hubbard sweeps under data/ as JSON, for the dashboard (hpc/condor/dashboard/server.py).
# One Julia call reads every stored state of a sweep, instead of one `--status` call per point.
#
# Usage (from the repo root):
#   julia --project=. hpc/condor/sweep_status.jl --list          all sweeps: [{tag, L, n_states, n_obs, mtime}]
#   julia --project=. hpc/condor/sweep_status.jl <tag> <L>       one sweep:
#     {tag, L, points: [{t, mu, state, error, converged, eps, nsteps, maxiter, convtol, resumed_from, mtime, obs, n_edge, obs_version}],
#      planned: [[t, mu], ...]}
# t and mu are the strings of the file names (%.4f). `state` is true if t<t>/mu<μ>.jld2 exists; `error` is set if it
# cannot be read. `converged` is the stored flag (final ϵ_conv ≤ CONVTOL of that run). `obs` is true if mu<μ>_obs.csv
# exists, `n_edge` its N_EDGE and `obs_version` the version stamp of the observables script. `planned` are the points
# of every points file the condor scripts wrote for this sweep (logs/.../<tag>/L<L>/condor/points_*.txt), so submitted points without a state are part of the grid too.

const ROOT = joinpath("data", "BoseHubbardCorrelated_NormalizedITE")
const LOGROOT = joinpath("logs", "BoseHubbardCorrelated_NormalizedITE")

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

function list_sweeps()
    out = []
    for tag in dirs(ROOT, r"."), Ldir in dirs(joinpath(ROOT, tag), r"^L\d+$")
        folder = joinpath(ROOT, tag, Ldir)
        files = [joinpath(folder, td, f) for td in dirs(folder, r"^t-?[\d.]+$") for f in readdir(joinpath(folder, td))]
        states = filter(f -> occursin(r"/mu-?[\d.]+\.jld2$", f), files)
        obs = filter(f -> endswith(f, "_obs.csv"), files)
        push!(out, (; tag, L = parse(Int, Ldir[2:end]), n_states = length(states), n_obs = length(obs),
                    mtime = isempty(files) ? mtime(folder) : maximum(mtime, files)))
    end
    return out
end

# n_edge and obs_version of an observables file (nothing if missing)
function read_obs(file)
    lines = filter(!isempty, readlines(file))
    length(lines) >= 2 || return nothing, nothing
    header, row = split(lines[1], ','), split(lines[2], ',')
    col(name) = (k = findfirst(==(name), header); isnothing(k) || k > length(row) ? nothing : String(row[k]))
    n_edge = col("n_edge")
    return isnothing(n_edge) ? nothing : tryparse(Int, n_edge), col("obs_version")
end

function sweep_status(tag, L)
    folder = joinpath(ROOT, tag, "L$L")
    points = []
    statefiles = [(td[2:end], m[1], joinpath(folder, td, f)) for td in dirs(folder, r"^t-?[\d.]+$")
                  for f in readdir(joinpath(folder, td)) for m in (match(r"^mu(-?[\d.]+)\.jld2$", f),) if !isnothing(m)]
    !isempty(statefiles) && @eval using JLD2
    for (t, mu, file) in statefiles
        p = Dict{String, Any}("t" => t, "mu" => mu, "state" => true, "mtime" => mtime(file), "error" => nothing)
        try
            Base.invokelatest() do
                jldopen(file, "r") do f
                    ϵs = f["info"].ϵsconv
                    params = f["params"]
                    p["eps"] = isempty(ϵs) ? nothing : last(ϵs)
                    p["nsteps"] = haskey(f, "nsteps") ? f["nsteps"] : length(ϵs)
                    p["maxiter"] = params.maxiter
                    p["convtol"] = params.convtol
                    p["resumed_from"] = haskey(params, :resumed_from) ? params.resumed_from : 0   # steps of the run it continued
                    p["converged"] = haskey(f, "converged") ? f["converged"] : (!isempty(ϵs) && last(ϵs) <= params.convtol)
                end
            end
        catch e
            p["error"] = sprint(showerror, e)
        end
        obsfile = replace(file, r"\.jld2$" => "_obs.csv")
        p["obs"] = isfile(obsfile)
        p["n_edge"], p["obs_version"] = p["obs"] ? read_obs(obsfile) : (nothing, nothing)
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
    return (; tag, L, points, planned = [collect(p) for p in sort!(collect(planned))])
end

if length(ARGS) == 1 && ARGS[1] == "--list"
    println(json(list_sweeps()))
elseif length(ARGS) == 2
    println(json(sweep_status(ARGS[1], parse(Int, ARGS[2]))))
else
    error("usage: julia --project=. hpc/condor/sweep_status.jl --list | <tag> <L>")
end
