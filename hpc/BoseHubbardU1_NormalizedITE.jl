# One point of a two-parameter sweep of the Bose-Hubbard chain with correlated hopping and potential disorder at fixed
# filling, with U(1)-symmetric (particle number conserving) tensors and disorder-normalized imaginary time evolution:
#   H_w = -Σ t_{wᵢ} (b†ᵢbᵢ₊₁ + h.c.) + U/2 Σ nᵢ(nᵢ-1) - Σ μ_{wᵢ} nᵢ,   t_{wᵢ} ∈ {t(1-Ω), t(1+Ω)}, μ_{wᵢ} ∈ {-Δ, +Δ},
# with the label correlation r (DisorderKit.correlated_bose_hubbard). The particle number is exactly N = FILLING·L for
# every disorder word (|n⟩ carries the U₁ charge n - FILLING, the bonds at the ends are trivial), so the mean chemical
# potential only adds a constant and is not a parameter. The evolution starts from the Mott state |f f … f⟩.
#
# The two swept parameters are named by XAXIS and YAXIS, each one of t, U, Delta, Omega, r; their values are the
# positional arguments <x> <y> (formatted %.4f by the sweep and used verbatim in the file names). The environment value
# of a swept parameter is ignored.
#
# Usage (from the repo root, normally through hpc/condor/sweep_BHU1.sh):
#   julia --project=. hpc/BoseHubbardU1_NormalizedITE.jl <L> <x> <y>            run and store
#   julia hpc/BoseHubbardU1_NormalizedITE.jl --path <L> <x> <y>                 only print the tag and the output file
#   julia --project=. hpc/BoseHubbardU1_NormalizedITE.jl --status <L> <x> <y>   print the tag, the output file, the status
#       (run, resume, rerun or uptodate) and the reason, as in hpc/BoseHubbardCorrelated_NormalizedITE.jl:
#   run       there is no (readable) stored state
#   uptodate  the stored state is converged for this job: final ϵ_conv ≤ CONVTOL
#   resume    it is not, and MAXITER > stored maxiter (and RESUME ≠ 0): the run continues the ITE from the stored state
#   rerun     it is not, and CONVTOL < stored convtol only (or RESUME = 0): the rerun starts from scratch
#   uptodate  it is not, but the same settings would give the same result
#
# Environment (required): TAG (the sweep tag, built by the dashboard: it names the axes and the fixed parameters),
#   XAXIS, YAXIS, T, U, DELTA, OMEGA, R, FILLING, NMAX, DTAU, MAXD, MAXITER, CONVTOL, CROSS_WIDTH, DATA_DIR
# Optional: BLAS_THREADS (default: as many as Julia threads), RESUME (default 1), SEED (default 20266001: seed of the
#   randomized compression; a run from scratch with another SEED gives an independent ITE trajectory)
#
# Output: <DATA_DIR>/BoseHubbardU1_NormalizedITE/<TAG>/L<L>/<XAXIS><x>/<YAXIS><y>.jld2 with ψ, info (per-step
#   diagnostics), converged, nsteps, params (all parameters and axes) and meta

const MODEL = "BoseHubbardU1_NormalizedITE"
const AXES = Dict("t" => "T", "U" => "U", "Delta" => "DELTA", "Omega" => "OMEGA", "r" => "R")   # axis name => env variable

# ---- Parameters and paths (before loading DisorderKit, so --path and --status are fast)
path_only = !isempty(ARGS) && ARGS[1] == "--path"
status_only = !isempty(ARGS) && ARGS[1] == "--status"
args = path_only || status_only ? ARGS[2:end] : ARGS
length(args) == 3 || error("usage: julia --project=. hpc/$MODEL.jl [--path | --status] <L> <x> <y>")
L_str, x_str, y_str = args
L = parse(Int, L_str)

param(k) = haskey(ENV, k) ? ENV[k] : error("environment variable $k not set; run parameters are set in hpc/condor/sweep_BHU1.sh")
xaxis, yaxis = param("XAXIS"), param("YAXIS")
(haskey(AXES, xaxis) && haskey(AXES, yaxis) && xaxis != yaxis) ||
    error("XAXIS and YAXIS must be two different ones of $(join(sort(collect(keys(AXES))), ", ")), got $xaxis and $yaxis")
tag = param("TAG")
resume_enabled = get(ENV, "RESUME", "1") != "0"

# Value of a parameter: from the positional arguments if it is swept, else from the environment
value_str(name) = name == xaxis ? x_str : name == yaxis ? y_str : param(AXES[name])

data_root = param("DATA_DIR")
folder = joinpath(data_root, MODEL, tag, "L$(L)", "$(xaxis)$(x_str)")
filename = joinpath(folder, "$(yaxis)$(y_str).jld2")

if path_only
    println(tag)
    println(filename)
    exit()
end

status_only && isfile(filename) && @eval using JLD2
function status(filename, maxiter, convtol)
    isfile(filename) || return "run", "no state"
    stored = try
        jldopen(filename, "r") do f
            ϵs = f["info"].ϵsconv
            (; maxiter = f["params"].maxiter, convtol = f["params"].convtol, ϵ = isempty(ϵs) ? Inf : last(ϵs))
        end
    catch e
        return "run", "unreadable state: $(sprint(showerror, e))"
    end
    ϵ = "ϵ = $(round(stored.ϵ; sigdigits = 2))"
    stored.ϵ <= convtol && return "uptodate", "converged: $ϵ ≤ convtol $convtol"
    changes = String[]
    maxiter > stored.maxiter && push!(changes, "maxiter $(stored.maxiter) → $maxiter")
    convtol < stored.convtol && push!(changes, "convtol $(stored.convtol) → $convtol")
    isempty(changes) && return "uptodate", "unconverged ($ϵ), but run with maxiter $(stored.maxiter) ≥ $maxiter and convtol $(stored.convtol) ≤ $convtol"
    resume_enabled && maxiter > stored.maxiter && return "resume", "unconverged ($ϵ), $(join(changes, ", ")), continues from the stored state"
    return "rerun", "unconverged ($ϵ), $(join(changes, ", "))"
end
if status_only
    st, reason = status(filename, parse(Int, param("MAXITER")), parse(Float64, param("CONVTOL")))
    println(tag)
    println(filename)
    println(st)
    println(reason)
    exit()
end

# ---- Simulation
using TensorKit, DisorderKit, TimerOutputs, JLD2, Dates, LinearAlgebra, Logging
BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", string(Threads.nthreads()))))

# Log messages start with the time (HH:MM:SS), so the condor dashboard can show how long a job is in its current step
struct TimestampLogger <: AbstractLogger
    inner::AbstractLogger
end
Logging.handle_message(l::TimestampLogger, level, message, args...; kwargs...) =
    Logging.handle_message(l.inner, level, "$(Dates.format(now(), "HH:MM:SS")) $message", args...; kwargs...)
Logging.shouldlog(l::TimestampLogger, args...) = Logging.shouldlog(l.inner, args...)
Logging.min_enabled_level(l::TimestampLogger) = Logging.min_enabled_level(l.inner)
Logging.catch_exceptions(l::TimestampLogger) = Logging.catch_exceptions(l.inner)
global_logger(TimestampLogger(global_logger()))

t, U, Δ, Ω, r = (parse(Float64, value_str(n)) for n in ("t", "U", "Delta", "Omega", "r"))
filling = parse(Int, param("FILLING"))
nmax = parse(Int, param("NMAX"))
Δτ = parse(Float64, param("DTAU"))
maxD = parse(Int, param("MAXD"))
maxiter = parse(Int, param("MAXITER"))
convtol = parse(Float64, param("CONVTOL"))
cross_width = parse(Int, param("CROSS_WIDTH"))
seed0 = parse(Int, get(ENV, "SEED", "20266001"))
Hs, ps = DisorderKit.correlated_bose_hubbard(t, U, 0.0, Δ, Ω, r; nmax, symmetry = U1Irrep, filling)

# Initial state: the stored state if it can be continued (see the header), else the Mott state
function load_resume(filename)
    resume_enabled || return nothing, "RESUME = 0"
    isfile(filename) || return nothing, "no stored state"
    stored = try
        jldopen(filename, "r") do f
            (; ψ = f["ψ"], info = f["info"], params = f["params"], meta = f["meta"], nsteps = f["nsteps"])
        end
    catch e
        return nothing, "unreadable stored state: $(sprint(showerror, e))"
    end
    sp = stored.params
    (sp.L == L && sp.nmax == nmax && sp.filling == filling && sp.maxD == maxD && sp.Δτ == Δτ && sp.ps ≈ ps &&
     sp.cross_width == cross_width) || return nothing, "stored state has other parameters"
    ϵ = isempty(stored.info.ϵsconv) ? Inf : last(stored.info.ϵsconv)
    ϵ <= convtol && return nothing, "stored state is converged (ϵ = $ϵ)"
    sp.maxiter < maxiter && stored.nsteps < maxiter || return nothing, "stored state already has maxiter $(sp.maxiter) ≥ $maxiter"
    return stored, "unconverged after $(stored.nsteps) steps (ϵ = $(round(ϵ; sigdigits = 2)))"
end
stored, why = load_resume(filename)
n0 = stored === nothing ? 0 : stored.nsteps
ψ0 = stored === nothing ? DisorderKit.mott_state(Hs, ps, L; filling) : stored.ψ
println(stored === nothing ? "Starting from the Mott state ($why)" : "Resuming from $filename: $why, at most $(maxiter - n0) more steps")

params = (; L, t, U, Δ, Ω, r, filling, nmax, ps, Δτ, maxD, maxiter, convtol, cross_width, resumed_from = n0, seed = seed0,
          axes = (xaxis, yaxis), tag)
println("U(1) correlated Bose-Hubbard normalized ITE: L = $L, filling = $filling, t = $t, U = $U, Δ = $Δ, Ω = $Ω, r = $r, ",
        "nmax = $nmax, maxD = $maxD, Δτ = $Δτ (axes $xaxis × $yaxis)")
println("Output: $filename")

to = TimerOutput()
alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD, order = 2, convtol, maxiter = maxiter - n0, cross_width, verbosity = 1,
                                                  timer_output = to, seed = seed0 + n0)
walltime = @elapsed ψ, _, info = DisorderKit.groundstate(ψ0, Hs, Δτ, alg)

info_nt = NamedTuple{fieldnames(typeof(info))}(Tuple(getfield(info, f) for f in fieldnames(typeof(info))))
stored === nothing || (info_nt = NamedTuple{keys(info_nt)}(Tuple(vcat(stored.info[k], info_nt[k]) for k in keys(info_nt))))
nsteps = length(info_nt.Es)
converged = !isempty(info_nt.ϵsconv) && info_nt.ϵsconv[end] <= convtol

gitcmd(args...) = try readchomp(Cmd(`git -C $(pkgdir(DisorderKit)) $args`)) catch; "unknown" end
meta = (; walltime, timer = TimerOutputs.todict(to), git_commit = gitcmd("rev-parse", "HEAD"),
        git_dirty = !isempty(gitcmd("status", "--porcelain", "--untracked-files=no")),
        julia_version = string(VERSION), hostname = gethostname(), threads = Threads.nthreads(),
        blas_threads = BLAS.get_num_threads(), date = string(now()),
        previous = stored === nothing ? nothing : stored.meta)

# Write to a temporary file and move it in place, so an interrupted job never leaves a partial result
mkpath(folder)
tmp = filename * ".tmp"
jldsave(tmp; ψ, info = info_nt, converged, nsteps, params, meta)
mv(tmp, filename; force = true)

println("Done: $nsteps steps$(n0 > 0 ? " ($n0 stored + $(nsteps - n0) new)" : ""), converged = $converged, walltime = $(round(walltime; digits = 1)) s")
println("Saved $filename")
show(to; allocations = false, compact = true); println()
