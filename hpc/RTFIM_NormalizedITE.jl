# One point of the RTFIM phase diagram with disorder-normalized imaginary time evolution (Groundstate_FiniteNormalizedITE).
# Model: h ∈ e^δ {1, e^{-2W}}, J ∈ {1, e^{-2W}}, p_u = 1/4 (δ distance to criticality, W disorder strength).
# Stores the final state, the state at the end of every cap stage and the per-step diagnostics; no observables, those
# are computed from the stored states when plotting.
#
# Usage (from the repo root):
#   julia --project=. hpc/RTFIM_NormalizedITE.jl <L> <δ> <W>          run and store
#   julia hpc/RTFIM_NormalizedITE.jl --path <L> <δ> <W>               only print the run tag and the output file
# δ and W are used verbatim in the file names (the sweep script passes them formatted as %.4f).
#
# Run parameters (environment, with defaults):
#   DTAU=0.02  ORDER=2  DMAX_SCHEDULE="300:8,100:16,100:32" (steps:cap, the last cap is kept)
#   MAXITER (default: total steps of the schedule)  CONVTOL=1e-8  FILTER_TAIL_TOL=1e-8  SEED=20266001
#   QUICK=1 (default schedule "3:4,2:8", for smoke tests)  DATA_DIR (default $VSC_DATA/data, else ./data)
#
# Output: <DATA_DIR>/RTFIM_NormalizedITE/<run tag>/L<L>/W<W>/delta<δ>.jld2, run tag dtau<Δτ>_order<N>_D<steps>x<cap>-...

# ---- Parameters and paths (before loading DisorderKit, so --path is fast)
path_only = !isempty(ARGS) && ARGS[1] == "--path"
args = path_only ? ARGS[2:end] : ARGS
length(args) == 3 || error("usage: julia --project=. hpc/RTFIM_NormalizedITE.jl [--path] <L> <δ> <W>")
L_str, δ_str, W_str = args
L, δ, W = parse(Int, L_str), parse(Float64, δ_str), parse(Float64, W_str)

quick = get(ENV, "QUICK", "0") == "1"
dtau_str = get(ENV, "DTAU", "0.02")
order = parse(Int, get(ENV, "ORDER", "2"))
schedule_str = get(ENV, "DMAX_SCHEDULE", quick ? "3:4,2:8" : "300:8,100:16,100:32")
schedule = map(split(schedule_str, ",")) do stage
    steps, cap = parse.(Int, split(stage, ":"))
    steps => cap
end
run_tag = "dtau$(dtau_str)_order$(order)_D" * join(["$(s.first)x$(s.second)" for s in schedule], "-")

data_root = get(ENV, "DATA_DIR", joinpath(get(ENV, "VSC_DATA", "."), "data"))
folder = joinpath(data_root, "RTFIM_NormalizedITE", run_tag, "L$(L)", "W$(W_str)")
filename = joinpath(folder, "delta$(δ_str).jld2")

if path_only
    println(run_tag)
    println(filename)
    exit()
end

# ---- Simulation
using TensorKit, DisorderKit, TimerOutputs, JLD2, Dates, LinearAlgebra
BLAS.set_num_threads(Threads.nthreads())   # --threads=PPN controls both Julia and BLAS threads

Δτ = parse(Float64, dtau_str)
maxiter = parse(Int, get(ENV, "MAXITER", string(sum(first, schedule))))
convtol = parse(Float64, get(ENV, "CONVTOL", "1e-8"))
filter_tail_tol = parse(Float64, get(ENV, "FILTER_TAIL_TOL", "1e-8"))
seed = parse(Int, get(ENV, "SEED", "20266001"))

hs = exp(δ) .* [1.0, exp(-2W)]
Js = [1.0, exp(-2W)]
ps = fill(0.25, 4)
Hs = DisorderKit.random_transverse_field_ising(Js, hs)
ψ0 = DisorderKit.plus_state(ps, L)

params = (; L, δ, W, hs, Js, ps, Δτ, order, schedule, maxiter, convtol, filter_tail_tol, seed)
println("RTFIM normalized ITE: L = $L, δ = $δ, W = $W, schedule = $schedule_str, Δτ = $Δτ")
println("Output: $filename")

# The finalizer counts the steps and keeps the state at the end of every cap stage
stage_ends = cumsum(first.(schedule))
stage_states = NamedTuple{(:step, :cap, :ψ), Tuple{Int, Int, DisorderKit.FiniteDisorderMPS}}[]
step = Ref(0)
function record_stage!(ψ, _)
    step[] += 1
    k = findfirst(==(step[]), stage_ends)
    isnothing(k) || push!(stage_states, (step = step[], cap = schedule[k].second, ψ = deepcopy(ψ)))
    return step[]
end

to = TimerOutput()
alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD = schedule, order, filter_tail_tol, convtol, maxiter, seed,
                                                  verbosity = 1, timer_output = to,
                                                  finalizer = DisorderKit.Finalizer(Int, record_stage!))
walltime = @elapsed ψ, _, info = DisorderKit.groundstate(ψ0, Hs, Δτ, alg)

nsteps = length(info.Es)
converged = !isempty(info.ϵsconv) && info.ϵsconv[end] <= convtol
info_nt = NamedTuple{fieldnames(typeof(info))}(Tuple(getfield(info, f) for f in fieldnames(typeof(info))))

gitcmd(args...) = try readchomp(Cmd(`git -C $(pkgdir(DisorderKit)) $args`)) catch; "unknown" end
meta = (; walltime, timer = TimerOutputs.todict(to), git_commit = gitcmd("rev-parse", "HEAD"),
        git_dirty = !isempty(gitcmd("status", "--porcelain", "--untracked-files=no")),
        julia_version = string(VERSION), hostname = gethostname(), threads = Threads.nthreads(),
        blas_threads = BLAS.get_num_threads(), date = string(now()))

# Write to a temporary file and move it in place, so an interrupted job never leaves a partial result
mkpath(folder)
tmp = filename * ".tmp"
jldsave(tmp; ψ, stage_states, info = info_nt, converged, nsteps, params, meta)
mv(tmp, filename; force = true)

println("Done: $nsteps steps, converged = $converged, walltime = $(round(walltime; digits = 1)) s")
println("Saved $filename")
show(to; allocations = false, compact = true); println()
