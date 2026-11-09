# One point (t, μ) of the Bose-Hubbard chain with correlated hopping and potential disorder, with disorder-normalized
# imaginary time evolution (Groundstate_FiniteNormalizedITE):
#   H_w = -Σ t_{wᵢ} (b†ᵢbᵢ₊₁ + h.c.) + U/2 Σ nᵢ(nᵢ-1) - Σ μ_{wᵢ} nᵢ,   t_{wᵢ} ∈ {t(1-Ω), t(1+Ω)}, μ_{wᵢ} ∈ {μ-Δ, μ+Δ},
# where the label wᵢ of site i sets both μᵢ and the hopping of bond (i, i+1). The hopping and the potential of a site are
# correlated by r ∈ (-1, 1): P = (1+r)/4 if both are high or both low, (1-r)/4 otherwise. The last site has no bond, so
# its sectors are merged into μ-Δ and μ+Δ with probability 1/2 each (DisorderKit.site_sectors).
# For Δ = 0 and/or Ω = 0 the identical sectors are merged: 2 sectors (p = 1/2) if one of them is 0, 1 if both are.
# H conserves the particle number, so the evolution starts from the uniform superposition of all occupations.
# Stores the final state and the per-step diagnostics; no observables, those are computed from the stored states by
# hpc/plot_BoseHubbardCorrelated_NormalizedITE.jl.
#
# Usage (from the repo root, normally through hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh):
#   julia --project=. hpc/BoseHubbardCorrelated_NormalizedITE.jl <L> <t> <μ>      run and store
#   julia hpc/BoseHubbardCorrelated_NormalizedITE.jl --path <L> <t> <μ>           only print the run tag and the output file
#   julia --project=. hpc/BoseHubbardCorrelated_NormalizedITE.jl --status <L> <t> <μ>   print the run tag, the output
#       file, the status (run, rerun or uptodate) and the reason for it; the sweep only submits run and rerun points
# The status compares the stored state (its final ϵ_conv and the maxiter and convtol it was run with) with the MAXITER
# and CONVTOL of the environment:
#   run       there is no (readable) stored state
#   uptodate  the stored state is converged for this job: final ϵ_conv ≤ CONVTOL
#   resume    it is not, and MAXITER > stored maxiter (and RESUME ≠ 0): the run continues the ITE from the stored state
#   rerun     it is not, and CONVTOL < stored convtol only (or RESUME = 0): the rerun starts from scratch
#   uptodate  it is not, but the same settings would give the same result
# Resume: MAXITER is the total number of steps, so a state stored after n steps continues for at most MAXITER - n steps;
# the per-step diagnostics (info) of both runs are concatenated and params.resumed_from = n (0 for a fresh run).
# t and μ (the mean values) are used verbatim in the file name (the sweep script passes them formatted as %.4f).
#
# Run parameters (environment, all required; their defaults are set in hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh):
#   U, R (correlation r), DELTA (potential disorder Δ), OMEGA (relative hopping disorder Ω), NMAX, DTAU, MAXD, MAXITER, CONVTOL, CROSS_WIDTH, DATA_DIR
# Optional: BLAS_THREADS (default: as many as Julia threads), RESUME (default 1: continue an unconverged stored state
#   when MAXITER > its maxiter, 0: always start from scratch)
#
# Output: <DATA_DIR>/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/t<t>/mu<μ>.jld2,
#   run tag U<U>_r<R>_Delta<DELTA>_Omega<OMEGA>_nmax<NMAX>_dtau<DTAU>_D<MAXD>

# ---- Parameters and paths (before loading DisorderKit, so --path and --status are fast)
path_only = !isempty(ARGS) && ARGS[1] == "--path"
status_only = !isempty(ARGS) && ARGS[1] == "--status"
args = path_only || status_only ? ARGS[2:end] : ARGS
length(args) == 3 || error("usage: julia --project=. hpc/BoseHubbardCorrelated_NormalizedITE.jl [--path | --status] <L> <t> <μ>")
L_str, t_str, μ_str = args
L, t, μ = parse(Int, L_str), parse(Float64, t_str), parse(Float64, μ_str)

param(k) = haskey(ENV, k) ? ENV[k] : error("environment variable $k not set; run parameters are set in hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh")
resume_enabled = get(ENV, "RESUME", "1") != "0"
U_str = param("U")
r_str = param("R")
Δ_str = param("DELTA")
Ω_str = param("OMEGA")
nmax = parse(Int, param("NMAX"))
dtau_str = param("DTAU")
maxD = parse(Int, param("MAXD"))
run_tag = "U$(U_str)_r$(r_str)_Delta$(Δ_str)_Omega$(Ω_str)_nmax$(nmax)_dtau$(dtau_str)_D$(maxD)"

data_root = param("DATA_DIR")
folder = joinpath(data_root, "BoseHubbardCorrelated_NormalizedITE", run_tag, "L$(L)", "t$(t_str)")
filename = joinpath(folder, "mu$(μ_str).jld2")

if path_only
    println(run_tag)
    println(filename)
    exit()
end

# Status of the stored state for this job's MAXITER and CONVTOL (see the header); JLD2 is only loaded if there is a
# state, and only params and info are read from it (not ψ)
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
    println(run_tag)
    println(filename)
    println(st)
    println(reason)
    exit()
end

# ---- Simulation
using TensorKit, DisorderKit, TimerOutputs, JLD2, Dates, LinearAlgebra, Logging
BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", string(Threads.nthreads()))))   # default: as many as Julia threads

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

U, r = parse(Float64, U_str), parse(Float64, r_str)
-1 < r < 1 || error("R = $r must lie in (-1, 1): SRC requires strictly positive disorder probabilities")
Δ, Ω = parse(Float64, Δ_str), parse(Float64, Ω_str)
Δ >= 0 || error("DELTA = $Δ must be ≥ 0")
0 <= Ω <= 1 || error("OMEGA = $Ω must lie in [0, 1], so that the hopping t(1-Ω) is not negative")
Δτ = parse(Float64, dtau_str)
maxiter = parse(Int, param("MAXITER"))
convtol = parse(Float64, param("CONVTOL"))
cross_width = parse(Int, param("CROSS_WIDTH"))
d = nmax + 1

# Joint probabilities P[a, b] of μ_a ∈ (μ-Δ, μ+Δ) and t_b ∈ (t(1-Ω), t(1+Ω)), correlated by r
P = [(1+r)/4 (1-r)/4; (1-r)/4 (1+r)/4]
# Identical values (Δ = 0 or Ω = 0) are merged into one with the marginal probability: 4 sectors for Δ, Ω > 0,
# 2 if one of them is 0 and 1 (clean chain) if both are
μs = iszero(Δ) ? [μ] : [μ-Δ, μ+Δ]
iszero(Δ) && (P = sum(P; dims = 1))
tvals = iszero(Ω) ? [t] : [t*(1-Ω), t*(1+Ω)]
iszero(Ω) && (P = sum(P; dims = 2))
# Sectors of random_bose_hubbard(ts, Us, μs) run over Iterators.product(μs, Us, ts), μ fastest (column-major order of P);
# for Δ, Ω > 0: 1 = (μ-Δ, t(1-Ω)), 2 = (μ+Δ, t(1-Ω)), 3 = (μ-Δ, t(1+Ω)), 4 = (μ+Δ, t(1+Ω))
Hs = DisorderKit.random_bose_hubbard(tvals, [U], μs; nmax)
ps = vec(P)

# Uniform superposition of 0..nmax bosons on every site and in every sector: every conditional state is normalized
function uniform_state(Hs, ps, L, d)
    W = TensorMap(fill(ComplexF64(1/sqrt(d)), d), ℂ^1 ⊗ ℂ^d ← ℂ^1)
    site_ps, sectors = DisorderKit.site_sectors(Hs, ps, L)
    return DisorderKit.FiniteDisorderMPS([[W for _ in p] for p in site_ps], site_ps, sectors)
end

# Initial state: the stored state if it can be continued (see the header), else the uniform superposition
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
    (sp.L == L && sp.nmax == nmax && sp.maxD == maxD && sp.Δτ == Δτ && sp.ps ≈ ps && sp.cross_width == cross_width) ||
        return nothing, "stored state has other parameters"
    ϵ = isempty(stored.info.ϵsconv) ? Inf : last(stored.info.ϵsconv)
    ϵ <= convtol && return nothing, "stored state is converged (ϵ = $ϵ)"
    sp.maxiter < maxiter && stored.nsteps < maxiter || return nothing, "stored state already has maxiter $(sp.maxiter) ≥ $maxiter"
    return stored, "unconverged after $(stored.nsteps) steps (ϵ = $(round(ϵ; sigdigits = 2)))"
end
stored, why = load_resume(filename)
n0 = stored === nothing ? 0 : stored.nsteps
ψ0 = stored === nothing ? uniform_state(Hs, ps, L, d) : stored.ψ
println(stored === nothing ? "Starting from the uniform state ($why)" : "Resuming from $filename: $why, at most $(maxiter - n0) more steps")

params = (; L, t, μ, U, r, Δ, Ω, nmax, ps, Δτ, maxD, maxiter, convtol, cross_width, resumed_from = n0)
println("Correlated Bose-Hubbard normalized ITE: L = $L, t = $t, μ = $μ, U = $U, r = $r, Δ = $Δ, Ω = $Ω, nmax = $nmax, maxD = $maxD, Δτ = $Δτ")
println("Output: $filename")

to = TimerOutput()
# A resumed run does the remaining maxiter - n0 steps, with other random streams than the steps already done
alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD, order = 2, convtol, maxiter = maxiter - n0, cross_width, verbosity = 1,
                                                  timer_output = to, seed = 20266001 + n0)
walltime = @elapsed ψ, _, info = DisorderKit.groundstate(ψ0, Hs, Δτ, alg)

info_nt = NamedTuple{fieldnames(typeof(info))}(Tuple(getfield(info, f) for f in fieldnames(typeof(info))))
# Resumed: the per-step diagnostics of the stored run followed by those of this one
stored === nothing || (info_nt = NamedTuple{keys(info_nt)}(Tuple(vcat(stored.info[k], info_nt[k]) for k in keys(info_nt))))
nsteps = length(info_nt.Es)
converged = !isempty(info_nt.ϵsconv) && info_nt.ϵsconv[end] <= convtol

gitcmd(args...) = try readchomp(Cmd(`git -C $(pkgdir(DisorderKit)) $args`)) catch; "unknown" end
meta = (; walltime, timer = TimerOutputs.todict(to), git_commit = gitcmd("rev-parse", "HEAD"),
        git_dirty = !isempty(gitcmd("status", "--porcelain", "--untracked-files=no")),
        julia_version = string(VERSION), hostname = gethostname(), threads = Threads.nthreads(),
        blas_threads = BLAS.get_num_threads(), date = string(now()),
        previous = stored === nothing ? nothing : stored.meta)   # meta of the run this one continues

# Write to a temporary file and move it in place, so an interrupted job never leaves a partial result
mkpath(folder)
tmp = filename * ".tmp"
jldsave(tmp; ψ, info = info_nt, converged, nsteps, params, meta)
mv(tmp, filename; force = true)

println("Done: $nsteps steps$(n0 > 0 ? " ($n0 stored + $(nsteps - n0) new)" : ""), converged = $converged, walltime = $(round(walltime; digits = 1)) s")
println("Saved $filename")
show(to; allocations = false, compact = true); println()
