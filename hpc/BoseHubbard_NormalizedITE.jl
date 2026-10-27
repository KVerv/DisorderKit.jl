# One chemical potential μ of the disordered Bose-Hubbard chain with disorder-normalized imaginary time evolution
# (Groundstate_FiniteNormalizedITE), as in examples/FiniteDisorderBoseHubbard.jl:
#   H_w = -Σ t_{wᵢ} (b†ᵢbᵢ₊₁ + h.c.) + U/2 Σ nᵢ(nᵢ-1) - μ Σ nᵢ,   t_{wᵢ} ∈ {t, t/3} with probability 1/2 each.
# H conserves the particle number, so the evolution starts from the uniform superposition of all occupations.
# Stores the final state and the per-step diagnostics; no observables, those are computed from the stored states by
# hpc/plot_BoseHubbard_NormalizedITE.jl.
#
# Usage (from the repo root):
#   julia --project=. hpc/BoseHubbard_NormalizedITE.jl <L> <μ>         run and store
#   julia hpc/BoseHubbard_NormalizedITE.jl --path <L> <μ>              only print the run tag and the output file
# μ is used verbatim in the file name (the sweep script passes it formatted as %.4f).
#
# Run parameters (environment, with defaults):
#   T=0.16 (strong hopping, the weak bonds have t/3)  U=1.0  NMAX=4  DTAU=0.05  MAXD=16
#   MAXITER=400  CONVTOL=1e-8  CROSS_WIDTH=50  DATA_DIR (default $VSC_DATA/data, else ./data)
#
# Output: <DATA_DIR>/BoseHubbard_NormalizedITE/<run tag>/L<L>/mu<μ>.jld2, run tag t<T>_U<U>_nmax<NMAX>_dtau<DTAU>_D<MAXD>

# ---- Parameters and paths (before loading DisorderKit, so --path is fast)
path_only = !isempty(ARGS) && ARGS[1] == "--path"
args = path_only ? ARGS[2:end] : ARGS
length(args) == 2 || error("usage: julia --project=. hpc/BoseHubbard_NormalizedITE.jl [--path] <L> <μ>")
L_str, μ_str = args
L, μ = parse(Int, L_str), parse(Float64, μ_str)

t_str = get(ENV, "T", "0.16")
U_str = get(ENV, "U", "1.0")
nmax = parse(Int, get(ENV, "NMAX", "4"))
dtau_str = get(ENV, "DTAU", "0.05")
maxD = parse(Int, get(ENV, "MAXD", "16"))
run_tag = "t$(t_str)_U$(U_str)_nmax$(nmax)_dtau$(dtau_str)_D$(maxD)"

data_root = get(ENV, "DATA_DIR", joinpath(get(ENV, "VSC_DATA", "."), "data"))
folder = joinpath(data_root, "BoseHubbard_NormalizedITE", run_tag, "L$(L)")
filename = joinpath(folder, "mu$(μ_str).jld2")

if path_only
    println(run_tag)
    println(filename)
    exit()
end

# ---- Simulation
using TensorKit, DisorderKit, TimerOutputs, JLD2, Dates, LinearAlgebra
BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", string(Threads.nthreads()))))   # default: as many as Julia threads

t, U = parse(Float64, t_str), parse(Float64, U_str)
Δτ = parse(Float64, dtau_str)
maxiter = parse(Int, get(ENV, "MAXITER", "500"))
convtol = parse(Float64, get(ENV, "CONVTOL", "1e-8"))
cross_width = parse(Int, get(ENV, "CROSS_WIDTH", "32"))
ps = [0.5, 0.5]
d = nmax + 1

# Uniform superposition of 0..nmax bosons on every site and in every sector: every conditional state is normalized
function uniform_state(ps, L, d)
    W = TensorMap(fill(ComplexF64(1/sqrt(d)), d), ℂ^1 ⊗ ℂ^d ← ℂ^1)
    return DisorderKit.FiniteDisorderMPS([[W for _ in ps] for _ in 1:L], ps)
end

Hs = DisorderKit.random_bose_hubbard([t, t/3], [U], [μ]; nmax)
ψ0 = uniform_state(ps, L, d)

params = (; L, μ, t, U, nmax, ps, Δτ, maxD, maxiter, convtol, cross_width)
println("Bose-Hubbard normalized ITE: L = $L, μ = $μ, t = $t, U = $U, nmax = $nmax, maxD = $maxD, Δτ = $Δτ")
println("Output: $filename")

to = TimerOutput()
alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD, order = 2, convtol, maxiter, cross_width, verbosity = 1,
                                                  timer_output = to)
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
jldsave(tmp; ψ, info = info_nt, converged, nsteps, params, meta)
mv(tmp, filename; force = true)

println("Done: $nsteps steps, converged = $converged, walltime = $(round(walltime; digits = 1)) s")
println("Saved $filename")
show(to; allocations = false, compact = true); println()
