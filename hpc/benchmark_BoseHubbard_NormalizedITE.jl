# Benchmark of BoseHubbard_NormalizedITE.jl to choose the job resources (WALLTIME, MEM, PPN) of the μ sweep.
#
# Run mode, once per split of nj Julia threads and nb BLAS threads (BLAS_THREADS, default nj):
#   BLAS_THREADS=<nb> julia --project=. --threads=<nj> hpc/benchmark_BoseHubbard_NormalizedITE.jl <L>
#   Runs BENCH_STEPS ITE steps at the cap MAXD from the initial state (never stopping early) at the chemical potential MU
#   and records the time per step once the bonds have saturated, the bond dimension reached and the peak memory (maxrss).
#   Output: <DATA_DIR>/BoseHubbard_NormalizedITE/benchmarks/<run tag>/L<L>/threads<nj>_blas<nb>.csv
#
# Summary mode, after the runs:
#   julia hpc/benchmark_BoseHubbard_NormalizedITE.jl --summary <L>
#   Estimates the full run time T = warm-up + MAXITER × (time per step) for every split, with speedup (relative to 1 Julia
#   and 1 BLAS thread) and parallel efficiency (speedup / cores, cores = max(nj, nb)). It recommends the split with the
#   most cores among those with efficiency ≥ EFF_MIN (the fastest one for that number of cores): PPN = cores,
#   JULIA_THREADS = nj, BLAS_THREADS = nb, WALLTIME = SAFETY × T and MEM = SAFETY × peak memory.
#   T is an upper bound: a production run stops early when it converges.
#
# Environment (all required, defaults set in hpc/benchmark_BoseHubbard_NormalizedITE.sh): T, U, NMAX, DTAU, MAXD,
#   MAXITER, DATA_DIR (as in hpc/BoseHubbard_NormalizedITE.jl, so the run tag matches); run mode also CROSS_WIDTH, MU,
#   BENCH_STEPS. Optional: EFF_MIN=0.5, SAFETY=1.5, NODE_CORES=96 (summary), BLAS_THREADS (run)

using Printf

summary_mode = !isempty(ARGS) && ARGS[1] == "--summary"
args = summary_mode ? ARGS[2:end] : ARGS
length(args) == 1 || error("usage: julia --project=. [--threads=n] hpc/benchmark_BoseHubbard_NormalizedITE.jl [--summary] <L>")
L = parse(Int, args[1])

# Same run tag as BoseHubbard_NormalizedITE.jl
param(k) = haskey(ENV, k) ? ENV[k] : error("environment variable $k not set; run parameters are set in hpc/benchmark_BoseHubbard_NormalizedITE.sh")
t_str = param("T")
U_str = param("U")
nmax = parse(Int, param("NMAX"))
dtau_str = param("DTAU")
maxD = parse(Int, param("MAXD"))
maxiter = parse(Int, param("MAXITER"))
run_tag = "t$(t_str)_U$(U_str)_nmax$(nmax)_dtau$(dtau_str)_D$(maxD)"
folder = joinpath(param("DATA_DIR"), "BoseHubbard_NormalizedITE", "benchmarks", run_tag, "L$(L)")

if summary_mode
    eff_min = parse(Float64, get(ENV, "EFF_MIN", "0.5"))
    safety = parse(Float64, get(ENV, "SAFETY", "1.5"))
    node_cores = parse(Int, get(ENV, "NODE_CORES", "96"))

    files = filter(f -> occursin(r"^threads\d+_blas\d+\.csv$", f), readdir(folder))
    isempty(files) && error("no benchmark results in $folder")
    # CSV columns: maxD, steps, sec_per_step, max_bond, maxrss_GB, warmup_s (one row)
    results = map(files) do f
        m = match(r"^threads(\d+)_blas(\d+)\.csv$", f)
        r = parse.(Float64, split(readlines(joinpath(folder, f))[2], ","))
        (parse(Int, m[1]), parse(Int, m[2])) => (sec = r[3], bond = Int(r[4]), rss = r[5], warmup = r[6])
    end

    # Splits: both (n Julia, n BLAS), Julia only (n, 1) and BLAS only (1, n); (1, 1) belongs to both
    mode(k) = k[1] == k[2] ? "both" : k[2] == 1 ? "julia" : k[1] == 1 ? "blas" : "mixed"
    cores(k) = max(k...)
    sort!(results; by = r -> (findfirst(==(mode(r.first)), ["both", "julia", "blas", "mixed"]), cores(r.first), r.first))

    # Steps only (what the threads speed up) and the full run including the one-time warm-up (compilation, set-up)
    Tsteps = Dict(k => maxiter * r.sec for (k, r) in results)
    T = Dict(k => r.warmup + Tsteps[k] for (k, r) in results)
    rss = Dict(k => r.rss for (k, r) in results)
    ref = haskey(Tsteps, (1, 1)) ? (1, 1) : argmin(k -> (cores(k), Tsteps[k]), keys(Tsteps))
    speedup(k) = Tsteps[ref] * cores(ref) / Tsteps[k]
    efficiency(k) = speedup(k) / cores(k)

    println("Benchmark: L = $L, run tag $run_tag ($folder)")
    println("Splits: both = n Julia and n BLAS threads, julia = n Julia threads and 1 BLAS thread, blas = 1 Julia thread ",
            "and n BLAS threads")
    println("Full run (MAXITER = $maxiter steps at D = $maxD), upper bound; speedup and efficiency of the steps relative to ",
            "$(ref[1]) Julia and $(ref[2]) BLAS threads:")
    @printf("  %6s  %6s  %6s  %10s  %10s  %10s  %8s  %10s  %10s  %8s\n", "split", "julia", "blas", "s/step", "steps [h]",
            "total [h]", "speedup", "efficiency", "peak [GB]", "max bond")
    for (k, r) in results
        @printf("  %6s  %6d  %6d  %10.4f  %10.3f  %10.3f  %8.2f  %10.2f  %10.2f  %8d\n", mode(k), k..., r.sec,
                Tsteps[k]/3600, T[k]/3600, speedup(k), efficiency(k), rss[k], r.bond)
    end
    maximum(r.bond for (_, r) in results) < maxD &&
        println("Warning: the bonds did not reach MAXD = $maxD at the benchmarked μ, the time per step may be underestimated")

    ok = [k for (k, _) in results if efficiency(k) >= eff_min]
    isempty(ok) && (ok = [ref])
    ppn = maximum(cores, ok)
    best = argmin(k -> Tsteps[k], filter(k -> cores(k) == ppn, ok))
    walltime_min = max(15, 15 * ceil(Int, safety * T[best] / 900))   # rounded up to 15 minutes
    walltime = @sprintf("%d:%02d:00", div(walltime_min, 60), mod(walltime_min, 60))
    mem_gb = max(1, ceil(Int, safety * rss[best]))
    println("Recommendation (efficiency ≥ $eff_min, safety factor $safety): nodes = 1 (one μ per job), PPN = $ppn ",
            "($(best[1]) Julia, $(best[2]) BLAS threads), WALLTIME = $walltime, MEM = $(mem_gb)GB")
    println("  $(div(node_cores, ppn)) jobs fit on a $node_cores-core node")
    println("  WALLTIME=$walltime MEM=$(mem_gb)GB PPN=$ppn JULIA_THREADS=$(best[1]) BLAS_THREADS=$(best[2]) ",
            "bash hpc/sweep_BoseHubbard_NormalizedITE.sh $L")
    exit()
end

# ---- Run mode
using TensorKit, DisorderKit, LinearAlgebra, Statistics
nthreads = Threads.nthreads()
nblas = parse(Int, get(ENV, "BLAS_THREADS", string(nthreads)))
BLAS.set_num_threads(nblas)
t, U = parse(Float64, t_str), parse(Float64, U_str)
μ = parse(Float64, param("MU"))
Δτ = parse(Float64, dtau_str)
cross_width = parse(Int, param("CROSS_WIDTH"))
bench_steps = parse(Int, param("BENCH_STEPS"))
bench_steps >= 2 || error("BENCH_STEPS must be at least 2")
ps = [0.5, 0.5]
d = nmax + 1

# Uniform superposition of 0..nmax bosons on every site and in every sector, as in BoseHubbard_NormalizedITE.jl
function uniform_state(ps, L, d)
    W = TensorMap(fill(ComplexF64(1/sqrt(d)), d), ℂ^1 ⊗ ℂ^d ← ℂ^1)
    return DisorderKit.FiniteDisorderMPS([[W for _ in ps] for _ in 1:L], ps)
end

Hs = DisorderKit.random_bose_hubbard([t, t/3], [U], [μ]; nmax)
ψ0 = uniform_state(ps, L, d)
println("Benchmark: L = $L, μ = $μ, t = $t, U = $U, nmax = $nmax, maxD = $maxD, $nthreads Julia threads, ",
        "$(BLAS.get_num_threads()) BLAS threads")

# Runs nsteps ITE steps and returns the time stamps after every step and the final state
function timed_run(nsteps)
    stamps = Float64[]
    record!(_, _) = (push!(stamps, time()); length(stamps))
    alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD, order = 2, cross_width, maxiter = nsteps, convtol = 0.0,
                                                      finalizer = DisorderKit.Finalizer(Int, record!))
    ψ, _, _ = DisorderKit.groundstate(ψ0, Hs, Δτ, alg)
    return stamps, ψ
end

# Warm-up: compilation and set-up, counted once as overhead of a job
warmup = @elapsed timed_run(1)
println("warm-up (compilation, set-up): $(round(warmup; digits = 1)) s")

stamps, ψ = timed_run(bench_steps)
dt = diff(stamps)
sec = median(dt[cld(length(dt), 2):end])            # second half: bonds saturated at the cap
bond = maximum(DisorderKit.bond_dimensions(ψ))
rss = Sys.maxrss() / 2^30

mkpath(folder)
file = joinpath(folder, "threads$(nthreads)_blas$(nblas).csv")
open(file, "w") do io
    println(io, "maxD,steps,sec_per_step,max_bond,maxrss_GB,warmup_s")
    println(io, "$maxD,$bench_steps,$sec,$bond,$rss,$warmup")
end
println("D = $maxD: $(round(sec; digits = 4)) s/step, max bond $bond, peak memory $(round(rss; digits = 2)) GB")
println("Saved $file")
