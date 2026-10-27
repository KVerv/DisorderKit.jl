# Benchmark of RTFIM_NormalizedITE.jl to choose the job resources (WALLTIME, MEM, PPN) of the phase-diagram sweep.
#
# Run mode, once per split of nj Julia threads and nb BLAS threads (BLAS_THREADS, default nj):
#   BLAS_THREADS=<nb> julia --project=. --threads=<nj> hpc/benchmark_RTFIM_NormalizedITE.jl <L>
#   For every distinct cap of the schedule it runs BENCH_STEPS ITE steps from the initial state (never stopping early) and
#   records the time per step once the bonds have saturated, the bond dimension reached and the peak memory (maxrss).
#   Output: <DATA_DIR>/RTFIM_NormalizedITE/benchmarks/<run tag>/L<L>/threads<nj>_blas<nb>.csv
#
# Summary mode, after the runs:
#   julia hpc/benchmark_RTFIM_NormalizedITE.jl --summary <L>
#   Estimates the full run time T = warm-up + Σ_stages steps × (time per step at the stage cap) for every split, with
#   speedup (relative to 1 Julia and 1 BLAS thread) and parallel efficiency (speedup / cores, cores = max(nj, nb)). It
#   recommends the split with the most cores among those with efficiency ≥ EFF_MIN (the fastest one for that number of
#   cores): PPN = cores, JULIA_THREADS = nj, BLAS_THREADS = nb, WALLTIME = SAFETY × T and MEM = SAFETY × peak memory.
#   T is an upper bound: a production run stops early when it converges. Old threads<n>.csv files count as (n, n).
#
# Environment: DTAU, ORDER, DMAX_SCHEDULE (as in RTFIM_NormalizedITE.jl), DELTA=0.0, W=0.1 (near criticality),
#   BENCH_STEPS=8, DATA_DIR, EFF_MIN=0.5, SAFETY=1.5, NODE_CORES=96

using Printf

summary_mode = !isempty(ARGS) && ARGS[1] == "--summary"
args = summary_mode ? ARGS[2:end] : ARGS
length(args) == 1 || error("usage: julia --project=. [--threads=n] hpc/benchmark_RTFIM_NormalizedITE.jl [--summary] <L>")
L = parse(Int, args[1])

# Same run tag as RTFIM_NormalizedITE.jl
dtau_str = get(ENV, "DTAU", "0.02")
order = parse(Int, get(ENV, "ORDER", "2"))
schedule_str = get(ENV, "DMAX_SCHEDULE", "100:8,100:16,100:32")
schedule = map(split(schedule_str, ",")) do stage
    steps, cap = parse.(Int, split(stage, ":"))
    steps => cap
end
run_tag = "dtau$(dtau_str)_order$(order)_D" * join(["$(s.first)x$(s.second)" for s in schedule], "-")
data_root = get(ENV, "DATA_DIR", joinpath(get(ENV, "VSC_DATA", "."), "data"))
folder = joinpath(data_root, "RTFIM_NormalizedITE", "benchmarks", run_tag, "L$(L)")

if summary_mode
    eff_min = parse(Float64, get(ENV, "EFF_MIN", "0.5"))
    safety = parse(Float64, get(ENV, "SAFETY", "1.5"))
    node_cores = parse(Int, get(ENV, "NODE_CORES", "96"))

    # New threads<nj>_blas<nb>.csv files first, so they win over an old threads<n>.csv of the same split
    files = sort(filter(f -> occursin(r"^threads\d+(_blas\d+)?\.csv$", f), readdir(folder)); by = f -> !occursin("_blas", f))
    isempty(files) && error("no benchmark results in $folder")
    # CSV columns: cap, steps, sec_per_step, max_bond, maxrss_GB, warmup_s
    results = map(files) do f
        m = match(r"^threads(\d+)(?:_blas(\d+))?\.csv$", f)
        nj = parse(Int, m[1])
        nb = isnothing(m[2]) ? nj : parse(Int, m[2])
        rows = [parse.(Float64, split(line, ",")) for line in readlines(joinpath(folder, f))[2:end]]
        (nj, nb) => Dict(Int(r[1]) => (sec = r[3], bond = Int(r[4]), rss = r[5], warmup = r[6]) for r in rows)
    end
    unique!(first, results)

    # Splits: both (n Julia, n BLAS), Julia only (n, 1) and BLAS only (1, n); (1, 1) belongs to both
    mode(k) = k[1] == k[2] ? "both" : k[2] == 1 ? "julia" : k[1] == 1 ? "blas" : "mixed"
    cores(k) = max(k...)
    sort!(results; by = r -> (findfirst(==(mode(r.first)), ["both", "julia", "blas", "mixed"]), cores(r.first), r.first))

    # Steps only (what the threads speed up) and the full run including the one-time warm-up (compilation, set-up)
    steps_time(rows) = sum(s.first * rows[s.second].sec for s in schedule)
    Tsteps = Dict(k => steps_time(rows) for (k, rows) in results)
    T = Dict(k => first(values(rows)).warmup + Tsteps[k] for (k, rows) in results)
    rss = Dict(k => maximum(r.rss for r in values(rows)) for (k, rows) in results)
    ref = haskey(Tsteps, (1, 1)) ? (1, 1) : argmin(k -> (cores(k), Tsteps[k]), keys(Tsteps))
    speedup(k) = Tsteps[ref] * cores(ref) / Tsteps[k]
    efficiency(k) = speedup(k) / cores(k)

    println("Benchmark: L = $L, run tag $run_tag ($folder)")
    println("Splits: both = n Julia and n BLAS threads, julia = n Julia threads and 1 BLAS thread, blas = 1 Julia thread ",
            "and n BLAS threads")
    println("Time per step [s] per cap:")
    caps = sort(unique(s.second for s in schedule))
    @printf("  %6s  %6s  %6s", "split", "julia", "blas"); foreach(c -> @printf("  %10s", "D=$c"), caps); println()
    for (k, rows) in results
        @printf("  %6s  %6d  %6d", mode(k), k...); foreach(c -> @printf("  %10.3f", rows[c].sec), caps); println()
    end
    println("Full run (schedule $schedule_str, $(sum(first, schedule)) steps), upper bound; speedup and efficiency of the ",
            "steps relative to $(ref[1]) Julia and $(ref[2]) BLAS threads:")
    @printf("  %6s  %6s  %6s  %12s  %12s  %8s  %10s  %10s\n", "split", "julia", "blas", "steps [h]", "total [h]", "speedup",
            "efficiency", "peak [GB]")
    for (k, _) in results
        @printf("  %6s  %6d  %6d  %12.2f  %12.2f  %8.2f  %10.2f  %10.2f\n", mode(k), k..., Tsteps[k]/3600, T[k]/3600,
                speedup(k), efficiency(k), rss[k])
    end

    ok = [k for (k, _) in results if efficiency(k) >= eff_min]
    isempty(ok) && (ok = [ref])
    ppn = maximum(cores, ok)
    best = argmin(k -> Tsteps[k], filter(k -> cores(k) == ppn, ok))
    walltime_h = max(1, ceil(Int, safety * T[best] / 3600))
    mem_gb = max(1, ceil(Int, safety * rss[best]))
    println("Recommendation (efficiency ≥ $eff_min, safety factor $safety): nodes = 1 (one point per job), PPN = $ppn ",
            "($(best[1]) Julia, $(best[2]) BLAS threads), WALLTIME = $(walltime_h):00:00, MEM = $(mem_gb)GB")
    println("  $(div(node_cores, ppn)) jobs fit on a $node_cores-core node")
    println("  WALLTIME=$(walltime_h):00:00 MEM=$(mem_gb)GB PPN=$ppn JULIA_THREADS=$(best[1]) BLAS_THREADS=$(best[2]) ",
            "bash hpc/sweep_RTFIM_NormalizedITE.sh $L")
    exit()
end

# ---- Run mode
using TensorKit, DisorderKit, LinearAlgebra, Statistics
nthreads = Threads.nthreads()
nblas = parse(Int, get(ENV, "BLAS_THREADS", string(nthreads)))
BLAS.set_num_threads(nblas)
δ = parse(Float64, get(ENV, "DELTA", "0.0"))
W = parse(Float64, get(ENV, "W", "0.1"))
Δτ = parse(Float64, dtau_str)
bench_steps = parse(Int, get(ENV, "BENCH_STEPS", "8"))
bench_steps >= 2 || error("BENCH_STEPS must be at least 2")

hs = exp(δ) .* [1.0, exp(-2W)]
Js = [1.0, exp(-2W)]
ps = fill(0.25, 4)
Hs = DisorderKit.random_transverse_field_ising(Js, hs)
ψ0 = DisorderKit.plus_state(ps, L)
println("Benchmark: L = $L, δ = $δ, W = $W, $nthreads Julia threads, $(BLAS.get_num_threads()) BLAS threads")

# Runs nsteps ITE steps at a fixed cap and returns the time stamps after every step and the final state
function timed_run(cap, nsteps)
    stamps = Float64[]
    record!(_, _) = (push!(stamps, time()); length(stamps))
    alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD = cap, order, maxiter = nsteps, convtol = 0.0,
                                                      finalizer = DisorderKit.Finalizer(Int, record!))
    ψ, _, _ = DisorderKit.groundstate(ψ0, Hs, Δτ, alg)
    return stamps, ψ
end

# Warm-up: compilation and set-up, counted once as overhead of a job
warmup = @elapsed timed_run(minimum(last, schedule), 1)
println("warm-up (compilation, set-up): $(round(warmup; digits = 1)) s")

mkpath(folder)
file = joinpath(folder, "threads$(nthreads)_blas$(nblas).csv")
open(file, "w") do io
    println(io, "cap,steps,sec_per_step,max_bond,maxrss_GB,warmup_s")
    for cap in sort(unique(last.(schedule)))
        stamps, ψ = timed_run(cap, bench_steps)
        dt = diff(stamps)
        sec = median(dt[cld(length(dt), 2):end])            # second half: bonds saturated at the cap
        bond = maximum(DisorderKit.bond_dimensions(ψ))
        rss = Sys.maxrss() / 2^30
        println(io, "$cap,$bench_steps,$sec,$bond,$rss,$warmup")
        println("cap $cap: $(round(sec; digits = 3)) s/step, max bond $bond, peak memory $(round(rss; digits = 2)) GB")
    end
end
println("Saved $file")
