# HPC scripts: RTFIM phase diagram with normalized ITE

Ground states of the random transverse-field Ising chain on an open chain of `L` sites, obtained with disorder-normalized
imaginary time evolution (`Groundstate_FiniteNormalizedITE`), for a grid of disorder distributions

    h ∈ e^δ {1, e^{-2W}},   J ∈ {1, e^{-2W}},   p_u = 1/4,

with δ the distance to criticality and W the disorder strength. The jobs store states and diagnostics only; observables
are computed from the stored states when plotting.

## Scripts

| script | what it does |
|---|---|
| `RTFIM_NormalizedITE.jl` | one point (L, δ, W): runs the ITE and stores the final state, the state at the end of every cap stage and the per-step diagnostics |
| `sweep_RTFIM_NormalizedITE.sh` | generates and submits one PBS job per (δ, W) for a given L, skipping points that already have a result |
| `benchmark_RTFIM_NormalizedITE.jl` / `.sh` | measures time per step, peak memory and thread scaling for a given L, and recommends WALLTIME, MEM and PPN |

## Usage (from the repo root, on the login node)

Every job of both `.sh` scripts first runs `Pkg.instantiate(); Pkg.precompile()` on its compute node, so dependencies
added after a `git pull` are installed before the computation starts.

    DRYRUN=1 bash hpc/sweep_RTFIM_NormalizedITE.sh 16      # show what would be submitted
    bash hpc/sweep_RTFIM_NormalizedITE.sh 16               # submit the default grid for L = 16
    DELTAS="-0.5 0 0.5" WS="0.5 1" bash hpc/sweep_RTFIM_NormalizedITE.sh 16
    FORCE=1 bash hpc/sweep_RTFIM_NormalizedITE.sh 16       # also rerun points that already have a result

    # a single point, without the scheduler (QUICK=1: a few steps only; instantiate first after a git pull)
    ~/.juliaup/bin/julia +1.12.6 --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
    QUICK=1 julia --project=. hpc/RTFIM_NormalizedITE.jl 6 0.0000 0.1000

Run parameters (environment): `DTAU` (0.02), `ORDER` (2), `DMAX_SCHEDULE` ("300:8,100:16,100:32", steps:cap),
`MAXITER` (total steps of the schedule), `CONVTOL` (1e-8), `FILTER_TAIL_TOL` (1e-8), `SEED` (20266001), `QUICK`,
`DATA_DIR` ($VSC_DATA/data). Job resources: `WALLTIME` (24:00:00), `MEM` (16GB), `PPN` (4, also the number of Julia
threads), `CLUSTER` (skiddo), `JULIA` (`~/.juliaup/bin/julia +1.12.6`). Grid: `DELTAS`, `WS`.

## Choosing resources

Run the benchmark once per L (and whenever the cap schedule changes), before the sweep:

    bash hpc/benchmark_RTFIM_NormalizedITE.sh 16           # one PBS job, threads 1 2 4 8 16 32
    LOCAL=1 bash hpc/benchmark_RTFIM_NormalizedITE.sh 16   # or directly, e.g. on an interactive node
    julia hpc/benchmark_RTFIM_NormalizedITE.jl --summary 16   # reprint the summary

For every thread count it runs `BENCH_STEPS` (8) ITE steps at every cap of `DMAX_SCHEDULE` near criticality (`DELTA`
= 0, `W` = 0.1) and records the time per step once the bonds have saturated and the peak memory. The summary prints:

- the time per step per cap and thread count;
- the estimated full run time T = warm-up + Σ_stages steps × time per step (an upper bound, since a run stops when it
  converges), with the speedup and parallel efficiency of the steps;
- a recommendation: `PPN` = the largest thread count with efficiency ≥ `EFF_MIN` (0.5), `WALLTIME` = `SAFETY` (1.5) × T,
  `MEM` = `SAFETY` × peak memory, and the matching sweep command.

Every job simulates one point on one node: the nodes per job is always 1, and `PPN` sets the number of Julia and BLAS
threads (`--threads=$PPN`, BLAS is set to the same number so jobs never oversubscribe a shared node). More points run in
parallel as separate jobs, so a small `PPN` with good efficiency is usually best. The benchmark results are stored in
`$DATA_DIR/RTFIM_NormalizedITE/benchmarks/<run tag>/L<L>/threads<n>.csv`.

## Data layout

    $DATA_DIR/RTFIM_NormalizedITE/
      dtau0.02_order2_D300x8-100x16-100x32/      run tag: Δτ, MPO order and cap schedule
        L16/
          W0.1000/
            delta-1.0000.jld2
            delta-0.7500.jld2
            ...
          W0.5000/
            ...
      benchmarks/<run tag>/L<L>/threads<n>.csv   benchmark results
    logs/RTFIM_NormalizedITE/<run tag>/L<L>/     PBS stdout/stderr of every job
    logs/RTFIM_NormalizedITE/benchmarks/<run tag>/L<L>/

Results are written to `<file>.tmp` and moved in place at the end, so a file that exists is always complete.

## Contents of a result file (load with `using DisorderKit, JLD2; d = load(file)`)

| key | content |
|---|---|
| `ψ` | final `FiniteDisorderMPS` |
| `stage_states` | `(step, cap, ψ)` at the end of every completed cap stage (accuracy vs D_max from one run) |
| `info` | per-step diagnostics (NamedTuple of arrays): `Es`, `ϵsconv`, `caps`, `bond_dimensions`, `partition_ranks`, `partition_met`, `filter_ranks`, `filter_met`, `src_ranks`, `src_cap_unmet`, `rounding_errors`, `residual_rms`, `residual_max`, `cross_max_phase`, `cross_negative_real` |
| `converged`, `nsteps` | whether `ϵ_conv ≤ convtol` was reached, and the number of steps |
| `params` | `L, δ, W, hs, Js, ps, Δτ, order, schedule, maxiter, convtol, filter_tail_tol, seed` (rebuild the model with `random_transverse_field_ising(params.Js, params.hs)`) |
| `meta` | wall time, timings (`TimerOutput` as a Dict), git commit and dirty flag, Julia version, host, threads, date |
