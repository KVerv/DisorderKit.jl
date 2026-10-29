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
| `benchmark_RTFIM_NormalizedITE.jl` / `.sh` | measures time per step, peak memory and thread scaling for a given L, and recommends WALLTIME, MEM, PPN and the split into Julia and BLAS threads |

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
`DATA_DIR` ($VSC_DATA/data). Job resources: `WALLTIME` (24:00:00), `MEM` (16GB), `PPN` (4), `JULIA_THREADS` and
`BLAS_THREADS` (both `PPN` by default), `CLUSTER` (skiddo), `JULIA` (`~/.juliaup/bin/julia +1.12.6`). Grid: `DELTAS`, `WS`.

## Choosing resources

Run the benchmark once per L (and whenever the cap schedule changes), before the sweep:

    bash hpc/benchmark_RTFIM_NormalizedITE.sh 16           # one PBS job, threads 1 2 4 8 16 32
    MODES="both blas" bash hpc/benchmark_RTFIM_NormalizedITE.sh 16   # only some of the thread splits
    LOCAL=1 bash hpc/benchmark_RTFIM_NormalizedITE.sh 16   # or directly, e.g. on an interactive node
    julia hpc/benchmark_RTFIM_NormalizedITE.jl --summary 16   # reprint the summary

For every thread count n it runs three splits (`MODES`, default all): `both` = n Julia and n BLAS threads, `julia` =
n Julia threads and 1 BLAS thread, `blas` = 1 Julia thread and n BLAS threads. Julia threads parallelize the loops over
disorder words, BLAS threads the dense linear algebra inside one word. Every split runs `BENCH_STEPS` (8) ITE steps at every cap of `DMAX_SCHEDULE` near criticality (`DELTA`
= 0, `W` = 0.1) and records the time per step once the bonds have saturated and the peak memory. The summary prints:

- the time per step per cap for every split;
- the estimated full run time T = warm-up + Σ_stages steps × time per step (an upper bound, since a run stops when it
  converges), with the speedup (relative to 1 Julia and 1 BLAS thread) and parallel efficiency (speedup / cores, cores =
  max(Julia, BLAS threads)) of the steps;
- a recommendation: `PPN` = the most cores among the splits with efficiency ≥ `EFF_MIN` (0.5), the fastest split for
  that number of cores (`JULIA_THREADS`, `BLAS_THREADS`), `WALLTIME` = `SAFETY` (1.5) × T, `MEM` = `SAFETY` × peak
  memory, and the matching sweep command.

Every job simulates one point on one node: the nodes per job is always 1, and `PPN` sets the cores per job; `JULIA_THREADS`
(`--threads`) and `BLAS_THREADS` should not exceed it, so jobs never oversubscribe a shared node. More points run in
parallel as separate jobs, so a small `PPN` with good efficiency is usually best. The benchmark results are stored in
`$DATA_DIR/RTFIM_NormalizedITE/benchmarks/<run tag>/L<L>/threads<nj>_blas<nb>.csv` (older `threads<n>.csv` files are
read as n Julia and n BLAS threads).

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
      benchmarks/<run tag>/L<L>/threads<nj>_blas<nb>.csv   benchmark results
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

# HPC scripts: disordered Bose-Hubbard chain with normalized ITE

The HPC version of `examples/FiniteDisorderBoseHubbard.jl`: ground states of

    H_w = -Σ t_{wᵢ} (b†ᵢbᵢ₊₁ + h.c.) + U/2 Σ nᵢ(nᵢ-1) - μ Σ nᵢ,   t_{wᵢ} ∈ {t, t/3} with probability 1/2 each,

on an open chain of `L` sites, one job per chemical potential μ. Every job starts from the uniform superposition of all
occupations (H conserves the particle number) and stores the final state and the per-step diagnostics only; the
observables are computed afterwards by the plot script.

| script | what it does |
|---|---|
| `BoseHubbard_NormalizedITE.jl` | one point (L, μ): runs the ITE and stores the final state and the per-step diagnostics |
| `sweep_BoseHubbard_NormalizedITE.sh` | submits one PBS job per μ for a given L, skipping points that already have a result |
| `benchmark_BoseHubbard_NormalizedITE.jl` / `.sh` | measures time per step, peak memory and thread scaling for a given L, and recommends WALLTIME, MEM, PPN and the split into Julia and BLAS threads |
| `plot_BoseHubbard_NormalizedITE.jl` | computes ρ, κ = dρ/dμ and \|⟨a⟩\|² from the stored states, prints and saves them as a CSV, and plots them against μ/U |

## Usage (from the repo root, on the login node)

    DRYRUN=1 bash hpc/sweep_BoseHubbard_NormalizedITE.sh 6     # show what would be submitted
    bash hpc/sweep_BoseHubbard_NormalizedITE.sh 6              # submit μ/U = 0, 0.05, ..., 1 for L = 6
    MUS="0.1 0.2 0.3" bash hpc/sweep_BoseHubbard_NormalizedITE.sh 6
    T=0.1 MAXD=32 MAXITER=1000 bash hpc/sweep_BoseHubbard_NormalizedITE.sh 6

    # afterwards, with the same run parameters (they set the run tag)
    bash hpc/submit_plot_BoseHubbard_NormalizedITE.sh 6
    T=0.1 MAXD=32 bash hpc/submit_plot_BoseHubbard_NormalizedITE.sh 6

Run parameters (environment, defaults set in the submit scripts; the Julia scripts have none and require them all):
`T` (0.16, strong hopping; the weak bonds have t/3), `U` (1.0), `NMAX` (4), `DTAU` (0.05), `MAXD` (16), `MAXITER` (500),
`CONVTOL` (1e-8), `CROSS_WIDTH` (32), `DATA_DIR` ($VSC_DATA/data). Job resources: `WALLTIME` (1:00:00), `MEM` (8GB),
`PPN` (8), `JULIA_THREADS` and `BLAS_THREADS` (both `PPN`), `CLUSTER` (skiddo), `JULIA`. Grid: `MUS`. Plot script:
`N_SAMPLES` (10000, disorder words are enumerated exactly if q^L ≤ `N_SAMPLES`), `S_MUS` ("0.3 0.6 1.0", the μ/U values
whose disorder-averaged entanglement entropy profile S(k) and its disorder variance Var[S](k) are plotted in the right
panels; the nearest grid points are used). The profile and variance on every bond are also stored in `observables.csv`
as the columns `S_1..S_{L-1}` and `varS_1..varS_{L-1}`. To run the plot
script directly, set these yourself, e.g. `T=0.16 U=1.0 NMAX=4 DTAU=0.05 MAXD=16 DATA_DIR=$VSC_DATA/data N_SAMPLES=10000 S_MUS="0.3 0.6 1.0" julia --project=. hpc/plot_BoseHubbard_NormalizedITE.jl 6`.

## Choosing resources

Run the benchmark once per L (and whenever `MAXD` or `NMAX` changes), before the sweep, with the same run parameters:

    bash hpc/benchmark_BoseHubbard_NormalizedITE.sh 6            # one PBS job, threads 1 2 4 8
    MODES="both blas" bash hpc/benchmark_BoseHubbard_NormalizedITE.sh 6   # only some of the thread splits
    LOCAL=1 bash hpc/benchmark_BoseHubbard_NormalizedITE.sh 6    # or directly, e.g. on an interactive node
    T=0.16 U=1.0 NMAX=4 DTAU=0.05 MAXD=16 MAXITER=500 DATA_DIR=$VSC_DATA/data \
        julia hpc/benchmark_BoseHubbard_NormalizedITE.jl --summary 6   # reprint the summary

As for the RTFIM, every thread count n runs the splits `both`, `julia` and `blas` (`MODES`). Every split runs
`BENCH_STEPS` (20) ITE steps at the cap `MAXD` at μ = `MU` (1.0, superfluid between the n = 1 and n = 2 Mott lobes, so
the bonds saturate at `MAXD`; the summary warns if they did not) and records the time per step once the bonds have
saturated and the peak memory. The summary estimates T = warm-up + `MAXITER` × time per step (an upper bound, a run stops
when it converges), the speedup and efficiency per split, and recommends `PPN`, `JULIA_THREADS`, `BLAS_THREADS`,
`WALLTIME` (`SAFETY` × T, rounded up to 15 minutes) and `MEM` (`SAFETY` × peak memory), with the matching sweep command.
The results are stored in `$DATA_DIR/BoseHubbard_NormalizedITE/benchmarks/<run tag>/L<L>/threads<nj>_blas<nb>.csv`.

## Data layout

    $DATA_DIR/BoseHubbard_NormalizedITE/
      t0.16_U1.0_nmax4_dtau0.05_D16/            run tag: t, U, nmax, Δτ, maxD
        L6/
          mu0.0000.jld2
          mu0.0500.jld2
          ...
          observables.csv                       written by the plot script
      benchmarks/<run tag>/L<L>/threads<nj>_blas<nb>.csv   benchmark results
    logs/BoseHubbard_NormalizedITE/<run tag>/L<L>/   PBS stdout/stderr of every job
    logs/BoseHubbard_NormalizedITE/benchmarks/<run tag>/L<L>/
    plots/BoseHubbard_NormalizedITE_<run tag>_L<L>.png

`MAXITER` and `CONVTOL` are not in the run tag: rerunning with more steps needs `FORCE=1` (or another `DATA_DIR`).

## Contents of a result file (load with `using DisorderKit, JLD2; d = load(file)`)

| key | content |
|---|---|
| `ψ` | final `FiniteDisorderMPS` |
| `info` | per-step diagnostics (NamedTuple of arrays), e.g. `Es`, `ϵsconv`, `bond_dimensions`, `residual_max` |
| `converged`, `nsteps` | whether `ϵ_conv ≤ convtol` was reached, and the number of steps |
| `params` | `L, μ, t, U, nmax, ps, Δτ, maxD, maxiter, convtol, cross_width` (rebuild the model with `random_bose_hubbard([t, t/3], [U], [μ]; nmax)`) |
| `meta` | wall time, timings (`TimerOutput` as a Dict), git commit and dirty flag, Julia version, host, threads, date |
