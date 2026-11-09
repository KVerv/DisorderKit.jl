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
`N_SAMPLES` (10000, disorder words are enumerated exactly if ∏ᵢ qᵢ ≤ `N_SAMPLES`, qᵢ the number of sectors of site i), `S_MUS` ("0.3 0.6 1.0", the μ/U values
whose disorder-averaged entanglement entropy profile S(k) and its disorder variance Var[S](k) are plotted in the right
panels; the nearest grid points are used). S and Var[S] are those of the conditional states projected onto their most
likely particle number (`average_entanglement_entropy(ψ; charges = 0:nmax)`): the ITE states are superpositions of
neighbouring particle numbers, which would lower S in the bulk. The profile and variance on every bond are also stored in `observables.csv`
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

# HPC scripts: Bose-Hubbard chain with correlated hopping and potential disorder

Ground states of

    H_w = -Σ t_{wᵢ} (b†ᵢbᵢ₊₁ + h.c.) + U/2 Σ nᵢ(nᵢ-1) - Σ μ_{wᵢ} nᵢ,   t_{wᵢ} ∈ {t(1-Ω), t(1+Ω)},  μ_{wᵢ} ∈ {μ-Δ, μ+Δ},

on an open chain of `L` sites, where the label wᵢ of site i sets both μᵢ and the hopping of bond (i, i+1). The hopping and
the potential of a site are correlated by r ∈ (-1, 1):

| sector | μᵢ | t_{i,i+1} | probability |
|---|---|---|---|
| 1 | μ-Δ | t(1-Ω) | (1+r)/4 |
| 2 | μ+Δ | t(1-Ω) | (1-r)/4 |
| 3 | μ-Δ | t(1+Ω) | (1-r)/4 |
| 4 | μ+Δ | t(1+Ω) | (1+r)/4 |

(the sector order of `random_bose_hubbard([t(1-Ω), t(1+Ω)], [U], [μ-Δ, μ+Δ]; nmax)`). t and μ are the swept mean values,
Δ (`DELTA`) and Ω (`OMEGA`, 0 ≤ Ω ≤ 1) are fixed per run. The last site has no bond, so
`site_sectors` merges its sectors into μ-Δ and μ+Δ with probability 1/2 each. One job per point (t, μ) of the phase
diagram; every job starts from the uniform superposition of all occupations and stores the final state and the per-step
diagnostics only.

| script | what it does |
|---|---|
| `BoseHubbardCorrelated_NormalizedITE.jl` | one point (L, t, μ): runs the ITE and stores the final state and the per-step diagnostics |
| `sweep_BoseHubbardCorrelated_NormalizedITE.sh` | submits one PBS job per (t, μ) for a given L, skipping points that already have a result |
| `observables_BoseHubbardCorrelated_NormalizedITE.jl` | one point (L, t, μ): computes ρ and \|⟨a⟩\|² exactly (`expectation_value`), S̄ and Var[S] per bond (enumerated or sampled words), their chain averages and the homogeneity `S_inhom` = Var_k[S̄(k)] from the stored state and writes them to `mu<μ>_obs.csv`; `--status` only reports whether that is needed |
| `submit_observables_BoseHubbardCorrelated_NormalizedITE.sh` | submits one PBS job per (t, μ) with a state whose observables are missing, older than the state, or sampled with fewer than `N_SAMPLES` words |
| `plot_BoseHubbardCorrelated_NormalizedITE.jl` | reads only the `mu<μ>_obs.csv` files, computes κ = dρ/dμ, prints the table, saves the combined `observables.csv` and plots one figure with six panels over (t/U, μ/U): κ, the coefficient of variation CV = 𝔼_w[std_k S_w(k)]/S̄ of the bond-averaged entropy (both over the bulk bonds, `N_EDGE`), the quasicondensate density \|⟨a⟩\|², ρ, the bond-averaged std 𝔼_w[std_k S_w(k)] and the MI/SF/BG classification from κ and S_bstd, with \|⟨a⟩\|² only for low κ and high S_bstd (`KAPPA_TOL`, `SBSTD_TOL`, `A2_TOL`) |
| `submit_plot_BoseHubbardCorrelated_NormalizedITE.sh` | submits the plot script as one PBS job (`WALLTIME` 0:30:00, `MEM` 4GB, `PPN` 1) |

On an HTCondor pool use the condor versions of the sweep, observables and plot
scripts instead: see [condor/README.md](condor/README.md).

## Usage (from the repo root, on the login node)

    DRYRUN=1 bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6     # show what would be submitted
    bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6              # submit t/U = 0.02, 0.06, ..., 0.30 × μ/U = 0, 0.2, ..., 1.6
    TS="0.05 0.1" MUS="0.2 0.4 0.6" bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6
    R=0.5 DELTA=0.3 OMEGA=0.5 MAXD=32 MAXITER=1000 bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6

    # afterwards, with the same run parameters (they set the run tag): observables, one job per point ...
    DRYRUN=1 bash hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh 6
    bash hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh 6
    N_SAMPLES=10000 bash hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh 6   # more words where sampled
    R=0.5 DELTA=0.3 OMEGA=0.5 MAXD=32 bash hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh 6
    # ... then, once those jobs have finished, the plot job
    bash hpc/submit_plot_BoseHubbardCorrelated_NormalizedITE.sh 6
    R=0.5 DELTA=0.3 OMEGA=0.5 MAXD=32 bash hpc/submit_plot_BoseHubbardCorrelated_NormalizedITE.sh 6

Run parameters (environment, defaults set in the submit scripts; the Julia scripts have none and require them all):
`U` (1.0), `R` (0.0, the correlation r), `DELTA` (0.2, Δ), `OMEGA` (0.5, Ω), `NMAX` (4), `DTAU` (0.05), `MAXD` (16), `MAXITER` (500), `CONVTOL` (1e-8),
`CROSS_WIDTH` (32), `DATA_DIR` ($VSC_DATA/data). Job resources as for the Bose-Hubbard sweep (`WALLTIME`, `MEM`, `PPN`,
`JULIA_THREADS`, `BLAS_THREADS`, `CLUSTER`, `JULIA`); `benchmark_BoseHubbard_NormalizedITE.sh` at the same L, `NMAX`
and `MAXD` gives a good estimate (4 instead of 2 sectors per site makes a step somewhat slower). Grid: `TS`, `MUS` (the
mean values t/U and μ/U). Observables: ρ and \|⟨a⟩\|² = (1/L) Σᵢ ⟨aᵢ⟩² are exact disorder averages; `N_SAMPLES` applies to the entropy only (1000, disorder words are enumerated exactly if ∏ᵢ qᵢ ≤
`N_SAMPLES`, qᵢ the number of sectors of site i), `N_EDGE` (1, sites excluded at each end: S_avg, S_inhom and S_bstd = 𝔼_w[std_k S_w(k)] use only the bulk bonds k = N_EDGE+1..L-N_EDGE-1), job resources `WALLTIME` (1:00:00), `MEM` (8GB), `PPN` (4), and
`TS`/`MUS` to restrict the points (default: every stored state); `FORCE=1` recomputes every point. A point is
(re)computed only if its `mu<μ>_obs.csv` is missing, unreadable or of an older format (no `S_inhom`, `S_bstd` or `n_edge`), computed with another `N_EDGE`, older than the state (the point was rerun), or its
words were sampled (`exact = false`) with fewer than `N_SAMPLES`; exactly enumerated points are never redone. κ = dρ/dμ is taken along μ at fixed t (central differences, one-sided
at the ends). Var[S] is computed for the conditional states projected onto their most likely particle number, as in the
Bose-Hubbard plot script. Unconverged points are marked with open circles, grid points without observables with crosses.

## Data layout

    $DATA_DIR/BoseHubbardCorrelated_NormalizedITE/
      U1.0_r0.0_Delta0.2_Omega0.5_nmax4_dtau0.05_D16/   run tag: U, r, Δ, Ω, nmax, Δτ, maxD
        L6/
          t0.0200/
            mu0.0000.jld2
            mu0.2000.jld2
            mu0.2000_obs.csv                    observables of the point (observables script)
            ...
          t0.0600/
          ...
          observables.csv                       all points with κ, written by the plot script
    logs/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/   PBS stdout/stderr of every job (observables jobs in obs/, plot job in plot/)
    plots/BoseHubbardCorrelated_NormalizedITE_<run tag>_L<L>.png

`MAXITER` and `CONVTOL` are not in the run tag: rerunning with more steps needs `FORCE=1` (or another `DATA_DIR`).

## Contents of a result file (load with `using DisorderKit, JLD2; d = load(file)`)

| key | content |
|---|---|
| `ψ` | final `FiniteDisorderMPS` |
| `info` | per-step diagnostics (NamedTuple of arrays), e.g. `Es`, `ϵsconv`, `bond_dimensions`, `residual_max` |
| `converged`, `nsteps` | whether `ϵ_conv ≤ convtol` was reached, and the number of steps |
| `params` | `L, t, μ, U, r, Δ, Ω, nmax, ps, Δτ, maxD, maxiter, convtol, cross_width` (rebuild the model with `random_bose_hubbard([t*(1-Ω), t*(1+Ω)], [U], [μ-Δ, μ+Δ]; nmax)`, `ps` the 4 sector probabilities) |
| `meta` | wall time, timings (`TimerOutput` as a Dict), git commit and dirty flag, Julia version, host, threads, date |
