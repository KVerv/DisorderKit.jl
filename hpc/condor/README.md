# Correlated Bose-Hubbard on the QuantumGhent HTCondor pool

Condor versions of the correlated Bose-Hubbard PBS scripts (`hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh`,
`hpc/submit_obs_BHCorr.sh`, `hpc/submit_plot_BHCorr.sh`). The Julia scripts are the same; only the submission differs.
General condor notes of the group: `docs/condor_notes-main/`.

| script | what it does |
|---|---|
| `sweep_BHCorr.sh <L>` | one ITE job per (t, μ) that needs a (re)run, all in one condor cluster |
| `obs_BHCorr.sh <L>` | one observables job per stored state whose `mu<μ>_obs.csv` is missing or outdated |
| `plot_BHCorr.sh <L>` | the phase-diagram plot, directly on the access point (or `CONDOR=1`: as one job) |
| `run_julia.sh` | the job executable: `Pkg.instantiate()` and run the script with the node Julia |

## How it works

The pool has no shared filesystem. Every job gets `Project.toml`, `Manifest.toml`, `src/`, its Julia script and (for
observables) its state file, runs in a sandbox on the execute node and only its result file is transferred back.
Julia is the version the Manifest was resolved with (1.12.6), through juliaup: the first job on a machine installs
juliaup and that version in `/tmp/<UGent login>/.juliaup` (`HOME=/tmp/<login>`); later jobs on that machine reuse it
and the package depot `/tmp/<login>/.julia`. `DATA_DIR` is always the relative path `data`, so with
`preserve_relative_paths` a result lands at the same path in the repo on the access point:

    data/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/t<t>/mu<μ>.jld2        (sweep)
    data/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/t<t>/mu<μ>_obs.csv     (observables)

Each script writes its points file and submit file to `logs/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/condor/`
(`…/condor/obs/`, `…/condor/plot/`), next to the job `.out`/`.err` and the cluster `.log`. The skip/rerun decisions are
the same as on the VSC (`--status` of the Julia scripts).

## One-time setup on the access point

    ssh -p <port> <login>@<access point>
    curl -fsSL https://install.julialang.org | sh -s -- --yes   # juliaup in ~/.juliaup (the access point' own Julia is 1.11)
    source ~/.bashrc
    juliaup add 1.12.6
    git clone https://github.com/KVerv/DisorderKit.jl.git && cd DisorderKit.jl
    julia +1.12.6 --project=. -e 'using Pkg; Pkg.instantiate()'   # for the --status checks and the plot on the access point

The scripts call `~/.juliaup/bin/julia +<Manifest version>` on the access point (override with `JULIA=...`). Check the pool:

    condor_status -constraint has_julia -af:h machine cpus memory cpu_performance | sort -u

## Workflow (on the access point, from the repo root)

    git pull
    # 1. a small test first
    TS=0.1 MUS="0.2 0.4" MAXITER=50 DELTA=0.3 OMEGA=0.5 bash hpc/condor/sweep_BHCorr.sh 6
    # 2. the sweep (DRYRUN=1 first shows the points and the submit file)
    DELTA=0.3 OMEGA=0.5 DRYRUN=1 bash hpc/condor/sweep_BHCorr.sh 6
    DELTA=0.3 OMEGA=0.5 bash hpc/condor/sweep_BHCorr.sh 6
    # 3. when the cluster is done: unconverged points with more steps (only those are resubmitted)
    DELTA=0.3 OMEGA=0.5 MAXITER=4000 bash hpc/condor/sweep_BHCorr.sh 6
    # 4. observables and plot (same run parameters, so the run tag matches)
    DELTA=0.3 OMEGA=0.5 bash hpc/condor/obs_BHCorr.sh 6
    DELTA=0.3 OMEGA=0.5 bash hpc/condor/plot_BHCorr.sh 6
    # 5. copy the results to the laptop (from the laptop)
    rsync -av -e "ssh -p <port>" <login>@<access point>:DisorderKit.jl/data/BoseHubbardCorrelated_NormalizedITE/ data/BoseHubbardCorrelated_NormalizedITE/
    rsync -av -e "ssh -p <port>" <login>@<access point>:DisorderKit.jl/plots/ plots/

Wait for a cluster to finish before running the same script again: points that are still queued or running have no
result yet and would be submitted twice.

Parameters (environment) are those of the PBS scripts: `TS`, `MUS`, `U`, `R`, `DELTA`, `OMEGA`, `NMAX`, `DTAU`, `MAXD`,
`MAXITER`, `CONVTOL`, `CROSS_WIDTH` (sweep), `N_SAMPLES`, `N_EDGE` (observables), `KAPPA_TOL`, `CV_TOL` (plot), `FORCE`,
`DRYRUN`. Condor resources: `PPN` (request_cpus; 8 sweep, 4 observables), `JULIA_THREADS`, `BLAS_THREADS` (sweep:
1 and `PPN`), `MEM` ("6 G"), `DISK` ("2 G"), `REQUIREMENTS` (extra requirements, e.g. `"cpu_performance > 7"`),
`NICE` (`nice_user`, TRUE: only idle slots), `UG_NAME` (<login>). There is no walltime on condor.

## Monitoring and troubleshooting

    condor_q -nobatch                         # status of every job (I idle, R running, H held)
    condor_q -run                             # which machine runs which job
    condor_tail -follow -stderr 123.4         # live output of a running job
    condor_q -hold -af:jh HoldReason          # why jobs are held
    condor_q -better-analyze 123.4            # why a job does not start
    condor_rm 123                             # remove a cluster (or 123.4, or -all)

A job that crashes does not produce its result file, and condor then puts it on hold ("failed to transfer output").
Read its `.err` in the logs folder, fix the cause, `condor_rm` the held jobs and run the script again: the point has no
result, so it is submitted again. The first job on a machine installs juliaup, Julia and the packages in
`/tmp/<login>` and is therefore slower; jobs on the same machine at the same time wait for each other (a lock for the
Julia install, Julia's own locks for precompilation).
