# Correlated Bose-Hubbard on an HTCondor pool

Condor versions of the correlated Bose-Hubbard PBS scripts (`hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh`,
`hpc/submit_obs_BHCorr.sh`, `hpc/submit_plot_BHCorr.sh`), plus a dashboard on the laptop that drives them. The Julia
scripts are the same as on the PBS cluster; only the submission differs.

| file | what it does |
|---|---|
| `dashboard/server.py`, `dashboard/index.html` | the dashboard (below): grid of the sweep, submit, gather, plot |
| `sweep_BHCorr.sh <L>` | one job per (t, μ): the ITE and then the observables of the new state; both files come back |
| `obs_BHCorr.sh <L>` | observables-only jobs, to recompute observables without rerunning the ITE |
| `plot_BHCorr.sh <L>` | the phase-diagram plot on the access point (manual use; the dashboard plots on the laptop) |
| `run_julia.sh` | the job executable: installs Julia (juliaup) and the packages (`Pkg.instantiate()`, with a timeout and retries), runs the scripts |
| `sweep_status.jl` | the state of every point of a sweep as JSON (used by the dashboard) |
| `condor.local.env.example` | template for the connection settings (the real `condor.local.env` is gitignored) |

## Dashboard (recommended)

On the laptop, from the repo root (on the university network or VPN):

    python3 hpc/condor/dashboard/server.py

The first start asks for the access point's host name, ssh port, your login and the repo folder there, and saves them
in `hpc/condor/condor.local.env`. That file is gitignored: connection details never end up in the (public) repo.
Then it asks for your password once (the ssh connection is kept open for 12 h) and opens `http://localhost:8765`.
Nothing to install (Python standard library); the one-time setup on the access point below is still needed.

The page shows:

- a **grid** of the sweep (columns t/U, rows μ/U), one square per point: ✓ done (converged), ~ unconverged,
  ▶ running, … queued, ! held, ✕ failed (job ended without a result), grey not submitted. A dot in the corner shows the
  observables (amber: outdated), a bar at the bottom that the result is still on the access point. Hover for ϵ_conv,
  steps, machine and run time; click for details and the job's log (live output for a running job). A running point
  shows its CPU use and current step (setup phase, or ITE iteration and step with the time it entered it); an amber
  corner marks a running job that used almost no CPU for more than 30 minutes (stuck);
- **Running jobs**: *Check running jobs* reads the live output of all running jobs (condor_tail) and lists their
  runtime, CPU use, phase/step and how long they are in it, longest first;
- counts, a progress bar, what is on the laptop and what still waits on the access point; it refreshes every 20 s;
- the **phase diagram** of the sweep once it is plotted;
- a theme switch (Auto / Light / Dark).

The workflow, one button each:

1. **Submit**: pulls the latest code on the access point (warns if the laptop has uncommitted or unpushed changes, since
   the cluster runs the code on GitHub) and submits every grid point without a result on the laptop or the access point.
   Each job runs the ITE and the observables. Points that are done or in the queue are skipped; raise MAXITER (or lower
   CONVTOL) to rerun unconverged points. *Preview* shows what would be submitted.
2. **Gather**: copies the finished results to this repo's `data/` (rsync), then deletes them on the access point, with
   their logs, and the cluster logs and submit files once no job of the sweep is left. A file is deleted only after it was
   copied completely, and results younger than 2 minutes or of points still in the queue wait for the next Gather. The
   laptop is the archive; nothing piles up on the access point. Logs of held or failed points are kept until resolved.
3. **Plot**: runs `hpc/plot_BoseHubbardCorrelated_NormalizedITE.jl` on the laptop's data and shows the figure.

- **Recompute observables**: for points whose observables are missing or outdated: another N_EDGE, or a changed
  observables script (every observables file stores the `git hash-object` of the script that made it, so after you change
  `hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl` and push, all points show as outdated). Uploads only the
  needed states to the access point and runs observables-only jobs; the ITE is not rerun. Gather afterwards.
- **Remove held** (then Submit resubmits them) and **Remove all jobs** of the sweep.

The parameters (grid, model, convergence, observables, plot, resources) are in the form and remembered per sweep. The
model parameters set the run tag, so changing them switches to another sweep. Sweeps on the laptop, on the access point
or in the queue are in the selector at the top; `http://localhost:8765/?tag=<run tag>&L=<L>` opens one directly.
Closing the dashboard does not affect the jobs. `--help` lists the options.

## How it works

The pool has no shared filesystem. Every job gets `Project.toml`, `Manifest.toml`, `src/` and its Julia scripts, runs in
a sandbox on the execute node and only its result files are transferred back. Julia is the version the Manifest was
resolved with, through juliaup, installed in `/tmp/<login>/.juliaup` with the package depot `/tmp/<login>/.julia`
(`HOME=/tmp/<login>`). On this pool `/tmp` is private to every job (`MOUNT_UNDER_SCRATCH`), so every job installs Julia
and the packages from scratch (about 8 minutes). `Pkg.instantiate()` sometimes hangs without output (a stalled
download), so `run_julia.sh` limits every attempt to `SETUP_TIMEOUT` seconds (1800) and retries it up to
`SETUP_ATTEMPTS` times (3) before the job fails. The dashboard shows the phase of every running job (setup or the ITE
step, see below). `DATA_DIR` is
always the relative path `data`, so with `preserve_relative_paths` a result lands at the same path in the repo on the
access point:

    data/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/t<t>/mu<μ>.jld2       (state)
    data/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/t<t>/mu<μ>_obs.csv    (observables)

Each script writes its points file and submit file to `logs/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/condor/`
(`…/condor/obs/` for observables-only jobs), next to the job `.out`/`.err` and the cluster `.log`. The jobs carry the
attributes `BHC_stage`, `BHC_tag`, `BHC_L`, `BHC_t`, `BHC_mu`, so a point is never submitted twice while it is queued.

## One-time setup on the access point

    ssh -p <port> <login>@<access point>
    curl -fsSL https://install.julialang.org | sh -s -- --yes   # juliaup in ~/.juliaup
    source ~/.bashrc
    juliaup add 1.12.6                                          # the Manifest's julia_version
    git clone https://github.com/KVerv/DisorderKit.jl.git && cd DisorderKit.jl
    julia +1.12.6 --project=. -e 'using Pkg; Pkg.instantiate()'

The scripts call `~/.juliaup/bin/julia +<Manifest version>` on the access point (override with `JULIA=...`). To see
the pool: `condor_status -constraint has_julia -af:h machine cpus memory cpu_performance | sort -u`.

## Without the dashboard (on the access point, from the repo root)

    git pull
    DELTA=0.3 OMEGA=0.5 DRYRUN=1 bash hpc/condor/sweep_BHCorr.sh 6    # the points and the submit file
    DELTA=0.3 OMEGA=0.5 bash hpc/condor/sweep_BHCorr.sh 6
    DELTA=0.3 OMEGA=0.5 MAXITER=4000 bash hpc/condor/sweep_BHCorr.sh 6   # rerun unconverged points
    DELTA=0.3 OMEGA=0.5 bash hpc/condor/plot_BHCorr.sh 6

Without `POINTS` the sweep script decides from the states on the access point only, so after a Gather (states only on the
laptop) it would submit those points again: use the dashboard, or pass `POINTS="t mu,t mu,…"` explicitly.

Parameters (environment): `TS`, `MUS`, `POINTS`, `U`, `R`, `DELTA`, `OMEGA`, `NMAX`, `DTAU`, `MAXD`, `MAXITER`,
`CONVTOL`, `CROSS_WIDTH`, `N_SAMPLES`, `N_EDGE` (observables), `KAPPA_TOL`, `SBSTD_TOL`, `A2_TOL` (plot), `FORCE`, `DRYRUN`. Condor
resources: `PPN` (request_cpus; 8 sweep, 4 observables-only), `JULIA_THREADS`, `BLAS_THREADS` (ITE: 1 and `PPN`),
`OBS_THREADS` (observables in the sweep job: `PPN`), `MEM` ("6 G"), `DISK` ("2 G"), `REQUIREMENTS` (extra requirements,
e.g. `"cpu_performance > 7"`), `NICE` (`nice_user`, TRUE: only idle slots), `UG_NAME` (default `$USER`). There is no
walltime on condor.

## Monitoring and troubleshooting

    condor_q -nobatch                         # status of every job (I idle, R running, H held)
    condor_q -run                             # which machine runs which job
    condor_tail -follow -stderr 123.4         # live output of a running job
    condor_q -hold -af:jh HoldReason          # why jobs are held
    condor_q -better-analyze 123.4            # why a job does not start
    condor_rm 123                             # remove a cluster (or 123.4, or -all)

A job that crashes does not produce its result files, and condor then puts it on hold ("failed to transfer output").
Read its log (click the point in the dashboard), fix the cause, remove the held jobs and submit again. The first job on
a machine installs juliaup, Julia and the packages in `/tmp/<login>` and is therefore slower; jobs on the same machine
at the same time wait for each other (a lock for the Julia install, Julia's own locks for precompilation).
