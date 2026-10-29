#!/bin/bash
#
# Submit one hpc/BoseHubbard_NormalizedITE.jl job per chemical potential μ of the disordered Bose-Hubbard chain.
# Run on the login node from the repo root:
#   bash hpc/sweep_BoseHubbard_NormalizedITE.sh 6                    # submit the default μ grid for L = 6
#   DRYRUN=1 bash hpc/sweep_BoseHubbard_NormalizedITE.sh 6           # only print what would be submitted
#   MUS="0.1 0.2 0.3" bash hpc/sweep_BoseHubbard_NormalizedITE.sh 6
#   T=0.1 MAXD=32 MAXITER=1000 bash hpc/sweep_BoseHubbard_NormalizedITE.sh 6
#   FORCE=1 bash hpc/sweep_BoseHubbard_NormalizedITE.sh 6            # also resubmit points that already have a result
#   PPN=4 JULIA_THREADS=1 BLAS_THREADS=4 bash hpc/sweep_BoseHubbard_NormalizedITE.sh 6   # thread split from hpc/benchmark_BoseHubbard_NormalizedITE.sh
#
# Run parameters (T, U, NMAX, DTAU, MAXD, MAXITER, CONVTOL, CROSS_WIDTH, DATA_DIR) and their defaults are set below;
# override them from the caller's environment. They are written into every job script, hpc/BoseHubbard_NormalizedITE.jl
# has no defaults of its own.
# Results: <DATA_DIR>/BoseHubbard_NormalizedITE/<run tag>/L<L>/mu<μ>.jld2
# Logs:    logs/BoseHubbard_NormalizedITE/<run tag>/L<L>/
# Observables and plot afterwards: bash hpc/submit_plot_BoseHubbard_NormalizedITE.sh <L> (with the same run parameters)
#
# Every job first runs Pkg.instantiate() and Pkg.precompile() on its compute node, so new dependencies are installed
# (concurrent jobs wait for each other through Julia's precompile locks).

set -euo pipefail

L="${1:?usage: bash hpc/sweep_BoseHubbard_NormalizedITE.sh <L>}"

# μ/U grid, formatted as %.4f in the file names
MUS="${MUS:-$(LC_ALL=C seq 0 0.05 1.5)}"

# Run parameters (T, U, NMAX, DTAU, MAXD set the run tag)
export T="${T:-0.16}"   # strong hopping, the weak bonds have t/3
export U="${U:-1.0}"
export NMAX="${NMAX:-4}"
export DTAU="${DTAU:-0.05}"
export MAXD="${MAXD:-16}"
export MAXITER="${MAXITER:-500}"
export CONVTOL="${CONVTOL:-1e-8}"
export CROSS_WIDTH="${CROSS_WIDTH:-32}"
export DATA_DIR="${DATA_DIR:-${VSC_DATA:-.}/data}"

# Resources per job
WALLTIME="${WALLTIME:-2:00:00}"
MEM="${MEM:-8GB}"
NODES=1
PPN="${PPN:-8}"   # CPUs per node
JULIA_THREADS="${JULIA_THREADS:-1}"   # Julia runs with --threads=$JULIA_THREADS
BLAS_THREADS="${BLAS_THREADS:-$PPN}"     # and BLAS.set_num_threads($BLAS_THREADS)
CLUSTER="${CLUSTER:-skiddo}"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia +1.12.6}"

if command -v module > /dev/null 2>&1; then
    module swap cluster/$CLUSTER   # qsub submits to the currently loaded cluster module
fi

submitted=0
skipped=0
shown_job=0
for mu in $MUS; do
    mu=$(LC_ALL=C printf "%.4f" "$mu")
    # The Julia script decides the run tag and the output file, so the paths always agree
    mapfile -t paths < <($JULIA --startup-file=no hpc/BoseHubbard_NormalizedITE.jl --path "$L" "$mu")
    tag="${paths[0]}"
    out="${paths[1]}"
    logdir="logs/BoseHubbard_NormalizedITE/$tag/L$L"

    if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "skip   L=$L μ=$mu (exists: $out)"
        skipped=$((skipped + 1))
        continue
    fi

    # Job script. Unescaped variables are filled in now, escaped ones (\$) when the job runs.
    job=$(cat <<EOF
#!/bin/bash
cd \$PBS_O_WORKDIR
module purge
echo "PBS: \$PBS_JOBID"
echo "Bose-Hubbard normalized ITE: L = $L, μ = $mu"
echo "Job started at : \$(date)"
export T=$T U=$U NMAX=$NMAX DTAU=$DTAU MAXD=$MAXD
export MAXITER=$MAXITER CONVTOL=$CONVTOL CROSS_WIDTH=$CROSS_WIDTH DATA_DIR="$DATA_DIR"
export BLAS_THREADS=$BLAS_THREADS
$JULIA --project=. --threads=$JULIA_THREADS -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
$JULIA --project=. --threads=$JULIA_THREADS hpc/BoseHubbard_NormalizedITE.jl $L $mu
echo "Job ended at : \$(date)"
EOF
)
    name="BH_L${L}_mu${mu}"
    cmd=(qsub -V -N "$name" -m a -o "$logdir/" -e "$logdir/" -l walltime=$WALLTIME -l mem=$MEM -l nodes=$NODES:ppn=$PPN)
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "submit L=$L μ=$mu -> $out"
        echo "       ${cmd[*]}"
        if [ "$shown_job" = "0" ]; then
            echo "$job" | sed 's/^/       | /'
            shown_job=1
        fi
    else
        mkdir -p "$logdir"
        echo "$job" | "${cmd[@]}"
    fi
    submitted=$((submitted + 1))
done

echo "L = $L: $submitted job(s) $([ "${DRYRUN:-0}" = "1" ] && echo "to submit" || echo "submitted"), $skipped skipped (result exists)"
