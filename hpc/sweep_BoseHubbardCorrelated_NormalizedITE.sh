#!/bin/bash
#
# Submit one hpc/BoseHubbardCorrelated_NormalizedITE.jl job per point (t, μ) of the phase diagram of the Bose-Hubbard
# chain with correlated hopping and potential disorder. Run on the login node from the repo root:
#   bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6                    # submit the default (t, μ) grid for L = 6
#   DRYRUN=1 bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6           # only print what would be submitted
#   TS="0.05 0.1" MUS="0.2 0.4 0.6" bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6
#   R=0.5 MAXD=32 MAXITER=1000 bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6
#   FORCE=1 bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6            # also resubmit points that already have a result
#   PPN=4 JULIA_THREADS=1 BLAS_THREADS=4 bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6
#
# Run parameters (U, R, NMAX, DTAU, MAXD, MAXITER, CONVTOL, CROSS_WIDTH, DATA_DIR) and their defaults are set below;
# override them from the caller's environment. They are written into every job script,
# hpc/BoseHubbardCorrelated_NormalizedITE.jl has no defaults of its own.
# Results: <DATA_DIR>/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/t<t>/mu<μ>.jld2
# Logs:    logs/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/
# Observables and plot afterwards: bash hpc/submit_plot_BoseHubbardCorrelated_NormalizedITE.sh <L> (with the same run
# parameters)
#
# Every job first runs Pkg.instantiate() and Pkg.precompile() on its compute node, so new dependencies are installed
# (concurrent jobs wait for each other through Julia's precompile locks).

set -euo pipefail

L="${1:?usage: bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh <L>}"

# (t/U, μ/U) grid of the strong values (the weak ones are t/3 and μ/3), formatted as %.4f in the file names
TS="${TS:-$(LC_ALL=C seq 0.02 0.04 0.30)}"
MUS="${MUS:-$(LC_ALL=C seq 0 0.2 1.6)}"

# Run parameters (U, R, NMAX, DTAU, MAXD set the run tag)
export U="${U:-1.0}"
export R="${R:-0.0}"   # correlation of hopping and potential: P = (1±R)/4
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
for t in $TS; do
    t=$(LC_ALL=C printf "%.4f" "$t")
    for mu in $MUS; do
        mu=$(LC_ALL=C printf "%.4f" "$mu")
        # The Julia script decides the run tag and the output file, so the paths always agree
        mapfile -t paths < <($JULIA --startup-file=no hpc/BoseHubbardCorrelated_NormalizedITE.jl --path "$L" "$t" "$mu")
        tag="${paths[0]}"
        out="${paths[1]}"
        logdir="logs/BoseHubbardCorrelated_NormalizedITE/$tag/L$L"

        if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "skip   L=$L t=$t μ=$mu (exists: $out)"
            skipped=$((skipped + 1))
            continue
        fi

        # Job script. Unescaped variables are filled in now, escaped ones (\$) when the job runs.
        job=$(cat <<EOF
#!/bin/bash
cd \$PBS_O_WORKDIR
module purge
echo "PBS: \$PBS_JOBID"
echo "Correlated Bose-Hubbard normalized ITE: L = $L, t = $t, μ = $mu, r = $R"
echo "Job started at : \$(date)"
export U=$U R=$R NMAX=$NMAX DTAU=$DTAU MAXD=$MAXD
export MAXITER=$MAXITER CONVTOL=$CONVTOL CROSS_WIDTH=$CROSS_WIDTH DATA_DIR="$DATA_DIR"
export BLAS_THREADS=$BLAS_THREADS
$JULIA --project=. --threads=$JULIA_THREADS -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
$JULIA --project=. --threads=$JULIA_THREADS hpc/BoseHubbardCorrelated_NormalizedITE.jl $L $t $mu
echo "Job ended at : \$(date)"
EOF
)
        name="BHC_L${L}_t${t}_mu${mu}"
        cmd=(qsub -V -N "$name" -m a -o "$logdir/" -e "$logdir/" -l walltime=$WALLTIME -l mem=$MEM -l nodes=$NODES:ppn=$PPN)
        if [ "${DRYRUN:-0}" = "1" ]; then
            echo "submit L=$L t=$t μ=$mu -> $out"
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
done

echo "L = $L: $submitted job(s) $([ "${DRYRUN:-0}" = "1" ] && echo "to submit" || echo "submitted"), $skipped skipped (result exists)"
