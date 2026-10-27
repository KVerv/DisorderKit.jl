#!/bin/bash
#
# Submit one hpc/RTFIM_NormalizedITE.jl job per point (δ, W) of the RTFIM phase diagram for a given L.
# Run on the login node from the repo root:
#   bash hpc/sweep_RTFIM_NormalizedITE.sh 16                         # submit the default grid for L = 16
#   DRYRUN=1 bash hpc/sweep_RTFIM_NormalizedITE.sh 16                # only print what would be submitted
#   DELTAS="-0.5 0 0.5" WS="0.5 1" bash hpc/sweep_RTFIM_NormalizedITE.sh 16
#   DMAX_SCHEDULE="300:8,200:16" DTAU=0.01 bash hpc/sweep_RTFIM_NormalizedITE.sh 16
#   FORCE=1 bash hpc/sweep_RTFIM_NormalizedITE.sh 16                 # also resubmit points that already have a result
#
# Run parameters (DTAU, ORDER, DMAX_SCHEDULE, MAXITER, CONVTOL, FILTER_TAIL_TOL, SEED, QUICK, DATA_DIR) are passed to the
# jobs through the environment (qsub -V); see hpc/RTFIM_NormalizedITE.jl for their defaults.
# Results: <DATA_DIR>/RTFIM_NormalizedITE/<run tag>/L<L>/W<W>/delta<δ>.jld2
# Logs:    logs/RTFIM_NormalizedITE/<run tag>/L<L>/
#
# Before submitting (not with DRYRUN), the script runs Pkg.instantiate() and Pkg.precompile() on the login node, so new
# dependencies are installed and the jobs do not race to precompile.

set -euo pipefail

L="${1:?usage: bash hpc/sweep_RTFIM_NormalizedITE.sh <L>}"

# Phase diagram grid: δ (distance to criticality) and W (disorder strength), formatted as %.4f in the file names
DELTAS="${DELTAS:--1.0 -0.75 -0.5 -0.25 0.0 0.25 0.5 0.75 1.0}"
WS="${WS:-0.1 0.5 1.0 2.0}"

# Resources per job
WALLTIME="${WALLTIME:-24:00:00}"
MEM="${MEM:-16GB}"
NODES=1
PPN="${PPN:-4}"   # CPUs per node, Julia runs with --threads=$PPN
CLUSTER="${CLUSTER:-skiddo}"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia +1.12.6}"

if command -v module > /dev/null 2>&1; then
    module swap cluster/$CLUSTER   # qsub submits to the currently loaded cluster module
fi

# Install and precompile the dependencies here (login node), so the jobs never miss a package after a git pull
if [ "${DRYRUN:-0}" != "1" ]; then
    $JULIA --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
fi

submitted=0
skipped=0
shown_job=0
for W in $WS; do
    W=$(printf "%.4f" "$W")
    for d in $DELTAS; do
        d=$(printf "%.4f" "$d")
        # The Julia script decides the run tag and the output file, so the paths always agree
        mapfile -t paths < <($JULIA --startup-file=no hpc/RTFIM_NormalizedITE.jl --path "$L" "$d" "$W")
        tag="${paths[0]}"
        out="${paths[1]}"
        logdir="logs/RTFIM_NormalizedITE/$tag/L$L"

        if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "skip   L=$L W=$W δ=$d (exists: $out)"
            skipped=$((skipped + 1))
            continue
        fi

        # Job script. Unescaped variables are filled in now, escaped ones (\$) when the job runs.
        job=$(cat <<EOF
#!/bin/bash
cd \$PBS_O_WORKDIR
module purge
echo "PBS: \$PBS_JOBID"
echo "RTFIM normalized ITE: L = $L, δ = $d, W = $W"
echo "Job started at : \$(date)"
$JULIA --project=. --threads=$PPN hpc/RTFIM_NormalizedITE.jl $L $d $W
echo "Job ended at : \$(date)"
EOF
)
        name="RTFIM_L${L}_W${W}_d${d}"
        cmd=(qsub -V -N "$name" -m a -o "$logdir/" -e "$logdir/" -l walltime=$WALLTIME -l mem=$MEM -l nodes=$NODES:ppn=$PPN)
        if [ "${DRYRUN:-0}" = "1" ]; then
            echo "submit L=$L W=$W δ=$d -> $out"
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
