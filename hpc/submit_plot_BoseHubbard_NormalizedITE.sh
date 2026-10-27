#!/bin/bash
#
# Submit one PBS job that runs hpc/plot_BoseHubbard_NormalizedITE.jl (observables, CSV and plot) for a given L.
# Run on the login node from the repo root, with the same run parameters as the sweep, so the run tag matches:
#   bash hpc/submit_plot_BoseHubbard_NormalizedITE.sh 6
#   T=0.1 MAXD=32 bash hpc/submit_plot_BoseHubbard_NormalizedITE.sh 6
#   DRYRUN=1 bash hpc/submit_plot_BoseHubbard_NormalizedITE.sh 6      # only print the job
#
# Run parameters (T, U, NMAX, DTAU, MAXD, DATA_DIR, N_SAMPLES) reach the job through the environment (qsub -V).
# Logs: logs/BoseHubbard_NormalizedITE/<run tag>/L<L>/plot/

set -euo pipefail

L="${1:?usage: bash hpc/submit_plot_BoseHubbard_NormalizedITE.sh <L>}"

WALLTIME="${WALLTIME:-1:00:00}"
MEM="${MEM:-8GB}"
PPN="${PPN:-4}"
CLUSTER="${CLUSTER:-skiddo}"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia +1.12.6}"

# Same run tag as the simulation script (via its --path mode, μ is irrelevant for the tag), for the log folder
mapfile -t paths < <($JULIA --startup-file=no hpc/BoseHubbard_NormalizedITE.jl --path "$L" 0)
tag="${paths[0]}"
logdir="logs/BoseHubbard_NormalizedITE/$tag/L$L/plot"

job=$(cat <<EOF
#!/bin/bash
cd \$PBS_O_WORKDIR
module purge
echo "PBS: \$PBS_JOBID"
echo "Bose-Hubbard observables and plot: L = $L, run tag $tag"
echo "Job started at : \$(date)"
$JULIA --project=. --threads=$PPN -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
$JULIA --project=. --threads=$PPN hpc/plot_BoseHubbard_NormalizedITE.jl $L
echo "Job ended at : \$(date)"
EOF
)

cmd=(qsub -V -N "BH_plot_L$L" -m a -o "$logdir/" -e "$logdir/" -l walltime=$WALLTIME -l mem=$MEM -l nodes=1:ppn=$PPN)
if [ "${DRYRUN:-0}" = "1" ]; then
    echo "${cmd[*]}"
    echo "$job"
else
    if command -v module > /dev/null 2>&1; then
        module swap cluster/$CLUSTER   # qsub submits to the currently loaded cluster module
    fi
    mkdir -p "$logdir"
    echo "$job" | "${cmd[@]}"
fi
