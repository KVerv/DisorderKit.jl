#!/bin/bash
#
# Submit one PBS job that runs hpc/plot_BoseHubbard_NormalizedITE.jl (observables, CSV and plot) for a given L.
# Run on the login node from the repo root, with the same run parameters as the sweep, so the run tag matches:
#   bash hpc/submit_plot_BoseHubbard_NormalizedITE.sh 6
#   T=0.1 MAXD=32 bash hpc/submit_plot_BoseHubbard_NormalizedITE.sh 6
#   DRYRUN=1 bash hpc/submit_plot_BoseHubbard_NormalizedITE.sh 6      # only print the job
#
# Run parameters (T, U, NMAX, DTAU, MAXD, DATA_DIR, N_SAMPLES, S_MUS) and their defaults are set below; override them from the
# caller's environment. They are written into the job script, hpc/plot_BoseHubbard_NormalizedITE.jl has no defaults.
# Logs: logs/BoseHubbard_NormalizedITE/<run tag>/L<L>/plot/

set -euo pipefail

L="${1:?usage: bash hpc/submit_plot_BoseHubbard_NormalizedITE.sh <L>}"

# Run parameters: T, U, NMAX, DTAU, MAXD and DATA_DIR as in hpc/sweep_BoseHubbard_NormalizedITE.sh (they set the run tag)
export T="${T:-0.16}"
export U="${U:-1.0}"
export NMAX="${NMAX:-4}"
export DTAU="${DTAU:-0.05}"
export MAXD="${MAXD:-24}"
export DATA_DIR="${DATA_DIR:-${VSC_DATA:-.}/data}"
export N_SAMPLES="${N_SAMPLES:-10000}"   # disorder words are enumerated exactly if q^L ≤ N_SAMPLES, else sampled
export S_MUS="${S_MUS:-0.3 0.6 1.0}"     # μ/U values of the entanglement entropy profiles in the plot

WALLTIME="${WALLTIME:-0:50:00}"
MEM="${MEM:-4GB}"
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
export T=$T U=$U NMAX=$NMAX DTAU=$DTAU MAXD=$MAXD DATA_DIR="$DATA_DIR" N_SAMPLES=$N_SAMPLES S_MUS="$S_MUS"
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
