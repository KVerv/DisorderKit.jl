#!/bin/bash
#
# Submit one PBS job that runs hpc/plot_BoseHubbardCorrelated_NormalizedITE.jl (κ, table, combined CSV and heatmaps from
# the per-point observables files) for a given L. Run on the login node from the repo root, after the jobs of
# hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh have finished, with the same run parameters as the sweep,
# so the run tag matches:
#   bash hpc/submit_plot_BoseHubbardCorrelated_NormalizedITE.sh 6
#   R=0.5 DELTA=0.3 OMEGA=0.5 MAXD=32 bash hpc/submit_plot_BoseHubbardCorrelated_NormalizedITE.sh 6
#   DRYRUN=1 bash hpc/submit_plot_BoseHubbardCorrelated_NormalizedITE.sh 6      # only print the job
#
# Run parameters (U, R, DELTA, OMEGA, NMAX, DTAU, MAXD, DATA_DIR) and their defaults are set below; override them from the caller's
# environment. They are written into the job script, hpc/plot_BoseHubbardCorrelated_NormalizedITE.jl has no defaults.
# Output: <DATA_DIR>/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/observables.csv and
#   plots/BoseHubbardCorrelated_NormalizedITE_<run tag>_L<L>.png (one figure: κ, CV, |⟨a⟩|², ρ, S_bstd and the MI/SF/BG
#   classification with the thresholds KAPPA_TOL, SBSTD_TOL and A2_TOL, see below)
# Logs: logs/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/plot/

set -euo pipefail

L="${1:?usage: bash hpc/submit_plot_BoseHubbardCorrelated_NormalizedITE.sh <L>}"

# Run parameters: as in hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh (they set the run tag)
export U="${U:-1.0}"
export R="${R:-0.0}"
export DELTA="${DELTA:-0.0}"   # potential disorder: μ ± DELTA
export OMEGA="${OMEGA:-0.0}"   # hopping disorder: t(1 ± OMEGA), 0 ≤ OMEGA ≤ 1
export NMAX="${NMAX:-4}"
export DTAU="${DTAU:-0.05}"
export MAXD="${MAXD:-16}"
export DATA_DIR="${DATA_DIR:-${VSC_DATA:-.}/data}"

# Phase classification thresholds for the phase-diagram plot (do not change the run tag):
#   κ ≤ KAPPA_TOL → κ low, S_bstd < SBSTD_TOL → std low, |⟨a⟩|² ≤ A2_TOL → |⟨a⟩|² low;
#   MI: κ, std and |⟨a⟩|² low; SF: κ high, std low, |⟨a⟩|² high; BG: κ high, std high, |⟨a⟩|² low; else unclassified
#   e.g. KAPPA_TOL=0.02 SBSTD_TOL=0.1 A2_TOL=0.02 bash hpc/submit_plot_BHCorr.sh 6
export KAPPA_TOL="${KAPPA_TOL:-0.05}"
export SBSTD_TOL="${SBSTD_TOL:-0.05}"
export A2_TOL="${A2_TOL:-0.05}"

# Resources: the plot script only reads small CSV files
WALLTIME="${WALLTIME:-0:15:00}"
MEM="${MEM:-4GB}"
PPN="${PPN:-1}"
CLUSTER="${CLUSTER:-skiddo}"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia +1.12.6}"

# Same run tag as the simulation script (via its --path mode, t and μ are irrelevant for the tag), for the log folder
mapfile -t paths < <($JULIA --startup-file=no hpc/BoseHubbardCorrelated_NormalizedITE.jl --path "$L" 0 0)
tag="${paths[0]}"
logdir="logs/BoseHubbardCorrelated_NormalizedITE/$tag/L$L/plot"

job=$(cat <<EOF
#!/bin/bash
cd \$PBS_O_WORKDIR
module purge
echo "PBS: \$PBS_JOBID"
echo "Correlated Bose-Hubbard plot: L = $L, run tag $tag"
echo "Job started at : \$(date)"
export U=$U R=$R DELTA=$DELTA OMEGA=$OMEGA NMAX=$NMAX DTAU=$DTAU MAXD=$MAXD DATA_DIR="$DATA_DIR" KAPPA_TOL=$KAPPA_TOL SBSTD_TOL=$SBSTD_TOL A2_TOL=$A2_TOL
$JULIA --project=. --threads=$PPN -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
$JULIA --project=. --threads=$PPN hpc/plot_BoseHubbardCorrelated_NormalizedITE.jl $L
echo "Job ended at : \$(date)"
EOF
)

cmd=(qsub -V -N "BHC_plot_L$L" -m a -o "$logdir/" -e "$logdir/" -l walltime=$WALLTIME -l mem=$MEM -l nodes=1:ppn=$PPN)
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
