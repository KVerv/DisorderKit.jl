#!/bin/bash
#
# Submit one EMPS phase-line job per (W, Dmax). Run on the login node from the repo root:
#   bash hpc/sweep_EMPS.sh            # submit
#   DRYRUN=1 bash hpc/sweep_EMPS.sh   # only print the qsub commands
#   DELTA_MIN=-1 DELTA_MAX=1 NDELTA=21 bash hpc/sweep_EMPS.sh   # δ = μ/W² grid, the same for every W
#
# One-time setup before the first sweep (avoids precompile races between jobs):
#   ~/.juliaup/bin/julia +1.12.6 --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

Ws=(0.1 0.2)   # W > 0, since δ = μ/W²
Dmaxs=(2)
export DELTA_MIN="${DELTA_MIN:--0.5}"
export DELTA_MAX="${DELTA_MAX:-0.5}"
export NDELTA="${NDELTA:-11}"
WALLTIME="1:00:00"
MEM="16GB"
NODES=1
PPN=1   # CPUs per node
CLUSTER="shinx"

module swap cluster/$CLUSTER   # qsub submits to the currently loaded cluster module

mkdir -p logs

for D in "${Dmaxs[@]}"; do
    for W in "${Ws[@]}"; do
        echo "Submitting job for W = $W and Dmax = $D (δ from $DELTA_MIN to $DELTA_MAX, $NDELTA points)"
        export W
        export DMAX="$D"
        cmd=(qsub -V -N "EMPS_W${W}_D${D}" -o logs/ -e logs/ -l walltime=$WALLTIME -l mem=$MEM -l nodes=$NODES:ppn=$PPN hpc/submit_EMPS.sh)
        if [ "${DRYRUN:-0}" = "1" ]; then
            echo "${cmd[@]}"
        else
            "${cmd[@]}"
        fi
    done
done
