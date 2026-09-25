#!/bin/bash
#
# Submit one EMPS phase-line job per (W, Dmax). Run on the login node from the repo root:
#   bash hpc/sweep_EMPS.sh            # submit
#   DRYRUN=1 bash hpc/sweep_EMPS.sh   # only print the qsub commands
#
# One-time setup before the first sweep (avoids precompile races between jobs):
#   ~/.juliaup/bin/julia +1.12.6 --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

Ws=(0.1 0.25 0.5 0.75 1.0)
Dmaxs=(2 4 8)
WALLTIME="12:00:00"
MEM="8GB"

mkdir -p logs

for D in "${Dmaxs[@]}"; do
    for W in "${Ws[@]}"; do
        echo "Submitting job for W = $W and Dmax = $D"
        export W
        export DMAX="$D"
        cmd=(qsub -V -N "EMPS_W${W}_D${D}" -o logs/ -e logs/ -l walltime=$WALLTIME -l mem=$MEM hpc/submit_EMPS.sh)
        if [ "${DRYRUN:-0}" = "1" ]; then
            echo "${cmd[@]}"
        else
            "${cmd[@]}"
        fi
    done
done
