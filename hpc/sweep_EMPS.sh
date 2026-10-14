#!/bin/bash
#
# Submit one EMPS phase-line job per Dmax. Run on the login node from the repo root:
#   bash hpc/sweep_EMPS.sh            # submit
#   DRYRUN=1 bash hpc/sweep_EMPS.sh   # only print the qsub commands
#   H0_MIN=0.9 H0_MAX=1.1 NH0=21 bash hpc/sweep_EMPS.sh   # h0 grid, the same for every Dmax
#
# One-time setup before the first sweep (avoids precompile races between jobs):
#   ~/.juliaup/bin/julia +1.12.6 --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

Dmaxs=(2)
export H0_MIN="${H0_MIN:-0.8}"   # H0_MIN > 0, since δ = ln h0/2
export H0_MAX="${H0_MAX:-1.2}"
export NH0="${NH0:-11}"
WALLTIME="1:00:00"
MEM="16GB"
NODES=1
PPN=16   # CPUs per node
CLUSTER="skiddo"

module swap cluster/$CLUSTER   # qsub submits to the currently loaded cluster module

mkdir -p logs

for D in "${Dmaxs[@]}"; do
    echo "Submitting job for Dmax = $D (h0 from $H0_MIN to $H0_MAX, $NH0 points)"
    export DMAX="$D"
    cmd=(qsub -V -N "EMPS_D${D}" -o logs/ -e logs/ -l walltime=$WALLTIME -l mem=$MEM -l nodes=$NODES:ppn=$PPN hpc/submit_EMPS.sh)
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
done
