#!/bin/bash
#
#PBS -m a

#
# Single EMPS phase-line job. Submit from the repo root:
#   DMAX=4 qsub -V -N EMPS_D4 hpc/submit_EMPS.sh

STARTDIR=$PBS_O_WORKDIR
export I_MPI_COMPATIBILITY=4

module purge

cd $STARTDIR
echo "PBS: $PBS_JOBID"
echo "Parameters: Dmax = $DMAX, h0 from $H0_MIN to $H0_MAX ($NH0 points)"

echo "Job started at : "`date`
~/.juliaup/bin/julia +1.12.6 --project=. --threads=1 hpc/EMPS_hpc.jl $DMAX
echo "Job ended at : "`date`
