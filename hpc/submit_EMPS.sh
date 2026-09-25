#!/bin/bash
#
#PBS -m a
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=8GB
#
# Single EMPS phase-line job. Submit from the repo root:
#   W=0.25 DMAX=4 qsub -V -N EMPS_W0.25_D4 hpc/submit_EMPS.sh

STARTDIR=$PBS_O_WORKDIR
export I_MPI_COMPATIBILITY=4

module purge

cd $STARTDIR
echo "PBS: $PBS_JOBID"
echo "Parameters: W = $W, Dmax = $DMAX"

echo "Job started at : "`date`
~/.juliaup/bin/julia +1.12.6 --project=. --threads=1 hpc/EMPS_hpc.jl $W $DMAX
echo "Job ended at : "`date`
