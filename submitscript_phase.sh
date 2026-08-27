#!/bin/bash
#
#PBS -m a
#PBS -l walltime=2:30:00
#PBS -l nodes=1:ppn=12
#PBS -l mem=64GB
#

STARTDIR=$PBS_O_WORKDIR
export I_MPI_COMPATIBILITY=4


#For doduo cluster
module purge

cd $STARTDIR
echo "PBS: $PBS_ID"
echo "Job number $ix"

ls

echo "Job started at : "`date`
~/.juliaup/bin/julia +1.12.6 --project=. --threads=1 RTFIM_phase.jl $ix $ij $iW
echo "Job ended at : "`date`