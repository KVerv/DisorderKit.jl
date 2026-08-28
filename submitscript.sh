#!/bin/bash
#
#PBS -m a
#PBS -l walltime=1:00:00
#PBS -l nodes=4:ppn=12
#PBS -l mem=64GB
#

STARTDIR=$PBS_O_WORKDIR
export I_MPI_COMPATIBILITY=4


#For doduo cluster
module purge
module load Julia/1.10.4-linux-x86_64 

cd $STARTDIR
echo "PBS: $PBS_ID"
echo "Job number $ix"

ls

echo "Job started at : "`date`
~/.juliaup/bin/julia +1.12.6 --project=. --threads=1 RTFIM_groundstate.jl $ix $ij $iW $id
echo "Job ended at : "`date`