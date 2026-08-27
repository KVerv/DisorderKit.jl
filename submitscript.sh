#!/bin/bash
#
#PBS -m a
#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=96
#PBS -l mem=264GB
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
julia --project=. --threads=96 density_matrices.jl $ix $ij $in
echo "Job ended at : "`date`