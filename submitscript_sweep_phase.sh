#!/bin/bash
#
#PBS -N Sweep
#PBS -m a
#PBS -l walltime=0:05:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=1GB
#

STARTDIR=$PBS_O_WORKDIR
export I_MPI_COMPATIBILITY=4


#For doduo cluster
module purge
# module load Julia/1.12.6-linux-x86_64 

cd $STARTDIR
echo "PBS: $PBS_ID"

ls

echo "Job started at : "`date`
for j in 2; do
    for i in 8 16 32; do
        for W in 0.1; do
            echo "Submitting job for parameter values $i and $j and $W"
            export ix="$i"
            export ij="$j"
            export iW="$W"
            qsub -N "RTFIM_phase_D_${i}_Z${j}_W${W}" -V submitscript_phase.sh 
        done
    done
done
echo "Job ended at : "`date`