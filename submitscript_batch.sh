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
    for i in 10; do
        for W in 0.1; do
            for delta in 0.0; do
                echo "Submitting job for parameter values $i, $j, $W and $delta"
                export ix="$i"
                export ij="$j"
                export iW="$W"
                export id="$delta"
                qsub -N "RTFIM_phase_D_${i}_R${j}_W${W}_delta${delta}" -V submitscript.sh 
            done
        done
    done
done
echo "Job ended at : "`date`