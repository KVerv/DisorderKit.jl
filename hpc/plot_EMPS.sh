#!/bin/bash
#
# Submit hpc/plot_EMPS.jl as a job. Run on the login node from the repo root:
#   bash hpc/plot_EMPS.sh                                  # plot the default data folder
#   FOLDER=$VSC_DATA/data/EMPS_phase_line/dtau0.05_maxiter20 bash hpc/plot_EMPS.sh
#   PLOT_MODE=fixW bash hpc/plot_EMPS.sh                   # one figure per W (fixD: one per Dmax, all: default)
#   DRYRUN=1 bash hpc/plot_EMPS.sh                         # only print the qsub command and job
#
# Plots end up in <PLOT_DIR>/EMPS_phase_line/<data folder name>/ (PLOT_DIR defaults to $VSC_DATA/plots).

WALLTIME="1:00:00"
MEM="8GB"
NODES=1
PPN=1   # CPUs per node
CLUSTER="shinx"

module swap cluster/$CLUSTER   # qsub submits to the currently loaded cluster module

mkdir -p logs

# Arguments for plot_EMPS.jl: [data_folder]
args=()
if [ -n "$FOLDER" ]; then
    args=("$FOLDER")
fi

# Job script. Unescaped variables are filled in now, escaped ones (\$) when the job runs.
job=$(cat <<EOF
#!/bin/bash
cd \$PBS_O_WORKDIR
module purge
echo "PBS: \$PBS_JOBID"
echo "Job started at : \$(date)"
~/.juliaup/bin/julia +1.12.6 --project=. --threads=1 hpc/plot_EMPS.jl ${args[@]}
echo "Job ended at : \$(date)"
EOF
)

cmd=(qsub -V -N "EMPS_plot" -m a -o logs/ -e logs/ -l walltime=$WALLTIME -l mem=$MEM -l nodes=$NODES:ppn=$PPN)
if [ "${DRYRUN:-0}" = "1" ]; then
    echo "${cmd[@]}"
    echo "$job"
else
    echo "$job" | "${cmd[@]}"
fi
