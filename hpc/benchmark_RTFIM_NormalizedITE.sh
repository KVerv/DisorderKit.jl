#!/bin/bash
#
# Benchmark hpc/RTFIM_NormalizedITE.jl for a given L to choose WALLTIME, MEM and PPN of the phase-diagram sweep.
# For every thread count n in THREADS_LIST it runs the benchmark with n Julia and n BLAS threads (both), n Julia threads
# and 1 BLAS thread (julia) and 1 Julia thread and n BLAS threads (blas), and then prints the summary with the
# recommended resources and thread split.
# Run from the repo root:
#   bash hpc/benchmark_RTFIM_NormalizedITE.sh 16                    # submit as one PBS job (ppn = largest thread count)
#   LOCAL=1 bash hpc/benchmark_RTFIM_NormalizedITE.sh 16            # run here, e.g. on an interactive node
#   DRYRUN=1 bash hpc/benchmark_RTFIM_NormalizedITE.sh 16           # only print the job
#   THREADS_LIST="1 4 16" BENCH_STEPS=6 DMAX_SCHEDULE="300:8,200:16" bash hpc/benchmark_RTFIM_NormalizedITE.sh 16
#   MODES="both blas" bash hpc/benchmark_RTFIM_NormalizedITE.sh 16  # only some of the thread splits
#
# Benchmark parameters: THREADS_LIST ("1 2 4 8 16 32"), MODES ("both julia blas"), BENCH_STEPS (8 steps per cap), DELTA (0.0), W (0.1), and the run
# parameters DTAU, ORDER, DMAX_SCHEDULE, DATA_DIR (as in the sweep, so the run tag matches). Summary: EFF_MIN (0.5),
# SAFETY (1.5), NODE_CORES (96). The summary is printed at the end of the job log and can be reprinted any time with
#   julia hpc/benchmark_RTFIM_NormalizedITE.jl --summary <L>
# The job (or LOCAL run) itself first runs Pkg.instantiate() and Pkg.precompile(), so new dependencies are installed.

set -euo pipefail

L="${1:?usage: bash hpc/benchmark_RTFIM_NormalizedITE.sh <L>}"

THREADS_LIST="${THREADS_LIST:-1 2 4 8}"
MODES="${MODES:-both julia blas}"
BENCH_WALLTIME="${BENCH_WALLTIME:-1:00:00}"
BENCH_MEM="${BENCH_MEM:-8GB}"
CLUSTER="${CLUSTER:-skiddo}"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia +1.12.6}"

maxthreads=$(echo $THREADS_LIST | tr ' ' '\n' | sort -n | tail -1)

# Commands of the benchmark: install and precompile the dependencies, one run per (Julia threads, BLAS threads), then the
# summary. For n = 1 the three splits coincide, so every split runs only once
body="set -e"
body+=$'\n'"$JULIA --project=. --threads=$maxthreads -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'"
splits=""
for n in $THREADS_LIST; do
    for mode in $MODES; do
        case $mode in
            both)  split="$n:$n" ;;
            julia) split="$n:1" ;;
            blas)  split="1:$n" ;;
            *)     echo "unknown mode '$mode' in MODES (both, julia, blas)" >&2; exit 1 ;;
        esac
        case " $splits " in *" $split "*) continue ;; esac
        splits+=" $split"
        body+=$'\n'"BLAS_THREADS=${split#*:} $JULIA --project=. --threads=${split%:*} hpc/benchmark_RTFIM_NormalizedITE.jl $L"
    done
done
body+=$'\n'"$JULIA --project=. hpc/benchmark_RTFIM_NormalizedITE.jl --summary $L"

if [ "${LOCAL:-0}" = "1" ]; then
    eval "$body"
    exit 0
fi

# Same run tag as the production script (via its --path mode), for the log folder
mapfile -t paths < <($JULIA --startup-file=no hpc/RTFIM_NormalizedITE.jl --path "$L" 0 0)   # read all output (head would close the pipe early)
tag="${paths[0]}"
logdir="logs/RTFIM_NormalizedITE/benchmarks/$tag/L$L"

job=$(cat <<EOF
#!/bin/bash
cd \$PBS_O_WORKDIR
module purge
echo "PBS: \$PBS_JOBID"
echo "Benchmark RTFIM normalized ITE: L = $L, threads $THREADS_LIST, modes $MODES"
echo "Job started at : \$(date)"
$body
echo "Job ended at : \$(date)"
EOF
)

cmd=(qsub -V -N "bench_RTFIM_L$L" -m a -o "$logdir/" -e "$logdir/" -l walltime=$BENCH_WALLTIME -l mem=$BENCH_MEM -l nodes=1:ppn=$maxthreads)
if [ "${DRYRUN:-0}" = "1" ]; then
    echo "${cmd[*]}"
    echo "$job"
else
    if command -v module > /dev/null 2>&1; then
        module swap cluster/$CLUSTER   # qsub submits to the currently loaded cluster module
    fi
    mkdir -p "$logdir"
    echo "$job" | "${cmd[@]}"
fi
