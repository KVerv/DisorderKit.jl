#!/bin/bash
#
# Benchmark hpc/BoseHubbard_NormalizedITE.jl for a given L to choose WALLTIME, MEM and PPN of the μ sweep.
# For every thread count n in THREADS_LIST it runs the benchmark with n Julia and n BLAS threads (both), n Julia threads
# and 1 BLAS thread (julia) and 1 Julia thread and n BLAS threads (blas), and then prints the summary with the
# recommended resources and thread split.
# Run from the repo root:
#   bash hpc/benchmark_BoseHubbard_NormalizedITE.sh 6                     # submit as one PBS job (ppn = largest thread count)
#   LOCAL=1 bash hpc/benchmark_BoseHubbard_NormalizedITE.sh 6             # run here, e.g. on an interactive node
#   DRYRUN=1 bash hpc/benchmark_BoseHubbard_NormalizedITE.sh 6            # only print the job
#   THREADS_LIST="1 4 16" BENCH_STEPS=10 MAXD=32 bash hpc/benchmark_BoseHubbard_NormalizedITE.sh 6
#   MODES="both blas" bash hpc/benchmark_BoseHubbard_NormalizedITE.sh 6   # only some of the thread splits
#
# Benchmark parameters: THREADS_LIST ("1 2 4 8"), MODES ("both julia blas"), BENCH_STEPS (20), MU (1.0, superfluid
# between the n = 1 and n = 2 Mott lobes, so the bonds saturate at MAXD), and the run parameters T, U, NMAX, DTAU, MAXD,
# MAXITER, CROSS_WIDTH, DATA_DIR with the defaults of hpc/sweep_BoseHubbard_NormalizedITE.sh (so the run tag matches).
# Summary: EFF_MIN (0.5), SAFETY (1.5), NODE_CORES (96). The summary is printed at the end of the job log and can be
# reprinted any time with the same run parameters set, e.g.
#   T=0.16 U=1.0 NMAX=4 DTAU=0.05 MAXD=16 MAXITER=500 DATA_DIR=$VSC_DATA/data julia hpc/benchmark_BoseHubbard_NormalizedITE.jl --summary <L>
# The job (or LOCAL run) itself first runs Pkg.instantiate() and Pkg.precompile(), so new dependencies are installed.

set -euo pipefail

L="${1:?usage: bash hpc/benchmark_BoseHubbard_NormalizedITE.sh <L>}"

# Run parameters as in hpc/sweep_BoseHubbard_NormalizedITE.sh (T, U, NMAX, DTAU, MAXD set the run tag)
export T="${T:-0.16}"
export U="${U:-1.0}"
export NMAX="${NMAX:-4}"
export DTAU="${DTAU:-0.05}"
export MAXD="${MAXD:-16}"
export MAXITER="${MAXITER:-500}"
export CROSS_WIDTH="${CROSS_WIDTH:-32}"
export DATA_DIR="${DATA_DIR:-${VSC_DATA:-.}/data}"
export MU="${MU:-1.0}"
export BENCH_STEPS="${BENCH_STEPS:-20}"

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
body+=$'\n'"export T=$T U=$U NMAX=$NMAX DTAU=$DTAU MAXD=$MAXD MAXITER=$MAXITER CROSS_WIDTH=$CROSS_WIDTH DATA_DIR=\"$DATA_DIR\""
body+=$'\n'"export MU=$MU BENCH_STEPS=$BENCH_STEPS"
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
        body+=$'\n'"BLAS_THREADS=${split#*:} $JULIA --project=. --threads=${split%:*} hpc/benchmark_BoseHubbard_NormalizedITE.jl $L"
    done
done
body+=$'\n'"$JULIA --project=. hpc/benchmark_BoseHubbard_NormalizedITE.jl --summary $L"

if [ "${LOCAL:-0}" = "1" ]; then
    eval "$body"
    exit 0
fi

# Same run tag as the production script (via its --path mode, μ is irrelevant for the tag), for the log folder
mapfile -t paths < <($JULIA --startup-file=no hpc/BoseHubbard_NormalizedITE.jl --path "$L" 0)
tag="${paths[0]}"
logdir="logs/BoseHubbard_NormalizedITE/benchmarks/$tag/L$L"

job=$(cat <<EOF
#!/bin/bash
cd \$PBS_O_WORKDIR
module purge
echo "PBS: \$PBS_JOBID"
echo "Benchmark Bose-Hubbard normalized ITE: L = $L, μ = $MU, threads $THREADS_LIST, modes $MODES"
echo "Job started at : \$(date)"
$body
echo "Job ended at : \$(date)"
EOF
)

cmd=(qsub -V -N "bench_BH_L$L" -m a -o "$logdir/" -e "$logdir/" -l walltime=$BENCH_WALLTIME -l mem=$BENCH_MEM -l nodes=1:ppn=$maxthreads)
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
