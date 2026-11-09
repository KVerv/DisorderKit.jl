#!/bin/bash
#
# Submit one hpc/BoseHubbardCorrelated_NormalizedITE.jl job per point (t, μ) of the phase diagram of the Bose-Hubbard
# chain with correlated hopping and potential disorder. Run on the login node from the repo root:
#   bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6                    # submit the default (t, μ) grid for L = 6
#   DRYRUN=1 bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6           # only print what would be submitted
#   TS="0.05 0.1" MUS="0.2 0.4 0.6" bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6
#   R=0.5 DELTA=0.3 OMEGA=0.5 MAXD=32 MAXITER=1000 bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6
#   MAXITER=4000 bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6       # continue the unconverged points up to 4000 steps
#   FORCE=1 bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6            # resubmit every point from scratch, also those that are done
#   PPN=4 JULIA_THREADS=1 BLAS_THREADS=4 bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh 6
#
# Run parameters (U, R, DELTA, OMEGA, NMAX, DTAU, MAXD, MAXITER, CONVTOL, CROSS_WIDTH, DATA_DIR) and their defaults are set below;
# override them from the caller's environment. They are written into every job script,
# hpc/BoseHubbardCorrelated_NormalizedITE.jl has no defaults of its own.
# A point is only submitted if it has no stored state (run), or if its stored state is not converged for this CONVTOL
# (final ϵ_conv > CONVTOL) and this sweep has a larger MAXITER (resume: the job continues the ITE from the stored state,
# up to MAXITER steps in total; RESUME=0 or FORCE=1 restart from scratch instead) or only a smaller CONVTOL (rerun, from
# scratch). Converged points, and unconverged points that the same settings would not improve, are skipped. The decision
# is made by hpc/BoseHubbardCorrelated_NormalizedITE.jl --status (it reads the stored params and ϵ_conv history).
# Results: <DATA_DIR>/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/t<t>/mu<μ>.jld2
# Logs:    logs/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/
# Observables afterwards: bash hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh <L> (one job per point),
# then the plot job: bash hpc/submit_plot_BoseHubbardCorrelated_NormalizedITE.sh <L> (both with the same run parameters)
#
# Every job first runs Pkg.instantiate() and Pkg.precompile() on its compute node, so new dependencies are installed
# (concurrent jobs wait for each other through Julia's precompile locks).

set -euo pipefail

L="${1:?usage: bash hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh <L>}"

# (t/U, μ/U) grid of the mean values (disorder values t(1±OMEGA) and μ±DELTA), formatted as %.4f in the file names
TS="${TS:-$(LC_ALL=C seq 0.02 0.02 0.30)}"
MUS="${MUS:-$(LC_ALL=C seq 0 0.1 1.6)}"

# Run parameters (U, R, DELTA, OMEGA, NMAX, DTAU, MAXD set the run tag)
export U="${U:-1.0}"
export R="${R:-0.0}"   # correlation of hopping and potential: P = (1±R)/4
export DELTA="${DELTA:-0.0}"   # potential disorder: μ ± DELTA
export OMEGA="${OMEGA:-0.0}"   # hopping disorder: t(1 ± OMEGA), 0 ≤ OMEGA ≤ 1
export NMAX="${NMAX:-4}"
export DTAU="${DTAU:-0.05}"
export MAXD="${MAXD:-16}"
export MAXITER="${MAXITER:-2000}"
export CONVTOL="${CONVTOL:-1e-6}"
export CROSS_WIDTH="${CROSS_WIDTH:-32}"
export DATA_DIR="${DATA_DIR:-${VSC_DATA:-.}/data}"
[ "${FORCE:-0}" = "1" ] && RESUME=0
export RESUME="${RESUME:-1}"   # continue unconverged stored states (see above)

# Resources per job
WALLTIME="${WALLTIME:-1:00:00}"
MEM="${MEM:-6GB}"
NODES=1
PPN="${PPN:-8}"   # CPUs per node
JULIA_THREADS="${JULIA_THREADS:-1}"   # Julia runs with --threads=$JULIA_THREADS
BLAS_THREADS="${BLAS_THREADS:-$PPN}"     # and BLAS.set_num_threads($BLAS_THREADS)
CLUSTER="${CLUSTER:-skiddo}"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia +1.12.6}"

if command -v module > /dev/null 2>&1; then
    module swap cluster/$CLUSTER   # qsub submits to the currently loaded cluster module
fi

submitted=0
rerun=0
skipped=0
shown_job=0
for t in $TS; do
    t=$(LC_ALL=C printf "%.4f" "$t")
    for mu in $MUS; do
        mu=$(LC_ALL=C printf "%.4f" "$mu")
        # The Julia script decides the run tag and the output file, so the paths always agree
        # and whether the point has to be (re)run for this MAXITER and CONVTOL
        mapfile -t paths < <($JULIA --project=. --startup-file=no hpc/BoseHubbardCorrelated_NormalizedITE.jl --status "$L" "$t" "$mu")
        tag="${paths[0]}"
        out="${paths[1]}"
        st="${paths[2]}"
        reason="${paths[3]}"
        logdir="logs/BoseHubbardCorrelated_NormalizedITE/$tag/L$L"

        if [ "$st" = "uptodate" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "skip   L=$L t=$t μ=$mu ($reason)"
            skipped=$((skipped + 1))
            continue
        fi
        if [ "$st" = "rerun" ] || [ "$st" = "resume" ]; then
            echo "$(printf '%-6s' "$st") L=$L t=$t μ=$mu ($reason)"
            rerun=$((rerun + 1))
        fi

        # Job script. Unescaped variables are filled in now, escaped ones (\$) when the job runs.
        job=$(cat <<EOF
#!/bin/bash
cd \$PBS_O_WORKDIR
module purge
echo "PBS: \$PBS_JOBID"
echo "Correlated Bose-Hubbard normalized ITE: L = $L, t = $t, μ = $mu, r = $R, Δ = $DELTA, Ω = $OMEGA"
echo "Job started at : \$(date)"
export U=$U R=$R DELTA=$DELTA OMEGA=$OMEGA NMAX=$NMAX DTAU=$DTAU MAXD=$MAXD
export MAXITER=$MAXITER CONVTOL=$CONVTOL CROSS_WIDTH=$CROSS_WIDTH RESUME=$RESUME DATA_DIR="$DATA_DIR"
export BLAS_THREADS=$BLAS_THREADS
$JULIA --project=. --threads=$JULIA_THREADS -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
$JULIA --project=. --threads=$JULIA_THREADS hpc/BoseHubbardCorrelated_NormalizedITE.jl $L $t $mu
echo "Job ended at : \$(date)"
EOF
)
        name="BHC_L${L}_t${t}_mu${mu}"
        cmd=(qsub -V -N "$name" -m a -o "$logdir/" -e "$logdir/" -l walltime=$WALLTIME -l mem=$MEM -l nodes=$NODES:ppn=$PPN)
        if [ "${DRYRUN:-0}" = "1" ]; then
            echo "submit L=$L t=$t μ=$mu -> $out"
            echo "       ${cmd[*]}"
            if [ "$shown_job" = "0" ]; then
                echo "$job" | sed 's/^/       | /'
                shown_job=1
            fi
        else
            mkdir -p "$logdir"
            echo "$job" | "${cmd[@]}"
        fi
        submitted=$((submitted + 1))
    done
done

echo "L = $L: $submitted job(s) $([ "${DRYRUN:-0}" = "1" ] && echo "to submit" || echo "submitted") ($rerun rerun), $skipped skipped (done)"
