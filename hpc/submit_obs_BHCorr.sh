#!/bin/bash
#
# Submit one hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl job per point (t, μ) of the phase diagram that has a
# stored state and whose observables are missing, older than the state, or were sampled with fewer than N_SAMPLES words.
# Run on the login node from the repo root, with the same run parameters as the sweep, so the run tag matches:
#   bash hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh 6
#   N_SAMPLES=10000 bash hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh 6   # more words where sampled
#   R=0.5 DELTA=0.3 OMEGA=0.5 MAXD=32 bash hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh 6
#   TS="0.06 0.10" MUS="0.2 0.4" bash hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh 6   # only these points
#   DRYRUN=1 bash hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh 6      # only print what would be submitted
#   FORCE=1 bash hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh 6       # recompute every point with a state
#
# By default every point with a state t<t>/mu<μ>.jld2 under the L<L> folder is considered; TS and MUS (the mean values
# t/U and μ/U, formatted as %.4f) restrict them.
#   N_EDGE=2 bash hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh 8   # S_bstd without 2 sites at each end
# Run parameters (U, R, DELTA, OMEGA, NMAX, DTAU, MAXD, DATA_DIR, N_SAMPLES, N_EDGE) and their defaults are set below; override them from the
# caller's environment. They are written into every job script,
# hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl has no defaults of its own.
# Results: <DATA_DIR>/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/t<t>/mu<μ>_obs.csv
# Logs:    logs/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/obs/
# Plot afterwards (reads only the observables files): bash hpc/submit_plot_BoseHubbardCorrelated_NormalizedITE.sh <L>

set -euo pipefail

L="${1:?usage: bash hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh <L>}"

# Run parameters: U, R, NMAX, DTAU, MAXD and DATA_DIR as in hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh (they set
# the run tag)
export U="${U:-1.0}"
export R="${R:-0.0}"
export DELTA="${DELTA:-0.0}"   # potential disorder: μ ± DELTA
export OMEGA="${OMEGA:-0.0}"   # hopping disorder: t(1 ± OMEGA), 0 ≤ OMEGA ≤ 1
export NMAX="${NMAX:-4}"
export DTAU="${DTAU:-0.05}"
export MAXD="${MAXD:-16}"
export DATA_DIR="${DATA_DIR:-${VSC_DATA:-.}/data}"
export N_SAMPLES="${N_SAMPLES:-1000}"   # disorder words are enumerated exactly if ∏ᵢ qᵢ ≤ N_SAMPLES, else sampled
# Sites excluded at each end for S_bstd = 𝔼_w[std_k S_w(k)] over the bulk bonds k = N_EDGE+1..L-N_EDGE-1 (needs
# L ≥ 2 N_EDGE + 3; 0 = all bonds). Points whose observables were computed with another N_EDGE are recomputed.
export N_EDGE="${N_EDGE:-2}"

# Resources per job
WALLTIME="${WALLTIME:-0:15:00}"
MEM="${MEM:-6GB}"
PPN="${PPN:-4}"
CLUSTER="${CLUSTER:-skiddo}"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia +1.12.6}"

script=hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl

# Run tag and L folder (t and μ are irrelevant for them)
mapfile -t paths < <($JULIA --startup-file=no $script --status "$L" 0 0)
tag="${paths[0]}"
folder="$DATA_DIR/BoseHubbardCorrelated_NormalizedITE/$tag/L$L"
logdir="logs/BoseHubbardCorrelated_NormalizedITE/$tag/L$L/obs"
[ -d "$folder" ] || { echo "no results in $folder"; exit 1; }

# Points (t, μ): from TS × MUS if given, else every stored state
points=()
if [ -n "${TS:-}" ] || [ -n "${MUS:-}" ]; then
    for t in ${TS:-$(ls "$folder" | sed -n 's/^t\(-\{0,1\}[0-9.]*\)$/\1/p')}; do
        t=$(LC_ALL=C printf "%.4f" "$t")
        for mu in ${MUS:-$(ls "$folder/t$t" 2> /dev/null | sed -n 's/^mu\(-\{0,1\}[0-9.]*\)\.jld2$/\1/p')}; do
            points+=("$t $(LC_ALL=C printf "%.4f" "$mu")")
        done
    done
else
    for f in "$folder"/t*/mu*.jld2; do
        [ -e "$f" ] || continue
        t=$(basename "$(dirname "$f")"); t="${t#t}"
        mu=$(basename "$f" .jld2); mu="${mu#mu}"
        points+=("$t $mu")
    done
fi

if [ "${DRYRUN:-0}" != "1" ] && command -v module > /dev/null 2>&1; then
    module swap cluster/$CLUSTER   # qsub submits to the currently loaded cluster module
fi

submitted=0
uptodate=0
nostate=0
shown_job=0
for p in "${points[@]}"; do
    read -r t mu <<< "$p"
    # The Julia script decides the files and whether the observables need to be (re)computed
    mapfile -t paths < <($JULIA --startup-file=no $script --status "$L" "$t" "$mu")
    obs="${paths[2]}"
    st="${paths[3]}"
    if [ "$st" = "nostate" ]; then
        echo "skip   L=$L t=$t μ=$mu (no state)"
        nostate=$((nostate + 1))
        continue
    fi
    if [ "$st" = "uptodate" ] && [ "${FORCE:-0}" != "1" ]; then
        uptodate=$((uptodate + 1))
        continue
    fi
    # With FORCE=1 the job must recompute too: remove the observables file when the job starts
    rmcmd=$([ "${FORCE:-0}" = "1" ] && echo "rm -f \"$obs\"" || echo "")

    # Job script. Unescaped variables are filled in now, escaped ones (\$) when the job runs.
    job=$(cat <<EOF
#!/bin/bash
cd \$PBS_O_WORKDIR
module purge
echo "PBS: \$PBS_JOBID"
echo "Correlated Bose-Hubbard observables: L = $L, t = $t, μ = $mu, run tag $tag, N_SAMPLES = $N_SAMPLES, N_EDGE = $N_EDGE"
echo "Job started at : \$(date)"
export U=$U R=$R DELTA=$DELTA OMEGA=$OMEGA NMAX=$NMAX DTAU=$DTAU MAXD=$MAXD DATA_DIR="$DATA_DIR" N_SAMPLES=$N_SAMPLES N_EDGE=$N_EDGE
$rmcmd
$JULIA --project=. --threads=$PPN -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
$JULIA --project=. --threads=$PPN $script $L $t $mu
echo "Job ended at : \$(date)"
EOF
)
    name="BHC_obs_L${L}_t${t}_mu${mu}"
    cmd=(qsub -V -N "$name" -m a -o "$logdir/" -e "$logdir/" -l walltime=$WALLTIME -l mem=$MEM -l nodes=1:ppn=$PPN)
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "submit L=$L t=$t μ=$mu -> $obs"
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

echo "L = $L, N_SAMPLES = $N_SAMPLES, N_EDGE = $N_EDGE: $submitted job(s) $([ "${DRYRUN:-0}" = "1" ] && echo "to submit" || echo "submitted"), $uptodate up to date, $nostate without a state"
