#!/bin/bash
#
# HTCondor version of hpc/submit_obs_BHCorr.sh: one hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl job per
# point (t, μ) that has a stored state and whose observables are missing or outdated, all in one condor cluster. Run on
# the access point (the access point) from the repo root, with the same run parameters as the sweep, so the run tag matches:
#   bash hpc/condor/obs_BHCorr.sh 6
#   DRYRUN=1 bash hpc/condor/obs_BHCorr.sh 6                  # only print the points and the submit file
#   N_SAMPLES=10000 bash hpc/condor/obs_BHCorr.sh 6           # more words where sampled
#   N_EDGE=1 bash hpc/condor/obs_BHCorr.sh 6                  # bulk = all but 1 site at each end
#   R=0.5 DELTA=0.3 OMEGA=0.5 MAXD=32 bash hpc/condor/obs_BHCorr.sh 6
#   TS="0.06 0.10" MUS="0.2 0.4" bash hpc/condor/obs_BHCorr.sh 6   # only these points
#   FORCE=1 bash hpc/condor/obs_BHCorr.sh 6                   # recompute every point with a state
#
# Which points are submitted is decided by hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl --status (nostate,
# uptodate or compute, see its header). Every job gets Project.toml, Manifest.toml, src/, the script and the point's
# state t<t>/mu<μ>.jld2; its mu<μ>_obs.csv is transferred back next to the state under data/.
# Logs, the points file and the submit file: logs/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/condor/obs/
# Plot afterwards: bash hpc/condor/plot_BHCorr.sh <L>

set -euo pipefail

L="${1:?usage: bash hpc/condor/obs_BHCorr.sh <L>}"
[ -f Project.toml ] && [ -d hpc/condor ] || { echo "run from the repo root"; exit 1; }

# Run parameters (they set the run tag), as in hpc/condor/sweep_BHCorr.sh
export U="${U:-1.0}"
export R="${R:-0.0}"
export DELTA="${DELTA:-0.0}"
export OMEGA="${OMEGA:-0.0}"
export NMAX="${NMAX:-4}"
export DTAU="${DTAU:-0.05}"
export MAXD="${MAXD:-16}"
export DATA_DIR=data
export N_SAMPLES="${N_SAMPLES:-1000}"   # disorder words are enumerated exactly if ∏ᵢ qᵢ ≤ N_SAMPLES, else sampled
# Sites excluded at each end for S_avg, S_inhom and S_bstd (bulk bonds k = N_EDGE+1..L-N_EDGE-1, needs L ≥ 2 N_EDGE + 3)
export N_EDGE="${N_EDGE:-2}"

# Resources per job
PPN="${PPN:-4}"
JULIA_THREADS="${JULIA_THREADS:-$PPN}"
MEM="${MEM:-6 G}"
DISK="${DISK:-2 G}"
REQUIREMENTS="${REQUIREMENTS:-}"
NICE="${NICE:-TRUE}"
UG_NAME="${UG_NAME:-<login>}"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia +$(sed -n 's/^julia_version = "\(.*\)"/\1/p' Manifest.toml)}"   # Julia on the access point (juliaup, Manifest version)

script=hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl

# Run tag and L folder (t and μ are irrelevant for them)
mapfile -t paths < <($JULIA --startup-file=no $script --status "$L" 0 0)
tag="${paths[0]}"
folder="$DATA_DIR/BoseHubbardCorrelated_NormalizedITE/$tag/L$L"
logdir="logs/BoseHubbardCorrelated_NormalizedITE/$tag/L$L/condor/obs"
[ -d "$folder" ] || { echo "no results in $folder"; exit 1; }

# Candidate points (t, μ): from TS × MUS if given, else every stored state
candidates=()
if [ -n "${TS:-}" ] || [ -n "${MUS:-}" ]; then
    for t in ${TS:-$(ls "$folder" | sed -n 's/^t\(-\{0,1\}[0-9.]*\)$/\1/p')}; do
        t=$(LC_ALL=C printf "%.4f" "$t")
        for mu in ${MUS:-$(ls "$folder/t$t" 2> /dev/null | sed -n 's/^mu\(-\{0,1\}[0-9.]*\)\.jld2$/\1/p')}; do
            candidates+=("$t $(LC_ALL=C printf "%.4f" "$mu")")
        done
    done
else
    for f in "$folder"/t*/mu*.jld2; do
        [ -e "$f" ] || continue
        t=$(basename "$(dirname "$f")"); t="${t#t}"
        mu=$(basename "$f" .jld2); mu="${mu#mu}"
        candidates+=("$t $mu")
    done
fi

# Points of this sweep that still have an observables job in the queue ("t mu" per line): skipped
active=""
if command -v condor_q > /dev/null 2>&1; then
    active=$(condor_q "$USER" -constraint "BHC_stage =?= \"obs\" && BHC_tag =?= \"$tag\" && BHC_L =?= $L" \
             -af BHC_t BHC_mu 2> /dev/null || true)
fi

points=()
uptodate=0
queued=0
nostate=0
for p in "${candidates[@]}"; do
    read -r t mu <<< "$p"
    if [ -n "$active" ] && grep -qx "$t $mu" <<< "$active"; then
        echo "queued L=$L t=$t μ=$mu (job in the queue)"
        queued=$((queued + 1))
        continue
    fi
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
    # With FORCE=1 the job must recompute too: remove the observables file before submitting, so it is not uptodate
    if [ "${FORCE:-0}" = "1" ] && [ "${DRYRUN:-0}" != "1" ]; then
        rm -f "$obs"
    fi
    points+=("$t $mu")
done
if [ "${#points[@]}" = "0" ]; then
    echo "L = $L: nothing to submit, $uptodate up to date, $queued already queued, $nostate without a state"
    exit 0
fi

stamp=$(date +%Y%m%d_%H%M%S)
pointsfile="$logdir/points_$stamp.txt"
submitfile="$logdir/obs_$stamp.submit"
reqs="has_julia${REQUIREMENTS:+ && ($REQUIREMENTS)}"
pointdir="$folder/t\$(t)"

submit=$(cat <<EOF
# Correlated Bose-Hubbard observables, L = $L, run tag $tag (generated by hpc/condor/obs_BHCorr.sh)
JOBNAME = BHC_obs_L${L}
UG_NAME = $UG_NAME

executable          = hpc/condor/run_julia.sh
transfer_executable = TRUE
arguments           = "$script $L \$(t) \$(mu)"
environment         = "HOME=/tmp/\$(UG_NAME) JULIA_THREADS=$JULIA_THREADS U=$U R=$R DELTA=$DELTA OMEGA=$OMEGA NMAX=$NMAX DTAU=$DTAU MAXD=$MAXD DATA_DIR=$DATA_DIR N_SAMPLES=$N_SAMPLES N_EDGE=$N_EDGE"

should_transfer_files   = TRUE
preserve_relative_paths = TRUE
when_to_transfer_output = ON_EXIT
transfer_input_files    = Project.toml, Manifest.toml, src, $script, $pointdir/mu\$(mu).jld2
transfer_output_files   = $pointdir/mu\$(mu)_obs.csv

nice_user      = $NICE
request_cpus   = $PPN
request_memory = $MEM
request_disk   = $DISK
requirements   = $reqs
rank           = cpu_performance

# Labels for matching jobs to sweep points (hpc/condor/*.sh, dashboard)
+BHC_stage = "obs"
+BHC_tag   = "$tag"
+BHC_L     = $L
+BHC_t     = "\$(t)"
+BHC_mu    = "\$(mu)"

output = $logdir/\$(JOBNAME)_t\$(t)_mu\$(mu)_\$(Cluster).out
error  = $logdir/\$(JOBNAME)_t\$(t)_mu\$(mu)_\$(Cluster).err
log    = $logdir/\$(Cluster).log

queue t, mu from $pointsfile
EOF
)

if [ "${DRYRUN:-0}" = "1" ]; then
    printf '%s\n' "${points[@]}" | sed 's/^/point  /'
    echo "$submit" | sed 's/^/       | /'
    echo "L = $L, N_SAMPLES = $N_SAMPLES, N_EDGE = $N_EDGE: ${#points[@]} job(s) to submit, $uptodate up to date, $queued already queued, $nostate without a state"
else
    mkdir -p "$logdir"
    printf '%s\n' "${points[@]}" > "$pointsfile"
    echo "$submit" > "$submitfile"
    condor_submit "$submitfile"
    echo "L = $L, N_SAMPLES = $N_SAMPLES, N_EDGE = $N_EDGE: ${#points[@]} job(s) submitted, $uptodate up to date, $queued already queued, $nostate without a state; submit file $submitfile"
fi
