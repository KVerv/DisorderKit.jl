#!/bin/bash
#
# HTCondor version of hpc/submit_obs_BHCorr.sh: hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl jobs for the
# points (t, μ) that have a stored state and whose observables are missing or outdated, all in one condor cluster, CHUNK
# points per job (default 20; CHUNK=1 = one job per point). A job spends ~10 min (up to ~45 min) installing Julia and
# the packages in its private /tmp and only ~0.5-1 min per point (L = 10), so batching saves most of the time. The
# sweep jobs already compute the observables; this is for recomputing them (other N_EDGE or N_SAMPLES, or a changed
# observables script) without rerunning the ITE. Run on the access point from the repo root (normally through the
# dashboard, which first uploads the needed states), with the same run parameters as the sweep:
#   bash hpc/condor/obs_BHCorr.sh 6
#   DRYRUN=1 bash hpc/condor/obs_BHCorr.sh 6                  # only print the points and the submit file
#   N_SAMPLES=10000 bash hpc/condor/obs_BHCorr.sh 6           # more words where sampled
#   N_EDGE=1 bash hpc/condor/obs_BHCorr.sh 6                  # bulk = all but 1 site at each end
#   R=0.5 DELTA=0.3 OMEGA=0.5 MAXD=32 bash hpc/condor/obs_BHCorr.sh 6
#   TS="0.06 0.10" MUS="0.2 0.4" bash hpc/condor/obs_BHCorr.sh 6   # only these points
#   FORCE=1 bash hpc/condor/obs_BHCorr.sh 6                   # recompute every point with a state
#   POINTS="0.1000 0.3000,0.1200 0.3000" bash hpc/condor/obs_BHCorr.sh 6   # only these points
#   CHUNK=5 bash hpc/condor/obs_BHCorr.sh 6                   # 5 points per job (more, shorter jobs)
#
# Which points are submitted is decided by hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl --status (nostate,
# uptodate or compute, see its header; OBS_VERSION = git hash-object of the script, so a changed script recomputes).
# Every job gets Project.toml, Manifest.toml, src/, the script and the states t<t>/mu<μ>.jld2 of its points, and runs
# the script once for all of them (a failing point does not stop the others). On exit it deletes its copies of the
# states (REMOVE_STATES=1 in hpc/condor/run_julia.sh) and transfers back the L folder, i.e. the mu<μ>_obs.csv files, which land next to the states
# under data/. Job attribute BHC_points = "t1_mu1,t2_mu2,..." (the dashboard matches jobs to points with it).
# Logs (one .out/.err per job, BHC_obs_L<L>_<cluster>.<proc>), the points file and the submit file:
# logs/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/condor/obs/
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
[ "$L" -ge $((2 * N_EDGE + 3)) ] || { echo "N_EDGE = $N_EDGE leaves fewer than 2 bulk bonds for L = $L (need L ≥ 2 N_EDGE + 3)"; exit 1; }

# Resources per job
PPN="${PPN:-4}"
CHUNK="${CHUNK:-20}"   # points per job
[ "$CHUNK" -ge 1 ] || { echo "CHUNK = $CHUNK must be ≥ 1"; exit 1; }
JULIA_THREADS="${JULIA_THREADS:-$PPN}"
MEM="${MEM:-6 G}"
DISK="${DISK:-2 G}"   # per job, holds the states of its CHUNK points
REQUIREMENTS="${REQUIREMENTS:-}"
NICE="${NICE:-TRUE}"
UG_NAME="${UG_NAME:-$USER}"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia +$(sed -n 's/^julia_version = "\(.*\)"/\1/p' Manifest.toml)}"   # Julia on the access point (juliaup, Manifest version)

script=hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl
export OBS_VERSION=$(git hash-object "$script" 2> /dev/null || echo "")

# Run tag and L folder (t and μ are irrelevant for them)
mapfile -t paths < <($JULIA --startup-file=no $script --status "$L" 0 0)
tag="${paths[0]}"
folder="$DATA_DIR/BoseHubbardCorrelated_NormalizedITE/$tag/L$L"
logdir="logs/BoseHubbardCorrelated_NormalizedITE/$tag/L$L/condor/obs"
[ -d "$folder" ] || { echo "no results in $folder"; exit 1; }

# Candidate points (t, μ): POINTS if given, else TS × MUS if given, else every stored state
candidates=()
if [ -n "${POINTS:-}" ]; then
    IFS=',' read -ra pairs <<< "$POINTS"
    for p in "${pairs[@]}"; do
        read -r t mu <<< "$p"
        [ -n "${mu:-}" ] && candidates+=("$(LC_ALL=C printf "%.4f %.4f" "$t" "$mu")")
    done
elif [ -n "${TS:-}" ] || [ -n "${MUS:-}" ]; then
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

# Points of this sweep that still have an observables job in the queue ("t mu" per line): skipped. Jobs carry their
# points in BHC_points ("t1_mu1,t2_mu2,..."); single-point jobs of older submits in BHC_t and BHC_mu.
active=""
if command -v condor_q > /dev/null 2>&1; then
    active=$(condor_q "$USER" -constraint "BHC_stage =?= \"obs\" && BHC_tag =?= \"$tag\" && BHC_L =?= $L" \
             -af BHC_t BHC_mu BHC_points 2> /dev/null |
             awk '$3 != "undefined" { n = split($3, p, ","); for (i = 1; i <= n; i++) { sub("_", " ", p[i]); print p[i] } next }
                  $1 != "undefined" { print $1, $2 }' || true)
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
njobs=$(( (${#points[@]} + CHUNK - 1) / CHUNK ))

submit=$(cat <<EOF
# Correlated Bose-Hubbard observables, L = $L, run tag $tag, ${#points[@]} point(s) in $njobs job(s) of at most $CHUNK
# (generated by hpc/condor/obs_BHCorr.sh)
JOBNAME = BHC_obs_L${L}
UG_NAME = $UG_NAME

executable          = hpc/condor/run_julia.sh
transfer_executable = TRUE
environment         = "HOME=/tmp/\$(UG_NAME) JULIA_THREADS=$JULIA_THREADS U=$U R=$R DELTA=$DELTA OMEGA=$OMEGA NMAX=$NMAX DTAU=$DTAU MAXD=$MAXD DATA_DIR=$DATA_DIR N_SAMPLES=$N_SAMPLES N_EDGE=$N_EDGE OBS_VERSION=$OBS_VERSION REMOVE_STATES=1"

should_transfer_files   = TRUE
preserve_relative_paths = TRUE
when_to_transfer_output = ON_EXIT
# run_julia.sh deletes the job's copies of the states on exit (REMOVE_STATES), so the L folder carries back only the _obs.csv files
transfer_output_files   = $folder

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

output = $logdir/\$(JOBNAME)_\$(Cluster).\$(Process).out
error  = $logdir/\$(JOBNAME)_\$(Cluster).\$(Process).err
log    = $logdir/\$(Cluster).log
EOF
)
# One block per job: its points as arguments, their states as input files
for ((i = 0; i < ${#points[@]}; i += CHUNK)); do
    args="" inputs="" labels=""
    for p in "${points[@]:i:CHUNK}"; do
        read -r t mu <<< "$p"
        args+=" $t $mu"
        inputs+=", $folder/t$t/mu$mu.jld2"
        labels+="${labels:+,}${t}_${mu}"
    done
    submit+=$(cat <<EOF


arguments            = "$script $L$args"
transfer_input_files = Project.toml, Manifest.toml, src, $script$inputs
+BHC_points          = "$labels"
queue
EOF
)
done

if [ "${DRYRUN:-0}" = "1" ]; then
    printf '%s\n' "${points[@]}" | sed 's/^/point  /'
    echo "$submit" | sed 's/^/       | /'
    echo "L = $L, N_SAMPLES = $N_SAMPLES, N_EDGE = $N_EDGE: ${#points[@]} point(s) to submit in $njobs job(s), $uptodate up to date, $queued already queued, $nostate without a state"
else
    mkdir -p "$logdir"
    printf '%s\n' "${points[@]}" > "$pointsfile"
    echo "$submit" > "$submitfile"
    condor_submit "$submitfile"
    echo "L = $L, N_SAMPLES = $N_SAMPLES, N_EDGE = $N_EDGE: ${#points[@]} point(s) submitted in $njobs job(s), $uptodate up to date, $queued already queued, $nostate without a state; submit file $submitfile"
fi
