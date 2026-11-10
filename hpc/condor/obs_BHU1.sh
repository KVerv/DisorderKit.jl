#!/bin/bash
#
# Observables-only HTCondor jobs for a hpc/condor/sweep_BHU1.sh sweep (the generic counterpart of
# hpc/condor/obs_BHCorr.sh): hpc/observables_BoseHubbardU1_NormalizedITE.jl for the points with a stored state whose
# observables are missing or outdated, CHUNK points per job (Julia setup dominates a job, so batching saves time).
# The sweep jobs already compute the observables; this recomputes them (another N_EDGE, N_SAMPLES or OBS selection, or
# a changed observables script) without rerunning the ITE. Run on the access point from the repo root, normally
# through the dashboard (which first uploads the needed states):
#   TAG=... XAXIS=t YAXIS=Delta bash hpc/condor/obs_BHU1.sh 8            every stored state
#   POINTS="0.0500 0.2000,0.1000 0.2000" ...                            only these points
#   OBS=entropy N_SAMPLES=10000 ...                                     only the entropy, more sampled words
#   FORCE=1 ... / DRYRUN=1 ... / CHUNK=5 ...
#
# Which points are submitted is decided by the script's --status (nostate, uptodate or compute; OBS_VERSION = git
# hash-object of the script, so a changed script recomputes). Every job gets Project.toml, Manifest.toml, src/, the
# script and the states of its points, runs the script once for all of them, deletes its copies of the states on exit
# (REMOVE_STATES=1) and transfers back the L folder, i.e. the _obs.csv files. Job attributes: BHC_model = "BHU1",
# BHC_stage = "obs", BHC_tag, BHC_L, BHC_points = "x1_y1,x2_y2,...".
# Logs: logs/BoseHubbardU1_NormalizedITE/<TAG>/L<L>/condor/obs/

set -euo pipefail

L="${1:?usage: bash hpc/condor/obs_BHU1.sh <L>}"
[ -f Project.toml ] && [ -d hpc/condor ] || { echo "run from the repo root"; exit 1; }

export TAG="${TAG:?TAG not set (the dashboard builds it)}"
export XAXIS="${XAXIS:?XAXIS not set}"
export YAXIS="${YAXIS:?YAXIS not set}"
export DATA_DIR=data
export N_SAMPLES="${N_SAMPLES:-1000}"
export N_EDGE="${N_EDGE:-1}"
export OBS="${OBS:-}"
[ "$L" -ge $((2 * N_EDGE + 3)) ] || { echo "N_EDGE = $N_EDGE leaves fewer than 2 bulk bonds for L = $L (need L ≥ 2 N_EDGE + 3)"; exit 1; }

PPN="${PPN:-4}"
CHUNK="${CHUNK:-20}"
[ "$CHUNK" -ge 1 ] || { echo "CHUNK = $CHUNK must be ≥ 1"; exit 1; }
JULIA_THREADS="${JULIA_THREADS:-$PPN}"
MEM="${MEM:-6 G}"
DISK="${DISK:-2 G}"
REQUIREMENTS="${REQUIREMENTS:-}"
NICE="${NICE:-TRUE}"
UG_NAME="${UG_NAME:-$USER}"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia +$(sed -n 's/^julia_version = "\(.*\)"/\1/p' Manifest.toml)}"

script=hpc/observables_BoseHubbardU1_NormalizedITE.jl
export OBS_VERSION=$(git hash-object "$script" 2> /dev/null || echo "")
folder="$DATA_DIR/BoseHubbardU1_NormalizedITE/$TAG/L$L"
logdir="logs/BoseHubbardU1_NormalizedITE/$TAG/L$L/condor/obs"
[ -d "$folder" ] || { echo "no results in $folder"; exit 1; }

# Candidate points (x, y): POINTS if given, else every stored state
candidates=()
if [ -n "${POINTS:-}" ]; then
    IFS=',' read -ra pairs <<< "$POINTS"
    for p in "${pairs[@]}"; do
        read -r x y <<< "$p"
        [ -n "${y:-}" ] && candidates+=("$(LC_ALL=C printf "%.4f %.4f" "$x" "$y")")
    done
else
    for f in "$folder"/"$XAXIS"*/"$YAXIS"*.jld2; do
        [ -e "$f" ] || continue
        x=$(basename "$(dirname "$f")"); x="${x#$XAXIS}"
        y=$(basename "$f" .jld2); y="${y#$YAXIS}"
        candidates+=("$x $y")
    done
fi

# Points that still have an observables job in the queue: skipped
active=""
if command -v condor_q > /dev/null 2>&1; then
    active=$(condor_q "$USER" -constraint "BHC_model =?= \"BHU1\" && BHC_stage =?= \"obs\" && BHC_tag =?= \"$TAG\" && BHC_L =?= $L" \
             -af BHC_points 2> /dev/null |
             awk '$1 != "undefined" { n = split($1, p, ","); for (i = 1; i <= n; i++) { sub("_", " ", p[i]); print p[i] } }' || true)
fi

points=()
uptodate=0
queued=0
nostate=0
for p in "${candidates[@]}"; do
    read -r x y <<< "$p"
    if [ -n "$active" ] && grep -qx "$x $y" <<< "$active"; then
        echo "queued L=$L $XAXIS=$x $YAXIS=$y (job in the queue)"
        queued=$((queued + 1))
        continue
    fi
    mapfile -t paths < <($JULIA --startup-file=no $script --status "$L" "$x" "$y")
    obs="${paths[2]}"
    st="${paths[3]}"
    if [ "$st" = "nostate" ]; then
        echo "skip   L=$L $XAXIS=$x $YAXIS=$y (no state)"
        nostate=$((nostate + 1))
        continue
    fi
    if [ "$st" = "uptodate" ] && [ "${FORCE:-0}" != "1" ]; then
        uptodate=$((uptodate + 1))
        continue
    fi
    if [ "${FORCE:-0}" = "1" ] && [ "${DRYRUN:-0}" != "1" ]; then
        rm -f "$obs"
    fi
    points+=("$x $y")
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
# U(1) correlated Bose-Hubbard observables, L = $L, tag $TAG, ${#points[@]} point(s) in $njobs job(s) of at most $CHUNK
# (generated by hpc/condor/obs_BHU1.sh)
JOBNAME = BHU1_obs_L${L}
UG_NAME = $UG_NAME

executable          = hpc/condor/run_julia.sh
transfer_executable = TRUE
environment         = "HOME=/tmp/\$(UG_NAME) JULIA_THREADS=$JULIA_THREADS TAG=$TAG XAXIS=$XAXIS YAXIS=$YAXIS DATA_DIR=$DATA_DIR N_SAMPLES=$N_SAMPLES N_EDGE=$N_EDGE OBS=$OBS OBS_VERSION=$OBS_VERSION REMOVE_STATES=1"

should_transfer_files   = TRUE
preserve_relative_paths = TRUE
when_to_transfer_output = ON_EXIT
transfer_output_files   = $folder

nice_user      = $NICE
request_cpus   = $PPN
request_memory = $MEM
request_disk   = $DISK
requirements   = $reqs
rank           = cpu_performance

+BHC_model = "BHU1"
+BHC_stage = "obs"
+BHC_tag   = "$TAG"
+BHC_L     = $L

output = $logdir/\$(JOBNAME)_\$(Cluster).\$(Process).out
error  = $logdir/\$(JOBNAME)_\$(Cluster).\$(Process).err
log    = $logdir/\$(Cluster).log
EOF
)
for ((i = 0; i < ${#points[@]}; i += CHUNK)); do
    args="" inputs="" labels=""
    for p in "${points[@]:i:CHUNK}"; do
        read -r x y <<< "$p"
        args+=" $x $y"
        inputs+=", $folder/$XAXIS$x/$YAXIS$y.jld2"
        labels+="${labels:+,}${x}_${y}"
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
