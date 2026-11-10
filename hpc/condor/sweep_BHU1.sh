#!/bin/bash
#
# HTCondor sweep of the U(1)-symmetric correlated Bose-Hubbard chain at fixed filling over two freely chosen parameters
# (hpc/BoseHubbardU1_NormalizedITE.jl): one job per point (x, y), all in one condor cluster. Every job runs the ITE and
# then the observables of the new state (hpc/observables_BoseHubbardU1_NormalizedITE.jl) and returns both files. The
# generic counterpart of hpc/condor/sweep_BHCorr.sh (same resume and status logic). Run on the access point from the
# repo root, normally through the dashboard (hpc/condor/dashboard/server.py), which builds TAG:
#   TAG=t-Delta_U1.0_Omega0.0_r0.0_f1_nmax4_dtau0.05_D32 XAXIS=t YAXIS=Delta XS="0.05 0.1" YS="0 0.2 0.4" \
#       bash hpc/condor/sweep_BHU1.sh 8
#   DRYRUN=1 ...                         only print the points and the submit file
#   POINTS="0.0500 0.2000,0.1000 0.2000" exactly these points ("x y" pairs separated by commas; no status check)
#   MAXITER=4000 ...                     continue the unconverged points up to 4000 steps in total
#   FORCE=1 ...                          resubmit every point from scratch
#
# Axes: XAXIS and YAXIS are two different ones of t, U, Delta, Omega, r; XS and YS their values (formatted %.4f in the
# file names). The other parameters are fixed: T, U, DELTA, OMEGA, R (the swept ones are ignored), FILLING, NMAX, DTAU,
# MAXD. TAG names the sweep (axes and fixed parameters); the dashboard builds it, so that one place defines it.
#
# Files (no shared filesystem: the job gets Project.toml, Manifest.toml, src/ and the scripts, and its files are
# transferred back to the same relative paths under this repo, DATA_DIR = data):
#   data/BoseHubbardU1_NormalizedITE/<TAG>/L<L>/<XAXIS><x>/<YAXIS><y>.jld2 and <YAXIS><y>_obs.csv
#   logs/BoseHubbardU1_NormalizedITE/<TAG>/L<L>/condor/   logs, points and submit files
# Job attributes: BHC_model = "BHU1", BHC_stage = "ite", BHC_tag, BHC_L, BHC_x, BHC_y (the dashboard matches jobs with them).

set -euo pipefail

L="${1:?usage: bash hpc/condor/sweep_BHU1.sh <L>}"
[ -f Project.toml ] && [ -d hpc/condor ] || { echo "run from the repo root"; exit 1; }

TAG="${TAG:?TAG not set (the dashboard builds it)}"
XAXIS="${XAXIS:?XAXIS not set}"
YAXIS="${YAXIS:?YAXIS not set}"
case " t U Delta Omega r " in *" $XAXIS "*) ;; *) echo "XAXIS = $XAXIS must be one of t, U, Delta, Omega, r"; exit 1;; esac
case " t U Delta Omega r " in *" $YAXIS "*) ;; *) echo "YAXIS = $YAXIS must be one of t, U, Delta, Omega, r"; exit 1;; esac
[ "$XAXIS" != "$YAXIS" ] || { echo "XAXIS and YAXIS must differ"; exit 1; }
export TAG XAXIS YAXIS

# Fixed parameters (the swept ones are ignored by the job)
export T="${T:-0.1}"
export U="${U:-1.0}"
export DELTA="${DELTA:-0.0}"   # potential disorder: ±DELTA
export OMEGA="${OMEGA:-0.0}"   # hopping disorder: t(1 ± OMEGA), 0 ≤ OMEGA ≤ 1
export R="${R:-0.0}"           # correlation of hopping and potential: P = (1±R)/4
export FILLING="${FILLING:-1}" # bosons per site, N = FILLING·L
export NMAX="${NMAX:-4}"
export DTAU="${DTAU:-0.05}"
export MAXD="${MAXD:-32}"
export MAXITER="${MAXITER:-2000}"
export CONVTOL="${CONVTOL:-1e-6}"
export CROSS_WIDTH="${CROSS_WIDTH:-32}"
export DATA_DIR=data
[ "${FORCE:-0}" = "1" ] && RESUME=0
export RESUME="${RESUME:-1}"
# Observables computed in the same job
N_SAMPLES="${N_SAMPLES:-1000}"
N_EDGE="${N_EDGE:-1}"
OBS="${OBS:-}"
[ "$L" -ge $((2 * N_EDGE + 3)) ] || { echo "N_EDGE = $N_EDGE leaves fewer than 2 bulk bonds for L = $L (need L ≥ 2 N_EDGE + 3)"; exit 1; }

# Resources per job
PPN="${PPN:-8}"
JULIA_THREADS="${JULIA_THREADS:-1}"
BLAS_THREADS="${BLAS_THREADS:-$PPN}"
OBS_THREADS="${OBS_THREADS:-$PPN}"
MEM="${MEM:-6 G}"
DISK="${DISK:-2 G}"
REQUIREMENTS="${REQUIREMENTS:-}"
NICE="${NICE:-TRUE}"
UG_NAME="${UG_NAME:-$USER}"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia +$(sed -n 's/^julia_version = "\(.*\)"/\1/p' Manifest.toml)}"

script=hpc/BoseHubbardU1_NormalizedITE.jl
obsscript=hpc/observables_BoseHubbardU1_NormalizedITE.jl
OBS_VERSION=$(git hash-object "$obsscript" 2> /dev/null || echo "")
folder="$DATA_DIR/BoseHubbardU1_NormalizedITE/$TAG/L$L"

# Points of this sweep that still have a job in the queue ("x y" per line)
active=""
if command -v condor_q > /dev/null 2>&1; then
    active=$(condor_q "$USER" -constraint "BHC_model =?= \"BHU1\" && BHC_stage =?= \"ite\" && BHC_tag =?= \"$TAG\" && BHC_L =?= $L" \
             -af BHC_x BHC_y 2> /dev/null || true)
fi

candidates=()
if [ -n "${POINTS:-}" ]; then
    IFS=',' read -ra pairs <<< "$POINTS"
    for p in "${pairs[@]}"; do
        read -r x y <<< "$p"
        [ -n "${y:-}" ] || continue
        candidates+=("$(LC_ALL=C printf "%.4f %.4f" "$x" "$y")")
    done
else
    for x in ${XS:?XS not set}; do
        for y in ${YS:?YS not set}; do
            candidates+=("$(LC_ALL=C printf "%.4f %.4f" "$x" "$y")")
        done
    done
fi

points=()
resume=()
rerun=0
skipped=0
queued=0
for p in "${candidates[@]}"; do
    read -r x y <<< "$p"
    if [ -n "$active" ] && grep -qx "$x $y" <<< "$active"; then
        echo "queued L=$L $XAXIS=$x $YAXIS=$y (job in the queue)"
        queued=$((queued + 1))
        continue
    fi
    if [ -z "${POINTS:-}" ] && [ "${FORCE:-0}" != "1" ]; then
        mapfile -t st < <($JULIA --project=. --startup-file=no $script --status "$L" "$x" "$y")
        case "${st[2]}" in
            uptodate) echo "skip   L=$L $XAXIS=$x $YAXIS=$y (${st[3]})"; skipped=$((skipped + 1)); continue;;
            rerun)    echo "rerun  L=$L $XAXIS=$x $YAXIS=$y (${st[3]})"; rerun=$((rerun + 1));;
            resume)   echo "resume L=$L $XAXIS=$x $YAXIS=$y (${st[3]})"; resume+=("$x $y"); continue;;
        esac
    elif [ "$RESUME" = "1" ] && [ -f "$folder/$XAXIS$x/$YAXIS$y.jld2" ]; then
        resume+=("$x $y")   # POINTS: the job decides whether the state can be continued
        continue
    fi
    points+=("$x $y")
done
total=$((${#points[@]} + ${#resume[@]}))
counts="${#points[@]} from scratch ($rerun rerun), ${#resume[@]} resumed"
if [ "$total" = "0" ]; then
    echo "L = $L: nothing to submit, $skipped skipped (done), $queued already queued"
    exit 0
fi

logdir="logs/BoseHubbardU1_NormalizedITE/$TAG/L$L/condor"
stamp=$(date +%Y%m%d_%H%M%S)
pointsfile="$logdir/points_$stamp.txt"
resumefile="$logdir/resume_$stamp.txt"
submitfile="$logdir/sweep_$stamp.submit"
reqs="has_julia${REQUIREMENTS:+ && ($REQUIREMENTS)}"
pointdir="$folder/$XAXIS\$(x)"
pointfile="$pointdir/$YAXIS\$(y)"

submit=$(cat <<EOF
# U(1) correlated Bose-Hubbard normalized ITE + observables, L = $L, tag $TAG (generated by hpc/condor/sweep_BHU1.sh)
JOBNAME = BHU1_L${L}
UG_NAME = $UG_NAME

executable          = hpc/condor/run_julia.sh
transfer_executable = TRUE
arguments           = "$script $L \$(x) \$(y) :: --threads=$OBS_THREADS $obsscript $L \$(x) \$(y)"
environment         = "HOME=/tmp/\$(UG_NAME) JULIA_THREADS=$JULIA_THREADS BLAS_THREADS=$BLAS_THREADS TAG=$TAG XAXIS=$XAXIS YAXIS=$YAXIS T=$T U=$U DELTA=$DELTA OMEGA=$OMEGA R=$R FILLING=$FILLING NMAX=$NMAX DTAU=$DTAU MAXD=$MAXD MAXITER=$MAXITER CONVTOL=$CONVTOL CROSS_WIDTH=$CROSS_WIDTH RESUME=$RESUME DATA_DIR=$DATA_DIR N_SAMPLES=$N_SAMPLES N_EDGE=$N_EDGE OBS=$OBS OBS_VERSION=$OBS_VERSION"

should_transfer_files   = TRUE
preserve_relative_paths = TRUE
when_to_transfer_output = ON_EXIT
transfer_input_files    = Project.toml, Manifest.toml, src, $script, $obsscript
transfer_output_files   = $pointfile.jld2, ${pointfile}_obs.csv

nice_user      = $NICE
request_cpus   = $PPN
request_memory = $MEM
request_disk   = $DISK
requirements   = $reqs
rank           = cpu_performance

# Labels for matching jobs to sweep points (hpc/condor/*.sh, dashboard)
+BHC_model = "BHU1"
+BHC_stage = "ite"
+BHC_tag   = "$TAG"
+BHC_L     = $L
+BHC_x     = "\$(x)"
+BHC_y     = "\$(y)"

output = $logdir/\$(JOBNAME)_$XAXIS\$(x)_$YAXIS\$(y)_\$(Cluster).out
error  = $logdir/\$(JOBNAME)_$XAXIS\$(x)_$YAXIS\$(y)_\$(Cluster).err
log    = $logdir/\$(Cluster).log

$([ "${#points[@]}" = "0" ] || echo "queue x, y from $pointsfile")
$([ "${#resume[@]}" = "0" ] || printf '%s\n' "" "# Resumed points: their stored state is transferred to the job, which continues the ITE from it" \
    "transfer_input_files    = Project.toml, Manifest.toml, src, $script, $obsscript, $pointfile.jld2" \
    "queue x, y from $resumefile")
EOF
)

if [ "${DRYRUN:-0}" = "1" ]; then
    [ "${#points[@]}" = "0" ] || printf '%s\n' "${points[@]}" | sed 's/^/point  /'
    [ "${#resume[@]}" = "0" ] || printf '%s\n' "${resume[@]}" | sed 's/^/resume /'
    echo "$submit" | sed 's/^/       | /'
    echo "L = $L: $total job(s) to submit ($counts), $skipped skipped (done), $queued already queued"
else
    mkdir -p "$logdir"
    [ "${#points[@]}" = "0" ] || printf '%s\n' "${points[@]}" > "$pointsfile"
    [ "${#resume[@]}" = "0" ] || printf '%s\n' "${resume[@]}" > "$resumefile"
    echo "$submit" > "$submitfile"
    condor_submit "$submitfile"
    echo "L = $L: $total job(s) submitted ($counts), $skipped skipped (done), $queued already queued; submit file $submitfile"
fi
