#!/bin/bash
#
# HTCondor version of hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh for the correlated Bose-Hubbard phase diagram:
# one job per point (t, μ), all in one condor cluster. Every job runs the ITE (hpc/BoseHubbardCorrelated_NormalizedITE.jl)
# and then the observables of the new state (hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl) and returns both
# files. Run on the access point from the repo root (normally through the dashboard, hpc/condor/dashboard/server.py):
#   bash hpc/condor/sweep_BHCorr.sh 6                         # the default (t, μ) grid for L = 6
#   DRYRUN=1 bash hpc/condor/sweep_BHCorr.sh 6                # only print the points and the submit file
#   TS="0.05 0.1" MUS="0.2 0.4 0.6" bash hpc/condor/sweep_BHCorr.sh 6
#   R=0.5 DELTA=0.3 OMEGA=0.5 MAXD=32 MAXITER=1000 bash hpc/condor/sweep_BHCorr.sh 6
#   MAXITER=4000 bash hpc/condor/sweep_BHCorr.sh 6            # rerun the unconverged points with more steps
#   FORCE=1 bash hpc/condor/sweep_BHCorr.sh 6                 # resubmit every point, also those that are done
#   POINTS="0.1000 0.3000,0.1200 0.3000" bash hpc/condor/sweep_BHCorr.sh 6   # exactly these points (the dashboard)
#   PPN=4 REQUIREMENTS="cpu_performance > 7" bash hpc/condor/sweep_BHCorr.sh 6
#
# Which points are submitted: with POINTS ("t mu" pairs separated by commas) exactly those, without any status check
# (the dashboard decides, since gathered states only exist on the laptop). Otherwise every point of TS × MUS whose state
# here is missing or can be improved, as decided by hpc/BoseHubbardCorrelated_NormalizedITE.jl --status (run: no
# state, rerun: unconverged and MAXITER/CONVTOL improved, uptodate: skipped). In both cases points that still have a
# job in the queue (idle, running or held) are skipped, so a point is never submitted twice. The jobs carry the
# attributes BHC_stage = "ite", BHC_tag, BHC_L, BHC_t and BHC_mu (used for this and by the dashboard).
#
# The pool has no shared filesystem: every job gets Project.toml, Manifest.toml, src/ and the two scripts, runs them
# with hpc/condor/run_julia.sh, and its state and observables files are transferred back to the same relative paths
# data/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/t<t>/mu<μ>.jld2 and mu<μ>_obs.csv under this repo
# (DATA_DIR is always "data"). OBS_VERSION (git hash-object of the observables script) is stored with the observables.
# Logs, the points file and the submit file: logs/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/condor/

set -euo pipefail

L="${1:?usage: bash hpc/condor/sweep_BHCorr.sh <L>}"
[ -f Project.toml ] && [ -d hpc/condor ] || { echo "run from the repo root"; exit 1; }

# (t/U, μ/U) grid of the mean values (disorder values t(1±OMEGA) and μ±DELTA), formatted as %.4f in the file names
TS="${TS:-$(LC_ALL=C seq 0.02 0.02 0.30)}"
MUS="${MUS:-$(LC_ALL=C seq 0 0.1 1.6)}"

# Run parameters (U, R, DELTA, OMEGA, NMAX, DTAU, MAXD set the run tag), as in the PBS sweep
export U="${U:-1.0}"
export R="${R:-0.0}"           # correlation of hopping and potential: P = (1±R)/4
export DELTA="${DELTA:-0.0}"   # potential disorder: μ ± DELTA
export OMEGA="${OMEGA:-0.0}"   # hopping disorder: t(1 ± OMEGA), 0 ≤ OMEGA ≤ 1
export NMAX="${NMAX:-4}"
export DTAU="${DTAU:-0.05}"
export MAXD="${MAXD:-16}"
export MAXITER="${MAXITER:-2000}"
export CONVTOL="${CONVTOL:-1e-6}"
export CROSS_WIDTH="${CROSS_WIDTH:-32}"
export DATA_DIR=data           # relative, so the transferred files land at the same path here and in the sandbox
# Observables computed in the same job (see hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl)
N_SAMPLES="${N_SAMPLES:-1000}"
N_EDGE="${N_EDGE:-2}"
[ "$L" -ge $((2 * N_EDGE + 3)) ] || { echo "N_EDGE = $N_EDGE leaves fewer than 2 bulk bonds for L = $L (need L ≥ 2 N_EDGE + 3)"; exit 1; }

# Resources per job
PPN="${PPN:-8}"                          # request_cpus
JULIA_THREADS="${JULIA_THREADS:-1}"      # ITE: Julia runs with --threads=$JULIA_THREADS
BLAS_THREADS="${BLAS_THREADS:-$PPN}"     # and BLAS.set_num_threads($BLAS_THREADS)
OBS_THREADS="${OBS_THREADS:-$PPN}"       # observables: Julia threads (loops over disorder words)
MEM="${MEM:-6 G}"
DISK="${DISK:-2 G}"
REQUIREMENTS="${REQUIREMENTS:-}"         # extra requirements, e.g. "cpu_performance > 7"
NICE="${NICE:-TRUE}"                     # nice_user: only use idle slots
UG_NAME="${UG_NAME:-$USER}"              # login on the access point: HOME=/tmp/$UG_NAME on the execute node
JULIA="${JULIA:-$HOME/.juliaup/bin/julia +$(sed -n 's/^julia_version = "\(.*\)"/\1/p' Manifest.toml)}"   # Julia on the access point (juliaup, Manifest version)

script=hpc/BoseHubbardCorrelated_NormalizedITE.jl
obsscript=hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl
OBS_VERSION=$(git hash-object "$obsscript" 2> /dev/null || echo "")
mapfile -t paths < <($JULIA --startup-file=no $script --path "$L" 0 0)
tag="${paths[0]}"

# Points of this sweep that still have a job in the queue ("t mu" per line)
active=""
if command -v condor_q > /dev/null 2>&1; then
    active=$(condor_q "$USER" -constraint "BHC_stage =?= \"ite\" && BHC_tag =?= \"$tag\" && BHC_L =?= $L" \
             -af BHC_t BHC_mu 2> /dev/null || true)
fi

candidates=()
if [ -n "${POINTS:-}" ]; then
    IFS=',' read -ra pairs <<< "$POINTS"
    for p in "${pairs[@]}"; do
        read -r t mu <<< "$p"
        [ -n "${mu:-}" ] || continue
        candidates+=("$(LC_ALL=C printf "%.4f %.4f" "$t" "$mu")")
    done
else
    for t in $TS; do
        for mu in $MUS; do
            candidates+=("$(LC_ALL=C printf "%.4f %.4f" "$t" "$mu")")
        done
    done
fi

points=()
rerun=0
skipped=0
queued=0
for p in "${candidates[@]}"; do
    read -r t mu <<< "$p"
    if [ -n "$active" ] && grep -qx "$t $mu" <<< "$active"; then
        echo "queued L=$L t=$t μ=$mu (job in the queue)"
        queued=$((queued + 1))
        continue
    fi
    if [ -z "${POINTS:-}" ] && [ "${FORCE:-0}" != "1" ]; then
        mapfile -t st < <($JULIA --project=. --startup-file=no $script --status "$L" "$t" "$mu")
        if [ "${st[2]}" = "uptodate" ]; then
            echo "skip   L=$L t=$t μ=$mu (${st[3]})"
            skipped=$((skipped + 1))
            continue
        fi
        if [ "${st[2]}" = "rerun" ]; then
            echo "rerun  L=$L t=$t μ=$mu (${st[3]})"
            rerun=$((rerun + 1))
        fi
    fi
    points+=("$t $mu")
done
if [ "${#points[@]}" = "0" ]; then
    echo "L = $L: nothing to submit, $skipped skipped (done), $queued already queued"
    exit 0
fi

logdir="logs/BoseHubbardCorrelated_NormalizedITE/$tag/L$L/condor"
stamp=$(date +%Y%m%d_%H%M%S)
pointsfile="$logdir/points_$stamp.txt"
submitfile="$logdir/sweep_$stamp.submit"
reqs="has_julia${REQUIREMENTS:+ && ($REQUIREMENTS)}"
pointdir="$DATA_DIR/BoseHubbardCorrelated_NormalizedITE/$tag/L$L/t\$(t)"

submit=$(cat <<EOF
# Correlated Bose-Hubbard normalized ITE + observables, L = $L, run tag $tag (generated by hpc/condor/sweep_BHCorr.sh)
JOBNAME = BHC_L${L}
UG_NAME = $UG_NAME

executable          = hpc/condor/run_julia.sh
transfer_executable = TRUE
arguments           = "$script $L \$(t) \$(mu) :: --threads=$OBS_THREADS $obsscript $L \$(t) \$(mu)"
environment         = "HOME=/tmp/\$(UG_NAME) JULIA_THREADS=$JULIA_THREADS BLAS_THREADS=$BLAS_THREADS U=$U R=$R DELTA=$DELTA OMEGA=$OMEGA NMAX=$NMAX DTAU=$DTAU MAXD=$MAXD MAXITER=$MAXITER CONVTOL=$CONVTOL CROSS_WIDTH=$CROSS_WIDTH DATA_DIR=$DATA_DIR N_SAMPLES=$N_SAMPLES N_EDGE=$N_EDGE OBS_VERSION=$OBS_VERSION"

should_transfer_files   = TRUE
preserve_relative_paths = TRUE
when_to_transfer_output = ON_EXIT
transfer_input_files    = Project.toml, Manifest.toml, src, $script, $obsscript
transfer_output_files   = $pointdir/mu\$(mu).jld2, $pointdir/mu\$(mu)_obs.csv

nice_user      = $NICE
request_cpus   = $PPN
request_memory = $MEM
request_disk   = $DISK
requirements   = $reqs
rank           = cpu_performance

# Labels for matching jobs to sweep points (hpc/condor/*.sh, dashboard)
+BHC_stage = "ite"
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
    echo "L = $L: ${#points[@]} job(s) to submit ($rerun rerun), $skipped skipped (done), $queued already queued"
else
    mkdir -p "$logdir"
    printf '%s\n' "${points[@]}" > "$pointsfile"
    echo "$submit" > "$submitfile"
    condor_submit "$submitfile"
    echo "L = $L: ${#points[@]} job(s) submitted ($rerun rerun), $skipped skipped (done), $queued already queued; submit file $submitfile"
fi
