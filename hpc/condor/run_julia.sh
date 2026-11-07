#!/bin/bash
#
# Executable of the HTCondor jobs of hpc/condor/*.sh: runs Julia scripts of this repo in the job sandbox, which holds
# the transferred Project.toml, Manifest.toml, src/ and the scripts (at their relative paths, preserve_relative_paths).
#   run_julia.sh <script.jl> <args...> [:: <script2.jl> <args...> ...]
# Several scripts separated by `::` run one after the other in the same Julia setup; the job stops at the first failure.
# Environment (set by the submit file): HOME (/tmp/<login>), JULIA_THREADS (default 1), and the scripts' own variables.
#
# Julia: the version the Manifest was resolved with (julia_version in Manifest.toml; override with JULIA_VERSION), run
# through juliaup. The first job on a machine installs juliaup and that version in $HOME/.juliaup; later jobs reuse it
# (and the package depot $HOME/.julia). On the UGent pool /tmp is private to every job (MOUNT_UNDER_SCRATCH), so every
# job installs from scratch (about 8 minutes); on a pool with a shared /tmp, jobs on the same machine set up one at a time
# (flock on $HOME/.juliaup.lock). Pkg.instantiate() sometimes hangs forever without output (a stalled download; Pkg has
# no timeout), so every attempt is limited to SETUP_TIMEOUT seconds (default 1800) and retried up to SETUP_ATTEMPTS
# times (default 3); then the job fails with a clear message. Only Pkg.instantiate() is run, no Pkg.precompile(): the
# script's `using` precompiles just the packages it needs.
# Progress markers on stdout (read by the dashboard through condor_tail): "== HH:MM:SS setup: <phase>" and
# "== HH:MM:SS julia <script> <args>" before every script.

set -euo pipefail

echo "Host: $(hostname), started at $(date)"
echo "Sandbox: $(pwd)"
echo "Arguments: $*"

manifest=$(sed -n 's/^julia_version = "\(.*\)"/\1/p' Manifest.toml)
version="${JULIA_VERSION:-$manifest}"
juliaup="$HOME/.juliaup/bin/juliaup"
JULIA=("$HOME/.juliaup/bin/julia" "+$version")
mkdir -p "$HOME"
echo "== $(date +%T) setup: waiting for the setup lock"
(
    flock 9
    echo "== $(date +%T) setup: julia install"
    if [ ! -x "$juliaup" ]; then
        echo "Installing juliaup in $HOME/.juliaup"
        curl -fsSL https://install.julialang.org | sh -s -- --yes --add-to-path=no --default-channel "$version"
    fi
    if ! "$juliaup" status | grep -q " $version "; then
        echo "Installing Julia $version"
        "$juliaup" add "$version"
    fi
    echo "Julia $("${JULIA[@]}" --startup-file=no -e 'print(VERSION)') (juliaup), HOME = $HOME"
    attempts="${SETUP_ATTEMPTS:-3}"
    for ((i = 1; i <= attempts; i++)); do
        echo "== $(date +%T) setup: Pkg.instantiate (attempt $i of $attempts)"
        rc=0
        timeout "${SETUP_TIMEOUT:-1800}" "${JULIA[@]}" --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()' || rc=$?
        [ "$rc" = "124" ] || exit "$rc"
        echo "Pkg.instantiate() did not finish within ${SETUP_TIMEOUT:-1800} s on $(hostname) (attempt $i of $attempts)" | tee /dev/stderr
    done
    echo "Pkg.instantiate() hung $attempts times: giving up" | tee /dev/stderr
    exit 1
) 9> "$HOME/.juliaup.lock"
cmd=()
for a in "$@" ::; do
    if [ "$a" = "::" ]; then
        if [ "${#cmd[@]}" -gt 0 ]; then
            echo "== $(date +%T) julia ${cmd[*]}"
            "${JULIA[@]}" --project=. --startup-file=no --threads="${JULIA_THREADS:-1}" "${cmd[@]}"
        fi
        cmd=()
    else
        cmd+=("$a")
    fi
done
echo "Ended at $(date)"
