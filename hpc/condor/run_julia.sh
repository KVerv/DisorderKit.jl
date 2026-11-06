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
# (and the package depot $HOME/.julia). Jobs on the same machine install one at a time (flock on $HOME/.juliaup.lock).
# Only Pkg.instantiate() is run, no Pkg.precompile(): the script's `using` precompiles just the packages it needs.

set -euo pipefail

echo "Host: $(hostname), started at $(date)"
echo "Sandbox: $(pwd)"
echo "Arguments: $*"

manifest=$(sed -n 's/^julia_version = "\(.*\)"/\1/p' Manifest.toml)
version="${JULIA_VERSION:-$manifest}"
juliaup="$HOME/.juliaup/bin/juliaup"
mkdir -p "$HOME"
(
    flock 9
    if [ ! -x "$juliaup" ]; then
        echo "Installing juliaup in $HOME/.juliaup"
        curl -fsSL https://install.julialang.org | sh -s -- --yes --add-to-path=no --default-channel "$version"
    fi
    if ! "$juliaup" status | grep -q " $version "; then
        echo "Installing Julia $version"
        "$juliaup" add "$version"
    fi
) 9> "$HOME/.juliaup.lock"
JULIA=("$HOME/.juliaup/bin/julia" "+$version")
echo "Julia $("${JULIA[@]}" --startup-file=no -e 'print(VERSION)') (juliaup), HOME = $HOME"

"${JULIA[@]}" --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'
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
