#!/bin/bash
#
# Executable of the HTCondor jobs of hpc/condor/*.sh: runs one Julia script of this repo in the job sandbox, which holds
# the transferred Project.toml, Manifest.toml, src/ and the script (at their relative paths, preserve_relative_paths).
#   run_julia.sh <script.jl> <args...>
# Environment (set by the submit file): HOME (/tmp/<UGent login>: Julia keeps its depot in $HOME/.julia, so packages
# installed by an earlier job on the same machine are reused), JULIA_THREADS (default 1), and the script's own variables.
# Only Pkg.instantiate() is run, no Pkg.precompile(): the script's `using` precompiles just the packages it needs.

set -euo pipefail

JULIA="${JULIA:-/usr/local/bin/julia}"

echo "Host: $(hostname), started at $(date)"
echo "Sandbox: $(pwd)"
echo "Arguments: $*"
version=$($JULIA --startup-file=no -e 'print(VERSION)')
echo "Julia $version ($JULIA), HOME = $HOME"
manifest=$(sed -n 's/^julia_version = "\(.*\)"/\1/p' Manifest.toml)
if [ "${version%.*}" != "${manifest%.*}" ]; then
    echo "WARNING: node Julia $version, Manifest.toml was resolved with Julia $manifest" >&2
fi

mkdir -p "$HOME"
$JULIA --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'
$JULIA --project=. --startup-file=no --threads="${JULIA_THREADS:-1}" "$@"
echo "Ended at $(date)"
