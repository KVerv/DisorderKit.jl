# DisorderKit.jl
Julia code for studying random quantum systems with tensor networks

## Environments

- `Project.toml` (repo root): the package itself, only what `src/` and the cluster job scripts in `hpc/` load. The
  cluster jobs install and precompile this one, so keep it slim (no plotting or development packages).
- `examples/Project.toml`: DisorderKit (from `src/`) plus CairoMakie, LsqFit, Revise, Distributions, Symbolics, …
  Use it for `examples/`, `plot_make/`, `test/`, the plot scripts in `hpc/` and the REPL / VS Code
  (Julia: Activate This Environment on `examples/`):

      julia --project=examples -e 'using Pkg; Pkg.instantiate()'   # once, and after a pull that changes it
      julia --project=examples examples/FiniteBoseHubbard.jl

  To add a package for scripts: `julia --project=examples -e 'using Pkg; Pkg.add("X")'`; only add it to the root
  `Project.toml` if `src/` needs it.
