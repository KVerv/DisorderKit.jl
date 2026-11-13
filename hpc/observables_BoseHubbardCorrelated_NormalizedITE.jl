# Observables of one point (t, μ) of a hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh run: from the stored state it
# computes the disorder-averaged density ρ and quasi-condensate density |⟨a⟩|² (exactly, with DisorderKit.expectation_value),
# and the disorder-averaged entanglement entropy S̄(k) and its disorder variance Var[S](k) on every bond k (from
# enumerated or sampled disorder words), and writes them to a one-row CSV next to the state. Derived from the entropy:
#   varS_avg = (1/(L-1)) Σₖ Var[S](k)                     chain average of the disorder variance
#   S_avg    = (1/N_b) Σ_{k ∈ bulk} S̄(k)                  bulk average of the entropy (N_b bulk bonds, see N_EDGE below)
#   S_inhom  = (1/N_b) Σ_{k ∈ bulk} (S̄(k) - S_avg)²       homogeneity of the average entropy along the bulk
# and, per disorder word w, the standard deviation of S_w(k) across the bulk bonds, averaged over the words
# (same words as S̄, all from one DisorderKit.disorder_average pass):
#   S_bstd   = 𝔼_w[std_{k ∈ bulk} S_w(k)]                  bond-to-bond fluctuation of the entropy in one realization
#   S_bstd_err                                             its standard error (0 if the words are enumerated)
# and the effective Luttinger parameter from the block number fluctuations of the same words: for the blocks
# A_l = N_EDGE+1 .. N_EDGE+l (l = 1..N_s, N_s bulk sites, a common left edge) of every conditional state projected onto its
# most likely particle number (as for the entropy, see below),
#   F_l      = 𝔼_w[⟨N_A²⟩_w - ⟨N_A⟩_w²]                     (DisorderKit.conditional_block_fluctuations with n, projected)
# and K from the open-chain bosonization fit F = K/(2π²) X + c, X = ln[d(2x₁) d(2x₂) d(x₂-x₁)² / d(x₁+x₂)²] with
# d(x) = (2(L+1)/π) sin(πx/(2(L+1))) and the cuts x₁, x₂ of the block (luttinger_parameter below, over the blocks
# with l ≥ 2 whose right edge is at least N_EDGE sites from the wall, i.e. all l ≥ 2). In a gapped (Mott) phase F
# saturates and K → 0; on short chains K is an effective, finite-size value.
#   K        effective Luttinger parameter (NaN if fewer than 2 blocks qualify)
#   F_1..F_{N_s}                                           the block fluctuations
# The bulk (for S_avg, S_inhom, S_bstd and the blocks of K) excludes N_EDGE sites at each end: sites N_EDGE+1..L-N_EDGE, bonds
# k = N_EDGE+1..L-N_EDGE-1 (at least 2, so L ≥ 2 N_EDGE + 3). N_EDGE = 0 uses all bonds. varS_avg and the per-bond
# columns S_k, varS_k cover the whole chain.
# The compressibility κ = dρ/dμ needs the neighbouring points and is computed by hpc/plot_BoseHubbardCorrelated_NormalizedITE.jl.
#
# Usage (from the repo root, normally through hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh):
#   julia --project=. hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl <L> <t> <μ>      compute and store
#   julia --project=. hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl <L> <t1> <μ1> <t2> <μ2> ...
#       several points in one process (one Julia start and compilation; hpc/condor/obs_BHCorr.sh runs CHUNK points per
#       job). Every point is tried, a failing one does not stop the others; the exit code is 1 if any point failed.
#       "== HH:MM:SS point k of n: t = …, μ = …" on stdout before every point.
#   julia hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl --status <L> <t> <μ>         only print the run tag, the
#       state file, the observables file and the status: nostate, uptodate or compute
# t and μ are used verbatim in the file names, as in hpc/BoseHubbardCorrelated_NormalizedITE.jl (formatted as %.4f).
#
# Environment (all required; defaults in hpc/submit_obs_BHCorr.sh): U, R, DELTA, OMEGA, NMAX,
#   DTAU, MAXD, DATA_DIR (the same as for the sweep, so the run tag matches), N_SAMPLES (for the entropy only: disorder
#   words are enumerated exactly if ∏ᵢ qᵢ ≤ N_SAMPLES, qᵢ the number of sectors of site i, else N_SAMPLES words are sampled),
#   N_EDGE (sites excluded at each end for S_bstd and the blocks of K)
# Optional: OBS_VERSION, a version stamp of this script (the condor scripts pass `git hash-object` of it), stored in
#   the obs_version column, so observables computed with an older version of the script are recomputed
#
# Input:  <DATA_DIR>/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/t<t>/mu<μ>.jld2
# Output: <DATA_DIR>/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/t<t>/mu<μ>_obs.csv, columns
#   t_over_U, mu_over_U, rho, a2, varS_avg, S_avg, S_inhom, S_bstd, S_bstd_err, K, n_edge, obs_version, n_samples, exact, nsteps, eps_conv, converged, S_1..S_{L-1},
#   varS_1..varS_{L-1}, F_1..F_{N_s} (n_samples and exact refer to the entropy and F)
#
# The observables are (re)computed only if (status compute)
#   - there is no readable observables file, or it is of an older format (no S_bstd, K or n_edge column), or
#   - its n_edge differs from N_EDGE, or OBS_VERSION is set and its obs_version differs, or
#   - the state is newer than the observables file (the point was rerun), or
#   - the stored words were sampled (exact = false) and N_SAMPLES is larger than the stored n_samples.
# Exactly enumerated points are never recomputed for a larger N_SAMPLES.
#
# H conserves N, so an exact ground state with a fixed particle number has ⟨aᵢ⟩ = 0. ⟨aᵢ⟩ is nonzero only when the state
# mixes neighbouring particle numbers, which the evolution from the uniform state keeps where they are nearly degenerate
# (superfluid) and the bond cap maxD favours. |⟨a⟩|² is thus a superfluid indicator (≈ 0 on a Mott plateau), not a
# converged ground-state value. It is the chain average of the square of the disorder-averaged ⟨aᵢ⟩:
#   |⟨a⟩|² = (1/L) Σᵢ ⟨aᵢ⟩²,   ⟨aᵢ⟩ = Σ_w P(w) ⟨aᵢ⟩_w (real part, DisorderKit.expectation_value).
# The same mixing of particle numbers lowers the entanglement entropy (most in the bulk), so S and Var[S] are computed
# for every conditional state projected onto its most likely particle number, the number of its exact ground state.

# ---- Parameters, paths and status (before loading any package, so --status is fast)
status_only = !isempty(ARGS) && ARGS[1] == "--status"
args = status_only ? ARGS[2:end] : ARGS
usage = "usage: julia --project=. hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl [--status] <L> <t> <μ> [<t2> <μ2> ...] (--status: one point)"
(length(args) >= 3 && isodd(length(args)) && !(status_only && length(args) > 3)) || error(usage)
L_str = args[1]
points = [(args[k], args[k+1]) for k in 2:2:length(args)]   # (t, μ) as strings, used verbatim in the file names
L = parse(Int, L_str)

# Same run tag and folder as hpc/BoseHubbardCorrelated_NormalizedITE.jl
param(k) = haskey(ENV, k) ? ENV[k] : error("environment variable $k not set; run parameters are set in hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh")
U_str = param("U")
r_str = param("R")
Δ_str = param("DELTA")
Ω_str = param("OMEGA")
nmax = parse(Int, param("NMAX"))
dtau_str = param("DTAU")
maxD = parse(Int, param("MAXD"))
run_tag = "U$(U_str)_r$(r_str)_Delta$(Δ_str)_Omega$(Ω_str)_nmax$(nmax)_dtau$(dtau_str)_D$(maxD)"
data_root = param("DATA_DIR")
n_samples = parse(Int, param("N_SAMPLES"))
n_edge = parse(Int, param("N_EDGE"))
obs_version = get(ENV, "OBS_VERSION", "")
n_edge >= 0 || error("N_EDGE = $n_edge must be ≥ 0")
L >= 2n_edge + 3 || error("N_EDGE = $n_edge leaves fewer than 2 bulk bonds for L = $L (need L ≥ 2 N_EDGE + 3)")
bulk_bonds = n_edge+1:L-n_edge-1
bulk_sites = n_edge+1:L-n_edge
blocks = [n_edge+1:n_edge+l for l in 1:length(bulk_sites)]   # blocks of K: common left edge, growing into the bulk

# State and observables file of a point
function point_files(t_str, μ_str)
    folder = joinpath(data_root, "BoseHubbardCorrelated_NormalizedITE", run_tag, "L$(L)", "t$(t_str)")
    return joinpath(folder, "mu$(μ_str).jld2"), joinpath(folder, "mu$(μ_str)_obs.csv")
end

# Header and first row of a CSV as column => value (strings), nothing if the file is missing or malformed
function read_row(file)
    isfile(file) || return nothing
    lines = filter(!isempty, readlines(file))
    length(lines) >= 2 || return nothing
    header, row = split(lines[1], ','), split(lines[2], ',')
    length(header) == length(row) || return nothing
    return Dict(String(h) => String(v) for (h, v) in zip(header, row))
end

function status(statefile, obsfile, n_samples, n_edge, obs_version)
    isfile(statefile) || return "nostate"
    row = read_row(obsfile)
    (isnothing(row) || !all(k -> haskey(row, k), ("n_samples", "exact", "S_inhom", "S_bstd", "K", "n_edge"))) && return "compute"
    tryparse(Int, row["n_edge"]) == n_edge || return "compute"
    isempty(obs_version) || get(row, "obs_version", "") == obs_version || return "compute"
    mtime(statefile) > mtime(obsfile) && return "compute"
    stored = tryparse(Int, row["n_samples"])
    exact = tryparse(Bool, row["exact"])
    (isnothing(stored) || isnothing(exact)) && return "compute"
    return !exact && n_samples > stored ? "compute" : "uptodate"
end

if status_only
    statefile, obsfile = point_files(points[1]...)
    println(run_tag)
    println(statefile)
    println(obsfile)
    println(status(statefile, obsfile, n_samples, n_edge, obs_version))
    exit()
end

# ---- Observables
using TensorKit, DisorderKit, LinearAlgebra, Random, JLD2, Printf, StatsBase, Dates

# Effective Luttinger parameter from the block fluctuations F[k] of the particle number in the blocks[k] (sites a:b) of
# an open chain of L sites. Bosonization with hard walls at 0 and L+1 (L′ = L+1) gives, for the block between the cuts
# x₁ = a-1 and x₂ = b,
#   F = K/(2π²) X + c,   X = ln[d(2x₁) d(2x₂) d(x₂-x₁)² / d(x₁+x₂)²],   d(x) = (2L′/π) sin(πx/(2L′)),
# which reduces to F = (K/π²) ln l + c far from the walls. K and c from a linear least-squares fit of F against X over the
# blocks with at least lmin sites whose edges are at least edge_min sites from the walls (single sites and cuts next to a
# wall carry large lattice corrections). Returns (K, c, X, fit) with fit the indices of the fitted blocks; K = c = NaN if
# fewer than 2 blocks qualify. In a gapped phase F saturates and K → 0
function luttinger_parameter(F::AbstractVector{<:Real}, blocks::AbstractVector{<:UnitRange{<:Integer}}, L::Int;
                             lmin::Int = 2, edge_min::Int = 2)
    length(F) == length(blocks) || throw(ArgumentError("F and blocks should have the same length"))
    Lp = L + 1
    d(x) = (2Lp / π) * sin(π * x / (2Lp))
    X = [log(d(2(first(b) - 1)) * d(2last(b)) * d(length(b))^2 / d(first(b) - 1 + last(b))^2) for b in blocks]
    fit = findall(k -> length(blocks[k]) >= lmin && first(blocks[k]) - 1 >= edge_min && L - last(blocks[k]) >= edge_min &&
                       isfinite(X[k]) && isfinite(F[k]), eachindex(blocks))
    length(unique(X[fit])) >= 2 || return NaN, NaN, X, fit
    slope, c = hcat(X[fit], ones(length(fit))) \ F[fit]
    return 2π^2 * slope, c, X, fit
end

d = nmax + 1
n_op = TensorMap(ComplexF64.(diagm(0:nmax)), ℂ^d ← ℂ^d)
a_op = TensorMap(ComplexF64.(diagm(1 => sqrt.(1:nmax))), ℂ^d ← ℂ^d)   # a|n⟩ = √n |n-1⟩
U = parse(Float64, U_str)

function compute_point(t_str, μ_str)
    statefile, obsfile = point_files(t_str, μ_str)
    st = status(statefile, obsfile, n_samples, n_edge, obs_version)
    st == "nostate" && error("no state $statefile")
    if st == "uptodate"
        println("Up to date (N_SAMPLES = $n_samples, N_EDGE = $n_edge): $obsfile")
        return
    end
    println("Correlated Bose-Hubbard observables: L = $L, t = $t_str, μ = $μ_str, run tag $run_tag, N_SAMPLES = $n_samples, N_EDGE = $n_edge (bulk bonds $bulk_bonds)")
    println("State: $statefile")

    data = load(statefile)
    ψ = data["ψ"]
    exact = prod(float.(length.(ψ.ps))) <= n_samples
    local ρ, φ, S, S_var, S_bstd, S_bstd_err, F, K
    walltime = @elapsed begin
        # Exact disorder averages of the local operators on every site
        ρ = mean(DisorderKit.expectation_value(ψ, n_op))
        φ = mean(abs2, DisorderKit.expectation_value(ψ, a_op))
        # One pass over the words (enumerated or sampled) of the conditional states projected onto their most likely N:
        # the entropy S_w(k) on every bond, its std over the bulk bonds and the block number fluctuations
        nb = L - 1
        function per_word(w)
            S_w = DisorderKit.conditional_entanglement_entropy(ψ, w; charges = 0:nmax)
            F_w = DisorderKit.conditional_block_fluctuations(ψ, w, n_op, blocks; charges = 0:nmax)
            return vcat(S_w, std(S_w[bulk_bonds]), F_w)
        end
        avg, err, var_w = DisorderKit.disorder_average(per_word, ψ; n_samples, rng = Xoshiro(1))
        S, S_var = avg[1:nb], var_w[1:nb]
        S_bstd, S_bstd_err = avg[nb+1], err[nb+1]
        F = avg[nb+2:end]
        K, _, _, _ = luttinger_parameter(F, blocks, L; edge_min = n_edge)
    end
    ϵs = data["info"].ϵsconv
    nsteps = get(data, "nsteps", -1)
    converged = get(data, "converged", false)
    ϵconv = isempty(ϵs) ? NaN : last(ϵs)
    varS = mean(S_var)
    S_avg = mean(S[bulk_bonds])
    S_inhom = mean(abs2, S[bulk_bonds] .- S_avg)

    # Write to a temporary file and move it in place, so an interrupted job never leaves a partial result
    header = ["t_over_U", "mu_over_U", "rho", "a2", "varS_avg", "S_avg", "S_inhom", "S_bstd", "S_bstd_err", "K", "n_edge", "obs_version", "n_samples", "exact", "nsteps", "eps_conv",
              "converged",
              ("S_$k" for k in eachindex(S))..., ("varS_$k" for k in eachindex(S_var))..., ("F_$l" for l in eachindex(F))...]
    row = Any[parse(Float64, t_str) / U, parse(Float64, μ_str) / U, ρ, φ, varS, S_avg, S_inhom, S_bstd, S_bstd_err, K, n_edge, obs_version, n_samples, exact, nsteps, ϵconv, converged,
           S..., S_var..., F...]
    tmp = obsfile * ".tmp"
    open(tmp, "w") do io
        println(io, join(header, ','))
        println(io, join(string.(row), ','))
    end
    mv(tmp, obsfile; force = true)

    @printf("ρ = %.6f, |⟨a⟩|² = %.6f, ⟨Var S⟩ = %.6f, ⟨S⟩_bulk = %.6f, Var_{k∈bulk}[S] = %.6f, 𝔼_w[std_{k∈bulk} S] = %.6f ± %.6f, K = %.4f (entropy and K from %s, %.1f s)\n", ρ, φ,
            varS, S_avg, S_inhom, S_bstd, S_bstd_err, K, exact ? "exact enumeration" : "$n_samples sampled words", walltime)
    println("Saved $obsfile")
end

# Every point is tried; a failing point does not stop the others, but makes the exit code 1
failed = Tuple{String, String}[]
for (k, (t_str, μ_str)) in enumerate(points)
    println("== $(Dates.format(now(), "HH:MM:SS")) point $k of $(length(points)): t = $t_str, μ = $μ_str")
    flush(stdout)
    try
        compute_point(t_str, μ_str)
    catch e
        push!(failed, (t_str, μ_str))
        println(stderr, "ERROR for point t = $t_str, μ = $μ_str:")
        showerror(stderr, e, catch_backtrace())
        println(stderr)
    end
    flush(stdout); flush(stderr)
end
if !isempty(failed)
    println(stderr, "$(length(failed)) of $(length(points)) point(s) failed: ", join(("t = $t, μ = $μ" for (t, μ) in failed), "; "))
    exit(1)
end
