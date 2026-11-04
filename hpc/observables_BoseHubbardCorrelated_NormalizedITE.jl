# Observables of one point (t, μ) of a hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh run: from the stored state it
# computes the disorder-averaged density ρ, the quasi-condensate density |⟨a⟩|² and the disorder-averaged entanglement
# entropy S and its disorder variance Var[S] on every bond, and writes them to a one-row CSV next to the state. The
# compressibility κ = dρ/dμ needs the neighbouring points and is computed by hpc/plot_BoseHubbardCorrelated_NormalizedITE.jl.
#
# Usage (from the repo root, normally through hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh):
#   julia --project=. hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl <L> <t> <μ>      compute and store
#   julia hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl --status <L> <t> <μ>         only print the run tag, the
#       state file, the observables file and the status: nostate, uptodate or compute
# t and μ are used verbatim in the file names, as in hpc/BoseHubbardCorrelated_NormalizedITE.jl (formatted as %.4f).
#
# Environment (all required; defaults in hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh): U, R, NMAX,
#   DTAU, MAXD, DATA_DIR (the same as for the sweep, so the run tag matches), N_SAMPLES (disorder words are enumerated
#   exactly if ∏ᵢ qᵢ ≤ N_SAMPLES, qᵢ the number of sectors of site i, else N_SAMPLES words are sampled)
#
# Input:  <DATA_DIR>/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/t<t>/mu<μ>.jld2
# Output: <DATA_DIR>/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/t<t>/mu<μ>_obs.csv, columns
#   t_over_U, mu_over_U, rho, a2, varS_avg, n_samples, exact, nsteps, eps_conv, converged, S_1..S_{L-1}, varS_1..varS_{L-1}
#
# The observables are (re)computed only if (status compute)
#   - there is no readable observables file, or
#   - the state is newer than the observables file (the point was rerun), or
#   - the stored words were sampled (exact = false) and N_SAMPLES is larger than the stored n_samples.
# Exactly enumerated points are never recomputed for a larger N_SAMPLES.
#
# H conserves N, so an exact ground state with a fixed particle number has ⟨aᵢ⟩ = 0. ⟨aᵢ⟩ is nonzero only when the state
# mixes neighbouring particle numbers, which the evolution from the uniform state keeps where they are nearly degenerate
# (superfluid) and the bond cap maxD favours. |⟨a⟩|² is thus a superfluid indicator (≈ 0 on a Mott plateau), not a
# converged ground-state value. Every conditional state has its own global phase, so the squared modulus is averaged:
#   |⟨a⟩|² = Σ_w P(w) (1/L) Σᵢ |⟨aᵢ⟩_w|².
# The same mixing of particle numbers lowers the entanglement entropy (most in the bulk), so S and Var[S] are computed
# for every conditional state projected onto its most likely particle number, the number of its exact ground state.

# ---- Parameters, paths and status (before loading any package, so --status is fast)
status_only = !isempty(ARGS) && ARGS[1] == "--status"
args = status_only ? ARGS[2:end] : ARGS
length(args) == 3 || error("usage: julia --project=. hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl [--status] <L> <t> <μ>")
L_str, t_str, μ_str = args
L = parse(Int, L_str)

# Same run tag and folder as hpc/BoseHubbardCorrelated_NormalizedITE.jl
param(k) = haskey(ENV, k) ? ENV[k] : error("environment variable $k not set; run parameters are set in hpc/submit_observables_BoseHubbardCorrelated_NormalizedITE.sh")
U_str = param("U")
r_str = param("R")
nmax = parse(Int, param("NMAX"))
dtau_str = param("DTAU")
maxD = parse(Int, param("MAXD"))
run_tag = "U$(U_str)_r$(r_str)_nmax$(nmax)_dtau$(dtau_str)_D$(maxD)"
data_root = param("DATA_DIR")
n_samples = parse(Int, param("N_SAMPLES"))
folder = joinpath(data_root, "BoseHubbardCorrelated_NormalizedITE", run_tag, "L$(L)", "t$(t_str)")
statefile = joinpath(folder, "mu$(μ_str).jld2")
obsfile = joinpath(folder, "mu$(μ_str)_obs.csv")

# Header and first row of a CSV as column => value (strings), nothing if the file is missing or malformed
function read_row(file)
    isfile(file) || return nothing
    lines = filter(!isempty, readlines(file))
    length(lines) >= 2 || return nothing
    header, row = split(lines[1], ','), split(lines[2], ',')
    length(header) == length(row) || return nothing
    return Dict(String(h) => String(v) for (h, v) in zip(header, row))
end

function status(statefile, obsfile, n_samples)
    isfile(statefile) || return "nostate"
    row = read_row(obsfile)
    (isnothing(row) || !haskey(row, "n_samples") || !haskey(row, "exact")) && return "compute"
    mtime(statefile) > mtime(obsfile) && return "compute"
    stored = tryparse(Int, row["n_samples"])
    exact = tryparse(Bool, row["exact"])
    (isnothing(stored) || isnothing(exact)) && return "compute"
    return !exact && n_samples > stored ? "compute" : "uptodate"
end

st = status(statefile, obsfile, n_samples)
if status_only
    println(run_tag)
    println(statefile)
    println(obsfile)
    println(st)
    exit()
end
st == "nostate" && error("no state $statefile")
if st == "uptodate"
    println("Up to date (N_SAMPLES = $n_samples): $obsfile")
    exit()
end

# ---- Observables
using TensorKit, DisorderKit, LinearAlgebra, Random, JLD2, Printf

d = nmax + 1
n_op = TensorMap(ComplexF64.(diagm(0:nmax)), ℂ^d ← ℂ^d)
a_op = TensorMap(ComplexF64.(diagm(1 => sqrt.(1:nmax))), ℂ^d ← ℂ^d)   # a|n⟩ = √n |n-1⟩

# Left transfer of an environment E[bra; ket] by the site tensor A[a s; b] with the operator X applied to the ket
transfer(E, A, X) = @tensor E′[-2; -1] := A[1 3; -1] * X[4; 3] * conj(A[2 4; -2]) * E[2; 1]

# ⟨N⟩ and ⟨aᵢ⟩ of the conditional state of word w, as in examples/FiniteDisorderBoseHubbard.jl. One left sweep carries
# the norm (E0) and one inserted n (E1), with a common scale, and stores the norm environment before every site; ⟨aᵢ⟩
# then follows from a right sweep
function word_observables(ψ, w)
    L = length(w)
    E0 = id(ComplexF64, space(ψ[1, 1], 1))
    E1 = zero(E0)
    lefts = Vector{Any}(undef, L)
    for (i, p) in enumerate(w)
        lefts[i] = E0
        A, I = ψ[i, p], id(ComplexF64, space(ψ[i, p], 2))
        E0, E1 = transfer(E0, A, I), transfer(E1, A, I) + transfer(E0, A, n_op)
        m = LinearAlgebra.norm(E0, Inf)
        E0, E1 = E0 / m, E1 / m
    end
    R = first(DisorderKit._boundary_environment(domain(ψ[L, 1])[1]))
    N = real(tr(R * E1) / tr(R * E0))
    as = Vector{ComplexF64}(undef, L)
    for i in L:-1:1
        A, I = ψ[i, w[i]], id(ComplexF64, space(ψ[i, w[i]], 2))
        as[i] = tr(R * transfer(lefts[i], A, a_op)) / tr(R * transfer(lefts[i], A, I))
        R = first(DisorderKit._advance((R, 0.0), A; right = true))
    end
    return N, as
end

# Disorder-averaged density and |⟨a⟩|²: all words if there are at most n_samples, else sampled words
function disorder_average(ψ, exact; rng = Xoshiro(1))
    L = length(ψ)
    words = DisorderKit.assessment_words(ψ.ps, n_samples; rng)
    weights = exact ? [prod(ψ.ps[i][u] for (i, u) in enumerate(w)) for w in words] : fill(1/length(words), length(words))
    ρ, φ = 0.0, 0.0
    for (w, P) in zip(words, weights)
        N, as = word_observables(ψ, w)
        ρ += P * N / L
        φ += P * sum(abs2, as) / L
    end
    return ρ, φ
end

U = parse(Float64, U_str)
println("Correlated Bose-Hubbard observables: L = $L, t = $t_str, μ = $μ_str, run tag $run_tag, N_SAMPLES = $n_samples")
println("State: $statefile")

data = load(statefile)
ψ = data["ψ"]
exact = prod(float.(length.(ψ.ps))) <= n_samples
walltime = @elapsed begin
    ρ, φ = disorder_average(ψ, exact)
    S, _, S_var = DisorderKit.average_entanglement_entropy(ψ; n_samples, rng = Xoshiro(1), charges = 0:nmax)
end
ϵs = data["info"].ϵsconv
nsteps = get(data, "nsteps", -1)
converged = get(data, "converged", false)
ϵconv = isempty(ϵs) ? NaN : last(ϵs)
varS = sum(S_var) / length(S_var)

# Write to a temporary file and move it in place, so an interrupted job never leaves a partial result
header = ["t_over_U", "mu_over_U", "rho", "a2", "varS_avg", "n_samples", "exact", "nsteps", "eps_conv", "converged",
          ("S_$k" for k in eachindex(S))..., ("varS_$k" for k in eachindex(S_var))...]
row = Any[parse(Float64, t_str) / U, parse(Float64, μ_str) / U, ρ, φ, varS, n_samples, exact, nsteps, ϵconv, converged,
       S..., S_var...]
tmp = obsfile * ".tmp"
open(tmp, "w") do io
    println(io, join(header, ','))
    println(io, join(string.(row), ','))
end
mv(tmp, obsfile; force = true)

@printf("ρ = %.6f, |⟨a⟩|² = %.6f, ⟨Var S⟩ = %.6f (%s, %.1f s)\n", ρ, φ, varS,
        exact ? "exact enumeration" : "$n_samples sampled words", walltime)
println("Saved $obsfile")
