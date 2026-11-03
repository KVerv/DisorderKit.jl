# Observables and phase diagram of a hpc/sweep_BoseHubbardCorrelated_NormalizedITE.sh run: from the stored states on the
# (t, μ) grid it computes the disorder-averaged density ρ, the compressibility κ = dρ/dμ (along μ at fixed t), the
# quasi-condensate density |⟨a⟩|² and the chain average of the disorder variance of the entanglement entropy, and plots
# the four as heatmaps over (t/U, μ/U). t and μ are the strong values, the weak ones are t/3 and μ/3.
#
# Usage (from the repo root, normally through hpc/submit_plot_BoseHubbardCorrelated_NormalizedITE.sh, which sets the
# environment):
#   U=1.0 R=0.0 NMAX=4 DTAU=0.05 MAXD=16 DATA_DIR=$VSC_DATA/data N_SAMPLES=1000 \
#       julia --project=. hpc/plot_BoseHubbardCorrelated_NormalizedITE.jl <L>
#
# Environment (all required; defaults in hpc/submit_plot_BoseHubbardCorrelated_NormalizedITE.sh): U, R, NMAX, DTAU, MAXD,
#   DATA_DIR (the same as for the sweep, so the run tag matches), N_SAMPLES (disorder words are enumerated exactly if
#   ∏ᵢ qᵢ ≤ N_SAMPLES, qᵢ the number of sectors of site i, else sampled)
# Output: the table on stdout, <DATA_DIR>/BoseHubbardCorrelated_NormalizedITE/<run tag>/L<L>/observables.csv and
#   plots/BoseHubbardCorrelated_NormalizedITE_<run tag>_L<L>.png. Grid points without a usable result (no file,
#   or one that cannot be read) are set to 0 and marked with a cross; the grid is set by the t<t>/mu<μ>.jld2 names.
#
# H conserves N, so an exact ground state with a fixed particle number has ⟨aᵢ⟩ = 0. ⟨aᵢ⟩ is nonzero only when the state
# mixes neighbouring particle numbers, which the evolution from the uniform state keeps where they are nearly degenerate
# (superfluid) and the bond cap maxD favours. |⟨a⟩|² is thus a superfluid indicator (≈ 0 on a Mott plateau), not a
# converged ground-state value. Every conditional state has its own global phase, so the squared modulus is averaged:
#   |⟨a⟩|² = Σ_w P(w) (1/L) Σᵢ |⟨aᵢ⟩_w|².
# The same mixing of particle numbers lowers the entanglement entropy (most in the bulk), so Var[S] is computed for
# every conditional state projected onto its most likely particle number, the number of its exact ground state.

length(ARGS) == 1 || error("usage: julia --project=. hpc/plot_BoseHubbardCorrelated_NormalizedITE.jl <L>")
L = parse(Int, ARGS[1])

# Same run tag as BoseHubbardCorrelated_NormalizedITE.jl
param(k) = haskey(ENV, k) ? ENV[k] : error("environment variable $k not set; run parameters are set in hpc/submit_plot_BoseHubbardCorrelated_NormalizedITE.sh")
U_str = param("U")
r_str = param("R")
nmax = parse(Int, param("NMAX"))
dtau_str = param("DTAU")
maxD = parse(Int, param("MAXD"))
run_tag = "U$(U_str)_r$(r_str)_nmax$(nmax)_dtau$(dtau_str)_D$(maxD)"
data_root = param("DATA_DIR")
folder = joinpath(data_root, "BoseHubbardCorrelated_NormalizedITE", run_tag, "L$(L)")
n_samples = parse(Int, param("N_SAMPLES"))

using TensorKit, DisorderKit, LinearAlgebra, Random, StatsBase, CairoMakie, Printf, JLD2

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
function disorder_average(ψ; rng = Xoshiro(1))
    L = length(ψ)
    exact = prod(float.(length.(ψ.ps))) <= n_samples
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

# (t/U, μ/U) grid from the folder and file names, so that points whose job failed, is still running or never ran are
# part of it too
U, r = parse(Float64, U_str), parse(Float64, r_str)
isdir(folder) || error("no results in $folder")
tdirs = filter(s -> occursin(r"^t-?[\d.]+$", s) && isdir(joinpath(folder, s)), readdir(folder))
tvalue(td) = parse(Float64, td[2:end]) / U
μvalue(f) = parse(Float64, f[3:end-5]) / U
files = [(td, f) for td in tdirs for f in filter(f -> occursin(r"^mu-?[\d.]+\.jld2$", f), readdir(joinpath(folder, td)))]
isempty(files) && error("no results in $folder")
ts = sort(unique(tvalue.(tdirs)))
μs = sort(unique(μvalue(f) for (_, f) in files))

# Load every (t, μ) of the run; a file that cannot be read or evaluated is skipped with a warning
results = map(files) do (td, f)
    file = joinpath(folder, td, f)
    try
        data = load(file)
        ψ = data["ψ"]
        ρ, φ = disorder_average(ψ)
        # Disorder variance of the entanglement entropy on every bond, averaged over the chain, of the conditional
        # states projected onto their most likely particle number
        _, _, S_var = DisorderKit.average_entanglement_entropy(ψ; n_samples, rng = Xoshiro(1), charges = 0:nmax)
        ϵs = data["info"].ϵsconv
        (; t = tvalue(td), μ = μvalue(f), ρ, φ, varS = sum(S_var) / length(S_var), nsteps = get(data, "nsteps", -1),
         converged = get(data, "converged", false), ϵconv = isempty(ϵs) ? NaN : last(ϵs))
    catch e
        @warn "skipping $file: $(sprint(showerror, e))"
        nothing
    end
end
results = filter(!isnothing, results)

# Observables on the grid, 0 where there is no result (marked with a cross in the plot)
grid() = zeros(length(ts), length(μs))
ρs, φs, varSs, κs = grid(), grid(), grid(), grid()
converged = fill(true, length(ts), length(μs))
hasdata = fill(false, length(ts), length(μs))
index(x) = (searchsortedfirst(ts, x.t), searchsortedfirst(μs, x.μ))
for x in results
    i, j = index(x)
    ρs[i, j], φs[i, j], varSs[i, j], converged[i, j], hasdata[i, j] = x.ρ, x.φ, x.varS, x.converged, true
end

# Compressibility κ = dρ/dμ along μ at fixed t: central differences over the nearest neighbours with a result
# (one-sided at the ends and next to missing points), 0 where there is no result or no neighbour
for i in eachindex(ts), j in eachindex(μs)
    hasdata[i, j] || continue
    lo = j > 1 && hasdata[i, j-1] ? j - 1 : j
    hi = j < length(μs) && hasdata[i, j+1] ? j + 1 : j
    κs[i, j] = hi == lo ? 0.0 : (ρs[i, hi] - ρs[i, lo]) / (U * (μs[hi] - μs[lo]))
end

println("Run tag $run_tag, L = $L ($folder)")
println("    t/U      μ/U        ρ        κ   |⟨a⟩|²  ⟨Var S⟩   steps   ϵ_conv   converged")
sort!(results; by = x -> (x.t, x.μ))
for x in results
    i, j = index(x)
    @printf("%7.4f  %7.4f  %7.4f  %7.3f  %7.4f  %7.4f  %6d  %7.1e   %s\n", ts[i], μs[j], ρs[i, j], κs[i, j], φs[i, j],
            varSs[i, j], x.nsteps, x.ϵconv, x.converged)
end
nconv = count(!, (x.converged for x in results))
nconv > 0 && println("$nconv of $(length(results)) points not converged (open circles in the plot)")
missingpts = [(ts[i], μs[j]) for i in eachindex(ts), j in eachindex(μs) if !hasdata[i, j]]
if !isempty(missingpts)
    println("$(length(missingpts)) grid points without a result (0, marked × in the plot), (t/U, μ/U):")
    foreach(p -> @printf("  %7.4f  %7.4f\n", p...), missingpts)
end

csv = joinpath(folder, "observables.csv")
open(csv, "w") do io
    println(io, "t_over_U,mu_over_U,rho,kappa,a2,varS_avg,nsteps,eps_conv,converged")
    for x in results
        i, j = index(x)
        println(io, "$(ts[i]),$(μs[j]),$(ρs[i, j]),$(κs[i, j]),$(φs[i, j]),$(varSs[i, j]),$(x.nsteps),$(x.ϵconv),$(x.converged)")
    end
end
println("Saved $csv")

# Figure: four heatmaps over (t/U, μ/U); unconverged points are marked with open circles, points without a result
# with crosses
set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor = :white, fontsize = 30, size = (2000, 1600))
Label(fig[0, 1:4], L"$L=%$L$, $r=%$r$, $t' = t/3$, $μ' = μ/3$, $D=%$maxD$"; fontsize = 36)
unconv = [(ts[i], μs[j]) for i in eachindex(ts), j in eachindex(μs) if !converged[i, j]]
panels = [(κs, L"κ = dρ/dμ", :viridis), (varSs, L"\overline{\mathrm{Var}[S]}", :magma),
          (ρs, L"ρ", :cividis), (φs, L"|\langle a \rangle|^2", :plasma)]
for (k, (M, label, cmap)) in enumerate(panels)
    row, col = fldmod1(k, 2)
    ax = Axis(fig[row, 2col-1]; xlabel = L"t/U", ylabel = L"μ/U", title = label)
    finite = filter(isfinite, M[hasdata])
    colorrange = isempty(finite) || minimum(finite) == maximum(finite) ? (0, 1) : extrema(finite)
    hm = heatmap!(ax, ts, μs, M; colormap = cmap, colorrange, nan_color = :white)
    isempty(unconv) || scatter!(ax, unconv; color = :transparent, strokecolor = :white, strokewidth = 2, markersize = 12)
    isempty(missingpts) || scatter!(ax, missingpts; marker = :xcross, color = :white, strokecolor = :black,
                                    strokewidth = 1.5, markersize = 22)
    Colorbar(fig[row, 2col], hm)
end

mkpath("plots")
file = joinpath("plots", "BoseHubbardCorrelated_NormalizedITE_$(run_tag)_L$(L).png")
save(file, fig)
println("Saved $file")
