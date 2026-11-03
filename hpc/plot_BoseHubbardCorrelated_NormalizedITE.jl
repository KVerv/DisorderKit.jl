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
#   plots/BoseHubbardCorrelated_NormalizedITE_<run tag>_L<L>.png. Grid points without a result are left blank.
#
# H conserves N, so an exact ground state with a fixed particle number has ⟨aᵢ⟩ = 0. ⟨aᵢ⟩ is nonzero only when the state
# mixes neighbouring particle numbers, which the evolution from the uniform state keeps where they are nearly degenerate
# (superfluid) and the bond cap maxD favours. |⟨a⟩|² is thus a superfluid indicator (≈ 0 on a Mott plateau), not a
# converged ground-state value. Every conditional state has its own global phase, so the squared modulus is averaged:
#   |⟨a⟩|² = Σ_w P(w) (1/L) Σᵢ |⟨aᵢ⟩_w|².

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

# Load every (t, μ) of the run
isdir(folder) || error("no results in $folder")
files = [joinpath(folder, td, f) for td in filter(s -> occursin(r"^t-?[\d.]+$", s), readdir(folder))
         for f in filter(f -> occursin(r"^mu-?[\d.]+\.jld2$", f), readdir(joinpath(folder, td)))]
isempty(files) && error("no results in $folder")
results = map(files) do f
    data = load(f)
    ψ = data["ψ"]
    ρ, φ = disorder_average(ψ)
    # Disorder variance of the entanglement entropy on every bond, averaged over the chain
    _, _, S_var = DisorderKit.average_entanglement_entropy(ψ; n_samples, rng = Xoshiro(1))
    p = data["params"]
    (; t = p.t, μ = p.μ, U = p.U, r = p.r, ρ, φ, varS = sum(S_var) / length(S_var), nsteps = data["nsteps"],
     converged = data["converged"], ϵconv = data["info"].ϵsconv[end])
end
U, r = results[1].U, results[1].r

# (t/U, μ/U) grid, NaN where there is no result
ts = sort(unique(x.t / U for x in results))
μs = sort(unique(x.μ / U for x in results))
grid() = fill(NaN, length(ts), length(μs))
ρs, φs, varSs, κs = grid(), grid(), grid(), grid()
converged = fill(true, length(ts), length(μs))
index(x) = (searchsortedfirst(ts, x.t / U), searchsortedfirst(μs, x.μ / U))
for x in results
    i, j = index(x)
    ρs[i, j], φs[i, j], varSs[i, j], converged[i, j] = x.ρ, x.φ, x.varS, x.converged
end

# Compressibility κ = dρ/dμ along μ at fixed t (central differences, one-sided at the ends)
for i in eachindex(ts), j in eachindex(μs)
    lo, hi = max(j - 1, 1), min(j + 1, length(μs))
    κs[i, j] = hi == lo ? NaN : (ρs[i, hi] - ρs[i, lo]) / (U * (μs[hi] - μs[lo]))
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
nmissing = length(ts) * length(μs) - length(results)
nmissing > 0 && println("$nmissing grid points without a result (blank in the plot)")

csv = joinpath(folder, "observables.csv")
open(csv, "w") do io
    println(io, "t_over_U,mu_over_U,rho,kappa,a2,varS_avg,nsteps,eps_conv,converged")
    for x in results
        i, j = index(x)
        println(io, "$(ts[i]),$(μs[j]),$(ρs[i, j]),$(κs[i, j]),$(φs[i, j]),$(varSs[i, j]),$(x.nsteps),$(x.ϵconv),$(x.converged)")
    end
end
println("Saved $csv")

# Figure: four heatmaps over (t/U, μ/U); unconverged points are marked with open circles
set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor = :white, fontsize = 30, size = (2000, 1600))
Label(fig[0, 1:4], L"$L=%$L$, $r=%$r$, $t' = t/3$, $μ' = μ/3$, $D=%$maxD$"; fontsize = 36)
unconv = [(ts[i], μs[j]) for i in eachindex(ts), j in eachindex(μs) if !converged[i, j]]
panels = [(κs, L"κ = dρ/dμ", :viridis), (varSs, L"\overline{\mathrm{Var}[S]}", :magma),
          (ρs, L"ρ", :cividis), (φs, L"|\langle a \rangle|^2", :plasma)]
for (k, (M, label, cmap)) in enumerate(panels)
    row, col = fldmod1(k, 2)
    ax = Axis(fig[row, 2col-1]; xlabel = L"t/U", ylabel = L"μ/U", title = label)
    finite = filter(isfinite, M)
    colorrange = isempty(finite) || minimum(finite) == maximum(finite) ? (0, 1) : extrema(finite)
    hm = heatmap!(ax, ts, μs, M; colormap = cmap, colorrange, nan_color = :white)
    isempty(unconv) || scatter!(ax, unconv; color = :transparent, strokecolor = :white, strokewidth = 2, markersize = 12)
    Colorbar(fig[row, 2col], hm)
end

mkpath("plots")
file = joinpath("plots", "BoseHubbardCorrelated_NormalizedITE_$(run_tag)_L$(L).png")
save(file, fig)
println("Saved $file")
