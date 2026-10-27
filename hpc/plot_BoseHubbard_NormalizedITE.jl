# Observables and plot of a hpc/sweep_BoseHubbard_NormalizedITE.sh run: from the stored states it computes the
# disorder-averaged density ρ, compressibility κ = dρ/dμ and condensate weight |⟨a⟩|², and plots them against μ/U.
#
# Usage (from the repo root, with the same run parameters as the sweep, so the run tag matches):
#   julia --project=. hpc/plot_BoseHubbard_NormalizedITE.jl <L>
#   T=0.1 MAXD=32 julia --project=. hpc/plot_BoseHubbard_NormalizedITE.jl 6
#
# Environment: T, U, NMAX, DTAU, MAXD, DATA_DIR (as in hpc/BoseHubbard_NormalizedITE.jl), N_SAMPLES=1000 (disorder words
#   are enumerated exactly if q^L ≤ N_SAMPLES, else sampled)
# Output: the table on stdout, <DATA_DIR>/BoseHubbard_NormalizedITE/<run tag>/L<L>/observables.csv and
#   plots/BoseHubbard_NormalizedITE_<run tag>_L<L>.png
#
# H conserves N, so an exact ground state with a fixed particle number has ⟨aᵢ⟩ = 0. ⟨aᵢ⟩ is nonzero only when the state
# mixes neighbouring particle numbers, which the evolution from the uniform state keeps where they are nearly degenerate
# (superfluid) and the bond cap maxD favours. |⟨a⟩|² is thus a superfluid indicator (≈ 0 on a Mott plateau), not a
# converged ground-state value. Every conditional state has its own global phase, so the squared modulus is averaged:
#   |⟨a⟩|² = Σ_w P(w) (1/L) Σᵢ |⟨aᵢ⟩_w|².

length(ARGS) == 1 || error("usage: julia --project=. hpc/plot_BoseHubbard_NormalizedITE.jl <L>")
L = parse(Int, ARGS[1])

# Same run tag as BoseHubbard_NormalizedITE.jl
t_str = get(ENV, "T", "0.16")
U_str = get(ENV, "U", "1.0")
nmax = parse(Int, get(ENV, "NMAX", "4"))
dtau_str = get(ENV, "DTAU", "0.05")
maxD = parse(Int, get(ENV, "MAXD", "16"))
run_tag = "t$(t_str)_U$(U_str)_nmax$(nmax)_dtau$(dtau_str)_D$(maxD)"
data_root = get(ENV, "DATA_DIR", joinpath(get(ENV, "VSC_DATA", "."), "data"))
folder = joinpath(data_root, "BoseHubbard_NormalizedITE", run_tag, "L$(L)")
n_samples = parse(Int, get(ENV, "N_SAMPLES", "10000"))

using TensorKit, DisorderKit, LinearAlgebra, Random, StatsBase, CairoMakie, Printf, JLD2

d = nmax + 1
n_op = TensorMap(ComplexF64.(diagm(0:nmax)), ℂ^d ← ℂ^d)
a_op = TensorMap(ComplexF64.(diagm(1 => sqrt.(1:nmax))), ℂ^d ← ℂ^d)   # a|n⟩ = √n |n-1⟩

# Left transfer of an environment E[bra; ket] by the site tensor A[a s; b] with the operator X applied to the ket
transfer(E, A, X) = @tensor E′[-2; -1] := A[1 3; -1] * X[4; 3] * conj(A[2 4; -2]) * E[2; 1]

# ⟨N⟩ and ⟨aᵢ⟩ of the conditional state of word w. One left sweep carries the norm (E0) and one inserted n (E1), with a
# common scale, and stores the norm environment before every site; ⟨aᵢ⟩ then follows from a right sweep
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
    q = length(ψ.ps)
    words = float(q)^L <= n_samples ? [collect(w) for w in vec(collect(Iterators.product(fill(1:q, L)...)))] :
                                       [sample(rng, 1:q, Weights(ψ.ps), L) for _ in 1:n_samples]
    weights = float(q)^L <= n_samples ? [prod(ψ.ps[w]) for w in words] : fill(1/length(words), length(words))
    ρ, φ = 0.0, 0.0
    for (w, P) in zip(words, weights)
        N, as = word_observables(ψ, w)
        ρ += P * N / L
        φ += P * sum(abs2, as) / L
    end
    return ρ, φ
end

# Load every μ of the run, sorted by μ
isdir(folder) || error("no results in $folder")
files = filter(f -> occursin(r"^mu-?[\d.]+\.jld2$", f), readdir(folder))
isempty(files) && error("no results in $folder")
results = map(files) do f
    data = load(joinpath(folder, f))
    ρ, φ = disorder_average(data["ψ"])
    p = data["params"]
    (; μ = p.μ, U = p.U, t = p.t, ρ, φ, nsteps = data["nsteps"], converged = data["converged"],
     ϵconv = data["info"].ϵsconv[end])
end
sort!(results; by = r -> r.μ)
μs = [r.μ / r.U for r in results]
ρs = [r.ρ for r in results]
φs = [r.φ for r in results]
converged = [r.converged for r in results]
t, U = results[1].t, results[1].U

# Compressibility κ = dρ/dμ (central differences, one-sided at the ends)
κs = similar(ρs)
for k in eachindex(μs)
    lo, hi = max(k - 1, 1), min(k + 1, length(μs))
    κs[k] = hi == lo ? NaN : (ρs[hi] - ρs[lo]) / (U * (μs[hi] - μs[lo]))
end

println("Run tag $run_tag, L = $L ($folder)")
println("    μ/U        ρ        κ   |⟨a⟩|²   steps   ϵ_conv   converged")
for (k, r) in enumerate(results)
    @printf("%7.3f  %7.4f  %7.3f  %7.4f  %6d  %7.1e   %s\n", μs[k], ρs[k], κs[k], φs[k], r.nsteps, r.ϵconv, r.converged)
end
all(converged) || println("$(count(!, converged)) of $(length(converged)) points not converged (open markers in the plot)")

csv = joinpath(folder, "observables.csv")
open(csv, "w") do io
    println(io, "mu_over_U,rho,kappa,a2,nsteps,eps_conv,converged")
    for (k, r) in enumerate(results)
        println(io, "$(μs[k]),$(ρs[k]),$(κs[k]),$(φs[k]),$(r.nsteps),$(r.ϵconv),$(r.converged)")
    end
end
println("Saved $csv")

# Figure: ρ and |⟨a⟩|² on the left axis, κ on the right axis, sharing μ/U. Unconverged points have open markers
set_theme!(theme_latexfonts())
cρ, cκ, cφ = Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3]
fig = Figure(backgroundcolor = :white, fontsize = 30, size = (1200, 800))
title = L"$L=%$L$, $t/U=%$(t/U)$, $t' = t/3$, $D=%$maxD$"
axρ = Axis(fig[1, 1]; xlabel = L"μ/U", ylabel = L"ρ,\ |\langle a \rangle|^2", title)
axκ = Axis(fig[1, 1]; ylabel = L"κ = dρ/dμ", yaxisposition = :right, ylabelcolor = cκ, yticklabelcolor = cκ,
           ygridvisible = false)
hidespines!(axκ)
hidexdecorations!(axκ)
linkxaxes!(axρ, axκ)
hlines!(axρ, [1.0]; color = :gray, linestyle = :dash)

markerfill(c) = [conv ? c : :white for conv in converged]
lines!(axρ, μs, ρs; color = cρ, linewidth = 3)
lines!(axρ, μs, φs; color = cφ, linewidth = 3)
lines!(axκ, μs, κs; color = cκ, linewidth = 3)
scatter!(axρ, μs, ρs; color = markerfill(cρ), strokecolor = cρ, strokewidth = 2, markersize = 14)
scatter!(axρ, μs, φs; color = markerfill(cφ), strokecolor = cφ, strokewidth = 2, marker = :diamond, markersize = 16)
scatter!(axκ, μs, κs; color = markerfill(cκ), strokecolor = cκ, strokewidth = 2, marker = :rect, markersize = 14)
ylims!(axρ, 0, nothing)
ylims!(axκ, 0, nothing)
entry(c, m, ms) = [LineElement(color = c, linewidth = 3), MarkerElement(color = c, marker = m, markersize = ms)]
axislegend(axρ, [entry(cρ, :circle, 14), entry(cφ, :diamond, 16), entry(cκ, :rect, 14)],
           [L"ρ", L"|\langle a \rangle|^2", L"κ"]; position = :lt)

mkpath("plots")
file = joinpath("plots", "BoseHubbard_NormalizedITE_$(run_tag)_L$(L).png")
save(file, fig)
println("Saved $file")
