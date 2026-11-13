# Golden chain: Fibonacci anyons τ on an open chain of L sites with
#   H = -J Σᵢ P_I(i, i+1),
# P_I the projector of τ ⊗ τ onto the trivial fusion channel. For J > 0 the chain is critical, described by the
# tricritical Ising CFT with central charge c = 7/10 (J < 0: 3-state Potts, c = 4/5). The tensors live on
# Vect[FibonacciAnyon] graded spaces (anyonic braiding, no dense representation), so every contraction in DisorderKit is
# planar. The chain is clean here (one coupling, q = 1); DisorderKit.golden_chain(Js) with several couplings and
# probabilities ps gives random couplings.
#
# 1. Ground states for several L with disorder-normalized imaginary time evolution (Groundstate_FiniteNormalizedITE),
#    checked against MPSKit DMRG on the same chain.
# 2. Central charge from the entanglement entropy of the open chain, S(l) = (c/6) ln d(l) + s₀ + oscillation, with the
#    chord length d(l) = (2L/π) sin(πl/L):
#      (a) for every L, a fit of S(l) over the bulk bonds, with an alternating term b(-1)^l for the parity oscillation
#          of the open boundaries,
#      (b) across L, a fit of S(L/2) = (c/6) ln(2L/π) + s₀ (L/2 even for every L, so the oscillation is the same),
#      (c) finite-entanglement check: at the largest L the final cap D_max of the bond (number of multiplets) is scanned,
#          showing how the fitted c converges with the bond dimension.
using TensorKit, MPSKit, DisorderKit, LinearAlgebra, Printf, CairoMakie

# Model
J = 1.0
Hs = DisorderKit.golden_chain([J, -J])
# ps = [1.0]
ps = [1/2, 1/2]  # probabilities of the two couplings (J, -J) for the random chain
P = space(Hs.Ds[1], 1)                                          # Vect[FibonacciAnyon](:τ => 1)
c_exact = 7 / 10

# Normalized ITE
# Bond caps count multiplets (sectors I and τ together). The time evolution MPO is SVD compressed (mpo_compress_tol,
# 78 → 11 multiplets at order 4), ~0.1 s per step at L = 12
Ls = [8]
Δτ = 0.01
order = 2
maxiter = 1000
convtol = 1e-9
maxD = [100 => 8, 1 => 16]                                       # stage schedule (steps => cap), the last cap is kept
D_scan = [16, 32, 64]                                          # smaller final caps of the finite-entanglement scan at the
                                                                # largest L (the largest cap comes from step 1)
V0 = Vect[FibonacciAnyon](:I => 4, :τ => 4)                     # bond space of the random start

# Reference Hamiltonian for MPSKit DMRG
P_I = zeros(ComplexF64, P ⊗ P ← P ⊗ P)
block(P_I, FibonacciAnyon(:I)) .= 1
mpskit_hamiltonian(L) = FiniteMPOHamiltonian(fill(P, L), ((i, i + 1) => -J * P_I for i in 1:L-1)...)

chord(l, L) = (2L / π) * sin(π * l / L)

# Least-squares fit of S(l) = (c/6) ln d(l) + s₀ + b(-1)^l over the bulk bonds l = lmin..L-lmin. Returns (c, s₀, b)
function fit_chord(S, L; lmin = 2)
    ls = lmin:L-lmin
    X = hcat(log.(chord.(ls, L)) ./ 6, ones(length(ls)), (-1.0) .^ ls)
    c, s0, b = X \ S[ls]
    return c, s0, b
end

# Ground state of the chain of length L with the final cap D_final (the stages of maxD with a larger cap are dropped)
function ite_groundstate(L, D_final)
    schedule = [s for s in maxD if s.second < D_final]
    push!(schedule, 1 => D_final)
    ψ0 = DisorderKit.FiniteDisorderMPS(FiniteMPS(randn, ComplexF64, L, P, V0), Hs, ps)
    alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD = schedule, order, maxiter, convtol)
    ψ, _, info = DisorderKit.groundstate(ψ0, Hs, Δτ, alg)
    info.ϵsconv[end] <= convtol || @warn "L = $L, D = $D_final: not converged after $maxiter steps (ϵ = $(info.ϵsconv[end]))"
    return ψ, info
end

entropies(ψ) = DisorderKit.conditional_entanglement_entropy(ψ, ones(Int, length(ψ)))   # the single word of the clean chain

# 1. Ground states for every L at the largest cap, and the DMRG reference
D_max = maxD[end].second
results = map(Ls) do L
    t = @elapsed ψ, info = ite_groundstate(L, D_max)
    S = entropies(ψ)
    H = mpskit_hamiltonian(L)
    ψd, = find_groundstate(FiniteMPS(randn, ComplexF64, L, P, Vect[FibonacciAnyon](:I => D_max, :τ => D_max)), H,
                           DMRG(; verbosity = 0, tol = 1e-10))
    E_dmrg = real(expectation_value(ψd, H)) / L
    S_dmrg = [real(entropy(ψd, l)) for l in 1:L-1]
    @printf("L = %2d: E/L = %.10f (ITE, %4d steps, %5.1f s)  %.10f (DMRG)  ΔE = %.1e  max |ΔS| = %.1e  bonds ≤ %d\n",
            L, info.Es[end], length(info.Es), t, E_dmrg, info.Es[end] - E_dmrg, maximum(abs, S .- S_dmrg),
            maximum(info.bond_dimensions[end]))
    (; L, S, S_dmrg, E = info.Es[end], E_dmrg, fit = fit_chord(S, L), fit_dmrg = fit_chord(S_dmrg, L),
       bond = maximum(info.bond_dimensions[end]))
end

# 2a. Central charge per L from the chord fit
println("\nCentral charge from S(l) = (c/6) ln[(2L/π) sin(πl/L)] + s₀ + b(-1)^l (bulk bonds 2..L-2), c = 7/10 = 0.7:")
for r in results
    @printf("  L = %2d: c = %.4f (ITE)  %.4f (DMRG),  s₀ = %.4f, b = %+.4f\n", r.L, r.fit[1], r.fit_dmrg[1], r.fit[2], r.fit[3])
end

# 2b. Central charge across L from the middle bond
S_mid = [r.S[r.L ÷ 2] for r in results]
X_mid = log.(2 .* Ls ./ π) ./ 6
c_mid, s0_mid = hcat(X_mid, ones(length(Ls))) \ S_mid
@printf("\nS(L/2) = (c/6) ln(2L/π) + s₀ across L = %s: c = %.4f, s₀ = %.4f\n", Ls, c_mid, s0_mid)

# 2c. Finite-entanglement scan at the largest L: smaller final caps, plus the run of step 1 at D_max
L_scan = Ls[end]
println("\nFinite-entanglement scan at L = $L_scan (DMRG: E/L = $(round(results[end].E_dmrg; digits = 10)), c = $(round(results[end].fit_dmrg[1]; digits = 4))):")
scan = map(D_scan) do D
    ψ, info = ite_groundstate(L_scan, D)
    S = entropies(ψ)
    (; D, c = fit_chord(S, L_scan)[1], S_mid = S[L_scan ÷ 2], E = info.Es[end], bond = maximum(info.bond_dimensions[end]))
end
push!(scan, (; D = D_max, c = results[end].fit[1], S_mid = results[end].S[L_scan ÷ 2], E = results[end].E, bond = results[end].bond))
for s in scan
    @printf("  D_max = %2d: E/L = %.10f, S(L/2) = %.6f, c = %.4f, bonds ≤ %d\n", s.D, s.E, s.S_mid, s.c, s.bond)
end

# Figure: S(l) against the chord log for every L, S(L/2) against ln L, and c against D_max
set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor = :white, fontsize = 22, size = (1800, 600))
Label(fig[0, 1:3], L"Golden chain $H = -J\sum_i P_I(i,i+1)$, $J = %$J$, open boundaries, $Δτ = %$Δτ$"; fontsize = 26)

ax1 = Axis(fig[1, 1]; xlabel = L"\frac{1}{6}\ln\left[\frac{2L}{\pi}\sin\frac{\pi l}{L}\right]", ylabel = L"S(l)",
           title = "entanglement entropy (markers: ITE, lines: chord fit)")
for (k, r) in enumerate(results)
    ls = 1:r.L-1
    col = Makie.wong_colors()[mod1(k, 7)]
    scatter!(ax1, log.(chord.(ls, r.L)) ./ 6, r.S; color = col, label = L"L = %$(r.L)")
    c, s0, b = r.fit
    lf = 2:r.L-2
    lines!(ax1, log.(chord.(lf, r.L)) ./ 6, c .* log.(chord.(lf, r.L)) ./ 6 .+ s0 .+ b .* (-1.0) .^ lf; color = col)
end
axislegend(ax1; position = :lt)

ax2 = Axis(fig[1, 2]; xlabel = L"\frac{1}{6}\ln(2L/\pi)", ylabel = L"S(L/2)", title = "middle bond across L")
scatter!(ax2, X_mid, S_mid; markersize = 14, label = "ITE")
scatter!(ax2, X_mid, [r.S_dmrg[r.L ÷ 2] for r in results]; marker = :xcross, markersize = 14, label = "DMRG")
xs = range(extrema(X_mid)...; length = 50)
lines!(ax2, xs, c_mid .* xs .+ s0_mid; label = L"c = %$(round(c_mid; digits = 3))")
lines!(ax2, xs, c_exact .* xs .+ (S_mid[end] - c_exact * X_mid[end]); linestyle = :dash, color = :gray, label = L"c = 7/10")
axislegend(ax2; position = :lt)

ax3 = Axis(fig[1, 3]; xlabel = L"D_{\max}\ \mathrm{(multiplets)}", ylabel = L"c", xscale = log2,
           title = L"finite-entanglement scan, $L = %$L_scan$")
scatterlines!(ax3, [s.D for s in scan], [s.c for s in scan]; markersize = 14, label = "chord fit (ITE)")
hlines!(ax3, [c_exact]; linestyle = :dash, color = :gray, label = L"c = 7/10")
axislegend(ax3; position = :rb)

# file = joinpath(@__DIR__, "..", "plots", "golden_chain.png")
# save(file, fig)
# println("\nsaved $file")
fig
