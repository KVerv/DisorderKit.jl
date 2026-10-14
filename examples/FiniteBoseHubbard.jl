# Finite-chain version of the MPSKit 1D Bose-Hubbard example
# (https://quantumkithub.github.io/MPSKit.jl/dev/examples/quantum1d/8.bose-hubbard/)
# H = -t Σ (a†ᵢaᵢ₊₁ + h.c.) + U/2 Σ nᵢ(nᵢ-1) - μ Σ nᵢ on an open chain of L sites, solved with DMRG.
using TensorKit, MPSKit, MPSKitModels, CairoMakie

# Setup
L = 20
cutoff, D = 5, 5
initial_state = FiniteMPS(L, ℂ^(cutoff + 1), ℂ^D)

# Bosonic operators
a_op = a_min(; cutoff = cutoff)
n_op = a_number(; cutoff = cutoff)
a_pm = a_plusmin(; cutoff = cutoff) # a† ⊗ a, used for two-point correlations
display(a_op[])
display((a_op' * a_op)[])

# Ground state
hamiltonian = bose_hubbard_model(FiniteChain(L); cutoff = cutoff, U = 1, mu = 0.5, t = 0.2)
ground_state, _, _ = find_groundstate(initial_state, hamiltonian, DMRG(; tol = 1.0e-6, verbosity = 2, maxiter = 200))
E = real(expectation_value(ground_state, hamiltonian))
println("Energy: ", E, "  energy per site: ", E / L)

function get_ground_state(mu, t, cutoff, D, L; kwargs...)
    hamiltonian = bose_hubbard_model(FiniteChain(L); cutoff = cutoff, U = 1, mu = mu, t = t)
    state = FiniteMPS(L, ℂ^(cutoff + 1), ℂ^D)
    state, _, _ = find_groundstate(state, hamiltonian, DMRG(; kwargs...))
    return state
end

# ⟨a†_{i0} a_{i0+r}⟩ for r in rs, measured from a bulk reference site to avoid the edges
correlation_profile(ψ, i0, rs) = real.(correlator(ψ, a_pm, i0, i0 .+ rs))

i0 = L ÷ 4
rs = 1:(L ÷ 2)

# Mott insulator: exponential decay of correlations
ground_state = get_ground_state(0.5, 0.01, cutoff, D, L; tol = 1.0e-6, verbosity = 2, maxiter = 500)
fig = Figure(size = (900, 350))
ax1 = Axis(fig[1, 1]; xlabel = "Distance r", ylabel = "⟨a†ᵢ aᵢ₊ᵣ⟩", yscale = log10, title = "Mott (t = 0.01)")
lines!(ax1, rs, abs.(correlation_profile(ground_state, i0, rs)); linewidth = 2)
ax2 = Axis(fig[1, 2]; xlabel = "Site i", ylabel = "⟨nᵢ⟩")
lines!(ax2, 1:L, [real(expectation_value(ground_state, i => n_op)) for i in 1:L]; linewidth = 2)
display(fig)

# Superfluid: algebraic decay of correlations
ground_state = get_ground_state(0.5, 0.2, cutoff, D, L; tol = 1.0e-6, verbosity = 2, maxiter = 500)
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "Distance r", ylabel = "⟨a†ᵢ aᵢ₊ᵣ⟩", xscale = log10, yscale = log10, title = "Superfluid (t = 0.2)")
lines!(ax, rs, abs.(correlation_profile(ground_state, i0, rs)); linewidth = 2)
display(fig)

# Bond-dimension dependence
Ds = 20:5:50
mu, t = 0.5, 0.2
states = Vector{FiniteMPS}(undef, length(Ds))
Threads.@threads for idx in eachindex(Ds)
    states[idx] = get_ground_state(mu, t, cutoff, Ds[idx], L; tol = 1.0e-7, verbosity = 1, maxiter = 500)
end

two_point_correlation = zeros(length(Ds), length(rs))
Threads.@threads for idx in eachindex(Ds)
    two_point_correlation[idx, :] .= correlation_profile(states[idx], i0, rs)
end

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "Distance |i-j|", ylabel = "⟨a†ᵢ aⱼ⟩", xscale = log10, yscale = log10)
for (idx, Dval) in enumerate(Ds)
    lines!(ax, rs, abs.(two_point_correlation[idx, :]); linewidth = 2, label = "D = $Dval")
end
axislegend(ax; position = :rt)
# inset: half-chain entanglement entropy, which saturates once D is large enough
inset = Axis(fig[1, 1]; width = Relative(0.3), height = Relative(0.3), halign = 0.15, valign = 0.1,
    xlabel = "D", ylabel = "S(L/2)", xlabelsize = 10, ylabelsize = 10)
scatter!(inset, Ds, [real(entropy(ψ, L ÷ 2)) for ψ in states])
display(fig)

# On a finite chain the U(1) symmetry is not broken: ⟨a⟩ → 0 as D grows
quasicondensate_density = map(ψ -> abs2(expectation_value(ψ, (L ÷ 2) => a_op)), states)
for (Dval, qc) in zip(Ds, quasicondensate_density)
    println("D = $Dval: |⟨a_{L/2}⟩|² = $qc")
end

# Momentum distribution n_k = 1/L_b Σ_{ij} e^{ik(i-j)} ⟨a†ᵢ aⱼ⟩ over the bulk sites
function correlation_matrix(ψ, sites)
    C = zeros(ComplexF64, length(sites), length(sites))
    for (a, i) in enumerate(sites)
        C[a, a] = expectation_value(ψ, i => n_op)
        a < length(sites) || continue
        C[a, (a + 1):end] .= correlator(ψ, a_pm, i, sites[(a + 1):end])
        C[(a + 1):end, a] .= conj.(C[a, (a + 1):end])
    end
    return C
end

bulk = (L ÷ 10 + 1):(L - L ÷ 10)
ks = range(-0.3, 0.3, 500)
momentum_distribution = map(states) do ψ
    C = correlation_matrix(ψ, bulk)
    map(ks) do k
        phase = exp.(1im * k .* bulk)
        real(phase' * C * phase) / length(bulk)
    end
end

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "Momentum k", ylabel = "⟨n_k⟩")
for (idx, Dval) in enumerate(Ds)
    lines!(ax, ks, momentum_distribution[idx]; linewidth = 1.5, label = "D = $Dval")
end
axislegend(ax)
display(fig)

# Charge gap Δ = E(N+1) + E(N-1) - 2E(N) at fixed particle number (U(1) symmetry).
# This replaces the twisted-boundary stiffness of the infinite example: on an open chain a uniform phase twist
# can be gauged away, so E(φ) is flat in both phases. Δ is finite in the Mott phase and ~1/L in the superfluid.
function fixed_number_energy(t, mu, N, D, cutoff, L)
    hamiltonian = bose_hubbard_model(U1Irrep, FiniteChain(L); cutoff = cutoff, U = 1, mu = mu, t = t)
    P = Vect[U1Irrep](n => 1 for n in 0:cutoff)
    V = Vect[U1Irrep](n => D for n in 0:N)
    state = FiniteMPS(L, P, V; right = Vect[U1Irrep](N => 1))
    state, _, _ = find_groundstate(state, hamiltonian, DMRG2(; trscheme = truncrank(D), tol = 1.0e-8, verbosity = 0))
    return real(expectation_value(state, hamiltonian))
end

function charge_gap(t, mu, D, cutoff, L; N = L)
    Es = map(n -> fixed_number_energy(t, mu, n, D, cutoff, L), (N - 1, N, N + 1))
    return Es[1] + Es[3] - 2 * Es[2]
end

L_gap, D_gap = 10, 20
println("Charge gap superfluid (t = 0.2):  ", charge_gap(0.2, 0.3, D_gap, cutoff, L_gap))
println("Charge gap Mott       (t = 0.01): ", charge_gap(0.01, 0.3, D_gap, cutoff, L_gap))

# Phase diagram: local density fluctuations ⟨n²⟩ - ⟨n⟩² at the center site
# (≈ 0 in the Mott lobes, finite in the superfluid) replace the order parameter |⟨a⟩|.
L_pd, D_pd = 10, 10
mus = range(0, 0.75, 20)
ts = range(0, 0.3, 20)
n2_op = n_op * n_op
density_fluctuations = zeros(length(mus), length(ts))

Threads.@threads for (i, j) in collect(Iterators.product(eachindex(mus), eachindex(ts)))
    ψ = get_ground_state(mus[i], ts[j], cutoff, D_pd, L_pd; tol = 1.0e-8, verbosity = 0)
    c = L_pd ÷ 2
    density_fluctuations[i, j] = real(expectation_value(ψ, c => n2_op) - expectation_value(ψ, c => n_op)^2)
end

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "t/U", ylabel = "μ/U", title = "⟨n²⟩ - ⟨n⟩²")
hm = heatmap!(ax, ts, mus, density_fluctuations')
Colorbar(fig[1, 2], hm)
display(fig)
