# Clean Bose-Hubbard chain on an open chain of L sites, solved with DMRG (MPSKit):
#   H = -t Σ (a†ᵢaᵢ₊₁ + h.c.) + U/2 Σ nᵢ(nᵢ-1) - μ Σ nᵢ.
# The entanglement entropy S(k) on every bond k = 1..L-1 is compared for a point in the n = 1 Mott insulator (small t/U,
# S ≈ 0, close to the product state |1 1 … 1⟩) and one in the superfluid (larger t/U, S larger with a dome over the
# chain). Clean reference for the disordered runs in hpc/*BoseHubbard*.
using TensorKit, MPSKit, MPSKitModels, CairoMakie

# Model
L = 6
cutoff = 4                  # at most cutoff bosons per site
U, μ = 1.0, 0.5
t_mott, t_sf = 0.02, 0.2    # at μ/U = 0.5 the n = 1 Mott lobe ends around t/U ≈ 0.1-0.15
D = 64                      # FiniteMPS caps the bonds near the edges at their exact maximum

n_op = a_number(; cutoff)

function ground_state(t)
    H = bose_hubbard_model(FiniteChain(L); cutoff, U, mu = μ, t)
    ψ0 = FiniteMPS(L, ℂ^(cutoff + 1), ℂ^D)
    ψ, _, _ = find_groundstate(ψ0, H, DMRG(; tol = 1e-10, maxiter = 200, verbosity = 1))
    N = sum(real(expectation_value(ψ, i => n_op)) for i in 1:L)
    println("t/U = $(t/U): E = $(real(expectation_value(ψ, H))), ⟨N⟩ = $N")
    return ψ
end

# Entanglement entropy on the bond between site k and k+1
entropy_profile(ψ) = [real(entropy(ψ, k)) for k in 1:L-1]

S_mott = entropy_profile(ground_state(t_mott))
S_sf = entropy_profile(ground_state(t_sf))
println("S(k) Mott:       ", round.(S_mott; digits = 4))
println("S(k) superfluid: ", round.(S_sf; digits = 4))

bonds = 1:L-1
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "bond k", ylabel = "S(k)", xticks = bonds, title = "Bose-Hubbard, L = $L, μ/U = $(μ/U)")
scatterlines!(ax, bonds, S_mott; linewidth = 2, markersize = 12, label = "Mott (t/U = $(t_mott/U))")
scatterlines!(ax, bonds, S_sf; linewidth = 2, markersize = 12, label = "superfluid (t/U = $(t_sf/U))")
ylims!(ax, 0, nothing)
axislegend(ax; position = :ct)
display(fig)
