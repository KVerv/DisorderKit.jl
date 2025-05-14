using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie

# Define model
N = 3
a = 0.7
b = 1.33

Js = [1.0]
hs = Vector(a:(b-a)/(N-1):b)
ps = ones(N)./N
dτ = 0.05

# Us = random_transverse_field_ising_evolution(Js, hs, dτ; order=2)
# Us = DisorderMPO([Us[1],Us[2]])
Us = TFIM_time_evolution_with_disorder(dτ, hs, Js)

# Define algorithms
invtol = 1e-8
D_max = 50
alg_inversion = VOMPS_Inversion(1; tol = 1e-8, maxiter = 250, verbosity = 2)
alg_trunc_Z = StandardTruncation(trunc_method = truncerr(invtol))
alg_trunc_disordermpo = DisorderTracedTruncation(trunc_method = truncdim(D_max))

βs = 1:10
ξs = zeros(length(βs))
# Evolve density matrix
function get_ξ()
    ρ0 = nothing
    nsteps = round(Int, βs[1]/dτ)
    for (i,β) in enumerate(βs)
        @show (i,β)
        if β > 1
            dβ = βs[i] - βs[i-1]
            nsteps = round(Int, dβ/dτ)
        end
        alg_evolution = iDTEBD(alg_inversion, alg_trunc_Z, alg_trunc_disordermpo; invtol = invtol, nsteps = nsteps, verbosity = 2, truncfrequency = 1, timer_output = TimerOutput())
        ρs = evolve_densitymatrix(Us, ps, alg_evolution; ρ0 = ρ0)
        ξs[i] = average_correlation_length(ρs, ps)
        ρ0 = ρs
    end
    return ξs
end

ξs = get_ξ()

# Plot correlation lengths in function of β
set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=30, size=(600, 600))
ax1 = Axis(fig[1, 1], 
        xlabel = L"$(\ln{β})^2$",
        ylabel = L"$ξ$",
        # xscale = log10,
        # yscale = log10
        )
scatter!(ax1,log.(βs).^2,ξs, label=L"$D=%$D_max$, $Δτ=%$dτ$",markersize = 16)
fig