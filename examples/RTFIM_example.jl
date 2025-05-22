using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie

# Define model
N = 5 
a = 0.7
b = 1.33

Js = [1.0]
hs = Vector(a:(b-a)/(N-1):b)
# hs = [1.0]
ps = ones(N)./N
dτ = 1e-1

Us = random_transverse_field_ising_evolution(Js, hs, dτ; order=2)
# U = disorder_average(Us, [1.])

# Us = RTFIM_time_evolution_Trotter(dτ, hs, Js)
# Us = DisorderMPO([Us[1]])
Hs = RTFIM_hamiltonian(Js, hs)

# Define algorithms
invtol = 5e-1
trunctol = 1e-8
D_max = 20
alg_inversion = VOMPS_Inversion(1; tol = 1e-8, maxiter = 250, verbosity = 2)
alg_trunc_Z = StandardTruncation(trunc_method = truncerr(trunctol))
alg_trunc_disordermpo = DisorderTracedTruncation(trunc_method = truncdim(D_max))

βs = 1:0.4:4
# Evolve density matrix
function get_ξ(βs, Us)
    ξs = zeros(length(βs))
    Es = zeros(ComplexF64,length(βs))
    ϵs = []
    ρ0 = nothing
    nsteps = round(Int, βs[1]/dτ)
    for (i,β) in enumerate(βs)
        @show (i,β)
        if β > βs[1]
            dβ = βs[i] - βs[i-1]
            nsteps = round(Int, dβ/dτ)
        end
        inversion_frequency = 1
        alg_evolution = iDTEBD(alg_inversion, alg_trunc_Z, alg_trunc_disordermpo; invtol = invtol, nsteps = nsteps, verbosity = 2, truncfrequency = 1, inversion_frequency = inversion_frequency, timer_output = TimerOutput())
        ρs, ϵ = evolve_densitymatrix(Us, ps, alg_evolution; ρ0 = ρ0)
        ξs[i] = average_correlation_length(ρs, ps)
        Es[i] = (measure(ρs, ps, Hs, 1))
        ρ0 = ρs
        push!(ϵs, ϵ...)
    end
    return ξs, Es, ϵs
end

ξs, Es, ϵs = get_ξ(βs, Us)

Cvs = real.(-βs[1:end-1].^2 .*diff(Es)./diff(βs))

# Plot correlation lengths in function of β
set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=30, size=(1000, 1000))
ax1 = Axis(fig[1, 1], 
        xlabel = L"$(\ln{β})^2$",
        ylabel = L"$ξ$",
        # xscale = log10,
        # yscale = log10
        )
ax2 = Axis(fig[1, 2], 
        xlabel = L"$(\ln{β})$",
        ylabel = L"$C_v$",
        # xscale = log10,
        # yscale = log10
        )
scatter!(ax1,log.(βs).^2,ξs, label=L"$D=%$D_max$, $Δτ=%$dτ$",markersize = 16)
# scatter!(ax1,log.(βs),ξs, label=L"$D=%$D_max$, $Δτ=%$dτ$",markersize = 16)
scatter!(ax2,log.(βs[1:end-1]),Cvs[1:end], label=L"$D=%$D_max$, $Δτ=%$dτ$",markersize = 16)
fig
