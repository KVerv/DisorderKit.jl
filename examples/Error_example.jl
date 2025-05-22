using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie

# Define model
N = 3 
a = 0.7
b = 1.33

Js = [1.0]
hs = Vector(a:(b-a)/(N-1):b)
# hs = [1.0]
ps = ones(N)./N
dτ = 2e-2

# U = disorder_average(Us, [1.])

# Us = RTFIM_time_evolution_Trotter(dτ, hs, Js)
# Us = DisorderMPO([Us[1]])
Hs = RTFIM_hamiltonian(Js, hs)

# Define algorithms
invtol = 1e-1
trunctol = 1e-8
D_max = 25
alg_inversion = VOMPS_Inversion(1; tol = 1e-8, maxiter = 250, verbosity = 2)
alg_trunc_Z = StandardTruncation(trunc_method = truncerr(trunctol))
alg_trunc_disordermpo = DisorderTracedTruncation(trunc_method = truncdim(D_max))

βs = 0.1:0.1:10
# Evolve density matrix
function get_ξ(βs, Us, m)
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
        alg_evolution = iDTEBD(alg_inversion, alg_trunc_Z, alg_trunc_disordermpo; invtol = invtol, nsteps = nsteps, verbosity = 2, truncfrequency = 1, inversion_frequency = m, timer_output = TimerOutput())
        ρs, ϵ = evolve_densitymatrix(Us, ps, alg_evolution; ρ0 = ρ0)
        ξs[i] = average_correlation_length(ρs, ps)
        Es[i] = (measure(ρs, ps, Hs, 1))
        ρ0 = ρs
        push!(ϵs, ϵ...)
    end
    return ξs, Es, ϵs
end

fig2 = Figure(backgroundcolor=:white, fontsize=30, size=(1000, 1000))
ax3 = Axis(fig2[1, 1], 
        xlabel = L"$β$",
        ylabel = L"$ϵ$",
        # xscale = log10,
        yscale = log10
        )

for m in 10:10
    n = 2
    if n == 0
        Us = RTFIM_time_evolution_Trotter(dτ, hs, Js)
        Us = DisorderMPO([Us[1]])
    else
        Us = random_transverse_field_ising_evolution(Js, hs, dτ; order=n)
    end
    ξs, Es, ϵs = get_ξ(βs, Us, m)

    Cvs = real.(-βs[1:end-1].^2 .*diff(Es)./diff(βs))


    scatter!(ax3,dτ:dτ:βs[end],ϵs.+1e-16, label=L"$n = %$m$ $Δτ=%$dτ$",markersize = 16)
end
lines!(ax3,dτ:dτ:βs[end],ones(length(dτ:dτ:βs[end]))*invtol)
axislegend(ax3, position=:lt)
fig2
