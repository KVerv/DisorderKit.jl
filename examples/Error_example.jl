using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie

# Define model
N = 4
a = 0.8
b = 1.2

# Js = [1.0]
# hs = Vector(a:(b-a)/(N-1):b)
# # hs = [1.0]
# ps = ones(N)./N
dτ = 5e-2


# U = disorder_average(Us, [1.])

# Us = RTFIM_time_evolution_Trotter(dτ, hs, Js)
# Us = DisorderMPO([Us[1]])
# Hs = RTFIM_hamiltonian(Js, hs)

# Define algorithms
invtol = 1e-6
trunctol = 1e-6
D_max = 50
alg_inversion = VOMPS_Inversion(1; tol = 1e-8, maxiter = 250, verbosity = 2)
alg_trunc_Z = StandardTruncation(trunc_method = truncerr(trunctol))
alg_trunc_disordermpo = DisorderTracedTruncation(trunc_method = truncdim(D_max))

βs = 0.1:0.5:2
# Evolve density matrix
function fixed_point_distribution_h(h,β)
    Γ = log(β)
    y = h/Γ
    return  1/Γ*exp(-y)
end

function get_ξ(βs, Us, ps)
    ξs = zeros(length(βs))
    Es = zeros(ComplexF64,length(βs))
    DZs = zeros(Int,length(βs))
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
        alg_evolution = iDTEBD(alg_inversion, alg_trunc_Z, alg_trunc_disordermpo; invtol = invtol, nsteps = nsteps, verbosity = 2, truncfrequency = 1, inversion_frequency = inversion_frequency, timer_output = TimerOutput(), max_inverse_dim = 2)
        ρs, ϵ = evolve_densitymatrix(Us, ps, alg_evolution; ρ0 = ρ0)
        ξs[i] = average_correlation_length(ρs, ps)
        # Es[i] = (measure(ρs, ps, Hs, 1))
        Es[i] = 1
        Z = partition_functions(ρs)
        Z = truncate_mpo(Z, alg_trunc_Z)
        @show DisorderKit.test_identity_random(Z)
        DZs[i] = dim(space(Z[1])[1])
        ρ0 = ρs
        push!(ϵs, ϵ...)
    end
    return ξs, Es, ϵs, DZs
end
set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(1000, 1000))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$β$",
            ylabel = L"$ξ$",
            # xscale = log10,
            # yscale = log10
            )
    # ax2 = Axis(fig[1, 2], 
    #         xlabel = L"$(\ln{β})$",
    #         ylabel = L"$C_v$",
    #         # xscale = log10,
    #         # yscale = log10
    #         )
    ax2 = Axis(fig[1, 2], 
            xlabel = L"$\ln β$",
            ylabel = L"$ϵ$",
    #         # xscale = log10,
            yscale = log10
            )
    ax3 = Axis(fig[2, 1], 
    xlabel = L"$(\ln β)^2$",
    ylabel = L"$\frac{d\xi}{d\ln{\beta}^2}$",
#         # xscale = log10,
    # yscale = log10
    )
    ax4 = Axis(fig[2, 2], 
    xlabel = L"$β$",
    ylabel = L"$D_\mathcal{Z}$",
#         # xscale = log10,
    # yscale = log10
    )
for n in [0, 2]
    if n == 0
        # Us = RTFIM_time_evolution_Trotter(dτ, hs, Js)
        Us = RTFIM_time_evolution_Trotter(dτ, [1.0], [1.0])
        Us = DisorderMPO([Us[1]])
        ps = [1.]
    else
        Js = [1.0]
        hs = Vector(a:(b-a)/(N-1):b)
        # hs = [1.0]
        ps = ones(N)./N

        # hs = [0.9, 1.1]
        # ps = [0.5, 0.5]
        ps = ps ./ sum(ps)
        @show ps
        # Us = random_transverse_field_ising_evolution(Js, hs, dτ; order=n)
        # Us = RTFIM_time_evolution_Trotter(dτ, [1.], [1.1])
        Us = RTFIM_time_evolution_Trotter(dτ, hs, Js)
        Us = DisorderMPO([Us[1]])
        # ps = [1.]
    end
    ξs, Es, ϵs, DZs = get_ξ(βs, Us, ps)

    Cvs = real.(-βs[1:end-1].^2 .*diff(Es)./diff(βs))

    # dxidlnβ2 = diff(ξs)./diff(log.(βs).^2)
    dxidlnβ2 = diff(ξs)./diff(βs)
    # Plot correlation lengths in function of β
    # scatter!(ax1,log.(βs).^2,ξs, label=L"$n=%$n$, $Δτ=%$dτ$",markersize = 16)
    scatter!(ax1,βs,ξs, label=L"$n=%$n$, $Δτ=%$dτ$",markersize = 16)
    # scatter!(ax2,log.(βs[1:end-1]),Cvs[1:end], label=L"$D=%$D_max$, $Δτ=%$dτ$",markersize = 16)
    scatter!(ax2,log.(dτ:dτ:βs[end]).^2,ϵs.+1e-16, label=L"$n=%$n$, $Δτ=%$dτ$",markersize = 16)
    scatter!(ax3,log.(βs[1:end-1]).^2, dxidlnβ2, label=L"$n=%$n$, $Δτ=%$dτ$",markersize = 16)
    scatter!(ax4,log.(βs).^2,DZs, label=L"$n=%$n$, $Δτ=%$dτ$",markersize = 16)

end
axislegend(ax1, position=:lt)
# axislegend(ax2, position=:lt)
fig
