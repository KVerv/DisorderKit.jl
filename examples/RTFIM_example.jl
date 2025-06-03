using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit

# Define model
N = 3
a = 0.7
b = 1.3

Js = [1.0]
# hs = Vector(a:(b-a)/(N-1):b)
# ps = ones(N)./N
hs = [0.99]
ps = [1.0]
dτ = 5e-2

# U = disorder_average(Us, [1.])

# Us = RTFIM_time_evolution_Trotter(dτ, hs, Js)
# Us = DisorderMPO([Us[1]])
Hs = RTFIM_hamiltonian(Js, hs)

# Define algorithms
invtol = 1e-6
trunctol = 1e-6
D_max = 30
alg_inversion = VOMPS_Inversion(1; tol = 1e-8, maxiter = 250, verbosity = 2)
alg_trunc_Z = StandardTruncation(trunc_method = truncerr(trunctol))
alg_trunc_disordermpo = DisorderTracedTruncation(trunc_method = truncdim(D_max))

βs = 1:1:25
# Evolve density matrix
function get_ξ(βs, Us)
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
        alg_evolution = iDTEBD(alg_inversion, alg_trunc_Z, alg_trunc_disordermpo; invtol = invtol, nsteps = nsteps, verbosity = 2, truncfrequency = 1, inversion_frequency = inversion_frequency, timer_output = TimerOutput(), max_inverse_dim = 4)
        ρs, ϵ = evolve_densitymatrix(Us, ps, alg_evolution; ρ0 = ρ0)
        ξs[i] = average_correlation_length(ρs, ps)
        Es[i] = (measure(ρs, ps, Hs, 1))
        Z = partition_functions(ρs)
        Z = truncate_mpo(Z, alg_trunc_Z)
        @show DisorderKit.test_identity_random(Z)
        ρ0 = ρs
        push!(ϵs, ϵ...)
    end
    return ξs, Es, ϵs, DZs
end
set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(1000, 1000))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$(\ln{β})^2$",
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
            ylabel = L"$\langle E\rangle$",
    #         # xscale = log10,
            # yscale = log10
            )
    ax3 = Axis(fig[2, 1], 
    xlabel = L"$(\ln β)^2$",
    ylabel = L"$\frac{d\xi}{d\ln{\beta}^2}$",
#         # xscale = log10,
    # yscale = log10
    )
    ax4 = Axis(fig[2, 2], 
    xlabel = L"$β$",
    ylabel = L"$C_v$",
#         # xscale = log10,
    # yscale = log10
    )
for n in [0]
    if n == 0
        Us = RTFIM_time_evolution_Trotter(dτ, hs, Js)
        Us = DisorderMPO([Us[1]])
    else
        Us = random_transverse_field_ising_evolution(Js, hs, dτ; order=n)
    end
    ξs, Es, ϵs, DZs = get_ξ(βs, Us)

    Cvs = real.(-βs[1:end-1].^2 .*diff(Es)./diff(βs))

    dxidlnβ2 = diff(log.(ξs))./diff(log.(βs))
    # Plot correlation lengths in function of β
    # scatter!(ax1,log.(βs),ξs, label=L"$n=%$n$, $Δτ=%$dτ$",markersize = 16)
    # scatter!(ax1,βs,ξs, label=L"$n=%$n$, $Δτ=%$dτ$",markersize = 16)
    scatter!(ax1,log.(βs),log.(ξs), label=L"$n=%$n$, $Δτ=%$dτ$",markersize = 16)
    # scatter!(ax1,log.(βs),ξs, label=L"$D=%$D_max$, $Δτ=%$dτ$",markersize = 16)
    # scatter!(ax2,log.(βs[1:end-1]),Cvs[1:end], label=L"$D=%$D_max$, $Δτ=%$dτ$",markersize = 16)
    scatter!(ax2,βs,real.(Es), label=L"$n=%$n$, $Δτ=%$dτ$",markersize = 16)
    scatter!(ax3,log.(βs[1:end-1]), dxidlnβ2, label=L"$n=%$n$, $Δτ=%$dτ$",markersize = 16)
    scatter!(ax4,βs[1:end-1],Cvs, label=L"$n=%$n$, $Δτ=%$dτ$",markersize = 16)

    xs = log.(βs)
    ys = log.(ξs)
    fitmax = 6
    fitmin = 1
    p0q = [1.,1.]
    quadmodel(t,p) = p[1].+p[2]*t
    quadfit = curve_fit(quadmodel, xs[fitmin:fitmax], ys[fitmin:fitmax], p0q)
    b, a = quadfit.param
    @show a, b
    lines!(ax1,xs,a*xs.+b,color =:black)
end
axislegend(ax1, position=:lt)
axislegend(ax2, position=:lt)
fig
