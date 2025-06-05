using Revise, TensorKit, MPSKit, MPSKitModels, KrylovKit
using DisorderKit, TimerOutputs, CairoMakie, LsqFit
# Define model
Ns = [1, 2]
# Js = [1.0]
a = 0.7
b = 1.3
hmean = (a+b)/2

dτ = 5e-2

# Define algorithms
invtol = 1e-6
trunctol = 1e-8
D_max = 20
alg_inversion = VOMPS_Inversion(1; tol = 1e-8, maxiter = 250, verbosity = 2)
alg_trunc_Z = StandardTruncation(trunc_method = truncerr(trunctol))
# alg_trunc_disordermpo = DisorderTracedTruncation(trunc_method = truncdim(D_max))
alg_trunc_disordermpo = DisorderOpenTruncation(trunc_method = truncdim(D_max))
# alg_trunc_disordermpo = SVDUpdateTruncation(D_max, tol = trunctol, maxit = 10, verbosity = 2)


βs = 1:0.5:20
# Evolve density matrix
function get_ξ(βs, Us, ps)
    ξs = zeros(length(βs))
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
        Z = partition_functions(ρs)
        Z = truncate_mpo(Z, alg_trunc_Z)
        DZs[i] = dim(space(Z[1])[1])
        ρ0 = ρs
        push!(ϵs, ϵ...)
    end
    return ξs, ϵs, DZs
end
set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(1000, 1000))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$β$",
            ylabel = L"$ξ$",
            # xscale = log10,
            # yscale = log10
            )
    ax2 = Axis(fig[1, 2], 
            xlabel = L"$β$",
            ylabel = L"$ϵ$",
    #         # xscale = log10,
            yscale = log10
            )
    ax3 = Axis(fig[2, 1], 
            xlabel = L"$(\ln{β})^2$",
            ylabel = L"$ξ$",
            # xscale = log10,
            # yscale = log10
    )
#     ax4 = Axis(fig[2, 2], 
#     xlabel = L"$β$",
#     ylabel = L"$D_\mathcal{Z}$",
#        # xscale = log10,
#     # yscale = log10
#     )
ax4 = Axis(fig[2, 2], 
xlabel = L"$\ln{β}$",
ylabel = L"$\xi$",
#         # xscale = log10,
# yscale = log10
)

linparams = zeros(2)
logparams = zeros(3)

for (iN, N) in enumerate(Ns)
    if N == 1
        Us = RTFIM_time_evolution_Trotter(dτ, [hmean], [1.0])
        Us = DisorderMPO([Us[1]])
        ps = [1.]
    else
        hs = Vector(a:(b-a)/(N-1):b)
        ps = ones(N^2)./N^2

        Us = RTFIM_time_evolution_Trotter(dτ, hs, hs)
        Us = DisorderMPO([Us[1]])
    end

    ξs, ϵs, DZs = get_ξ(βs, Us, ps)

    # Make fits
    minfit = length(ξs)-10
    maxfit = length(ξs)
    linmodel(t, p) = p[1] .+ p[2]*t
    p0 = [1., 1.]
    linfit = curve_fit(linmodel, βs[1:maxfit], ξs[1:maxfit], p0)
    global linparams = linfit.param

    logmodel(t, p) = p[1] .+ p[2]*log.(t) .+ p[3]*log.(t).^2
    p0 = [1., 1., 1.]
    logfit = curve_fit(logmodel, βs[1:maxfit], ξs[1:maxfit], p0)
    global logparams = logfit.param

    # Plot correlation lengths in function of β
    colorslin = [:black, :grey]
    colorslog = [:red, :purple]
    colorserror = [:blue, :orange]
    scatter!(ax1,βs,ξs, label=L"$N=%$N$, $Δτ=%$dτ$",markersize = 16)
    lines!(ax1, βs, linmodel(βs, linfit.param), label=L"linear fit", color=colorslin[iN], linewidth=2)
    lines!(ax1, βs, logmodel(βs, logfit.param), label=L"log fit", color=colorslog[iN], linewidth=2)
    
    # scatter!(ax2,dτ:dτ:βs[end],ϵs.+1e-16, label=L"$N=%$N$, $Δτ=%$dτ$",markersize = 16)
    scatter!(ax2,βs, abs.(linmodel(βs, linfit.param).-ξs), label=L"\text{linear}",color=colorserror[iN],markersize = 16)
    scatter!(ax2,βs, abs.(logmodel(βs, logfit.param).-ξs), label=L"\text{log}", marker = :star5,color=colorserror[iN],markersize = 16)
    
    
    scatter!(ax3,log.(βs).^2, ξs, label=L"$N=%$N$, $Δτ=%$dτ$",markersize = 16)
    lines!(ax3, log.(βs).^2, linmodel(βs, linfit.param), label=L"linear fit", color=colorslin[iN], linewidth=2)
    lines!(ax3, log.(βs).^2, logmodel(βs, logfit.param), label=L"log fit", color=colorslog[iN], linewidth=2)
    
    # scatter!(ax4,βs,DZs, label=L"$N=%$N$, $Δτ=%$dτ$",markersize = 16)
    scatter!(ax4,log.(βs), ξs, label=L"$N=%$N$, $Δτ=%$dτ$",markersize = 16)
    lines!(ax4, log.(βs), linmodel(βs, linfit.param), label=L"linear fit", color=colorslin[iN], linewidth=2)
    lines!(ax4, log.(βs), logmodel(βs, logfit.param), label=L"log fit", color=colorslog[iN], linewidth=2)

end

# axislegend(ax1, position=:lt)
# axislegend(ax2, position=:lt)
fig[1, 3] = Legend(fig, ax2, framevisible = false)
fig
