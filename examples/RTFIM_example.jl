using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit

function local_E(ρ::DisorderMPO,ps::Vector{Float64}, Js::Vector{Float64}, hs::Vector{Float64}, i::Int)
    X, Z, Id = TensorMap(zeros(ComplexF64, 2, 2),ℂ^2,ℂ^2), TensorMap(zeros(ComplexF64, 2, 2),ℂ^2,ℂ^2), TensorMap(zeros(ComplexF64, 2, 2),ℂ^2,ℂ^2)
    X[1, 2], X[2, 1] = 1, 1
    Z[1, 1], Z[2, 2] = 1, -1
    Id[1, 1], Id[2, 2] = 1, 1
    Etotal = 0
    for (j, (h, J)) in enumerate(Iterators.product(hs, Js)) 
        Eh = -measure(ρ, ps, X, i)*h
        EJ = -measure(ρ, ps, Z, Z, i, 1)*J/2-measure(ρ, ps, Z, Z, i-1, 1)*J/2
        Etotal += Eh + EJ
    end
    return Etotal
end

# Define model
N = 2
a = 0.7
b = 1.3

Js = Vector(a:(b-a)/(N-1):b)
hs = Vector(a:(b-a)/(N-1):b)
ps = ones(N^2)./N^2

# Define Time-Evolution Operator
dτ = 5e-2
Us = RTFIM_time_evolution_Trotter(dτ, hs, Js)
Us = DisorderMPO([Us[1]])

# Define algorithms
invtol = 1e-6
D_max = 30
D_z = 2
alg_inversion = VOMPS_Inversion(1; tol = 1e-8, maxiter = 250, verbosity = 2)
alg_trunc_Z = StandardTruncation(trunc_method = truncdim(D_z))
alg_trunc_disordermpo = DisorderOpenTruncation(trunc_method = truncdim(D_max))

βs = 1:0.5:1
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
        alg_evolution = iDTEBD(alg_inversion, alg_trunc_Z, alg_trunc_disordermpo; invtol = invtol, nsteps = nsteps, verbosity = 2, truncfrequency = 1, inversion_frequency = inversion_frequency, timer_output = TimerOutput(), max_inverse_dim = 2)
        ρs, ϵ = evolve_densitymatrix(Us, ps, alg_evolution; ρ0 = ρ0)
        # Compute correlation length
        ξs[i] = average_correlation_length(ρs, ps)
        # Measure local energy density
        Es[i] = local_E(ρs, ps, Js, hs, 1)
        ρ0 = ρs
        push!(ϵs, ϵ...)
    end
    return ξs, Es, ϵs
end

ξs, Es, ϵs = get_ξ(βs, Us)

# Compute Specific Heat
Cvs = real.(-βs[1:end-1].^2 .*diff(Es)./diff(βs))

# Make fits
minfit = 10
maxfit = length(ξs)
linmodel(t, p) = p[1] .+ p[2]*t
p0 = [1., 1.]
linfit = curve_fit(linmodel, log.(βs[minfit:maxfit]).^2, ξs[minfit:maxfit], p0)
linparams = linfit.param

# Set up the figure and axes
set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(1000, 1000))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$(\ln{β})^2$",
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
            xlabel = L"$β$",
            ylabel = L"$<E>$",
            # xscale = log10,
            # yscale = log10
    )
    ax4 = Axis(fig[2, 2], 
        xlabel = L"$β$",
        ylabel = L"$C_v$",
        # xscale = log10,
        # yscale = log10
    )

# Plot correlation lengths in function of β
scatter!(ax1,log.(βs).^2, ξs, label=L"D=%$(D_max)",markersize = 16)
lines!(ax1,log.(βs).^2, linmodel(log.(βs).^2,linparams), label=L"$p_2=%$(linparams[2])$")

# Plot inversion accuracy in function of β
scatter!(ax2,dτ:dτ:βs[end],ϵs.+1e-16, label=L"D=%$(D_max)",markersize = 16)

# Plot average thermal energy in function of β
scatter!(ax3, βs[3:end], real.(Es[3:end]), label=L"Real",markersize = 16)
# scatter!(ax3, βs, imag.(Es), label=L"Imaginary",markersize = 16)

# Plot specific heat in function of β
scatter!(ax4, log.(βs[6:end-1]),Cvs[6:end], label=L"D=%$(D_max)",markersize = 16)

# fig[1, 3] = Legend(fig, ax1, framevisible = false)
# fig[2, 3] = Legend(fig, ax3, framevisible = false)
fig
