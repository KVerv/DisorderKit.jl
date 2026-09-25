using TensorKit
using DelimitedFiles, LinearAlgebra, CairoMakie, LsqFit

using Revise, JLD2
using DisorderKit

function double_ρ(ρ ,ps)
    invtol = 1e-8
    D_max = 20
    D_Z = 2
    alg_inversion = VOMPS_Inversion(2; tol = 1e-8, maxiter = 50, verbosity = 2)
    alg_trunc_Z = StandardTruncation(trunc_method = truncdim(D_Z))
    alg_trunc_disordermpo = DisorderOpenTruncation(trunc_method = truncdim(D_max))
    
    t = ρ*ρ
    ρs_normalized, ϵ_acc, mpoZinv = normalize_each_disorder_sector(t, ps, alg_trunc_Z, alg_inversion; init_guess = nothing, verbosity = 2, invtol = invtol)
    ρs = truncate_mpo(ρs_normalized, ps, alg_trunc_disordermpo)
    return ρs
end

function get_ξs(folder_name, βs, Ds, τs, invtols, freqs, ps)

    ξs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))
    ξs2 = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))
    δs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))

    fill!(ξs,0)
    fill!(δs,0)

    for (ib, β) in enumerate(βs)
        for (iD, Dmax) in enumerate(Ds)
            for (it, τ) in enumerate(τs)
                for (ii, invtol) in enumerate(invtols)
                    for (ifr, freq) in enumerate(freqs)
                        # folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_truncfreq$(freq)"
                         folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)"
                        isdir("data/$folder_name1") || @error("Folder $folder_name1 does not exist.")

                        @show Dmax, β, τ, invtol

                        β = convert(Int, round(β))
                        if isfile("data/$folder_name1/ρ_β$β.jld2")
                            ρ = load_object("data/$folder_name1/ρ_β$β.jld2")
                            ξ = average_correlation_length(ρ, ps)
                            ξs[ib,iD,it,ii,ifr] = ξ
                            ρ2 = double_ρ(ρ, ps)
                            ξ2 = average_correlation_length(ρ2, ps)
                            ξs2[ib,iD,it,ii,ifr] = ξ2
                        else
                            @warn("data/$folder_name1/ρ_β$β does not exist")
                        end
                    end
                end
            end
        end
    end
    return ξs, ξs2
end

function plot_ξ_β(βs,Ds,Dts,ξs,ξs2)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(600, 600))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"(\ln{β})^2",
            ylabel = L"$ξ$",
            # xscale = log10,
            # yscale = log10
            )
    for (iD, Dmax) in enumerate(Ds)
        for (it, t) in enumerate(Dts)
            nzxi = findall(x -> x > 0, ξs[:,iD,it])
            scatter!(ax1,log.(βs[nzxi]).^2,(ξs[:,iD,it][nzxi]), label=L"$D=%$Dmax$, $Δτ=%$t$",markersize = 16)
            scatter!(ax1,log.(2*βs[nzxi]).^2,(ξs2[:,iD,it][nzxi]), label=L"$D=%$Dmax$, $Δτ=%$t$",markersize = 16)
            # scatter!(ax1,βs[nzxi],(ξs[:,iD,it][nzxi]), label=L"$D=%$Dmax$, $Δτ=%$t$",markersize = 16)
        end
    end
    minfit = 5
    maxfit = 10
    p0q = [1.,1.,1.]
    quadmodel(t,p) = p[1].+p[2]*t .+p[3]*t.^2
    quadfit = curve_fit(quadmodel, log.(βs[minfit:maxfit]), ξs[minfit:maxfit,end,1], p0q)
    @show quadfit.param
    lines!(ax1,log.(βs).^2,quadmodel(log.(βs),quadfit.param), color=:black, linewidth=2)
    axislegend(ax1, position=:lt)
    return fig, ax1
end



N = 3
a = 0.7
b = 1.3
hs = Vector(a:(b-a)/(N-1):b)
Js = hs
ps = ones(N^2)./N^2

βs = 1:1.0:20
Ds = [80]
τs = [0.05]
invtols = [1e-6]
truncfrequency = [1]

folder_name ="ρ_a$(a)_b$(b)_N$(N)"

ξs, ξs2 = get_ξs(folder_name,βs,Ds,τs, invtols, truncfrequency, ps)

ξsplot = ξs[:,:,:,1,1]
ξsplot2 = ξs2[:,:,:,1,1]

fig1, ax1 = plot_ξ_β(βs, Ds, τs, ξsplot, ξsplot2)
fig1
