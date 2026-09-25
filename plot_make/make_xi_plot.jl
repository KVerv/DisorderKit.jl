using TensorKit
using DelimitedFiles, LinearAlgebra, CairoMakie, LsqFit

using Revise, JLD2
using DisorderKit

function get_ξs(folder_name, βs, Ds, τs, invtols, freqs, ps)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)

    ξs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))
    δs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))

    fill!(ξs,0)
    fill!(δs,0)

    for (ib, β) in enumerate(βs)
        for (iD, Dmax) in enumerate(Ds)
            for (it, τ) in enumerate(τs)
                for (ii, invtol) in enumerate(invtols)
                    for (ifr, freq) in enumerate(freqs)
                        # folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_truncfreq$(freq)"
                        # if Dmax == 80 || Dmax == 100
                        folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_Z4"
                        # else
                        #     folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)"
                        # end

                        isdir("data/$folder_name1") || @error("Folder $folder_name1 does not exist.")

                        @show Dmax, β, τ, invtol

                        β = convert(Int, round(β))
                        if isfile("data/$folder_name1/ρ_β$β.jld2")
                            ρ = load_object("data/$folder_name1/ρ_β$β.jld2")
                            ρ = DisorderKit.fix_phase(ρ,ps)
                            ξ = average_correlation_length(ρ, ps)
                            ξs[ib,iD,it,ii,ifr] = ξ
                        else
                            @warn("data/$folder_name1/ρ_β$β does not exist")
                        end
                    end
                end
            end
        end
    end
    return ξs
end

function plot_ξ_β(βs,Ds,Dts,ξs)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(600, 600))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"(\ln{β})^2",
            ylabel = L"$ξ$",
            # xscale = log10,
            # yscale = log10
            )
    minfit = 4
    for (iD, Dmax) in enumerate(Ds)
        for (it, t) in enumerate(Dts)
            nzxi = findall(x -> x > 0, ξs[:,iD,it])
            nzxi = nzxi[minfit:end]
            scatter!(ax1,log.(βs[nzxi]).^2,(ξs[:,iD,it][nzxi]), label=L"$D=%$Dmax$, $Δτ=%$t$",markersize = 20)
            # scatter!(ax1,βs[nzxi],(ξs[:,iD,it][nzxi]), label=L"$D=%$Dmax$, $Δτ=%$t$",markersize = 16)
        end
    end
    minplot = 4
    minfit = 30
    maxfit = 50
    p0q = [1.,1.]
    linmodel(t,p) = p[1].+p[2]*t
    linfit = curve_fit(linmodel, log.(βs[minfit:maxfit]).^2, ξs[minfit:maxfit,end,end], p0q)
    @show linfit.param
    lines!(ax1,log.(βs[minplot:end]).^2,linmodel(log.(βs[minplot:end]).^2,linfit.param), color=:black, linewidth=2)

    # maxfit = 50
    # p0q = [1.,1.,1.]
    # quadmodel(t,p) = p[1].+p[2]*t .+p[3]*t.^2
    # quadfit = curve_fit(quadmodel, log.(βs[minfit:maxfit]), ξs[minfit:maxfit,end,1], p0q)
    # @show quadfit.param
    # lines!(ax1,log.(βs[minfit:end]).^2,quadmodel(log.(βs[minfit:end]),quadfit.param), color=:black, linewidth=2)
    # p0q = [1.,1.]
    # logmodel(t,p) = p[1].*log.(t./p[2]).^2
    # logfit = curve_fit(logmodel, βs[minfit:maxfit], ξs[minfit:maxfit,end,1], p0q)
    # @show logfit.param
    # lines!(ax1,log.(βs).^2,logmodel(βs,logfit.param), color=:black, linewidth=2)
    axislegend(ax1, position=:lt)
    return fig, ax1
end

function extrapolate_ξ(βs,Ds,Dts,ξs, δs)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(600, 600))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$δ$",
            ylabel = L"$ξ$",
            xscale = log10,
            yscale = log10
            )
    for (iD, Dmax) in enumerate(Ds)
        for (it, t) in enumerate(Dts)
            nzxi = findall(x -> (x > 0), ξs[:,iD,it])
            scatter!(ax1,δs[:,iD,it],ξs[:,iD,it], label=L"$D=%$Dmax$, $Δτ=%$t$",markersize = 12)
            # scatter!(ax1,βs[nzxi],(ξs[:,iD,it][nzxi]), label=L"$D=%$Dmax$, $Δτ=%$t$",markersize = 16)
        end
    end
    # minfit = 3
    # maxfit = 15
    # p0q = [1.,1.,1.]
    # quadmodel(t,p) = p[1].+p[2]*t .+p[3]*t.^2
    # quadfit = curve_fit(quadmodel, log.(βs[minfit:maxfit]), ξs[minfit:maxfit,end,1], p0q)
    # @show quadfit.param
    # lines!(ax1,log.(βs).^2,quadmodel(log.(βs),quadfit.param), color=:black, linewidth=2)
    # fig[1, 2] = Legend(fig, ax1, framevisible = false)
    return fig, ax1
end

N = 3
a = 0.7
b = 1.3
# a = 0.5
# b = 1.5
hs = Vector(a:(b-a)/(N-1):b)
Js = hs
ps = ones(N^2)./N^2

βs = 1:1.0:50
Ds = [10 , 20, 40, 80, 100]
# Ds =[80]
# Ds = [80, 100]
τs = [0.05]
# τs = [0.01, 0.05, 0.1]
invtols = [1e-6]
truncfrequency = [1]

folder_name ="ρ_a$(a)_b$(b)_N$(N)"

ξs = get_ξs(folder_name,βs,Ds,τs, invtols, truncfrequency, ps)

ξsplot = ξs[:,:,:,1,1]

fig1, ax1 = plot_ξ_β(βs, Ds, τs, ξsplot)
fig1

save("xi_beta.pdf",fig1)
