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
                        if Ds == 80
                            folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_Z4"
                        else
                            folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)"
                        end

                        isdir("data/$folder_name1") || @error("Folder $folder_name1 does not exist.")

                        @show Dmax, β, τ, invtol

                        β = convert(Int, round(β))
                        if isfile("data/$folder_name1/ρ_β$β.jld2")
                            ρ = load_object("data/$folder_name1/ρ_β$β.jld2")
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
    fig = Figure(backgroundcolor=:white, fontsize=40, size=(1000, 600))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"β",
            ylabel = L"$ξ$",
            # xscale = log10,
            # yscale = log10
            )
    for ξss in ξs
        # scatter!(ax1,log.(βs).^2,(ξss),markersize = 20)
        scatter!(ax1,βs,(ξss),markersize = 20)
        # scatter!(ax1,βs[nzxi],(ξs[:,iD,it][nzxi]), label=L"$D=%$Dmax$, $Δτ=%$t$",markersize = 16)
    end
    # minfit = 5
    # maxfit = 20
    # p0q = [1.,1.,1.]
    # quadmodel(t,p) = p[1].+p[2]*t .+p[3]*t.^2
    # quadfit = curve_fit(quadmodel, log.(βs[minfit:maxfit]), ξs[minfit:maxfit,end,1], p0q)
    # @show quadfit.param
    # lines!(ax1,log.(βs).^2,quadmodel(log.(βs),quadfit.param), color=:black, linewidth=2)
    # axislegend(ax1, position=:lt)
    return fig, ax1
end

N = 1
hs = [1.]
Js = hs
ps = [1.]
βs = 1:1.0:20
Ds = [40]
τs = [0.05]
invtols = [1e-6]
truncfrequency = [1]

folder_name ="clean_ρ_N$(N)"
ξs = get_ξs(folder_name,βs,Ds,τs, invtols, truncfrequency, ps)

ξsplot = [ξs[:,1,1,1,1]]


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

ξs = get_ξs(folder_name,βs,Ds,τs, invtols, truncfrequency, ps)

push!(ξsplot, ξs[:,1,1,1,1])

fig1, ax1 = plot_ξ_β(βs, Ds, τs, ξsplot)
fig1

