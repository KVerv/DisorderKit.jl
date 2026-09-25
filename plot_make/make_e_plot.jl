using TensorKit
using DelimitedFiles, LinearAlgebra, CairoMakie, LsqFit

using Revise, JLD2
using DisorderKit

function get_ϵs(folder_name, Ds, τs, invtols, freqs, ps)

    ϵs = Array{Vector{Float64}}(undef,length(Ds), length(τs), length(invtols), length(freqs))

 
    for (iD, Dmax) in enumerate(Ds)
        for (it, τ) in enumerate(τs)
            for (ii, invtol) in enumerate(invtols)
                for (ifr, freq) in enumerate(freqs)
                    # folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_truncfreq$(freq)"
                    folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_Z4"
                    isdir("data/$folder_name1") || @error("Folder $folder_name1 does not exist.")

                    @show Dmax, τ, invtol

                    if isfile("data/$folder_name1/ϵs.jld2")
                        ϵ = load_object("data/$folder_name1/ϵs.jld2")
                        ϵs[iD,it,ii,ifr] = ϵ
                    else
                        @warn("data/$folder_name1/ρ_β$β does not exist")
                    end
                end
            end
        end
    end

    return ϵs
end

function plot_ϵ_β(Ds,Dts,ϵs)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(2000, 1000))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"β",
            ylabel = L"$ϵ$",
            # xscale = log10,
            yscale = log10,
            xticks = 0:1:50,
            yticks = [1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12]
            )
    for (iD, Dmax) in enumerate(Ds)
        for (it, t) in enumerate(Dts)
            ϵs[iD,it] = ϵs[iD,it][1:600]
            colors = [:red, :blue, :green, :orange, :purple, :brown, :pink, :gray]
            L = length(ϵs[iD,it])
            βs = t:t:L*t
            scatter!(ax1,βs,(ϵs[iD,it]), label=L"$D=%$Dmax$, $Δτ=%$t$",markersize = 16, color=colors[iD])
            lines!(ax1,βs,(ϵs[iD,it]), color=colors[iD])
            lines!(ax1,βs,1e-6*ones(length(βs)), color=:black)
        end
    end
    axislegend(ax1, position=:rb)
    return fig, ax1
end



N = 3
a = 0.7
b = 1.3
hs = Vector(a:(b-a)/(N-1):b)
Js = hs
ps = ones(N^2)./N^2

# Ds = [10,20,40,80,100]
Ds = [40,80,100]
τs = [0.05]
invtols = [1e-6]
truncfrequency = [1]

folder_name ="ρ_a$(a)_b$(b)_N$(N)"

ϵs = get_ϵs(folder_name,Ds,τs, invtols, truncfrequency, ps)

ϵsplot = ϵs[:,:,1,1]

fig1, ax1 = plot_ϵ_β(Ds, τs, ϵsplot)
fig1

save("errorD.pdf",fig1)

