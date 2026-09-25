using TensorKit
using DelimitedFiles, LinearAlgebra, CairoMakie, LsqFit

using Revise, JLD2
using DisorderKit

function get_ξs(folder_name, βs, Ds, τs, invtols, freqs, DZs, ps)
    ξs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs), length(DZs))
    ϵs = Array{Vector{Float64}}(undef,length(Ds), length(τs), length(invtols), length(freqs), length(DZs))

    fill!(ξs,0)
    # fill!(ϵs,0)

    for (ib, β) in enumerate(βs)
        for (iD, Dmax) in enumerate(Ds)
            for (it, τ) in enumerate(τs)
                for (ii, invtol) in enumerate(invtols)
                    for (ifr, freq) in enumerate(freqs)
                        for (iDZ, DZ) in enumerate(DZs)
                            # folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_truncfreq$(freq)"
                            folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_Z$(DZ)"
                            isdir("data/$folder_name1") || @error("Folder $folder_name1 does not exist.")

                            @show Dmax, β, τ, invtol, DZ

                            β = convert(Int, round(β))
                            if isfile("data/$folder_name1/ρ_β$β.jld2")
                                ρ = load_object("data/$folder_name1/ρ_β$β.jld2")
                                ξ = average_correlation_length(ρ, ps)
                                ξs[ib,iD,it,ii,ifr,iDZ] = ξ
                            else
                                @warn("data/$folder_name1/ρ_β$β does not exist")
                            end
                            if isfile("data/$folder_name1/ϵs.jld2")
                                ϵ = load_object("data/$folder_name1/ϵs.jld2")
                                ϵs[iD,it,ii,ifr,iDZ] = ϵ
                            else
                                @warn("data/$folder_name1/ϵs does not exist")
                            end
                        end
                    end
                end
            end
        end
    end
    return ξs, ϵs
end

function plot_ξ_β(βs,Ds,DZs,ξs)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(600, 600))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"(\ln{β})^2",
            ylabel = L"$ξ$",
            # xscale = log10,
            # yscale = log10
            )
    for (iD, Dmax) in enumerate(Ds)
        for (iDZ, DZ) in enumerate(DZs)
            nzxi = findall(x -> x > 0, ξs[:,iD,iDZ])
            scatter!(ax1,log.(βs[nzxi]).^2,(ξs[:,iD,iDZ][nzxi]), label=L"$D=%$Dmax$, $DZ=%$DZ$",markersize = 16)
        end
    end
    minfit = 5
    maxfit = 20
    p0q = [1.,1.,1.]
    quadmodel(t,p) = p[1].+p[2]*t .+p[3]*t.^2
    quadfit = curve_fit(quadmodel, log.(βs[minfit:maxfit]), ξs[minfit:maxfit,end,end], p0q)
    @show quadfit.param
    # lines!(ax1,log.(βs).^2,quadmodel(log.(βs),quadfit.param), color=:black, linewidth=2)
    # axislegend(ax1, position=:lt)
    fig[1, 2] = Legend(fig, ax1, framevisible = false)
    return fig, ax1
end


function plot_ϵ_β(Ds,DZs,ϵs,t)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(600, 600))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"(\ln{β})^2",
            ylabel = L"$ϵ$",
            # xscale = log10,
            yscale = log10
            )
    for (iD, Dmax) in enumerate(Ds)
        for (iDZ, DZ) in enumerate(DZs)
            L = length(ϵs[iD,iDZ])
            βs = t:t:L*t
            indices = 1:L
            scatter!(ax1,βs,(ϵs[iD,iDZ][indices]), label=L"$D=%$Dmax$, $DZ=%$DZ$",markersize = 16)
        end
    end
    fig[1, 2] = Legend(fig, ax1, framevisible = false)
    return fig, ax1
end

function heat_plot(βs,Ds,DZs,ξs)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(600, 600))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$β$",
            ylabel = L"$D_Z$",
            )
    ax2 = Axis(fig[1, 2], 
    xlabel = L"$β$",
    ylabel = L"$D_Z$",
    )
    ax3 = Axis(fig[2, 1], 
    xlabel = L"$β$",
    ylabel = L"$D_Z$",
    )
    axes = [ax1, ax2, ax3]
    for (id, DZ) in enumerate(DZs)
        heatmap!(axes[id], βs, Ds, ξs[:,:,id])
    end
    return fig, ax1
end

N = 3
a = 0.3
b = 1.3
hs = Vector(a:(b-a)/(N-1):b)
Js = hs
ps = ones(N^2)./N^2

βs = 1:1.0:30
Ds = [20, 40, 80]
DZs = [2]
τs = [0.05]
invtols = [1e-6]
truncfrequency = [1]

folder_name ="ρ_a$(a)_b$(b)_N$(N)"

ξs, ϵs = get_ξs(folder_name,βs,Ds,τs, invtols, truncfrequency, DZs, ps)

ξsplot = ξs[:,:,1,1,1,:]
ϵsplot = ϵs[:,1,1,1,:]

fig1, ax1 = plot_ξ_β(βs, Ds, DZs, ξsplot)
fig1
fig2, ax2 = heat_plot(βs, Ds, DZs, ξsplot)
fig2
fig3, ax3 = plot_ϵ_β(Ds, DZs, ϵsplot, τs[1])
fig3
# save("xi_beta.svg",fig1)
