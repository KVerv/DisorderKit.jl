using TensorKit
using DelimitedFiles, LinearAlgebra, CairoMakie, LsqFit, StatsBase

using Revise, JLD2
using DisorderKit

function get_ξs(folder_name, βs, Ds, τs, invtols, freqs, Δs, ps)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)

    ξs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))
    cs_list = Array{Vector{Float64}}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))

    fill!(ξs,0)
    fill!(cs_list,zeros(length(βs)))

    for (ib, β) in enumerate(βs)
        for (iD, Dmax) in enumerate(Ds)
            for (it, τ) in enumerate(τs)
                for (ii, invtol) in enumerate(invtols)
                    for (ifr, freq) in enumerate(freqs)
                        if Dmax == 40
                            folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)"
                        else
                            folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_Z4"
                        end
                        isdir("data/$folder_name1") || @error("Folder $folder_name1 does not exist.")

                        @show Dmax, β, τ, invtol

                        β = convert(Int, round(β))
                        if isfile("data/$folder_name1/ρ_β$β.jld2")
                            ρ = load_object("data/$folder_name1/ρ_β$β.jld2")
                            cs = ComplexF64[]
                            for Δ in Δs
                                # c = measure(ρ, ps, Z, Z, 0, Δ)-1/4*(measure(ρ, ps, Z, 0)+measure(ρ, ps, Z, 1))^2
                                m = measure(ρ, ps, Z, 0)
                                c = measure(ρ, ps, Z, Z, 0, Δ)-m^2
                                push!(cs,c)
                            end
                            ys = abs.(cs)
                            cs_list[ib,iD,it,ii,ifr]= ys

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
    return ξs, cs_list
end

function plot_correlations(rs, cs, βs, ξs, τs)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=40, size=(1000, 600))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$\ln(r)$",
            ylabel = L"$\ln(G(r))$",
            # xscale = log10,
            # yscale = log10
            )
    for (id, ξ) in enumerate(ξs)
        rmax = round(Integer,1.5*ξ)
        scatter!(ax1,log.(rs[1:rmax]),log.(cs[id][1:rmax]/cs[id][1]), label=L"$ξ=%$(round(ξ,digits=2))$", markersize = 20)
    end
    rmax = round(Integer,ξs[end])
    rs = rs[1:rmax]
    xmin = log(rs[12])
    ymin = log.(cs[end][12]/cs[end][1])
    lines!(ax1, log.(rs), -0.25*(log.(rs) .-log(rs[1])).+log.(cs[1][1]/cs[1][1]), color=:black, linestyle=:dash, linewidth=2, label=L"$r^{-0.25}$")
    lines!(ax1, log.(rs), -0.38*(log.(rs) .-xmin).+ymin, color=:black, linewidth=2, label=L"$r^{-0.38}$")
    axislegend(ax1, position=:lb)
    return fig, ax1
end



N = 1
hs = [1.]
Js = hs
ps = [1.]

rs = 1:1:80
βs = [17]
Ds = [40]
τs = [0.05]
invtols = [1e-6]
truncfrequency = [1]

folder_name ="clean_ρ_N$(N)"
ξs, cs = get_ξs(folder_name,βs,Ds,τs, invtols, truncfrequency, rs, ps)

ξsplot = [ξs[1,1,1,1,1]]
csplot = [cs[1,1,1,1,1]]
cbetas = [βs]

a = 0.7
b = 1.3
N = 3
hs = Vector(a:(b-a)/(N-1):b)
Js = hs
ps = ones(N^2)./N^2
l = 1/var(log.(Js))
rs = 1:1:80
βs = [38]
Ds = [40]
τs = [0.05]
invtols = [1e-6]
truncfrequency = [1]

folder_name ="ρ_a$(a)_b$(b)_N$(N)"
ξs, cs = get_ξs(folder_name,βs,Ds,τs, invtols, truncfrequency, rs, ps)


push!(ξsplot, ξs[1,1,1,1,1])
push!(csplot, cs[1,1,1,1,1])
push!(cbetas, βs)

# folder_name ="ρ_a$(a)_b$(b)_N$(N)"

fig, ax1 = plot_correlations(rs, csplot, cbetas, ξsplot, τs)
fig
