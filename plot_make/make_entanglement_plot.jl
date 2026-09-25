using TensorKit
using DelimitedFiles, LinearAlgebra, CairoMakie, LsqFit

using Revise, JLD2
using DisorderKit

function get_entanglement(folder_name, βs, Ds, τs, invtols, freqs, ps)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)

    es = Array{Vector{Float64}}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))
    eZs = Array{Vector{Float64}}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))

    fill!(es,[])
    fill!(eZs,[])

    for (ib, β) in enumerate(βs)
        for (iD, Dmax) in enumerate(Ds)
            for (it, τ) in enumerate(τs)
                for (ii, invtol) in enumerate(invtols)
                    for (ifr, freq) in enumerate(freqs)
                        # folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_truncfreq$(freq)"
                         folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_Z4"
                        isdir("data/$folder_name1") || @error("Folder $folder_name1 does not exist.")

                        @show Dmax, β, τ, invtol

                        β = convert(Int, round(β))
                        if isfile("data/$folder_name1/ρ_β$β.jld2")
                            ρ = load_object("data/$folder_name1/ρ_β$β.jld2")
                            av_ρ = disorder_average(ρ, ps)
                            ent = DisorderKit.entanglement_spectrum(av_ρ , 1)
                            es[ib,iD,it,ii,ifr] = ent
                            Z = partition_functions(ρ)
                            eZ = DisorderKit.entanglement_spectrum(Z, 1)
                            eZs[ib,iD,it,ii,ifr] = eZ
                            # Empo = measure(ρ, ps, Hs, 1)
                            # Esmpo[ib,iD,it,ii,ifr] = real.(Empo)
                            # CVs[ib,iD,it,ii,ifr] = -real(β^2 * diff(E)/diff(β))
                        else
                            @warn("data/$folder_name1/ρ_β$β does not exist")
                        end
                    end
                end
            end
        end
    end
    return es, eZs
end

function plot_ent_spec(βs,ents; number_eigs = 10)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(1000, 1000))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"#val",
            ylabel = L"$σ_i$",
            # xscale = log10,
            yscale = log10,
            )
    ax2 = Axis(fig[2, 1], 
    xlabel = L"$\beta$",
    ylabel = L"$σ_i$",
    # xscale = log10,
    yscale = log10
    )
    colors = [Makie.wong_colors()...,:red, :blue, :green, :orange, :purple, :brown, :pink, :cyan, :magenta, :yellow, :olive, :teal, :navy, :maroon]
    for i in eachindex(βs)
        label = "β = $(βs[i])"
        scatter!(ax1, 1:number_eigs, ents[i][1:number_eigs], label=label,markersize = 16, color=colors[i])
    end
    for i in 1:10
        label = "i = $(i)"
        ent_new = hcat(ents...)
        scatter!(ax2, βs, ent_new[i,:], label=label,markersize = 16, color=colors[i])
        lines!(ax2, βs, ent_new[i,:], color=colors[i])
    end
    fig[1, 2] = Legend(fig, ax1, framevisible = false)
    fig[2, 2] = Legend(fig, ax2, framevisible = false)
    return fig, ax1
end

function plot_ee_β(βs,ents; number_eigs = 10)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(1000, 1000))
    ax1 = Axis(fig[1, 1], 
    xlabel = L"$\beta$",
    ylabel = L"$dE/dβ$",
    # xscale = log10,
    yscale = log10
    )
    ax2 = Axis(fig[2, 1], 
    xlabel = L"$\ln(\beta)$",
    ylabel = L"$des$",
    # xscale = log10,
    yscale = log10
    )
    for i in 1:number_eigs
        label = "i = $(i)"
        ent_new = hcat(ents...)
        scatter!(ax1, βs[1:end-1], abs.(diff(ent_new[i,:])), label=label,markersize = 16)
        scatter!(ax2, log.(βs), abs.(ent_new[i+1,:]-ent_new[i,:]), label=label,markersize = 16)
    end
    fig[1, 2] = Legend(fig, ax1, framevisible = false)
    fig[2, 2] = Legend(fig, ax2, framevisible = false)
    return fig, ax1
end

N = 3
a = 0.7
b = 1.3
hs = Vector(a:(b-a)/(N-1):b)
Js = hs
ps = ones(N^2)./N^2

βs = 5:5:40
Ds = [80]
τs = [0.05]
invtols = [1e-6]
truncfrequency = [1]

folder_name ="ρ_a$(a)_b$(b)_N$(N)"

es, eZs = get_entanglement(folder_name,βs,Ds,τs, invtols, truncfrequency, ps)

esplot = es[:,1,1,1,1]
eZsplot = eZs[:,1,1,1,1]

@show  hcat(esplot...)[:,end]
fig1, ax1 = plot_ent_spec(βs, esplot; number_eigs = 80)
fig1
fig2, ax2 = plot_ent_spec(βs, eZsplot; number_eigs = 60)
fig2

fig3, ax3 = plot_ee_β(βs, esplot; number_eigs = 15)
fig3

save("rho_ent.pdf",fig1)
save("Z_ent.pdf",fig2)
