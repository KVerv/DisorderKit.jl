using TensorKit
using DelimitedFiles, LinearAlgebra, CairoMakie, LsqFit, StatsBase

using Revise, JLD2
using DisorderKit

function get_entanglement(folder_name, βs, Ds, τs, invtols, freqs, ps)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)

    es = Array{Vector{Float64}}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))
    eZs = Array{Vector{Float64}}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))
    ξs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))


    fill!(es,[])
    fill!(eZs,[])
    fill!(ξs,0)

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
    return es, eZs, ξs
end

function plot_ent_entro(bs, βs, ents, ξsb)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(1000, 1000))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$\ln(ξ)$",
            ylabel = L"$S$",
            # xscale = log10,
            # yscale = log10
            )
    ent_new = stack(ents)
    ibc = 1
    S = -vec(sum(ent_new[:,ibc,:] .*log.(ent_new[:,ibc,:]), dims=1))
    # S = -vec(sum(ent_new[:,ibc,:].^2 .*log.(ent_new[:,ibc,:].^2), dims=1))
    scatter!(ax1, log.(ξsb[ibc,:]), S, label="δ=$(δs[ibc])", markersize = 16)
    p0 = [1.,1.]
    linmodel(t,p) = p[1].+p[2] .*t
    linfit = curve_fit(linmodel, log.(ξsb[ibc,:]), S, p0)
    @show linfit.param
    lines!(ax1,log.(ξsb[ibc,:]),linmodel(log.(ξsb[ibc,:]),linfit.param),label=L"$%$(round(linfit.param[2],digits=2))\ln(ξ) + %$(round(linfit.param[1],digits=2))$", color=:red, linewidth=2)
    fig[1, 2] = Legend(fig, ax1, framevisible = false)
    return fig, ax1
end


N = 3
a = 0.7
bs = [1.3]
Js = Vector(a:(1.3-a)/(N-1):1.3)
ps = ones(N^2)./N^2

βs = 1:1:45
iβ = length(βs)
Ds = [80]
τs = [0.05]
invtols = [1e-6]
truncfrequency = [1]
δs = Float64[]

esb = Array{Vector{Float64}}(undef, length(bs), length(βs))
eZsb = Array{Vector{Float64}}(undef, length(bs), length(βs))
ξsb = Array{Float64}(undef, length(bs), length(βs))
for (ib,b) in enumerate(bs)
    hs = Vector(a:(b-a)/(N-1):b)
    VJ = var(Js; corrected = false)
    Vh = var(hs; corrected = false)
    mh = mean(hs)
    mJ = mean(Js)
    push!(δs, (mh-mJ)/(Vh+VJ))


    folder_name ="ρ_a$(a)_b$(b)_N$(N)"

    es, eZs, ξs = get_entanglement(folder_name,βs,Ds,τs, invtols, truncfrequency, ps)
    for ibeta in eachindex(βs)
        esb[ib, ibeta] = es[ibeta,1,1,1,1]
        eZsb[ib, ibeta] = eZs[ibeta,1,1,1,1]
        ξsb[ib, ibeta] = ξs[ibeta,1,1,1,1]
    end
end


fig1, ax1 = plot_ent_entro(δs, βs, esb, ξsb)
fig1

# save("c_charge.svg",fig1)