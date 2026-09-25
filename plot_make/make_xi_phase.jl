using TensorKit
using DelimitedFiles, LinearAlgebra, CairoMakie, LsqFit, StatsBase

using Revise, JLD2
using DisorderKit

function get_ξs(folder_name, βs, Ds, τs, invtols, freqs, ps)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)

    ξs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))

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

function plot_ξs(bs, βs, ξs)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(800, 600))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$δ$",
            ylabel = L"$ξ/(\ln(\beta)^2)$",
            # ylabel = L"$ξ/(A \cdot \ln(\beta)^2+ B\cdot \ln(\beta)+C)$",
            # xscale = log10,
            # yscale = log10
            )
    A = 4.52
    B = -6.11
    C = 8.73
    for (ib, β) in enumerate(βs)
        label = "β = $(β)"
        ξsnew = hcat(ξs...)
        # refξ = A*log.(β).^2 .+ B*log.(β) .+ C
        refξ = log.(β).^2
        scatter!(ax1, bs, ξsnew[ib,:]/refξ, label=label,markersize = 16)
        lines!(ax1, bs, ξsnew[ib,:]/refξ)
    end
    fig[1, 2] = Legend(fig, ax1, framevisible = false)
    # fig[2, 2] = Legend(fig, ax2, framevisible = false)
    return fig, ax1
end

N = 3
a = 0.7
bs = [1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.7]
Js = Vector(a:(1.3-a)/(N-1):1.3)
ps = ones(N^2)./N^2

βs = 10:2:20
iβ = length(βs)
Ds = [80]
τs = [0.05]
invtols = [1e-6]
truncfrequency = [1]
δs = Float64[]

ξss = Array{Vector{Float64}}(undef, length(bs))
for (ib,b) in enumerate(bs)
    hs = Vector(a:(b-a)/(N-1):b)
    VJ = var(Js; corrected = false)
    Vh = var(hs; corrected = false)
    mh = mean(hs)
    mJ = mean(Js)
    push!(δs, (mh-mJ)/(Vh+VJ))


    folder_name ="ρ_a$(a)_b$(b)_N$(N)"

    ξs = get_ξs(folder_name,βs,Ds,τs, invtols, truncfrequency, ps)
    ξss[ib] = ξs[:,1,1,1,1]
end


fig2, ax2 = plot_ξs(δs, βs, ξss)
fig2

# save("xi_phase.pdf",fig2)
