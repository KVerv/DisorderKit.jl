using TensorKit
using DelimitedFiles, LinearAlgebra, CairoMakie, LsqFit, StatsBase

using Revise, JLD2
using DisorderKit

function get_ξs(folder_name, βs, Ds, τs, invtols, freqs, p, Ls)

    ξs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))
    Ss = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs), length(Ls))

    fill!(ξs,0)
    fill!(Ss,0)

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
                            ρ = DisorderKit.fix_phase(ρ,ps)
                            ξ = average_correlation_length(ρ, ps)
                            ξs[ib,iD,it,ii,ifr] = ξ
                            ρ_av = disorder_average(ρ, ps)
                            for (il,L) in enumerate(Ls)
                                S = renyi_entropy2(ρ_av, L)
                                # S = average_renyi_entropy2(ρ, ps, 20)
                                @show S
                                imag(S) < 1e-4 || @warn("S has imaginary part: S = $S")
                                Ss[ib,iD,it,ii,ifr,il] = real.(S)
                            end
                        else
                            @warn("data/$folder_name1/ρ_β$β does not exist")
                        end
                    end
                end
            end
        end
    end
    return ξs, Ss
end

function plot_ξs(bs, ξs, Ss)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(800, 800))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$1/ξ$",
            ylabel = L"$S_2$",
            # ylabel = L"$\ln(ξ)$",
            # ylabel = L"$ξ/(A \cdot \ln(\beta)^2+ B\cdot \ln(\beta)+C)$",
            # xscale = log10,
            # yscale = log10
            )
    for (ib, b) in enumerate(bs)
        label = "δ = $(round(b,digits=3))"
        ξsnew = hcat(ξs...)
        Ssnew = hcat(Ss...)
        scatter!(ax1, 1 ./ξsnew[:,ib], Ssnew[:,ib], label=label,markersize = 16)
        # scatter!(ax1, log.(βs), log.(ξsnew[:,ib]), label=label,markersize = 16)
        p0 = [1.,1.]
        linmodel(t,p) = p[1].+p[2] .*t
        linfit = curve_fit(linmodel, 1 ./ξsnew[:,ib], Ssnew[:,ib], p0)
        @show linfit.param
        lines!(ax1,1 ./ξsnew[:,ib],linmodel(1 ./ξsnew[:,ib],linfit.param), color=:black, linewidth=2)
    
    end
    axislegend(ax1, position=:lt)
    return fig, ax1
end

function plot_S_b(bs, βs, Ss)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(800, 800))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$δ$",
            ylabel = L"$S_2$",
            # ylabel = L"$\ln(ξ)$",
            # ylabel = L"$ξ/(A \cdot \ln(\beta)^2+ B\cdot \ln(\beta)+C)$",
            # xscale = log10,
            # yscale = log10
            )
    for (ib, β) in enumerate(βs)
        label = "β = $(round(β,digits=3))"
        Ssnew = hcat(Ss...)
        scatter!(ax1, bs, Ssnew[ib,:], label=label,markersize = 16)
        lines!(ax1, bs, Ssnew[ib,:])
        # scatter!(ax1, log.(βs), log.(ξsnew[:,ib]), label=label,markersize = 16)
        # p0 = [1.,1.]
        # linmodel(t,p) = p[1].+p[2] .*t
        # linfit = curve_fit(linmodel, log.(βs), log.(ξsnew[:,ib]), p0)
        # @show linfit.param
        # lines!(ax1,log.(βs),linmodel(log.(βs),linfit.param), color=:black, linewidth=2)
    
    end
    axislegend(ax1, position=:lt)
    return fig, ax1
end

N = 3
a = 0.7
# bs = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.9]
bs = [1.3]
Js = Vector(a:(1.3-a)/(N-1):1.3)
ps = ones(N^2)./N^2

βs = 20:2:50
iβ = length(βs)
Ds = [80]
τs = [0.05]
invtols = [1e-6]
truncfrequency = [1]
δs = Float64[]
Ls = 1:50

ξss = Array{Vector{Float64}}(undef, length(bs))
Sss = Array{Vector{Float64}}(undef, length(bs), length(Ls))
for (ib,b) in enumerate(bs)
    hs = Vector(a:(b-a)/(N-1):b)
    VJ = var(Js; corrected = false)
    Vh = var(hs; corrected = false)
    mh = mean(hs)
    mJ = mean(Js)
    push!(δs, (mh-mJ)/(Vh+VJ))


    folder_name ="ρ_a$(a)_b$(b)_N$(N)"

    ξs, Ss = get_ξs(folder_name,βs,Ds,τs, invtols, truncfrequency, ps, Ls)
    @show Ss
    ξss[ib] = ξs[:,1,1,1,1]
    for il in 1:length(Ls)
        @show Ss[:,1,1,1,1,il]
        Sss[ib,il] = Ss[:,1,1,1,1,il]
    end
end

function plot_S_L(βs, Ls, Ss)
    Ls = Ls
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(800, 800))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$L$",
            ylabel = L"$S_2$",
            # ylabel = L"$\ln(ξ)$",
            # ylabel = L"$ξ/(A \cdot \ln(\beta)^2+ B\cdot \ln(\beta)+C)$",
            # xscale = log10,
            # yscale = log10
            )
    for (ib, b) in enumerate(βs)
        label = "β = $(round(b,digits=3))"
        Ssnew = hcat(Ss...)
        scatter!(ax1, Ls, Ssnew[ib,:].-0.028*Ls, label=label,markersize = 16)
        # scatter!(ax1, log.(βs), log.(ξsnew[:,ib]), label=label,markersize = 16)
        p0 = [1.,1.]
        linmodel(t,p) = p[1].+p[2] .*t
        linfit = curve_fit(linmodel, Ls[10:end], Ssnew[ib,10:end], p0)
        @show linfit.param
        # lines!(ax1,Ls,linmodel(Ls,linfit.param), color=:black, linewidth=2)
    
    end
    axislegend(ax1, position=:lt)
    return fig, ax1
end

iL = 10
fig1, ax1 = plot_S_b(δs, βs, Sss[:,10])
fig1

fig2, ax2 = plot_ξs(δs, ξss, Sss[:,10])
fig2

id = 1
fig3, ax3 = plot_S_L(βs, Ls, Sss[id,:])
fig3
# save("xi_ferro.svg",fig2)
# save("xi_para.svg",fig2)
