using TensorKit
using DelimitedFiles, LinearAlgebra, CairoMakie, LsqFit

using Revise, JLD2, StatsBase
using DisorderKit

function get_ξs(folder_name, βs, Ds, τs, invtols, freqs, Ns)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)

    ξs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs),length(Ns))
    δs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs),length(Ns))

    fill!(ξs,0)
    fill!(δs,0)

    for (ib, β) in enumerate(βs)
        for (iD, Dmax) in enumerate(Ds)
            for (it, τ) in enumerate(τs)
                for (ii, invtol) in enumerate(invtols)
                    for (ifr, freq) in enumerate(freqs)
                        for (iN, N) in enumerate(Ns)
                            ps = ones(N^2)./N^2
                            # folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_truncfreq$(freq)"
                            # if Dmax == 80 || Dmax == 100
                            folder_name1 = folder_name*"_N$(N)_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_Z4"
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
                                ξs[ib,iD,it,ii,ifr,iN] = ξ
                            else
                                @warn("data/$folder_name1/ρ_β$β does not exist")
                            end
                        end
                    end
                end
            end
        end
    end
    return ξs
end

function plot_ξ_β(βs,Ds,Ns,ξs)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(600, 600))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"(\ln{β})^2",
            ylabel = L"$ξ$",
            # xscale = log10,
            # yscale = log10
            )
    # ax2 = Axis(fig[1, 2], 
    # xlabel = L"(\ln{β})^2",
    # ylabel = L"$ξᵢ/ξ₄$",
    # # xscale = log10,
    # # yscale = log10
    # )   
    minfit = 4
    for (iD, Dmax) in enumerate(Ds)
        for (iN, N) in enumerate(Ns)
            Js = Vector(0.7:(1.3-0.7)/(N-1):1.3)
            ps = ones(N^2)./N^2
            V = 2*var(log.(Js);corrected = false)
            nzxi = findall(x -> x > 0, ξs[:,iD,iN])
            nzxi = nzxi[minfit:end]
            scatter!(ax1,log.(βs[nzxi]).^2,(ξs[:,iD,iN][nzxi]).*V, label=L"$D=%$Dmax$, $N=%$N$",markersize = 20)
            # scatter!(ax2,log.(βs[nzxi]).^2,(ξs[:,iD,iN][nzxi]./ξs[:,iD,3][nzxi]), label=L"$D=%$Dmax$, $N=%$N$",markersize = 20)
            # scatter!(ax1,βs[nzxi],(ξs[:,iD,it][nzxi]), label=L"$D=%$Dmax$, $Δτ=%$t$",markersize = 16)
        end
    end
    minplot = 4
    minfit = 20
    maxfit = 30
    p0q = [1.,1.]
    linmodel(t,p) = p[1].+p[2]*t
    linfit = curve_fit(linmodel, log.(βs[minfit:maxfit]).^2, ξs[minfit:maxfit,end,end], p0q)
    @show linfit.param
    lines!(ax1,log.(βs[minplot:end]).^2,linmodel(log.(βs[minplot:end]).^2,linfit.param), color=:black, linewidth=2)

    # maxfit = 30
    # p0q = [1.,1.,1.]
    # quadmodel(t,p) = p[1].+p[2]*t .+4/pi^2*p[3]*t.^2
    # quadfit = curve_fit(quadmodel, log.(βs[minfit:maxfit]), ξs[minfit:maxfit,end,3], p0q)
    # @show quadfit.param
    # confidence_inter = confint(quadfit; level=0.95)
    # @show confidence_inter
    # lines!(ax1,log.(βs[minfit:end]).^2,quadmodel(log.(βs[minfit:end]),quadfit.param), color=:black, linewidth=2)
    # p0q = [1.,1.]
    # logmodel(t,p) = p[1].*log.(t./p[2]).^2
    # logfit = curve_fit(logmodel, βs[minfit:maxfit], ξs[minfit:maxfit,end,1], p0q)
    # @show logfit.param
    # lines!(ax1,log.(βs).^2,logmodel(βs,logfit.param), color=:black, linewidth=2)
    axislegend(ax1, position=:lt)
    # axislegend(ax2, position=:rt)
    # a= quadfit.param[3]
    # VN = 0.0958
    # c = a*VN*2
    # @show c
    return fig, ax1
end


Ns = [2,3,4]
a = 0.7
b = 1.3


βs = 1:1.0:40
Ds = [80]
τs = [0.05]
invtols = [1e-6]
truncfrequency = [1]

folder_name ="ρ_a$(a)_b$(b)"

ξs = get_ξs(folder_name,βs,Ds,τs, invtols, truncfrequency, Ns)

ξsplot = ξs[:,:,1,1,1,:]

fig1, ax1 = plot_ξ_β(βs, Ds, Ns, ξsplot)
fig1

# save("xi_N.pdf",fig1)
