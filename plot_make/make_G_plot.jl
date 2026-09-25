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
                        # folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)"
                        folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_Z4"
                        isdir("data/$folder_name1") || @error("Folder $folder_name1 does not exist.")

                        @show Dmax, β, τ, invtol

                        β = convert(Int, round(β))
                        if isfile("data/$folder_name1/ρ_β$β.jld2")
                            ρ = load_object("data/$folder_name1/ρ_β$β.jld2")
                            ρ = DisorderKit.fix_phase(ρ,ps)
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
    fig = Figure(backgroundcolor=:white, fontsize=40, size=(800, 800))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$\ln(r)$",
            ylabel = L"$\ln(G(r))$",
            # xscale = log10,
            # yscale = log10
            )
            c = [1. , 1., 1., 1.]
    for (ib, β) in enumerate(βs)
        for (it, t) in enumerate(τs)
            ξ = ξs[ib,it]
            # ξ *= c[ib]
            rmax = round(Integer,ξs[ib,it])
            # rmax = round(Integer,2*ξs[ib,it])
            # rmax = 50
            if ξ>0
                # scatter!(ax1,log.(rs[1:rmax]),log.(cs[ib,it][1:rmax]./cs[ib,it][1]), label=L"$ξ=%$(round(ξ,digits=2))$", markersize = 16)
                scatter!(ax1,log.(rs[1:rmax]),log.(cs[ib,it][1:rmax]), label=L"$ξ=%$(round(ξ,digits=2))$", markersize = 20)
            end
        end
    end
    rmax = round(Integer,ξs[end,end])
    rs = rs[1:rmax]
    xmin = log(rs[13])
    ymin = log.(cs[end,end][13])
    lines!(ax1, log.(rs), -0.25*(log.(rs) .-log(rs[1])).+log.(cs[end,end][1]), color=:black, linestyle=:dash, linewidth=2, label=L"$r^{-0.25}$")
    lines!(ax1, log.(rs), -0.38*(log.(rs) .-xmin).+ymin, color=:black, linewidth=2, label=L"$r^{-0.38}$")
    axislegend(ax1, position=:lb)
    return fig, ax1
end

function plot_collapse(rs, cs, βs, ξs, τs, α; γ = 2-(1+sqrt(5))/2)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(1000, 800))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$\ln(r/ξ)$",
            ylabel = L"$\ln(G(r)⋅ξ^{%$(round(γ,digits=2))})$",
            # xscale = log10,
            # yscale = log10
            )
    for (ib, β) in enumerate(βs)
        for (it, t) in enumerate(τs)
            ξ = ξs[ib,it]
            # ξ = xis[ib]
            rmax = 2*floor(Integer,0.5*ξs[ib,it])
            rmin = 1
            # rmax = 30
            # rmax = length(rs)
            # rmax = 10
            if ξs[ib,it]>0
                # ys = cs[ib,it][rmin:rmax].*exp.(rs[rmin:rmax]/ξ)*ξ^γ.*rs[rmin:rmax].^α
                ys = cs[ib,it][rmin:rmax].*rs[rmin:rmax].^α.*ξ^γ
                # ys = cs[ib,it][rmin:rmax].*rs[rmin:rmax].^0.27.*exp.(rs[rmin:rmax]/ξ)
                # scatter!(ax1,log.(rs[rmin:rmax]/ξ),log.(ys), label=L"$ξ=%$(round(ξ,digits=2))$", markersize = 16)
                scatter!(ax1,log.(rs[rmin:rmax]/ξ),log.(ys), label=L"$ξ=%$(round(ξ,digits=2))$", markersize = 16)

            end
        end
    end
    rmin = 1
    rmax = floor(Integer,ξs[end,end])
    ξ = ξs[end,end]
    xs1 = rs[rmin:rmax]./ξ
    ys1 = xs1.^(-0.25).*exp.(-xs1)

    # lines!(ax1, log.(xs1), log.(ys1), color=:black, linewidth=2)

    return fig

end

function plot_deriv(rs, cs, βs, ξs, τs, α; γ = 2-(1+sqrt(5))/2)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=40, size=(1000, 1000))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$\ln(r)$",
            ylabel = L"$\ln(G(r))$",
            # xscale = log10,
            # yscale = log10
            )
    rmin = 1
    for (ib, β) in enumerate(βs)
        for (it, t) in enumerate(τs)
            ξ = ξs[ib,it]
            # ξ *= c[ib]
            # rmax = round(Integer,1.5*ξs[ib,it])
            # rmax = round(Integer,2*ξs[ib,it])
            rmax = 15
            if ξ>0
                xs = rs[rmin:rmax]
                xs = log.(xs)
                ys = cs[ib,it][rmin:rmax]#.*rs[rmin:rmax].^α.*ξ^γ#.*exp.(rs[1:rmax]/ξ)
                ys = log.(ys)
                dxs = diff(xs)
                dys = diff(ys)
                G = -dys./dxs
                scatter!(ax1,xs[rmin:rmax-1],G, label=L"$ξ=%$(round(ξ,digits=2))$", markersize = 20)
                # scatter!(ax1,xs[1:rmax],ys, label=L"$ξ=%$(round(ξ,digits=2))$", markersize = 20)
            end
        end
    end
    ξ = ξs[end,end]
    # axislegend(ax1, position=:lb)
    # lines!(ax1, log.(rs/ξ), 0.25.*ones(length(rs)), color=:black, linestyle=:dash, linewidth=2, label=L"$r^{-0.25}$")
    # lines!(ax1, log.(rs), 0.38.*ones(length(rs)), color=:black, linestyle=:dash, linewidth=2, label=L"$r^{-0.25}$")
    xs2 = rs[1:10]/ξ
    y2 = cs[end,end][1]

    ys2 = -0.25*(log.(xs2).-log(xs2[1])) .+ log.(y2)
    # lines!(ax1, log.(xs2), ys2, color=:black, linestyle=:dash, linewidth=2, label=L"$r^{-0.25}$")
    axislegend(ax1, position=:lt)
    return fig, ax1
end



a = 0.7
b = 1.3
N = 3
hs = Vector(a:(b-a)/(N-1):b)
Js = hs
ps = ones(N^2)./N^2
l = 1/var(log.(Js))
# N = 1fig, ax1 = plot_correlations(rs, csplot, cbetas, ξsplot, τs)
fig
# hs = [1.]
# Js = hs
# ps = [1.]

rs = 2:1:70
# βs = 10.0:2.0:18.0
βs = 20:10:50
Ds = [80]
τs = [0.05]
invtols = [1e-6]
truncfrequency = [1]

folder_name ="ρ_a$(a)_b$(b)_N$(N)"
# folder_name ="clean_ρ_N$(N)"

ξs, cs = get_ξs(folder_name,βs,Ds,τs, invtols, truncfrequency, rs, ps)
# cbetas = [16, 20]
cbetas = 1:length(βs)

ξsplot = ξs[cbetas,1,:,1,1]
csplot = cs[cbetas,1,:,1,1]

fig, ax1 = plot_correlations(rs, csplot, cbetas, ξsplot, τs)
fig
# fig2 = plot_collapse(rs, csplot, cbetas, ξsplot, τs)
fig2 = plot_collapse(rs, csplot, cbetas, ξsplot, τs, 0.0; γ = 0.29)
fig2
fig3, ax3 = plot_deriv(rs, csplot, cbetas, ξsplot, τs,0; γ=0.)
fig3
save("G1.pdf",fig)