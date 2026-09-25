using TensorKit, KrylovKit
using DelimitedFiles, LinearAlgebra, CairoMakie, LsqFit

using Revise, JLD2
using DisorderKit, Statistics, ProgressBars, StatsBase, HypothesisTests, Distributions

function typical_correlation_length(ρ, ps,L, nsamples)
    D_disorder = length(ps)

    ξs = Float64[]
    for _ in ProgressBar(1:nsamples)
        As = id(ComplexF64, space(ρ.opp[1],1))
        for _ in 1:L
            sample = rand(1:D_disorder, 1)
            W = zeros(ComplexF64, ℂ^(D_disorder))
            W[sample[1]] = 1.0
            
            @tensor Ap[-1; -2] := ρ.opp[1][-1 4 1;4 2 -2] * conj(W[1]) *W[2]

            As = Ap*As
        end
        v0 = rand(ComplexF64,space(ρ.opp[1],1))
        λs , vrs = eigsolve(x -> As*x, v0, 2, :LM)
        ξ = real(L/log(λs[1]/λs[2]))
        ξs = push!(ξs, ξ)
    end
    ξ_typ = median(ξs)

    return ξs
end

function lyapunovexp(ρ)
    D_disorder = length(ps)

    L = 100
    Nsamples = 20

    λfs1 = Float64[]
    λfs2 = Float64[]


    λs1 = Float64[]
    λs2 = Float64[]
    for _ in ProgressBar(1:Nsamples)

        λs1 = Float64[]
        λs2 = Float64[]

        u1 = rand(ComplexF64,space(ρ.opp[1],1))
        u2 = rand(ComplexF64,space(ρ.opp[1],1))

        for n in 1:L
            sample = rand(1:D_disorder, 1)
            W = zeros(ComplexF64, ℂ^(D_disorder))
            W[sample[1]] = 1.0
            
            @tensor Ap[-1; -2] := ρ.opp[1][-1 4 1;4 2 -2] * conj(W[1]) *W[2]

            u1 = Ap*u1
            u2 = Ap*u2
            push!(λs1, 1/n*log(TensorKit.norm(u1)))
            push!(λs2, 1/n*log(TensorKit.norm(u2)))
            if n==L
                push!(λfs1, 1/n*log(TensorKit.norm(u1)))
                push!(λfs2, 1/n*log(TensorKit.norm(u2)))
            end
            u2 = u2 - u1*u1'*u2
        end
    end
    ξs = 1 ./(λfs1 .- λfs2)
    filter!(x -> x<0, λfs2)
    filter!(x -> x>0, ξs)

    @show length(ξs)

    # @show λfs2
    ξt = median(ξs)

    return λs1, λs2, λfs1, λfs2, ξs, ξt
end


function get_ξs(folder_name, βs, Ds, τs, invtols, freqs, ps)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)

    ξs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))
    ξsav = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))

    δs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))

    nsamples = 20000
    L = 100
    fill!(ξs,0)
    fill!(δs,0)
    λs1 = []
    λs2 = []
    λfs1 = []
    λfs2 = []
    ξss = []


    for (ib, β) in enumerate(βs)
        for (iD, Dmax) in enumerate(Ds)
            for (it, τ) in enumerate(τs)
                for (ii, invtol) in enumerate(invtols)
                    for (ifr, freq) in enumerate(freqs)
                        # folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_truncfreq$(freq)"
                        # if Dmax == 80 || Dmax == 100
                        folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_Z4"
                        # else
                        #     folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)"
                        # end

                        isdir("data/$folder_name1") || @error("Folder $folder_name1 does not exist.")

                        @show Dmax, β, τ, invtol

                        β = convert(Int, round(β))
                        if isfile("data/$folder_name1/ρ_β$β.jld2")
                            ρ = load_object("data/$folder_name1/ρ_β$β.jld2")
                            ρ = DisorderKit.fix_phase(ρ,ps)
                            # ξ = typical_correlation_length(ρ, ps, L, nsamples)
                            # writedlm("data/typical/ξ_$(β)_$(L)_$nsamples.txt", ξ, ';')
                            λs1, λs2, λfs1, λfs2, ξss, ξt = lyapunovexp(ρ)
                            # writedlm("data/typical/ξ_$(β)_$(L)_$(nsamples).txt", ξss, ';')
                            # writedlm("data/typical/λ2_$(β)_$(L)_$(nsamples).txt", λfs2, ';')
                            # writedlm("data/typical/λ1_$(β)_$(L)_$(nsamples).txt", λfs1, ';')

                            ξs[ib,iD,it,ii,ifr] = ξt
                            ξ = average_correlation_length(ρ, ps)
                            ξsav[ib,iD,it,ii,ifr] = ξ
                        else
                            @warn("data/$folder_name1/ρ_β$β does not exist")
                        end
                    end
                end
            end
        end
    end
    return ξs, λs1, λs2, λfs1, λfs2, ξss, ξsav
end


function plot_ξs(bs, βs, ξs, ξsav)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(800, 600))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"$δ$",
            ylabel = L"$ξ/(\ln(\beta)^2)$",
            # ylabel = L"$ξ/(A \cdot \ln(\beta)^2+ B\cdot \ln(\beta)+C)$",
            # xscale = log10,
            # yscale = log10
            )

    colors = [Makie.wong_colors()...,:red, :blue, :green, :orange, :purple, :brown, :pink, :cyan, :magenta, :yellow, :olive, :teal, :navy, :maroon]

    for (ib, β) in enumerate(βs)
        label = "β = $(β)"
        ξsnew = hcat(ξs...)
        # refξ = A*log.(β).^2 .+ B*log.(β) .+ C
        # refξ = log.(β).^2
        refξ = 1
        scatterlines!(bs, ξsnew[ib,:]/refξ, color = colors[ib],  markersize = 16, label=label)
        ξsnewa = hcat(ξsav...)
        # refξ = A*log.(β).^2 .+ B*log.(β) .+ C
        # refξ = log.(β).^2
        refξ = 1
        scatterlines!(ax1, (bs), (ξsnewa[ib,:]/refξ), marker = :cross, color = colors[ib], markersize = 16)
        # scatterlines!(ax1, (bs), (ξsnew[ib,:]./ξsnewa[ib,:]), marker = :cross, color = colors[ib], markersize = 16)

    end
    # fig[1, 2] = Legend(fig, ax1, framevisible = false)
    # fig[2, 2] = Legend(fig, ax2, framevisible = false)
    return fig, ax1
end



N = 3
a = 0.7
b = 1.3
# a = 0.5
# b = 1.5
hs = Vector(a:(b-a)/(N-1):b)
Js = hs
ps = ones(N^2)./N^2

βs = [40]
iβ = length(βs)
Ds = [80]
τs = [0.05]
invtols = [1e-6]
truncfrequency = [1]
δs = Float64[]

folder_name ="ρ_a$(a)_b$(b)_N$(N)"

# ξs, λs1, λs2, λfs1, λfs2, ξss, ξsavv = get_ξs(folder_name,βs,Ds,τs, invtols, truncfrequency, ps)

ξss = [readdlm("data/typical/ξ_40_100_20000.txt", ';')...]
# ξss = [readdlm("data/typical/λ1_40_100_20000.txt", ';')...]
# ξss = [readdlm("data/typical/λ2_40_100_20000.txt", ';')...]
@show mean(ξss)
@show median(ξss)
# @show geomean(ξss)
@show harmmean(ξss)

ξssf = filter(x -> x < 500, ξss)
# ξssf = ξss

@show mean(ξssf)
@show median(ξssf)
# @show geomean(ξssf)
@show harmmean(ξssf)

 set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=40, size=(600, 600))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"ξ",
            ylabel = L"$P(ξ)$",
            # xscale = log10,
            # yscale = log10
            )
    #     fig = Figure(backgroundcolor=:white, fontsize=40, size=(600, 600))
    # ax1 = Axis(fig[1, 1], 
    #         xlabel = L"α_2",
    #         ylabel = L"$P(α_2)$",
    #         # xscale = log10,
    #         # yscale = log10
    #         )
hist!(ax1, ξssf, bins = 20, normalization = :pdf,
     strokewidth = 0.5, strokecolor = (:black, 0.5), color = :values)

fig

# save("l2_dist.pdf",fig)

 set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=40, size=(1000, 1000))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"n",
            ylabel = L"$\frac{1}{n}log|v_n|$",
            # xscale = log10,
            # yscale = log10
            )
#     minfit = 1
#     scatter!(ax1,1:length(λs1),λs1, label = "v_1", markersize = 20)
#     scatter!(ax1,1:length(λs2),λs2, label = "v_2", markersize = 20)
#     axislegend(ax1, position=:lb)
#     fig
# xs = 1 ./(ξssf)
# μ = mean(xs)
# σ = std(xs)
# α = 3.12
# θ = 0.0107
# d = Normal(0, 1)
# d = Gamma(α, θ)

# ApproximateOneSampleKSTest(xs, d)

cs = ecdf(ξssf)
xs = 1:198
Fs = 1. .-cs.(xs)

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=30, size=(700, 600))
ax1 = Axis(fig[1, 1], 
        xlabel = L"x",
        ylabel = L"$P(ξ>x)$",
        xscale = log10,
        yscale = log10
        )
minplot = 10
minfit = 60
maxfit = 120
p0q = [1.,1.]
linmodel(t,p) = p[1].+p[2]*t
linfit = curve_fit(linmodel, log.(xs[minfit:maxfit]), log.(Fs[minfit:maxfit]), p0q)
@show linfit.param
α = round(-linfit.param[2],digits=3)
scatterlines!(ax1, xs[minplot:end], Fs[minplot:end], markersize = 16)
lines!(ax1,(xs[minfit-40:end]),exp.(linmodel(log.(xs[minfit-40:end]),linfit.param)), color=:black, linewidth=2, label=L"$x^{-$%$α$}$")
axislegend(ax1, position=:lb, labelsize=60)
fig 

save("tail.pdf",fig)

# Zs = (xs .- μ)/σ
# ExactOneSampleKSTest(Zs, d)