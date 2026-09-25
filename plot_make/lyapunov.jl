using TensorKit, KrylovKit
using DelimitedFiles, LinearAlgebra, CairoMakie, LsqFit

using Revise, JLD2
using DisorderKit, Statistics, ProgressBars

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
    Nsamples = 100

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

    nsamples = 10
    L = 10
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
                            writedlm("data/typical/ξ_$(β)_$(L)_$nsamples.txt", ξss, ';')
                            writedlm("data/typical/λ2_$(β)_$(L)_$nsamples.txt", λfs2, ';')
                            writedlm("data/typical/λ1_$(β)_$(L)_$nsamples.txt", λfs1, ';')

                            
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
bs = [1.3,1.4, 1.5, 1.7]
# bs = [1.3, 1.4]
Js = Vector(a:(1.3-a)/(N-1):1.3)
ps = ones(N^2)./N^2

βs = [20]
iβ = length(βs)
Ds = [80]
τs = [0.05]
invtols = [1e-6]
truncfrequency = [1]
δs = Float64[]

function get_data()
    ξsss = Array{Vector{Float64}}(undef, length(bs))
    ξsav = Array{Vector{Float64}}(undef, length(bs))
    for (ib,b) in enumerate(bs)
        hs = Vector(a:(b-a)/(N-1):b)
        VJ = var(Js; corrected = false)
        Vh = var(hs; corrected = false)
        mh = mean(hs)
        mJ = mean(Js)
        push!(δs, (mh-mJ)/(Vh+VJ))


        folder_name ="ρ_a$(a)_b$(b)_N$(N)"

        ξs, λs1, λs2, λfs1, λfs2, ξss, ξsavv = get_ξs(folder_name,βs,Ds,τs, invtols, truncfrequency, ps)

        ξsss[ib] = ξs[:,1,1,1,1]
        ξsav[ib] = ξsavv[:,1,1,1,1]
    end
    return ξsss , δs, ξsav
end

ξss, δs, ξsav = get_data()
fig2, ax2 = plot_ξs(δs, βs, ξss, ξsav)
fig2

# ξss = filter(x -> x < 100, ξss)
# hist(ξss, bins = 10, bar_labels = :values,
#      label_formatter=x-> round(x, digits=2), label_size = 15,
#      strokewidth = 0.5, strokecolor = (:black, 0.5), color = :values)

#  set_theme!(theme_latexfonts())
#     fig = Figure(backgroundcolor=:white, fontsize=40, size=(1000, 1000))
#     ax1 = Axis(fig[1, 1], 
#             xlabel = L"n",
#             ylabel = L"$\frac{1}{n}log|v_n|$",
#             # xscale = log10,
#             # yscale = log10
#             )
#     minfit = 1
#     scatter!(ax1,1:length(λs1),λs1, label = "v_1", markersize = 20)
#     scatter!(ax1,1:length(λs2),λs2, label = "v_2", markersize = 20)
#     axislegend(ax1, position=:lb)
#     fig