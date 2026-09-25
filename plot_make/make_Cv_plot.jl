using TensorKit
using DelimitedFiles, LinearAlgebra, CairoMakie, LsqFit

using Revise, JLD2
using DisorderKit

function local_E(ρ::DisorderMPO,ps::Vector{Float64}, Js::Vector{Float64}, hs::Vector{Float64}, i::Int)
    X, Z, Id = TensorMap(zeros(ComplexF64, 2, 2),ℂ^2,ℂ^2), TensorMap(zeros(ComplexF64, 2, 2),ℂ^2,ℂ^2), TensorMap(zeros(ComplexF64, 2, 2),ℂ^2,ℂ^2)
    X[1, 2], X[2, 1] = 1, 1
    Z[1, 1], Z[2, 2] = 1, -1
    Id[1, 1], Id[2, 2] = 1, 1
    Etotal = 0
    for (j, (h, J)) in enumerate(Iterators.product(hs, Js)) 
        Eh = -measure(ρ, ps, X, i)*h
        EJ = -measure(ρ, ps, Z, Z, i, 1)*J/2-measure(ρ, ps, Z, Z, i-1, 1)*J/2
        Etotal += Eh + EJ
    end
    return Etotal
end

function RTFIM_hamiltonian2(Js::Vector{Float64}, hs::Vector{Float64})
    X, Z, Id = zeros(ComplexF64, 2, 2), zeros(ComplexF64, 2, 2), zeros(ComplexF64, 2, 2)
    X[1, 2], X[2, 1] = 1, 1
    Z[1, 1], Z[2, 2] = 1, -1
    Id[1, 1], Id[2, 2] = 1, 1
    D_disorder = length(Js) * length(hs)
    Hs = zeros(ComplexF64, 3, 2, D_disorder, 2, D_disorder, 3)
    # Hs = zeros(ComplexF64,ℂ^3⊗ℂ^2⊗ℂ^D_disorder,ℂ^2⊗ℂ^D_disorder⊗ℂ^3)
    for (i, (h, J)) in enumerate(Iterators.product(hs, Js))
        # H = transverse_field_ising(; J = J, g = h/J)
        # U = convert(TensorMap,H[1])
        # disordermap = DiagonalTensorMap(zeros(ComplexF64, D_disorder),ℂ^D_disorder)
        # disordermap[i,i] = 1.0
        # @tensor U_full[-1 -2 -3; -4 -5 -6] := U[-1 -2; -4 -6]*disordermap[-3; -5]
        # Hs += U_full
        # @show i, hh/(2(s+x))*exp(-a), J
        Hs[1,:,i, :, i, 1] = Id
        Hs[2,:,i, :, i, 1] = zeros(ComplexF64, 2, 2)
        Hs[3,:,i, :, i, 1] = zeros(ComplexF64, 2, 2)
        Hs[1,:,i, :, i, 2] = -J*Z
        Hs[2,:,i, :, i, 2] = zeros(ComplexF64, 2, 2)
        Hs[3,:,i, :, i, 2] = zeros(ComplexF64, 2, 2)
        Hs[1,:,i, :, i, 3] = -h*X
        Hs[2,:,i, :, i, 3] = Z
        Hs[3,:,i, :, i, 3] = Id
    end 

    Hs = TensorMap(Hs, ℂ^3⊗ℂ^2⊗ℂ^D_disorder,ℂ^2⊗ℂ^D_disorder⊗ℂ^3)

    return DisorderMPO([Hs])
end

function get_ξs(folder_name, βs, Ds, τs, invtols, freqs, ps, Hs)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)

    ξs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))
    Es = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))
    CVs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))
    Esmpo = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))

    fill!(ξs,0)
    fill!(Es,0)
    fill!(CVs,0)

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
                            E = real.(local_E(ρ, ps, Js, hs, 1))
                            Es[ib,iD,it,ii,ifr] = E
                            Empo = measure(ρ, ps, Hs, 1)
                            Esmpo[ib,iD,it,ii,ifr] = real.(Empo)
                            # CVs[ib,iD,it,ii,ifr] = -real(β^2 * diff(E)/1)
                        else
                            @warn("data/$folder_name1/ρ_β$β does not exist")
                        end
                    end
                end
            end
        end
    end
    return ξs, Es, Esmpo
end

function plot_E_β(βs,Ds,Dts,Es)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(600, 600))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"β",
            ylabel = L"$<E>$",
            # xscale = log10,
            # yscale = log10
            )
    for (iD, Dmax) in enumerate(Ds)
        for (it, t) in enumerate(Dts)
            nzxi = findall(x -> x < 0, Es[:,iD,it])
            scatter!(ax1,βs[nzxi],(Es[:,iD,it][nzxi]), label=L"$D=%$Dmax$, $Δτ=%$t$",markersize = 16)
            # scatter!(ax1,βs[nzxi],(ξs[:,iD,it][nzxi]), label=L"$D=%$Dmax$, $Δτ=%$t$",markersize = 16)
        end
    end
    axislegend(ax1, position=:rt)
    return fig, ax1
end

function plot_Cv_β(βs,Ds,Dts,Es)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(600, 600))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"(\ln{β})",
            ylabel = L"$C_v$",
            # xscale = log10,
            # yscale = log10
            )
    ys = []
    for (iD, Dmax) in enumerate(Ds)
        for (it, t) in enumerate(Dts)
            nzxi = findall(x -> x < 0, Es[:,iD,it])
            Cvs = -real.(βs[nzxi[1:end-1]].^2 .*diff(Es[:,iD,it][nzxi])./diff(βs[nzxi]))
            ys = Cvs
            scatter!(ax1, (log.(βs[nzxi[1:end-1]])),Cvs, label=L"$D=%$Dmax$, $Δτ=%$t$",markersize = 16)
            # scatter!(ax1,βs[nzxi],(ξs[:,iD,it][nzxi]), label=L"$D=%$Dmax$, $Δτ=%$t$",markersize = 16)
        end
    end
    xs = log.(βs[1:end-1])
    minfit = 5
    maxfit = 20
    p0 = [1.,1., 1.]
    invmodel(t,p) = p[1].+p[2] ./(t .+p[3]).^3
    invfit = curve_fit(invmodel, xs, ys, p0)
    @show invfit.param
    lines!(ax1,log.(βs),invmodel(log.(βs),invfit.param), color=:black, linewidth=2)
    # lines!(ax1, xs, ys, color=:black, linewidth=2)
    axislegend(ax1, position=:rt)
    return fig, ax1
end


N = 3
a = 0.7
b = 1.3
hs = Vector(a:(b-a)/(N-1):b)
Js = hs
ps = ones(N^2)./N^2
Hs = RTFIM_hamiltonian2(Js, hs)

βs = 10:1.0:20
Ds = [80]
τs = [0.05]
invtols = [1e-6]
truncfrequency = [1]

folder_name ="ρ_a$(a)_b$(b)_N$(N)"

ξs, Es, Esmpo = get_ξs(folder_name,βs,Ds,τs, invtols, truncfrequency, ps, Hs)

ξsplot = ξs[:,:,:,1,1]
Esplot = Es[:,:,:,1,1]

fig1, ax1 = plot_E_β(βs, Ds, τs, Esplot)
fig2, ax2 = plot_Cv_β(βs, Ds, τs, Esplot)
fig2