using TensorKit, KrylovKit,MPSKit
using DelimitedFiles, LinearAlgebra, CairoMakie, LsqFit

using Revise, JLD2
using DisorderKit
using Crayons

function get_ξs(folder_name, βs, Ds, τs, invtols, freqs, ps)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)

    ξs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))
    δs = Array{Float64}(undef,length(βs),length(Ds), length(τs), length(invtols), length(freqs))

    fill!(ξs,0)
    fill!(δs,0)

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
                            ξ = average_correlation_length(ρ, ps)
                            ξs[ib,iD,it,ii,ifr] = ξ
                            ϵ = load_object("data/$folder_name1/ϵs.jld2")
                            @show "-----"
                            # ie = convert(Int, round(β/τ))+10
                            # @show ϵ[ie]
                            test_MPU3(ρ)

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

function diagonal_traceless_orthonormal_basis(n::Int)
    basis = []

    for k in 1:(n-1)
        d = zeros(Float64, n)
        d[1:k] .= 1.0
        d[k+1] = -k
        d ./= sqrt(k * (k + 1))
        push!(basis, DiagonalTensorMap(d, ℂ^n))
    end


    # for i in 1:(n-1)
    #     for j in 1:(n-1)
    #         @show tr(basis[i]*basis[j])
    #     end
    # end
    return basis
end

function traceless_basis(n::Int)
    basis = []

    # 1. Symmetric off-diagonal: (E_ij + E_ji)/√2
    for i in 1:n
        for j in (i+1):n
            M = zeros(Float64, n, n)
            M[i, j] = 1.0
            M[j, i] = 1.0
            push!(basis, TensorMap(M / sqrt(2), ℂ^n, ℂ^n))
        end
    end

    # 2. Skew-symmetric off-diagonal: (E_ij - E_ji)/√2
    for i in 1:n
        for j in (i+1):n
            M = zeros(Float64, n, n)
            M[i, j] = 1.0
            M[j, i] = -1.0
            push!(basis, TensorMap(M / sqrt(2), ℂ^n, ℂ^n))
        end
    end

    # 3. Traceless diagonal (orthonormal)
    for k in 1:(n-1)
        d = zeros(Float64, n)
        d[1:k] .= 1.0
        d[k+1] = -k
        d ./= sqrt(k * (k + 1))
        push!(basis, DiagonalTensorMap(d, ℂ^n))
    end

    # for i in 1:(n-1)
    #     for j in 1:(n-1)
    #         @show tr(basis[i]*basis[j])
    #     end
    # end
    return basis

end

function test_MPU3(ρ)
    Nspace = space(ρ[1], 3)
    diagel = ones(ComplexF64, dim(Nspace))
    diagel *= 1/dim(Nspace)
    P = DiagonalTensorMap(diagel, Nspace)
    @tensor E[-1; -2] := ρ[1][-1 3 1; 3 1 -2]
    E *= 1/dim(Nspace)
    vl = rand(ComplexF64, space(E, 1)')
    vr = rand(ComplexF64, space(E, 2)')
    vl = permute(vl, ((), (1, )))

    λEs, vls = eigsolve(x->x*E, vl, 4, :LM)
    λEs, vrs = eigsolve(x->E*x, vr, 4, :LM)
    l = vls[1]
    l2 = vls[2]
    r = vrs[1]
    r2 = vrs[2]
    r3 = vrs[3]
    @show λEs[1:3]
    Δ = λEs[1] - λEs[2]

    Q = id(ComplexF64, space(E, 1)) - r*l
    global data = []

    for i in 1:dim(Nspace)
        diagel = zeros(ComplexF64, dim(Nspace))
        diagel[i] = 1.0
        P = DiagonalTensorMap(diagel, Nspace)
        @tensor S[-1; -2] := ρ[1][-1 3 1; 3 2 -2] * P[2; 1]
        @show (l*(S-E)*r)[1]
        @show norm(Q*(S-E)*r)
    end
    @tensor Q[-1; -2] := ρ[1][1 3 -1; 3 -2 2] * l[1] * r[2]
    push!(data, reshape(Q.data, dim(codomain(Q)), dim(domain(Q))))
end

function test_MPU2(ρ)
    Nspace = space(ρ[1], 3)
    diagel = ones(ComplexF64, dim(Nspace))
    diagel *= 1/dim(Nspace)
    P = DiagonalTensorMap(diagel, Nspace)
    σs = diagonal_traceless_orthonormal_basis(dim(Nspace))
    @tensor E[-1; -2] := ρ[1][-1 3 1; 3 1 -2]
    E *= 1/dim(Nspace)
    vl = rand(ComplexF64, space(E, 1)')
    vr = rand(ComplexF64, space(E, 2)')
    vl = permute(vl, ((), (1, )))

    @show space(vl)
    λEs, vls = eigsolve(x->x*E, vl, 1, :LM)
    λEs, vrs = eigsolve(x->E*x, vr, 1, :LM)
    l = vls[1]
    r = vrs[1]
    @show λEs[1]
    @show norm(l*r)
    σs = diagonal_traceless_orthonormal_basis(dim(Nspace))
    # σs = traceless_basis(dim(Nspace))

    Id = id(ComplexF64, Nspace)/dim(Nspace)
    @tensor Z[-1 -2; -3 -4] := ρ[1][-1 5 -2; 5 -3 -4]
    @tensor Zp[-1 -2; -3 -4] := ρ[1][-1 5 2; 5 2 -4] * Id[-2; -3]
    # Zp *= 1/dim(Nspace)
    @tensor ρp[-1 -2 -3; -4 -5 -6] := ρ[1][-1 -2 2; -4 2 -6] * Id[-3; -5]
    # ρp *= 1/dim(Nspace)
    F = 0
    global data = []
    push!(data, reshape(E.data, dim(codomain(E)), dim(domain(E))))
    for σ in σs
        @tensor S[-1; -2] :=  ρ[1][-1 5 2; 5 4 -2] * σ[4; 2]
        # @show norm(l*S)
        # @show norm(S*r)
        # @show norm(l*S*r)
        @show dot(l, l*S)/(norm(l)*norm(l*S))
        @tensor Sp[-1 -2 -3; -4 -5 -6] := ρ[1][-1 -2 2; -4 4 -6] * σ[4; 2] * σ[-3; -5]
        @tensor ZSp[-1 -2; -3 -4] := ρ[1][-1 5 3; 5 4 -4] * σ[4; 3] * σ[-2; -3]

        projector = id(ComplexF64, space(Z, 4)') - norm(l*S*r)/(norm(l*r))*r*l
        @tensor Spp[-1 -2 -3; -4 -5 -6] := Sp[-1 -2 -3; -4 -5 1] * projector[1; -6]
        ρp += Spp
        @tensor Zspp[-1 -2; -3 -4] := ZSp[-1 -2; -3 1] * projector[1; -4]
        Zp += Zspp
        F += norm(l*S*r)^2
        push!(data, reshape(S.data, dim(codomain(S)), dim(domain(S))))
        @show sort(real.(eig_vals(S)))
    end
    X = zeros(ComplexF64, ℂ^9, ℂ^9)
    X[1, 9] = 1.
    X[9, 1] = 1.
    @tensor S[-1; -2] :=  ρ[1][-1 5 2; 5 4 -2] * X[4; 3] * P[3; 2]

    # @show norm(l*S*r)
    proj = norm(ρp - ρ[1])
    projZ = norm(Zp - Z)
    @show proj
    @show projZ

    N = dim(Nspace)
    trO = norm(l*E*r)
    # @show trO
    trOO = (norm(l*E*r)^2 + F)
    # @show trOO
    @show trO*trO/(trOO)

    function transfer_left_mpo(O)
        function ftransfer(vl)
            @tensor vl[-1; -2] := O[2 4; 3 -2] * conj(O[1 4; 3 -1]) * vl[1; 2]
        end
        return ftransfer
    end

    function transfer_right_mpo(O)
        function ftransfer(vr)
            @tensor vr[-1; -2] := O[-1 4; 3 1] * conj(O[-2 4; 3 2]) * vr[1; 2]
            return vr
        end
        return ftransfer
    end


    # Entanglement spectrum of MPO
    function entanglement_spectrum(Os::InfiniteMPO, i::Int)
        unit_cell = length(Os)
        transfer_l = transfer_left_mpo(Os[i+1])
        transfer_r = transfer_right_mpo(Os[i])
        for j = i+2:i+unit_cell
            transfer_l = transfer_left_mpo(Os[j]) ∘ transfer_l
        end
        for j = i-1:-1:i-unit_cell+1
            transfer_r = transfer_right_mpo(Os[j]) ∘ transfer_r
        end

        Dl = space(Os[i+1], 1)
        Dr = space(Os[i+1], 1)

        ρl0 = rand(ComplexF64, Dl, Dl)
        ρr0 = rand(ComplexF64, Dr, Dr)

        _, ρls, infol = eigsolve(transfer_l, ρl0, 1, :LM)
        _, ρrs, infor = eigsolve(transfer_r, ρr0, 1, :LM)

        S = svd_vals((ρls[1] * ρrs[1]))
        es = S.data
        es /= sum(es)
        return es
    end

    esp =  entanglement_spectrum(InfiniteMPO([Zp]), 1)
    es = entanglement_spectrum(InfiniteMPO([Z]), 1)
    @show es[1:3]
    @show esp[1:3]

    ϵ = 0
    @show norm(r)
    @show tr(E)
    for σ in σs
        @show σ.data
        @tensor S[-1; -2] :=  ρ[1][-1 5 2; 5 4 -2] * σ[4; 2]

        ϵ += norm(l*S*r)
        @show norm(S*r/norm(S*r) - r)
        @tensor LL0[-1; -2] := ρ[1][1 2 -1; 2 -2 3] * l[1] *r[3]
        @tensor SS[-1; -2] := ρ[1][1 2 -1; 2 -2 3] * l[1] *S[3;4] * r[4]
        global LLdata = reshape(LL0.data, dim(space(LL0, 1)), dim(space(LL0, 2)))
        global SSdata = reshape(SS.data, dim(space(SS, 1)), dim(space(SS, 2)))
        global Sdata = reshape(S.data, dim(space(S, 1)), dim(space(S, 2)))
        global ldata = l.data
        global rdata = r.data
    end
    @show ϵ

        
    @tensor ET[-1 -2; -3 -4] := ρ[1][-1 1 2; 1 4 -3]*conj(ρ[1][-2 3 2;3 4 -4])
    ET *= 1/dim(Nspace)
    vl = rand(ComplexF64, codomain(ET))
    vr = rand(ComplexF64, domain(ET))
    vl = permute(vl, ((), (1, 2)))

    @show space(vl)
    # λETs, vls = eigsolve(x->x*ET, vl, 1, :LM)
    λETs, vrs = eigsolve(x->ET*x, vr, 3, :LM)
    @show λEs[1:3]
    @show λETs[1:3]
    rt = vrs[1]
    @tensor rtt[-1 -2] := r[-1] * conj(r[-2])
    @show norm(rt'*rtt)/(norm(rt)*norm(rtt))

    @show svd_vals(E)
end

function test_MPU(ρ)
    ρperm = permute(ρ[1], ((1, 2, 4), (3, 5, 6)))
    Nspace = space(ρ[1], 3)
    diagel = ones(ComplexF64, dim(Nspace))
    diagel *= 1/dim(Nspace)
    P = DiagonalTensorMap(diagel, Nspace)
    Q, R = qr_compact(ρperm)
    U, S, V = svd_trunc(ρperm; trunc = (maxerror = 1e-10,))
    Q = permute(Q, ((1, 2),(3, 4)))
    R = permute(R, ((1, 2),(3, 4)))
    @tensor EQ[-1 -2; -3 -4] := Q[-2 1; 2 -4] * conj(Q[-1 1; 2 -3])
    @tensor ER[-1 -2; -3 -4] := R[-2 4; 2 -4] * conj(R[-1 1; 2 -3]) * P[1; 4]

    vl = rand(ComplexF64, space(Q, 1)⊗space(Q, 1)')
    vr = rand(ComplexF64, space(R, 4)⊗space(R, 4)')
    vl = permute(vl, ((), (1, 2)))

    λEs, vls = eigsolve(x->x*EQ*ER, vl, 1, :LM)
    l = vls[1]
    EQ *= 1/λEs[1] 
    λEs, vrs = eigsolve(x->EQ*ER*x, vr, 1, :LM)
    r = vrs[1]
    @show λEs[1]
    @show norm(l*r)


    Id = id(ComplexF64, Nspace)
    @tensor Rp[-1 -2; -3 -4] := Q[-1 3; 3 4] * R[1; 2 -4] * P[2; 1] * Id[-2; -3] 
    σs = diagonal_traceless_orthonormal_basis(dim(Nspace))
    nullspaceproj = id(ComplexF64, space(R,4)') - r*r'
    @show norm(nullspaceproj*r)
    for σ in σs
        @tensor S[-1; -2] :=  Q[-1 1;1 2] * R[2 4; 3 -2] * σ[5; 4] * P[3; 5]
        @show norm(l*S)
        @show norm(S*r)
        @show norm(l*S*r)
        @show tr(σ^2)
        @tensor RSp[-1 -2; -3 -4] := S * σ[-2; -3]
        Rp += RSp
    end

    @show "Try projection"
    @tensor ρp[-1 -2 -3; -4 -5 -6] := Q[-1 -2; -4 1] * Rp[1 -3; -5 -6]
    if norm(ρp - ρ[1]) > 1e-10
        @show(crayon"magenta"("Projection failed"))
        @show norm(ρp - ρ[1])
    else
        @show(crayon"green"("Projection successful"))
    end
    @show "-------"
end

function plot_ξ_β(βs,Ds,Dts,ξs)
    set_theme!(theme_latexfonts())
    fig = Figure(backgroundcolor=:white, fontsize=30, size=(600, 600))
    ax1 = Axis(fig[1, 1], 
            xlabel = L"(\ln{β})^2",
            ylabel = L"$ξ$",
            # xscale = log10,
            # yscale = log10
            )
    minfit = 4
    for (iD, Dmax) in enumerate(Ds)
        for (it, t) in enumerate(Dts)
            nzxi = findall(x -> x > 0, ξs[:,iD,it])
            nzxi = nzxi[minfit:end]
            scatter!(ax1,log.(βs[nzxi]).^2,(ξs[:,iD,it][nzxi]), label=L"$D=%$Dmax$, $Δτ=%$t$",markersize = 20)
            # scatter!(ax1,βs[nzxi],(ξs[:,iD,it][nzxi]), label=L"$D=%$Dmax$, $Δτ=%$t$",markersize = 16)
        end
    end
    minplot = 4
    minfit = 30
    maxfit = 50
    p0q = [1.,1.]
    linmodel(t,p) = p[1].+p[2]*t
    linfit = curve_fit(linmodel, log.(βs[minfit:maxfit]).^2, ξs[minfit:maxfit,end,end], p0q)
    @show linfit.param
    lines!(ax1,log.(βs[minplot:end]).^2,linmodel(log.(βs[minplot:end]).^2,linfit.param), color=:black, linewidth=2)

    # maxfit = 50
    # p0q = [1.,1.,1.]
    # quadmodel(t,p) = p[1].+p[2]*t .+p[3]*t.^2
    # quadfit = curve_fit(quadmodel, log.(βs[minfit:maxfit]), ξs[minfit:maxfit,end,1], p0q)
    # @show quadfit.param
    # lines!(ax1,log.(βs[minfit:end]).^2,quadmodel(log.(βs[minfit:end]),quadfit.param), color=:black, linewidth=2)
    # p0q = [1.,1.]
    # logmodel(t,p) = p[1].*log.(t./p[2]).^2
    # logfit = curve_fit(logmodel, βs[minfit:maxfit], ξs[minfit:maxfit,end,1], p0q)
    # @show logfit.param
    # lines!(ax1,log.(βs).^2,logmodel(βs,logfit.param), color=:black, linewidth=2)
    axislegend(ax1, position=:lt)
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

# βs = 1:10.0:50
# Ds = [10 , 20, 40, 80, 100]
βs = [40]
# Ds = [40, 80, 100]
Ds = [80]
τs = [0.05]
# τs = [0.01, 0.05, 0.1]
invtols = [1e-6]
truncfrequency = [1]

folder_name ="ρ_a$(a)_b$(b)_N$(N)"

ξs = get_ξs(folder_name,βs,Ds,τs, invtols, truncfrequency, ps)

ξsplot = ξs[:,:,:,1,1]

fig1, ax1 = plot_ξ_β(βs, Ds, τs, ξsplot)
fig1

# save("xi_beta.pdf",fig1)
