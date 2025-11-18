# Algortihm for computing the groundstate of a disordered system
struct  StiefelOptim <: AbstractAlgorithm
    sweeps::Int
    gradtol::Float64
    verbosity::Int

    function StiefelOptim(sweeps::Int, gradtol::Float64, verbosity::Int)
        return new(sweeps, gradtol, verbosity)
    end
end

function disorder_retract(ρ::FiniteDisorderMPS, g::Vector{<:Vector{Union{Stiefel.StiefelTangent, Unitary.UnitaryTangent}}}, α::Real)
    opp = Vector{Vector{AbstractMPSTensor}}(undef, length(ρ))
    tangents = Vector{Vector{Union{Stiefel.StiefelTangent, Unitary.UnitaryTangent}}}(undef, length(ρ))
    for i in eachindex(ρ)
        As = Vector{AbstractMPSTensor}(undef, length(ρ[i]))
        ξs = Vector{Union{Stiefel.StiefelTangent, Unitary.UnitaryTangent}}(undef, length(ρ[i]))
        for p in eachindex(ρ[i])
            Dcod = dim(codomain(ρ[i][p]))
            Ddom = dim(domain(ρ[i][p]))
            if Dcod > Ddom
                As[p], ξs[p] = Stiefel.retract(ρ[i][p], g[i][p], α)
            else
                As[p], ξs[p] = Unitary.retract(ρ[i][p], g[i][p], α)
            end
        end
        opp[i] = As
        tangents[i] = ξs
    end
    return FiniteDisorderMPS(opp), tangents
end

function disorder_inner(ρ::FiniteDisorderMPS, g₁::Vector{<:Vector{Union{Stiefel.StiefelTangent, Unitary.UnitaryTangent}}}, g₂::Vector{<:Vector{Union{Stiefel.StiefelTangent, Unitary.UnitaryTangent}}})
    s = 0.0
    for i in eachindex(g₁)
        for j in eachindex(g₂)
            for p in eachindex(g₁[i])
                for q in eachindex(g₂[j])
                    Dcod1 = dim(codomain(g₁[i][p].W))
                    Ddom1 = dim(domain(g₁[i][p].W))
                    Dcod2 = dim(codomain(g₂[j][q].W))
                    Ddom2 = dim(domain(g₂[j][q].W))
                    if Dcod1 > Ddom1
                        X1 = g₁[i][p].W*g₁[i][p].A+ g₁[i][p].Z
                    else
                        X1 = g₁[i][p].W*g₁[i][p].A
                    end
                    if Dcod2 > Ddom2
                        X2 = g₂[j][q].W*g₂[j][q].A+ g₂[j][q].Z
                    else
                        X2 = g₂[j][q].W*g₂[j][q].A
                    end
                    if i<j
                        vr = right_env(ρ, j)
                        vl = TensorMap(zeros, ComplexF64, space(ρ[i][1],3)', space(X1,3)')  
                        @tensor stemp[-2; -1] := X1[1 2; -1]*conj(ρ[i][p][1 2; -2])
                        D = length(g₁[i])
                        vl += stemp/D
                        for k in i+1:j-1
                            f_t = transfer_left(ρ[k])
                            vl = f_t(vl)
                        end
                        vltemp = TensorMap(zeros, ComplexF64, space(X2,3)', space(ρ[j][q],3)')
                        @tensor stemp[-2; -1] := vl[3;4]*conj(X2[3 2; -2])*ρ[j][q][4 2; -1]
                        D = length(g₂[j])
                        vltemp += stemp/D
                        @tensor stemp = vltemp[ 2; 1]*vr[1; 2]
                        s += real.(stemp)
                    elseif i==j && p==q
                        vr = right_env(ρ, j)
                        vl = TensorMap(zeros, ComplexF64, space(X2,3)', space(X1,3)')
                        @tensor stemp[-2; -1] := X1[1 2; -1]*conj(X2[1 2; -2])
                        D = length(g₂[j])
                        vl += stemp/D
                        @tensor stemp = vr[1; 2]*vl[2; 1]
                        s += real.(stemp)
                    elseif i>j
                        vl = TensorMap(zeros, ComplexF64, space(X2,3)', space(ρ[j][q],3)')
                        vr = right_env(ρ, i)
                        @tensor stemp[-2; -1] := conj(X2[1 2; -2])*ρ[j][q][1 2; -1]
                        D = length(g₂[j])
                        vl += stemp/D
                        for k in j+1:1:i-1
                            f_t = transfer_left(ρ[k])
                            vl = f_t(vl)
                        end
                        vltemp = TensorMap(zeros, ComplexF64, space(ρ[i][p],3)', space(X1,3)')
                        @tensor stemp[-2; -1] := vl[3;4]*conj(ρ[i][p][3 2; -2])*X1[4 2; -1]
                        D = length(g₁[i])
                        vltemp += stemp/D   
                        @tensor stemp = vr[1; 2]*vltemp[2; 1]
                        s += real.(stemp)
                    end
                end
            end
        end
    end
    return s
end

function disorder_scale!(g::Vector{<:Vector{Union{Stiefel.StiefelTangent, Unitary.UnitaryTangent}}}, α::Real)
    for i in eachindex(g)
        for p in eachindex(g[i])
            if g[i][p] isa Stiefel.StiefelTangent
                Stiefel.rmul!(g[i][p], α)
            else
                Unitary.rmul!(g[i][p], α)
            end
        end
    end
    return g
end

function disorder_project!(X::Vector{<:Vector{<:AbstractMPSTensor}},ρ::FiniteDisorderMPS)
    opp = Vector{Vector{Union{Stiefel.StiefelTangent, Unitary.UnitaryTangent}}}(undef, length(ρ))
    for i in eachindex(ρ)
        As = Vector{Union{Stiefel.StiefelTangent, Unitary.UnitaryTangent}}(undef, length(ρ[i]))
        for p in eachindex(ρ[i])
            Dcod = dim(codomain(ρ[i][p]))
            Ddom = dim(domain(ρ[i][p]))
            if Dcod > Ddom
                As[p] = Stiefel.project!(X[i][p], ρ[i][p])
            else
               As[p] = Unitary.project!(X[i][p], ρ[i][p])
            end
        end
        opp[i] = As
    end
    return opp
end

function disorder_precond(ρ::FiniteDisorderMPS, g::Vector{<:Vector{Union{Stiefel.StiefelTangent, Unitary.UnitaryTangent}}})
    opp = Vector{Vector{Union{Stiefel.StiefelTangent, Unitary.UnitaryTangent}}}(undef, length(ρ))
    L = length(g)
    for i in eachindex(ρ)
        As = Vector{Union{Stiefel.StiefelTangent, Unitary.UnitaryTangent}}(undef, length(ρ[i]))
        vr = id(space(ρ[L][1],3)')
        for j in L:-1:i+1
            f_t = transfer_right(ρ[j])
            vr = f_t(vr)
        end
        U, S, V = tsvd(vr,trunc=truncerr(1e-4))
        # @show S
        Id = id(codomain(S))
        # @show S
        for p in eachindex(ρ[i])
            δ = norm(g[i][p])
            # δ = 0
            Sinv = sqrt(inv(S^2 + δ*Id))
            vrinv = V' * Sinv * U'
            # vrinv = inv(vr+δ*Id)
            # @show vrinv*vr

            Dcod = dim(codomain(ρ[i][p]))
            Ddom = dim(domain(ρ[i][p]))
            if Dcod > Ddom
                As[p] = Stiefel.StiefelTangent(g[i][p].W, g[i][p].A*vrinv, g[i][p].Z*vrinv)
            else
               As[p] = Unitary.UnitaryTangent(g[i][p].W, g[i][p].A*vrinv)
            end
        end
        opp[i] = As
    end
    return opp
end

function target_func(Hs::Vector{<:AbstractMPOTensor})
    function fg(ρs::FiniteDisorderMPS)
        target_val = measure(ρs, Hs)
        grad = gradient(x -> measure(x, Hs), ρs)
        # grad_pre = disorder_precond(grad[1].opp, ρs)
        grad_p = disorder_project!(grad[1].opp, ρs)
        # grad_p = disorder_project!(grad_pre, ρs)
        # grad_pre = disorder_precond(grad_p, ρs)
        # grad_p = disorder_precond(ρs, grad_p)
        return target_val, grad_p
    end
    return fg
end

function groundstate!(ρ::FiniteDisorderMPS, Hs::Vector{<:AbstractMPOTensor}, alg::StiefelOptim)
    fg = target_func(Hs)
    ρ_opt, _, _, _, gradhist = optimize(fg, ρ, GradientDescent(;verbosity=alg.verbosity, gradtol = alg.gradtol); retract = disorder_retract, inner = disorder_inner, (scale!) = disorder_scale!)#, precondition = disorder_precond)
    
    return ρ_opt, gradhist
end

