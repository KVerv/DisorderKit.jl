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
    for i in eachindex(ρ)
        for p in eachindex(ρ[i])
            Dcod = dim(codomain(ρ[i][p]))
            Ddom = dim(domain(ρ[i][p]))
            if Dcod > Ddom
                s += Stiefel.inner_euclidean(ρ[i][p], g₁[i][p], g₂[i][p])
            else
                s += Unitary.inner(ρ[i][p], g₁[i][p], g₂[i][p])
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

function target_func(Hs::Vector{<:AbstractMPOTensor})
    function fg(ρs::FiniteDisorderMPS)
        target_val = measure(ρs, Hs)
        grad = gradient(x -> measure(x, Hs), ρs)
        pgrad = disorder_project!(grad[1].opp, ρs)
        return target_val, pgrad
    end
    return fg
end

function groundstate!(ρ::FiniteDisorderMPS, Hs::Vector{<:AbstractMPOTensor}, alg::StiefelOptim)
    fg = target_func(Hs)
    ρ_opt, _ = optimize(fg, ρ, GradientDescent(;verbosity=alg.verbosity, gradtol = alg.gradtol); retract = disorder_retract, inner = disorder_inner, (scale!) = disorder_scale!)
    
    return ρ_opt
end

