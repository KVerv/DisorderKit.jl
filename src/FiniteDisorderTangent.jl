# Tangent vector at base point ρ{Wᵢₚ} in the manifold of finite disorder MPS
# With ∑ₚ PₚW'ᵢₚXᵢₚ = 0 and W'ᵢₚXᵢₚ + Xᵢₚ'Wᵢₚ = 0 on every site i
struct FiniteDisorderTangent{T<:AbstractMPSTensor}
    ρ::FiniteDisorderMPS{T}
    tangents::Vector{Vector{Stiefel.StiefelTangent}}
end

# Computes the inner product between two tangent vectors at the same base point
function inner(x::FiniteDisorderMPS, ξ₁::FiniteDisorderTangent, ξ₂::FiniteDisorderTangent)
    s = 0.0
    for i in eachindex(ξ₁.tangents)
        for p in eachindex(ξ₁.tangents[i])
            s += Stiefel.inner(x[i,p], ξ₁.tangents[i][p], ξ₂.tangents[i][p])
        end
    end
    return real.(s)
end

function scale!(ξ::FiniteDisorderTangent, β::Number)
    newtangents = [Stiefel.StiefelTangent[Stiefel.rmul!(t, β) for t in ts] for ts in ξ.tangents]
    return FiniteDisorderTangent(ξ.ρ, newtangents)
end

function retract(x::FiniteDisorderMPS, ξ::FiniteDisorderTangent, α::Real)
    Ws = [Vector{typeof(x[1,1])}(undef, length(x[i])) for i in eachindex(x)]
    newtangents = [Vector{Stiefel.StiefelTangent}(undef, length(x[i])) for i in eachindex(x)]
    for i in eachindex(ξ.tangents)
        for p in eachindex(ξ.tangents[i])
            Wa, tangenta = Stiefel.retract(x[i,p], ξ.tangents[i][p], α)
            Ws[i][p] = Wa
            newtangents[i][p] = tangenta
        end
    end
    ρa = FiniteDisorderMPS(Ws, x.ps, x.sectors)
    return ρa, FiniteDisorderTangent(ρa, newtangents)
end

function project(g::Vector{<:Vector{<:AbstractMPSTensor}}, ρ::FiniteDisorderMPS)
    tangents = [Stiefel.StiefelTangent[Stiefel.project(g[i][p], ρ[i,p]) for p in eachindex(ρ[i])] for i in eachindex(ρ)]

    return FiniteDisorderTangent(ρ, tangents)
end

function add!(ξ₁::FiniteDisorderTangent, ξ₂::FiniteDisorderTangent, β::Number)
    newtangents = [Stiefel.StiefelTangent[ξ₁.tangents[i][p] + ξ₂.tangents[i][p] * β for p in eachindex(ξ₁.tangents[i])] for i in eachindex(ξ₁.tangents)]
    return FiniteDisorderTangent(ξ₁.ρ, newtangents)
end

function transport!(ξ::FiniteDisorderTangent, ρ₀::FiniteDisorderMPS, η::FiniteDisorderTangent, α::Real, ρ₁::FiniteDisorderMPS)
    newtangents = [Stiefel.StiefelTangent[Stiefel.transport(ξ.tangents[i][p], ρ₀[i,p], η.tangents[i][p], α, ρ₁[i,p]) for p in eachindex(ξ.tangents[i])] for i in eachindex(ξ.tangents)]
    return FiniteDisorderTangent(ρ₁, newtangents)
end

function precondition(ρ::FiniteDisorderMPS, ξ::FiniteDisorderTangent)
    newtangents = [Vector{Stiefel.StiefelTangent}(undef, length(ρ[i])) for i in eachindex(ρ)]
    rs = right_environments(ρ)
    δ = sqrt(inner(ρ, ξ, ξ))

    for i in eachindex(ξ.tangents)
        # Right environment on the right bond of site i
        r = rs[i+1]
        Id = id(ComplexF64, space(r,1))
        rinv = inv(r+δ*Id)

        for p in eachindex(ξ.tangents[i])
            W = ρ[i,p]
            Z = ξ.tangents[i][p].Z * rinv

            A = sylvester(r,r,-2*ξ.tangents[i][p].A)

            newtangents[i][p] = Stiefel.StiefelTangent(W, A, Z)
        end
    end
    return FiniteDisorderTangent(ξ.ρ, newtangents)
end

function Ecost_func(Hs::DisorderMPOHam, ::FiniteDisorderMPS)
    function fg(ρ::FiniteDisorderMPS)
        E = DisorderKit.energy_density(ρ, Hs)
        grad = gradient(x -> DisorderKit.energy_density(x, Hs), ρ)
        gradp = project(grad[1].opp, ρ)

        return E, gradp
    end
    return fg
end

function groundstate!(ρ::FiniteDisorderMPS, Hs::DisorderMPOHam; gradtol = 1e-2, verbosity=1, maxiter = 1000)
    fg = Ecost_func(Hs, ρ)
    ρ_opt, _, _, _, gradhist = optimize(fg, ρ, LBFGS(;maxiter=maxiter,verbosity=verbosity, gradtol = gradtol); retract = retract, inner = inner, (scale!) = scale!, (transport!) = transport!, (add!) = add!, precondition = precondition)

    return ρ_opt, gradhist
end
