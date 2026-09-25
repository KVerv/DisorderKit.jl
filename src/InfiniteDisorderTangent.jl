# Tangent vector at base point ρ{Wₚ} in the manifold of infinite disorder MPS
# With ∑ₚ PₚW'ₚXₚ = 0 and W'ₚXₚ + Xₚ'Wₚ = 0
struct InfiniteDisorderTangent{T<:AbstractMPSTensor}
    ρ::InfiniteDisorderMPS{T}
    tangents::Vector{Stiefel.StiefelTangent}
end

# Computes the inner product between two tangent vectors at the same base point
function inner(x::InfiniteDisorderMPS, ξ₁::InfiniteDisorderTangent, ξ₂::InfiniteDisorderTangent)
    s = 0.0
    for p in eachindex(ξ₁.tangents)
        s += Stiefel.inner(x.opp[p], ξ₁.tangents[p], ξ₂.tangents[p])
    end
    return real.(s)
end

function scale!(ξ::InfiniteDisorderTangent, β::Number)
    newtangents = Vector{Stiefel.StiefelTangent}(undef, length(ξ.tangents))
    for p in eachindex(ξ.tangents)
        newtangents[p] = Stiefel.rmul!(ξ.tangents[p], β)
    end
    return InfiniteDisorderTangent(ξ.ρ, newtangents)
end

function retract(x::InfiniteDisorderMPS, ξ::InfiniteDisorderTangent, α::Real)
    Ws = Vector{typeof(x.opp[1])}(undef, length(x))
    newtangents = Vector{Stiefel.StiefelTangent}(undef, length(ξ.tangents))
    for p in eachindex(ξ.tangents)
        W = x.opp[p]
        Wa, tangenta = Stiefel.retract(W, ξ.tangents[p], α)
        Ws[p] = Wa
        newtangents[p] = tangenta
    end
    return InfiniteDisorderMPS(Ws, x.ps), InfiniteDisorderTangent(InfiniteDisorderMPS(Ws, x.ps), newtangents)
end

function project(g::Vector{<:AbstractMPSTensor}, ρ::InfiniteDisorderMPS)
    tangents = Vector{Stiefel.StiefelTangent}(undef, length(ρ.opp))
    for p in eachindex(g)
        W = ρ.opp[p]
        tangents[p] = Stiefel.project(g[p], W)
    end

    return InfiniteDisorderTangent(ρ, tangents)
end

function add!(ξ₁::InfiniteDisorderTangent, ξ₂::InfiniteDisorderTangent, β::Number)
    newtangents = Vector{Stiefel.StiefelTangent}(undef, length(ξ₁.tangents))
    for p in eachindex(ξ₁.tangents)
        newtangents[p] = ξ₁.tangents[p] + ξ₂.tangents[p] * β
    end
    return InfiniteDisorderTangent(ξ₁.ρ, newtangents)
end

function transport!(ξ::InfiniteDisorderTangent, ρ₀::InfiniteDisorderMPS, η::InfiniteDisorderTangent, α::Real, ρ₁::InfiniteDisorderMPS)
    newtangents = Vector{Stiefel.StiefelTangent}(undef, length(ξ.tangents))
    for p in eachindex(ξ.tangents)
        newtangents[p] = Stiefel.transport(ξ.tangents[p], ρ₀.opp[p], η.tangents[p], α, ρ₁.opp[p])
    end
    return InfiniteDisorderTangent(ρ₁, newtangents)
end

function precondition(ρ::InfiniteDisorderMPS, ξ::InfiniteDisorderTangent)
    newtangents = Vector{Stiefel.StiefelTangent}(undef, length(ξ.tangents))
    r = right_environment(ρ)[2]
    δ = sqrt(inner(ρ, ξ, ξ))
    Id = id(ComplexF64, space(r,1))
    # rinv = inv(sqrt((r^2 + δ^2*Id)))
    rinv = inv(r+δ*Id)
    # rinv = inv(r)

    for p in eachindex(ξ.tangents)
        W = ρ.opp[p]
        Z = ξ.tangents[p].Z * rinv

        A = sylvester(r,r,-2*ξ.tangents[p].A)

        newtangents[p] = Stiefel.StiefelTangent(W, A, Z)
    end
    return InfiniteDisorderTangent(ξ.ρ, newtangents)
end


function groundstate!(ρ::InfiniteDisorderMPS, Hs::DisorderMPOHam; λ::Real=1, gradtol = 1e-2, verbosity=1, maxiter = 1000)
    fg = icost_func(Hs; λ = λ)
    # ρ_opt, _, _, _, gradhist = optimize(fg, ρ, GradientDescent(;maxiter=maxiter,verbosity=verbosity, gradtol = gradtol); retract = retract, inner = inner, (scale!) = scale!, precondition = precondition)
    # ρ_opt, _, _, _, gradhist = optimize(fg, ρ, ConjugateGradient(;maxiter=maxiter,verbosity=verbosity, gradtol = gradtol); retract = retract, inner = inner, (scale!) = scale!, (transport!) = transport!, (add!) = add!, precondition = precondition)
    ρ_opt, _, _, _, gradhist = optimize(fg, ρ, LBFGS(;maxiter=maxiter,verbosity=verbosity, gradtol = gradtol); retract = retract, inner = inner, (scale!) = scale!, (transport!) = transport!, (add!) = add!)#, precondition = precondition)

    return ρ_opt, gradhist
end