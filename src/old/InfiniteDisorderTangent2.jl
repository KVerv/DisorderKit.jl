# Tangent vector at base point ρ{Wₚ} in the manifold of infinite disorder MPS
# With ∑ₚ PₚW'ₚXₚ = 0 and W'ₚXₚ + Xₚ'Wₚ = 0
struct InfiniteDisorderTangent2{T<:AbstractMPSTensor}
    ρ::InfiniteDisorderMPS2{T}
    tangentsW::Vector{Stiefel.StiefelTangent}
    tangentsV::Vector{Stiefel.StiefelTangent}
end

# Computes the inner product between two tangent vectors at the same base point
function inner(x::InfiniteDisorderMPS2, ξ₁::InfiniteDisorderTangent2, ξ₂::InfiniteDisorderTangent2)
    s = 0.0
    for p in eachindex(ξ₁.tangentsW)
        s += Stiefel.inner(x.Ws[p], ξ₁.tangentsW[p], ξ₂.tangentsW[p])
        s += Stiefel.inner(x.Vs[p], ξ₁.tangentsV[p], ξ₂.tangentsV[p])
    end
    return real.(s)
end

function scale!(ξ::InfiniteDisorderTangent2, β::Number)
    newtangentsW = Vector{Stiefel.StiefelTangent}(undef, length(ξ.tangentsW))
    newtangentsV = Vector{Stiefel.StiefelTangent}(undef, length(ξ.tangentsV))
    for p in eachindex(ξ.tangentsW)
        newtangentsW[p] = Stiefel.rmul!(ξ.tangentsW[p], β)
    end
    for p in eachindex(ξ.tangentsV)
        newtangentsV[p] = Stiefel.rmul!(ξ.tangentsV[p], β)
    end
    return InfiniteDisorderTangent2(ξ.ρ, newtangentsW, newtangentsV)
end

function retract(x::InfiniteDisorderMPS2, ξ::InfiniteDisorderTangent2, α::Real)
    Ws = Vector{typeof(x.Ws[1])}(undef, length(x.Ws))
    Vs = Vector{typeof(x.Vs[1])}(undef, length(x.Vs))
    newtangentsW = Vector{Stiefel.StiefelTangent}(undef, length(ξ.tangentsW))
    newtangentsV = Vector{Stiefel.StiefelTangent}(undef, length(ξ.tangentsV))
    for p in eachindex(ξ.tangentsW)
        W = x.Ws[p]
        Wa, tangenta = Stiefel.retract(W, ξ.tangentsW[p], α)
        Ws[p] = Wa
        newtangentsW[p] = tangenta
    end
    for p in eachindex(ξ.tangentsV)
        V = x.Vs[p]
        Va, tangenta = Stiefel.retract(V, ξ.tangentsV[p], α)
        Vs[p] = Va
        newtangentsV[p] = tangenta
    end
    return InfiniteDisorderMPS2(Ws, Vs, x.ps), InfiniteDisorderTangent2(InfiniteDisorderMPS2(Ws, Vs, x.ps), newtangentsW, newtangentsV)
end

function project(gW::Vector{<:AbstractMPSTensor}, gV::Vector{<:AbstractMPSTensor}, ρ::InfiniteDisorderMPS2)
    tangentsW = Vector{Stiefel.StiefelTangent}(undef, length(ρ.Ws))
    tangentsV = Vector{Stiefel.StiefelTangent}(undef, length(ρ.Vs))
    for p in eachindex(gW)
        W = ρ.Ws[p]
        V = ρ.Vs[p]
        tangentsW[p] = Stiefel.project(gW[p], W)
        tangentsV[p] = Stiefel.project(gV[p], V)
    end

    # t = tangents[1]
    # tangents[1] = Stiefel.StiefelTangent(t.W, t.A*0, t.Z)
    return InfiniteDisorderTangent2(ρ, tangentsW, tangentsV)
end

function add!(ξ₁::InfiniteDisorderTangent2, ξ₂::InfiniteDisorderTangent2, β::Number)
    newtangentsW = Vector{Stiefel.StiefelTangent}(undef, length(ξ₁.tangentsW))
    newtangentsV = Vector{Stiefel.StiefelTangent}(undef, length(ξ₁.tangentsV))
    for p in eachindex(ξ₁.tangentsW)
        newtangentsW[p] = ξ₁.tangentsW[p] + ξ₂.tangentsW[p] * β
    end
    for p in eachindex(ξ₁.tangentsV)
        newtangentsV[p] = ξ₁.tangentsV[p] + ξ₂.tangentsV[p] * β
    end
    return InfiniteDisorderTangent2(ξ₁.ρ, newtangentsW, newtangentsV)
end

function transport!(ξ::InfiniteDisorderTangent2, ρ₀::InfiniteDisorderMPS2, η::InfiniteDisorderTangent2, α::Real, ρ₁::InfiniteDisorderMPS2)
    newtangentsW = Vector{Stiefel.StiefelTangent}(undef, length(ξ.tangentsW))
    newtangentsV = Vector{Stiefel.StiefelTangent}(undef, length(ξ.tangentsV))
    for p in eachindex(ξ.tangentsW)
        newtangentsW[p] = Stiefel.transport(ξ.tangentsW[p], ρ₀.Ws[p], η.tangentsW[p], α, ρ₁.Ws[p])
    end
    for p in eachindex(ξ.tangentsV)
        newtangentsV[p] = Stiefel.transport(ξ.tangentsV[p], ρ₀.Vs[p], η.tangentsV[p], α, ρ₁.Vs[p])
    end
    return InfiniteDisorderTangent2(ρ₁, newtangentsW, newtangentsV)
end

function precondition(ρ::InfiniteDisorderMPS2, ξ::InfiniteDisorderTangent2)
    newtangentsW = Vector{Stiefel.StiefelTangent}(undef, length(ξ.tangentsW))
    newtangentsV = Vector{Stiefel.StiefelTangent}(undef, length(ξ.tangentsV))
    _, rW, _, rV = right_environment(ρ)
    δ = sqrt(inner(ρ, ξ, ξ))
    Id = id(ComplexF64, space(rW,1))
    rWinv = inv(sqrt((rW^2 + δ^2*Id)))
    rVinv = inv(sqrt((rV^2 + δ^2*Id)))
    # rinv = inv(r+δ*Id)
    # rinv = inv(r)

    for p in eachindex(ξ.tangentsW)
        W = ρ.Ws[p]
        V = ρ.Vs[p]
        ZW = ξ.tangentsW[p].Z * rVinv
        ZV = ξ.tangentsV[p].Z * rWinv

        # AW = ξ.tangentsW[p].A
        # AV = ξ.tangentsV[p].A
        AW = sylvester(rV,rV,-2*ξ.tangentsW[p].A)
        AV = sylvester(rW,rW,-2*ξ.tangentsV[p].A)

        newtangentsW[p] = Stiefel.StiefelTangent(W, AW, ZW)
        newtangentsV[p] = Stiefel.StiefelTangent(V, AV, ZV)
    end
    return InfiniteDisorderTangent2(ξ.ρ, newtangentsW, newtangentsV)
end


function icost_func2(Hs::DisorderMPOHam)
    function fg(ρ::InfiniteDisorderMPS2)
        target_val = energy_density(ρ, Hs)
        grad = gradient(x -> energy_density(x, Hs), ρ)
        # target_val = median_energy_density(ρ, Hs; λ = λ)
        # grad = gradient(x -> median_energy_density(x, Hs; λ = λ), ρ)
        # gradp = project(grad, ρ)
        gradp = project(grad[1].Ws, grad[1].Vs, ρ)
        # gradp = precondition(ρ,gradp)

        return target_val, gradp
    end
    return fg
end

function groundstate!(ρ::InfiniteDisorderMPS2, Hs::DisorderMPOHam; gradtol = 1e-2, verbosity=1, maxiter = 1000)
    fg = icost_func2(Hs)
    # ρ_opt, _, _, _, gradhist = optimize(fg, ρ, GradientDescent(;maxiter=maxiter,verbosity=verbosity, gradtol = gradtol); retract = retract, inner = inner, (scale!) = scale!)#, precondition = precondition)
    # ρ_opt, _, _, _, gradhist = optimize(fg, ρ, ConjugateGradient(;maxiter=maxiter,verbosity=verbosity, gradtol = gradtol); retract = retract, inner = inner, (scale!) = scale!, (transport!) = transport!, (add!) = add!, precondition = precondition)
    ρ_opt, _, _, _, gradhist = optimize(fg, ρ, LBFGS(10;maxiter=maxiter,verbosity=verbosity, gradtol = gradtol); retract = retract, inner = inner, (scale!) = scale!, (transport!) = transport!, (add!) = add!, precondition = precondition)

    return ρ_opt, gradhist
end