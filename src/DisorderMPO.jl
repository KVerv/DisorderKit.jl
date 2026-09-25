struct InfiniteDisorderMPO{T<:AbstractMPOTensor}
    opp::Vector{T}
end

Base.getindex(T::InfiniteDisorderMPO, ix::Int) = T.opp[ix]
Base.size(T::InfiniteDisorderMPO) = size(T.opp)
Base.length(T::InfiniteDisorderMPO) = length(T.opp)
Base.iterate(t::InfiniteDisorderMPO, i=1) = (i > length(t.opp)) ? nothing : (t[i], i + 1)

# Multiply DisorderMPS with DisorderMPO
function Base.:*(O::InfiniteDisorderMPO, ψ::InfiniteDisorderMPS)

    (length(ψ.opp) == length(O)) || throw(ArgumentError("O should have the same amount of disorder sectors as ψ"))
    
    opp = Vector{typeof(ψ.opp[1])}(undef, length(ψ.opp))
    for (p, W) in  enumerate(ψ.opp)
        iso = isomorphism(ComplexF64, fuse(space(W, 1), space(O[p],1)), space(W, 1) ⊗ space(O[p],1))
        @tensor Wnew[-1 -2; -4] := iso[-1; 1 2]*W[1 3; 4] *O[p][2 -2;3 5] * conj(iso[-4; 4 5])
        opp[p] = Wnew
    end

    return InfiniteDisorderMPS(opp, ψ.ps)
end

function approximate(ψ::InfiniteDisorderMPS, ϕ0::InfiniteDisorderMPS; gradtol=1e-12, maxiter=1000, verbosity=5)

    function fg(ρ::InfiniteDisorderMPS)
        # target_val = 1-fidelity(ρ, ψ)
        # grad = gradient(x -> 1-fidelity(x, ψ), ρ)

        target_val = -abs(overlap(ρ, ψ))
        grad = gradient(x -> -abs(overlap(x, ψ)), ρ)
        gradp = project(grad[1].opp, ρ)

        return target_val, gradp
    end

    ψ_opt, _, _, _, gradhist = optimize(fg, ϕ0, LBFGS(;maxiter=maxiter,verbosity=verbosity, gradtol = gradtol); retract = retract, inner = inner, (scale!) = scale!, (transport!) = transport!, (add!) = add!, precondition = precondition)
    return ψ_opt, gradhist
end

function DisorderMPO(Hs::DisorderMPOHam)
    Ws = Vector{typeof(Hs.As[1])}(undef, length(Hs.Bs))
    for (p, B) in enumerate(Hs.Bs)
        A = Hs.As[p]
        C = Hs.Cs[p]
        D = Hs.Ds[p]
        Wcodomain = BlockTensorKit.boxplus(ℂ^2) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, 3)...)
        Wdomain = BlockTensorKit.boxplus(fill(ℂ^1, 3)...) ⊗ BlockTensorKit.boxplus(ℂ^2)
        W = SparseBlockTensorMap{AbstractTensorMap{ComplexF64, ComplexSpace, 2, 2}}(undef, Wdomain, Wcodomain)

        W[1, 1, 1, 1] = BraidingTensor{ComplexF64, ComplexSpace}(ℂ^2, ℂ^1)
        W[1, 1, 1, 2] = BlockTensorKit.insertrightunit(C, 0)
        W[1, 1, 1, 3] = MPSKit.add_util_leg(D)
        W[2, 1, 1, 2] = A
        W[2, 1, 1, 3] = BlockTensorKit.insertrightunit(B, 3)
        W[3, 1, 1, 3] = BraidingTensor{ComplexF64, ComplexSpace}(ℂ^2, ℂ^1)
        Ws[p] = W
    end
    return InfiniteDisorderMPO(Ws)
end