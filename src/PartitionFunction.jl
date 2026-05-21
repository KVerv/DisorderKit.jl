struct PartitionFunction{T <: Number, S <: IndexSpace}
    ρ::InfiniteDisorderDensityMatrix
    A::AbstractTensorMap{T, S, 2, 2}
    L::AbstractTensorMap{T, S, 2, 1}
    R::AbstractTensorMap{T, S, 1, 2}
    D::AbstractTensorMap{T, S, 1, 1}
end

function PartitionFunction(ρ::InfiniteDisorderDensityMatrix)
    λ, l, r = DisorderKit.environments(ρ)

    D_dis = ℂ^length(ρ.opp)

    Tbar = zeros(ComplexF64, space(ρ[1], 1)'⊗space(ρ[1], 1), space(ρ[1], 1)'⊗space(ρ[1], 1))
    for (p, A) in enumerate(ρ)
        @tensor Tp[-1 -2; -3 -4] := A[-2 4; 3 -4] * conj(A[-1 4; 3 -3])
        Tbar += ρ.ps[p]*Tp
    end

    δTs = zeros(ComplexF64, length(ρ.opp))
    for (p, A) in enumerate(ρ)
        @tensor Tζ = l[1; 2] * A[2 4; 3 5] * conj(A[1 4; 3 6]) * r[5; 6]
        δTs[p] = Tζ - tr(l*r)
    end
    D = DiagonalTensorMap(δTs, D_dis)

    L = zeros(ComplexF64, ℂ^1 ⊗ ℂ^1, ℂ^1)
    R = zeros(ComplexF64, ℂ^1, ℂ^1 ⊗ ℂ^1)

    @show tr(l*r)
    return PartitionFunction(ρ, Tbar, L, R, D)
end

function renormalize(ρ::InfiniteDisorderDensityMatrix)
    Z = PartitionFunction(ρ)

    W = id(ComplexF64, ℂ^1)
    Id = id(ComplexF64, space(Z.D, 1))
    Dinv = inv(sqrt(Id + Z.D))
    @tensor O[-1 -2; -3 -4] := Dinv[-2; -3] * W[-1; -4]
    ODMPO = DisorderMPO(O, space(ρ.opp[1],2))
    ρ_product = ρ * ODMPO
    return ρ_product
end
