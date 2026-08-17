struct PartitionFunction{T <: Number, S <: IndexSpace}
    ρ::InfiniteDisorderDensityMatrix
    A::AbstractTensorMap{T, S, 2, 2}
    L::AbstractTensorMap{T, S, 1, 2}
    R::AbstractTensorMap{T, S, 2, 1}
    D::AbstractTensorMap{T, S, 1, 1}
    δ::Real
end


function PartitionFunction(ρ::InfiniteDisorderDensityMatrix)
    # @show space(ρ[1], 1)
    # Tζf, λ, lf, rf = compute_canonical_forms(ρ; maxiter = 100)
    iso = isomorphism(fuse(space(ρ[1], 1)'⊗space(ρ[1], 1)), space(ρ[1], 1)'⊗space(ρ[1], 1))
    @tensor Tζf[-1 -2; -3 -4] := iso[-1; 1 2] * ρ[1][2 5 -2; 6 7 4] * conj(ρ[1][1 5 -3; 6 7 3]) * conj(iso[-4; 3 4])
    @show space(Tζf)
    # Tζf, PL, PR = truncate_mpo(Tζf, MatrixAlgebraKit.truncerror(rtol = 1e-1))
    Tζf, PL, PR = truncate_mpo(Tζf, MatrixAlgebraKit.truncrank(2))
    @show space(Tζf)
    # es = entanglement_spectrum(InfiniteMPO([Tζf]), 1)
    # @show es
    dspace = space(ρ[1], 3)
    Id = id(ComplexF64, dspace)
    P = make_DiagonalBlockTensorMap(ρ.ps)

    # Tbar_fused = zeros(ComplexF64, space(Tζf, 1)[1], space(Tζf, 1)[1])
    # for i in eachindex(ρ.ps)
    #     Zx = TensorMap(Tζf[1,i,i,1].data, space(Tζf, 1)[1], space(Tζf, 1)[1])
    #     lZ = log(Zx)
    #     Tbar_fused += ρ.ps[i] * lZ
    # end
    # Tbar_fused = exp(Tbar_fused)


    @tensor Tbar_fused[-1; -2] := Tζf[-1 1; 2 -2] * P[2; 1]
    @tensor Tbar_tensor[-1 -2; -3 -4] := Tbar_fused[-1; -4] * Id[-2; -3]

    # vals = eig_vals(Zx)
    # @show vals

    λ, l, r = environments(ρ)
    # @tensor lf[-1] := l[1;2] * conj(iso[-1;1 2])
    # @tensor rf[-1] := r[1;2] * iso[-1;2 1]
    @tensor lf[-1] := l[1;2] * conj(iso[3;1 2]) * PR[3;-1]
    @tensor rf[-1] := r[1;2] * iso[3;2 1]  * PL[-1;3]

    # @tensor Pdom[-1; -2] := rf[-1] * lf[-2]
    # # @show space(Pdom)

    # QP = id(space(Tbar_fused, 1)) - Pdom
    # Q = QP*Tbar_fused*QP
    # Λ, U = eig_trunc(Q; trunc = truncrank(χ))
    # # vals = eig_vals(Q)
    # # @show abs.(vals)[1:min(6, length(vals))]
    # Uinv = pinv(U)

    χ = dim(space(Tζf,1))
    Λr, Vr = eig_trunc(Tbar_fused; trunc = truncrank(χ))
    Λ, Vl = eig_trunc(Tbar_fused'; trunc = truncrank(χ))
    @show abs.(Λ.data)
    Λ[1] = 0
    bdel = ones(ComplexF64, dim(space(Vl, 2)))
    bdel[1] = 0
    χeff = dim(space(Vl, 2))
    bd = TensorMap(diagm(bdel), ℂ^χeff, ℂ^χeff)
    Vr = Vr * bd

    # @tensor Vr[-1; -2] := Vr[-1; 1] * bd[1; -2]
    @tensor Vl[-1; -2] := Vl[-1; 1] * bd[1; -2]

    δT = Tζf - Tbar_tensor
    @tensor D[-1; -2] := lf[1] * δT[1 -1; -2 2] * rf[2]

    # @tensor L[-1; -2 -3] := lf[1] * δT[1 -1; -2 2] * QP[2; 3] * U[3; -3]
    # @tensor R[-1 -2; -3] := Uinv[-1; 2] * QP[2; 3] * δT[3 -2; -3 1] * rf[1]

    @tensor L[-1; -2 -3] := lf[1] * δT[1 -1; -2 2] * Vr[2; -3]
    @tensor R[-1 -2; -3] := conj(Vl[3; -1]) * δT[3 -2; -3 1] * rf[1]

    # @tensor L[-1; -2 -3] := lf[1] * δT[1 -1; -2 2] * QP[2; -3]
    # @tensor R[-1 -2; -3] := QP[-1; 3] * δT[3 -2; -3 1] * rf[1]
    @tensor A[-1 -2; -3 -4] := Λ[-1; -4] * Id[-2; -3]
    # @tensor A[-1 -2; -3 -4] := Q[-1; -4] * Id[-2; -3]


    # @tensor Dζ[-1; -2] := lf[1] * Tζf[1 -1; -2 2] * rf[2]
    # D = Dζ - λ*Id

    # @tensor L[-1; -2 -3] := lf[1] * Tζf[1 -1; -2 -3]
    # @tensor Ldom[-1; -2 -3] := lf[-3] * Id[-1; -2]
    # L = L -  λ*Ldom
    # @tensor Lt[-1; -2 -3] := L[-1; -2 1] * U[1; -3]
    # # @tensor Lt[-1; -2 -3] := L[-1; -2 -3]

    # @tensor R[-1 -2; -3] := Tζf[-1 -2; -3 1] * rf[1]
    # @tensor Rdom[-1 -2; -3] := rf[-1] * Id[-2; -3]
    # R = R - λ*Rdom

    # @tensor Rt[-1 -2; -3] := R[1 -2; -3] * Uinv[-1; 1]
    # # @tensor Rt[-1 -2; -3] := R[-1 -2; -3]

    # @tensor A[-1 -2; -3 -4] := Λ[-1; -4] * Id[-2; -3]
    # # @tensor A[-1 -2; -3 -4] := Q[-1; -4] * Id[-2; -3]

    # A *= 0
    # R *= 0
    # L *= 0
    δ = norm(δT[1,1,1,1])
    @info(crayon"red"("δ =  $(δ)"))
    return PartitionFunction(ρ, A, L, R, D, δ)
end


function inv_sqrt_MPO(Z::PartitionFunction; N::Int = 1)
    dspace = space(Z.D, 1)
    vspace = space(Z.R, 1)

    Wcodomain = BlockTensorKit.boxplus([ℂ^1, vspace]...) ⊗ BlockTensorKit.boxplus(dspace)
    Wdomain = BlockTensorKit.boxplus(dspace) ⊗ BlockTensorKit.boxplus([ℂ^1, vspace]...)

    W = SparseBlockTensorMap{AbstractTensorMap{ComplexF64, ComplexSpace, 2, 2}}(undef, Wcodomain, Wdomain)

    for i in 1:dim(dspace)
        # W[1, i, i, 1] = ones(ComplexF64, ℂ^1 ⊗ dspace[1], dspace[1] ⊗ ℂ^1) - 1/2 * TensorMap(Z.D[i, i].data, ℂ^1 ⊗ dspace[1], dspace[1] ⊗ ℂ^1) + 3/8 * TensorMap(Z.D[i, i].data.^2, ℂ^1 ⊗ dspace[1], dspace[1] ⊗ ℂ^1)
        W[1, i, i, 1] = TensorMap((1. .- 1/2*Z.D[i, i].data .+ 3/8*Z.D[i, i].data.^2), ℂ^1 ⊗ dspace[1], dspace[1] ⊗ ℂ^1)
        W[1, i, i, 2] = TensorMap(Z.L[i, i, 1].data, ℂ^1 ⊗ dspace[1], dspace[1] ⊗ vspace[1])
        W[2, i, i, 2] = TensorMap(Z.A[1, i, i, 1].data, vspace[1] ⊗ dspace[1], dspace[1] ⊗ vspace[1])
        W[2, i, i, 1] = TensorMap(-1/2*Z.R[1, i, i].data, vspace[1] ⊗ dspace[1], dspace[1] ⊗ ℂ^1)
    end


    return W
end

function compute_canonical_forms(ρ::InfiniteDisorderDensityMatrix; maxiter::Int=100)
    vspace = fuse(space(ρ[1], 1)', space(ρ[1], 1))
    dspace = space(ρ[1], 3)
    Zcodomain = BlockTensorKit.boxplus(vspace) ⊗ BlockTensorKit.boxplus(dspace)
    Zdomain = BlockTensorKit.boxplus(vspace)
    iso = isomorphism(vspace, space(ρ[1], 1)'⊗space(ρ[1], 1))

    # @show Zcodomain
    # @show Zdomain
    @tensor Z[-1 -2; -3 -4] := iso[-1; 1 2] * ρ[1][2 5 -2; 6 7 4] * conj(ρ[1][1 5 -3; 6 7 3]) * conj(iso[-4; 3 4])

    Zmps = spzeros(ComplexF64, Zcodomain, Zdomain)
    for i in 1:length(dspace)
        for j in 1:length(vspace)
            Zmps[1, i, 1] = TensorMap(Z[j,i,i,j].data, vspace[j] ⊗ ℂ^1, vspace[j])
        end
    end


    vspace = space(Zmps, 1)
    L = rand(ComplexF64, vspace, vspace)
    L /= norm(L)
    Lprev = deepcopy(L)
    ϵ = 1.0
    ZL = zeros(ComplexF64, space(Zmps))
    i = 0
    while (ϵ > 1e-8) && (i<maxiter)
        # @show ϵ
        @show space(L)
        @tensor Zi[-1 -2; -3] := L[-1; 1] * Zmps[1 -2; -3]
        U, S, V = svd_trunc(Zi; trunc = truncerror(atol = 1e-10))
        @show S
        L = S*V
        L /= norm(L)
        if space(L) == space(Lprev)
            ϵ = norm(L-Lprev)
        end
        Lprev = deepcopy(L)
        ZL = U
        i += 1
    end
    @show ϵ

    R = rand(ComplexF64, vspace, vspace)
    R /= norm(R)
    Rprev = deepcopy(R)
    ϵ = 1.0
    ZR = zeros(ComplexF64, space(Zmps))
    i = 0
    while (ϵ > 1e-8) && (i<200)
        # @show ϵ
        @tensor Zi[-1; -2 -3] := R[1; -3] * Zmps[-1 -2; 1]
        U, S, V = svd_trunc(Zi; trunc = truncerror(atol = 1e-10))
        R = U*S
        R /= norm(R)
        if space(R) == space(Rprev)
            ϵ = norm(R-Rprev)
        end
        Rprev = deepcopy(R)
        ZR = V
        i += 1
    end
    @show ϵ

    C = L*R
    U, S, V = svd_trunc(C; trunc = truncerror(atol = 1e-4))
    C = S
    @tensor ZL[-1 -2; -3] := conj(U[1; -1]) * ZL[1 -2; 2] * U[2; -3] 
    @tensor ZC[-1 -2; -3] := ZL[-1 -2; 1] * C[1; -3] 

    vspace = space(ZC, 1)
    Z = SparseBlockTensorMap{AbstractTensorMap{ComplexF64, ComplexSpace, 2, 2}}(undef, BlockTensorKit.boxplus(vspace) ⊗ BlockTensorKit.boxplus(dspace), BlockTensorKit.boxplus(dspace) ⊗ BlockTensorKit.boxplus(vspace))
    for i in 1:length(dspace)
        for j in 1:length(vspace)
            Z[j, i, i, j] = TensorMap(ZL[j,i,j].data, vspace[j] ⊗ dspace[i], dspace[i] ⊗ vspace[j])
        end
    end

    λ, l, r = environments(ρ)
    @show space(l)
    @show space(L)
    @show space(iso)
    @tensor lf[-1] := l[1;2] * conj(iso[3; 1 2]) * L[3; -1]
    @tensor rf[-1] := r[1;2] * iso[3; 2 1] * conj(L[3; -1])
    # @show space(Z)
    return Z, λ, lf, rf
end

function normalize(ρ::InfiniteDisorderDensityMatrix; N::Int=1)
    ρ = gauge(ρ)
    Z = PartitionFunction(ρ)

    O = inv_sqrt_MPO(Z; N=N)
    ρ_product = ρ * O

    return ρ_product, Z.δ
end


