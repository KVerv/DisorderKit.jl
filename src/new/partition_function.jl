function pf_left_transfer_matrix(ρs::InfiniteDisorderDensityMatrix)
    function ftransfer(v)
        vnew = zeros(ComplexF64, space(ρs[1], 1)⊗space(ρs[1], 1)', space(ρs[1], 1)⊗space(ρs[1], 1)')
        for (p, A) in enumerate(ρs)
            @tensor vp[-3 -4; -1 -2] := A[1 3;4 -1] * conj(A[2 3;4 -2]) * conj(A[5 8; 7 -3]) * A[6 8; 7 -4] * v[5 6; 1 2]
            vnew += ρs.ps[p] * vp
        end
        return vnew
    end
    return ftransfer
end

function pf_right_transfer_matrix(ρs::InfiniteDisorderDensityMatrix)
    function ftransfer(v)
        vnew = zeros(ComplexF64, space(ρs[1], 1)⊗space(ρs[1], 1)', space(ρs[1], 1)⊗space(ρs[1], 1)')
        for (p, A) in enumerate(ρs)
            @tensor vp[-1 -2; -3 -4] := A[-1 3;4 1] * conj(A[-2 3;4 2]) * conj(A[-3 8; 7 5]) * A[-4 8; 7 6] * v[1 2; 5 6]
            vnew += ρs.ps[p] * vp
        end
        return vnew
    end
    return ftransfer
end

function partition_function_mpo(ρs::InfiniteDisorderDensityMatrix, trunc_method::MatrixAlgebraKit.TruncationStrategy)
    ft_left = pf_left_transfer_matrix(ρs)
    ft_right = pf_right_transfer_matrix(ρs)
    v_left = rand(ComplexF64, space(ρs[1], 1)⊗space(ρs[1], 1)', space(ρs[1], 1)⊗space(ρs[1], 1)')
    v_right = rand(ComplexF64, space(ρs[1], 1)⊗space(ρs[1], 1)', space(ρs[1], 1)⊗space(ρs[1], 1)')
    vals_left, ls = eigsolve(x->ft_left(x), v_left, 1, :LM)
    vals_right, rs = eigsolve(x->ft_right(x), v_right, 1, :LM)

        _, Sl, Vl = svd_trunc(ls[1]; trunc = (atol = 1e-12,));
    X = sqrt(Sl) * Vl
    Xinv = Vl' * inv(sqrt(Sl))
    Ur, Sr, _ = svd_trunc(rs[1]; trunc = (atol = 1e-12,));
    Y = Ur * sqrt(Sr)
    Yinv = inv(sqrt(Sr)) * Ur'
    U, S, V, ϵ = svd_trunc(X*Y; trunc=trunc_method)
    PL = sqrt(S) * V * Yinv
    PR = Xinv * U * sqrt(S)

    vspace = space(PL, 1)
    D_disorder = length(ρs.opp)
    Z_full = zeros(ComplexF64, vspace ⊗ ℂ^D_disorder, ℂ^D_disorder ⊗ vspace)
    for (p, ρ) in enumerate(ρs)
        diagel = zeros(ComplexF64, D_disorder)
        diagel[p] = 1.
        W = DiagonalTensorMap(diagel, ℂ^D_disorder)
        @tensor Z[-1 -2; -3 -4] := PL[-1; 1 2] * ρ[1 3;4 5] * conj(ρ[2 3;4 6]) * PR[5 6; -4] * W[-2; -3]
        Z_full += Z
    end

    return Z_full
end

function canonical_forms(Z::AbstractMPSTensor)
    vspace = space(Z, 1)
    L = rand(ComplexF64, vspace, vspace)
    L /= norm(L)
    Lprev = deepcopy(L)
    ϵ = 1.0
    ZL = zeros(ComplexF64, space(Z))
    i = 0
    while (ϵ > 1e-8) && (i<200)
        @show ϵ, norm(L), norm(Lprev)
        @tensor Zi[-1 -2; -3] := L[-1; 1] * Z[1 -2; -3]
        U, S, V = svd_trunc(Zi; trunc = truncerror(atol = 1e-8))
        L = S*V
        L /= norm(L)
        if space(L) == space(Lprev)
            ϵ = norm(L-Lprev)
        end
        Lprev = deepcopy(L)
        ZL = U
        i += 1
    end
    R = rand(ComplexF64, vspace, vspace)
    R /= norm(R)
    Rprev = deepcopy(R)
    ϵ = 1.0
    ZR = zeros(ComplexF64, space(Z))
    i = 0
    while (ϵ > 1e-8) && (i<200)
        @show ϵ
        @tensor Zi[-1; -2 -3] := R[1; -3] * Z[-1 -2; 1]
        U, S, V = svd_trunc(Zi; trunc = truncerror(atol = 1e-8))
        R = U*S
        R /= norm(R)
        if space(R) == space(Rprev)
            ϵ = norm(R-Rprev)
        end
        Rprev = deepcopy(R)
        ZR = V
        i += 1
    end

    C = L*R
    @tensor ZC[-1 -2; -3] := ZL[-1 -2; 1] * C[1; -3] 
    return ZL, ZR, ZC, C
end

function partition_function_svd(ρs::InfiniteDisorderDensityMatrix, trunc_method::MatrixAlgebraKit.TruncationStrategy)
    PL = isomorphism(ComplexF64, fuse(space(ρs[1],1)',space(ρs[1], 1)),space(ρs[1],1)'⊗space(ρs[1], 1))
    vspace = space(PL, 1)
    D_disorder = length(ρs.opp)
    Z = zeros(ComplexF64, vspace ⊗ ℂ^D_disorder, vspace)
    for (p, ρ) in enumerate(ρs)
        diagel = zeros(ComplexF64, D_disorder)
        diagel[p] = 1.
        W = Tensor(diagel, ℂ^D_disorder)
        @tensor Zp[-1 -2; -3] := PL[-1; 1 2] * ρ[2 4;3 5] * conj(ρ[1 4;3 6]) * conj(PL[-3; 6 5]) * W[-2]
        Z += Zp * ρs.ps[p]
    end

    ZL, ZR, ZC, C = canonical_forms(Z)

    U, S, V = svd_trunc(C; trunc = trunc_method)
    @show S
    C = S

    @show space(ZL), space(ZR), space(ZC), space(C)
    @show space(U)
    @tensor ZLt[-1 -2; -3] := conj(U[2; -1]) * ZL[2 -2; 1] * U[1; -3]
    @tensor ZRt[-1 -2; -3] := V[-1; 2] * ZR[2; -2 1] * conj(V[-3; 1])
    @tensor ZCt[-1 -2; -3] := ZLt[-1 -2; 1] * S[1; -3]

    init = zeros(ComplexF64, D_disorder, D_disorder, D_disorder)
    for (i, j, k) in Iterators.product([1:D_disorder for _ in 1:3]...)
        if (i == j) && (j == k)
           init[i, j, k] = 1.
        end 
    end
    splitter = TensorMap(init, ℂ^D_disorder ← ℂ^D_disorder ⊗ ℂ^D_disorder)
    @tensor ZLt[-1 -2; -3 -4] := ZLt[-1 1; -4] * splitter[-2; 1 -3]
    @tensor ZRt[-1 -2; -3 -4] := ZRt[-1 1; -4] * splitter[-2; 1 -3]
    @tensor ZCt[-1 -2; -3 -4] := ZCt[-1 1; -4] * splitter[-2; 1 -3]
    return ZLt, ZRt, ZCt, C
end