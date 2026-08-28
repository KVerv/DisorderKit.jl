function transfer_left_mpo(O::AbstractMPOTensor)
    function ftransfer(vl)
        @tensor vl[-1; -2] := O[2 4; 3 -2] * conj(O[1 4; 3 -1]) * vl[1; 2]
    end
    return ftransfer
end

function transfer_right_mpo(O::AbstractMPOTensor)
    function ftransfer(vr)
        @tensor vr[-1; -2] := O[-1 4; 3 1] * conj(O[-2 4; 3 2]) * vr[1; 2]
        return vr
    end
    return ftransfer
end

# Compute truncation matrices
function truncation_matrices(O::AbstractMPOTensor, trunc_method::MatrixAlgebraKit.TruncationStrategy; timer::TimerOutput = TimerOutput())
    f_l = transfer_left_mpo(O)
    f_r = transfer_right_mpo(O)

    v_l = rand(ComplexF64, space(O, 1), space(O, 1))
    v_r = rand(ComplexF64, space(O, 1), space(O, 1))


    @timeit timer "eigsolves" begin
        _, ls = eigsolve(f_l, v_l, 1, :LM)
        _, rs = eigsolve(f_r, v_r, 1, :LM)
    end

    @timeit timer "SVDs" begin
        _, Sl, Vl = svd_trunc(ls[1]; trunc = (atol = 1e-12,));
        sqrtSl = sqrt(Sl)
        X = sqrtSl * Vl
        Xinv = Vl' * inv(sqrtSl)

        Ur, Sr, _ = svd_trunc(rs[1]; trunc = (atol = 1e-12,));
        sqrtSr = sqrt(Sr)
        Y = Ur * sqrtSr
        Yinv = inv(sqrtSr) * Ur'

        U, S, V, ϵ = svd_trunc(X*Y; trunc=trunc_method)
        sqrtS = sqrt(S)
        PL = sqrtS * V * Yinv
        PR = Xinv * U * sqrtS
    end

    return PL, PR, ϵ
end

# Truncate ordinary mpo with standard truncation algorithm
function truncate_mpo(O::AbstractMPOTensor, alg::StandardTruncation; timer::TimerOutput = TimerOutput())
    PL, PR, ϵ = truncation_matrices(O, alg.trunc_method; timer=timer)
    @tensor O_updated[-1 -2 ; -3 -4] := PL[-1; 1] * O[1 -2; -3 2] * PR[2; -4]
    return O_updated, ϵ, PL, PR
end

# Truncate disorder mpo with standard truncation algorithm
function truncate_disorder_mpo(Dmpo::InfiniteDisorderMPO, alg::StandardTruncation; timer::TimerOutput = TimerOutput())
    pspace = space(Dmpo[1], 2)
    dspace = space(Dmpo[1], 3)
    tspace = space(Dmpo[1], 4)'
    isop = isomorphism(ComplexF64, fuse(pspace ⊗ dspace), pspace ⊗ dspace)
    isot = isomorphism(ComplexF64, fuse(tspace ⊗ dspace), tspace ⊗ dspace)
    @tensor mpo_fused[-1 -2; -3 -4] :=  isop[-2; 1 2] * Dmpo[1][-1 1 2; 3 4 -4] * conj(isot[-3; 3 4])
    PL, PR, ϵ = truncation_matrices(mpo_fused, alg.trunc_method; timer=timer)
    L = length(Dmpo)
    mpo_updated = map(1:L) do ix
        PL = PL
        PR = PR
        @tensor O_updated[-1 -2 -3; -4 -5 -6] := PL[-1; 1] * Dmpo[ix][1 -2 -3; -4 -5 2] * PR[2; -6]
        return O_updated
    end
    return InfiniteDisorderMPO(mpo_updated), ϵ
end

# Truncate the density matrix in each disorder sector
function truncate(ρs::InfiniteDisorderDensityMatrix, alg::StandardTruncation; timer::TimerOutput = TimerOutput())
    mpo_truncated, ϵ = truncate_disorder_mpo(InfiniteDisorderMPO(ρs.opp), alg; timer=timer)
    return InfiniteDisorderDensityMatrix(mpo_truncated.opp, ρs.ps), ϵ
end

# Truncate the InfiniteDisorderMPS in each disorder sector
function truncate(ρ::InfiniteDisorderMPS, alg::StandardTruncation; timer::TimerOutput = TimerOutput())
    pspace = space(ρ[1], 2)
    dspace = space(ρ[1], 3)
    isop = isomorphism(ComplexF64, fuse(pspace ⊗ dspace), pspace ⊗ dspace)
    @tensor mpo_fused[-1 -2; -3 -4] :=  isop[-2; 1 2] * ρ[1][-1 1 2; -3 -4]
    PL, PR, ϵ = truncation_matrices(mpo_fused, alg.trunc_method; timer=timer)
    L = length(ρ)
    mpo_updated = map(1:L) do ix
        PL = PL
        PR = PR
        @tensor O_updated[-1 -2 -3; -4 -5] := PL[-1; 1] * ρ[ix][1 -2 -3; -4 2] * PR[2; -5]
        return O_updated
    end
    return InfiniteDisorderMPS(mpo_updated, ρ.ps), ϵ
end
