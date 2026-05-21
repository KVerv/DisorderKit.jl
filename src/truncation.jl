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
function truncation_matrices(O::AbstractMPOTensor, trunc_method::MatrixAlgebraKit.TruncationStrategy)
    f_l = transfer_left_mpo(O)
    f_r = transfer_right_mpo(O)
    v_l = rand(ComplexF64, space(O, 1), space(O, 1))
    v_r = rand(ComplexF64, space(O, 1), space(O, 1))
    _, ls = eigsolve(f_l, v_l, 1, :LM)
    _, rs = eigsolve(f_r, v_r, 1, :LM)
    _, Sl, Vl = svd_trunc(ls[1]; trunc = (atol = 1e-12,));
    X = sqrt(Sl) * Vl
    Xinv = Vl' * inv(sqrt(Sl))
    Ur, Sr, _ = svd_trunc(rs[1]; trunc = (atol = 1e-12,));
    Y = Ur * sqrt(Sr)
    Yinv = inv(sqrt(Sr)) * Ur'
    U, S, V = svd_trunc(X*Y; trunc=trunc_method)
    PL = sqrt(S) * V * Yinv
    PR = Xinv * U * sqrt(S)
    
    return PL, PR
end



# Truncate ordinary mpo with standard truncation algorithm
function truncate_disorder_mpo(Dmpo::DisorderMPO, trunc_method::MatrixAlgebraKit.TruncationStrategy)
    mpo = get_MPOTensor(Dmpo)
    PL, PR = truncation_matrices(mpo, trunc_method)
    L = length(Dmpo)
    mpo_updated = map(1:L) do ix
        PL = PL
        PR = PR
        @tensor O_updated[-1 -2 ; -3 -4] := PL[-1; 1] * Dmpo[ix][1 -2; -3 2] * PR[2; -4]
        return O_updated
    end
    return DisorderMPO(mpo_updated)
end


# Truncate the density matrix in each disorder sector
function truncate(ρs::InfiniteDisorderDensityMatrix, trunc_method::MatrixAlgebraKit.TruncationStrategy)
    mpo_truncated = truncate_disorder_mpo(DisorderMPO(ρs.opp), trunc_method)
    return InfiniteDisorderDensityMatrix(mpo_truncated.opp, ρs.ps)
end
