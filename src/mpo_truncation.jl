# Compute truncation matrices
function truncation_matrices(M::InfiniteMPO, trunc_method::TruncationScheme)
    L = length(M)

    envLs = map(ix -> env_left(M, ix), 1:L)
    envRs = map(ix -> env_right(M, ix), 1:L)

    Xs = map(envLs) do ρL
        _, SL, VL = tsvd(ρL; trunc=truncerr(1e-12));
        X = sqrt(SL) * VL
        Xinv = VL' * inv(sqrt(SL))
        return (X, Xinv)
    end

    Ys = map(envRs) do ρR
        UR, SR, _ = tsvd(ρR; trunc=truncerr(1e-12));
        Y = UR * sqrt(SR)
        Yinv = inv(sqrt(SR)) * UR'
        return (Y, Yinv)
    end

    truncations = map(1:L) do ix
        X, Xinv = Xs[ix]
        Y, Yinv = Ys[ix]
        U, S, V = tsvd(X*Y; trunc=trunc_method)
        PL = sqrt(S) * V * Yinv
        PR = Xinv * U * sqrt(S)
        return (PL, PR)
    end

    return PeriodicVector(truncations)
end

# Truncate ordinary mpo with standard truncation algorithm
function truncate_mpo(mpo::InfiniteMPO, alg::StandardTruncation)
    truncations = truncation_matrices(mpo, alg.trunc_method)
    L = length(mpo)
    mpo_updated = map(1:L) do ix
        PL = truncations[ix-1][1]
        PR = truncations[ix][2]
        @tensor O_updated[-1 -2 ; -3 -4] := PL[-1; 1] * mpo[ix][1 -2; -3 2] * PR[2; -4]
        return O_updated
    end
    return InfiniteMPO(mpo_updated)
end

# Truncate DisorderMPO by tracing disorder sectors
function truncate_mpo(ρ::DisorderMPO, ps::Vector{<:Real}, alg::DisorderTracedTruncation)
    @info(crayon"red"("Truncate DisorderMPO"))
    ρn_weighted = disorder_average(ρ, ps)
    
    truncations = truncation_matrices(ρn_weighted, alg.trunc_method)
    L = length(ρn_weighted)
    ρs_updated = map(1:L) do ix
        PL = truncations[ix-1][1]
        PR = truncations[ix][2]
        @tensor ρ1_updated[-1 -2 -3; -4 -5 -6] := PL[-1; 1] * ρ[ix][1 -2 -3; -4 -5 2] * PR[2; -6]
        return ρ1_updated
    end

    return DisorderMPO(ρs_updated)
end