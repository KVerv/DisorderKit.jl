@testset "test mpo truncation" for L in 1:6
    function expZZ(τ::Real)
        σz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)
        M = σz ⊗ σz
        L, S, R = tsvd(exp(-τ * M), (1, 3), (2, 4), trunc=truncerr(1e-10))
        L = permute(L * sqrt(S), (1, ), (2, 3))
        R = permute(sqrt(S) * R, (1, 2), (3, ))
        @tensor T1[-1 -2; -3 -4] := L[-2; 1 -4] * R[-1 1 ; -3]
        @tensor T2[-1 -2; -3 -4] := R[-1 -2; 1] * L[1; -3 -4]
        return InfiniteMPO([T1, T2])
    end

    mpoT = expZZ(0.1)
    mpoT2 = expZZ(0.2)
    
    
    @test mpo_fidelity(mpoT*mpoT, mpoT2) ≈ 1
    
    alg = StandardTruncation(; trunc_method = truncerr(1e-9))
    mpoT2_b = truncate_mpo(mpoT*mpoT, alg)
    @test mpo_fidelity(mpoT2_b, mpoT2) ≈ 1
end

@testset "test mpo truncation" for _ in 1:10
    χs = [4, 6, 8, 10, 2]
    Ts = [TensorMap(rand, ComplexF64, ℂ^χs[ix]*ℂ^2, ℂ^2*ℂ^χs[ix % 5 + 1]) for ix in 1:5]
    
    T = InfiniteMPO(Ts)
    alg = StandardTruncation(; trunc_method = truncerr(1e-12))

    T1 = truncate_mpo(T, alg);
    @test 1 - mpo_fidelity(T, T1) < 1e-12
    
    alg = StandardTruncation(; trunc_method = truncerr(1e-2))
    T2 = truncate_mpo(T, alg);
    @test 1 - mpo_fidelity(T, T2) < 1e-2
end