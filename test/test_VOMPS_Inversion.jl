@testset "test invert_mpo with VOMPS: random trivial bond dimension MPO" for ix in 1:10
    d = 2
    alg = VOMPS_Inversion(1; tol=1e-8, maxiter=20, verbose=true)
    for unit_cell = 1:5
        Os = [TensorMap(rand, ComplexF64, ℂ^1 ⊗ ℂ^d, ℂ^d ⊗ ℂ^1) for i in 1:unit_cell]

        mpo = InfiniteMPO(Os)
        Oinv, ϵ = invert_mpo(mpo, alg; init_guess = nothing)

        for i in 1:unit_cell
            O_times_Oinv = reshape(Os[i].data,(2,2)) * reshape(Oinv[i].data, (2,2))
            @test norm(O_times_Oinv - O_times_Oinv[1, 1] * Matrix{ComplexF64}(I, d, d)) < 1e-8
        end
    end
end

@testset "test invert_mpo with VOMPS: random trivial physical dimension MPO" for ix in 1:10
    d = 3
    alg = VOMPS_Inversion(1; tol=1e-8, maxiter=20, verbose=true)
    for unit_cell = 1:5
        Os = [TensorMap(rand, ComplexF64, ℂ^d ⊗ ℂ^1, ℂ^1 ⊗ ℂ^d) for i in 1:unit_cell]

        mpo = InfiniteMPO(Os)
        Oinv, ϵ = invert_mpo(mpo, alg; init_guess = nothing)

        for i in 1:unit_cell
            O_times_Oinv = mpo * Oinv
            ϵ = test_identity(O_times_Oinv)
            @test ϵ < 1e-8
        end
    end
end


@testset "test invert_mpo with VOMPS: exp(diagonal Hamiltonian) #1" for τ in (1:8) * 0.1
    #FIXME Does not work for τ>0.9 for inverse_dim = 2
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

    @show τ
    mpo = expZZ(τ)

    @show space(mpo[1])
    inverse_dim = 2
    alg = VOMPS_Inversion(inverse_dim; tol=1e-8, maxiter=500, verbose=true)
    
    Oinv, _ = invert_mpo(mpo, alg; init_guess = nothing)

    O_times_Oinv = mpo * Oinv
    mps = DisorderKit.transform_to_mps(mpo * Oinv)
    ϵ = test_identity(O_times_Oinv)
    @show τ
    @test ϵ < 1e-8
end