@testset "test invert_mpo with VOMPS: random trivial bond dimension MPO" for ix in 1:10
    d = 2
    alg = VOMPS_Inversion(1; tol=1e-8, maxiter=20, verbosity =1)
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
    alg = VOMPS_Inversion(1; tol=1e-8, maxiter=20, verbosity = 1)
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


@testset "test invert_mpo with VOMPS: exp(diagonal Hamiltonian) #1" for τ in (1:11) * 0.1
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
    t = 0.01
    @show τ,t
    mpo = expZZ(τ)
    mpo_inv = expZZ(-τ)
    
    # initial guess
    A = TensorMap(randn, ComplexF64, space(mpo_inv[2]))
    B = TensorMap(randn, ComplexF64, space(mpo_inv[1]))
    T1 = t*A + mpo_inv[2]
    T2 = t*B + mpo_inv[1]

    inverse_dim = 2
    alg = VOMPS_Inversion(inverse_dim; tol=1e-8, maxiter=200, verbosity = 1)
    
    Oinv, _ = invert_mpo(mpo, alg; init_guess = InfiniteMPO([T1, T2]))

    O_times_Oinv = mpo * Oinv
    ϵ = test_identity(O_times_Oinv)
    @show τ, t
    @test ϵ < 1e-8
end


@testset "test invert_mpo with VOMPS: trotter gates (Ising)" for τ in 0.1:0.1:0.7, t in 0:0.01:0.02
    function gen_expZZX(τ::Real)
        Id = TensorMap(ComplexF64[1 0; 0 1], ℂ^2, ℂ^2)
        σz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)
        σx = TensorMap(ComplexF64[0 1; 1 0], ℂ^2, ℂ^2)
        M = σz ⊗ σz + 1/2*σx⊗Id +1/2*Id⊗σx
        L, S, R = tsvd(exp(-τ * M), (1, 3), (2, 4), trunc=truncerr(1e-10))
        L = permute(L * sqrt(S), (1, ), (2, 3))
        R = permute(sqrt(S) * R, (1, 2), (3, ))
        @tensor T1[-1 -2; -3 -4] := L[-2; 1 -4] * R[-1 1 ; -3]
        @tensor T2[-1 -2; -3 -4] := R[-1 -2; 1] * L[1; -3 -4]
        return InfiniteMPO([T1, T2])
    end

    @show τ, t
    mpo = gen_expZZX(τ)
    mpo_inv = gen_expZZX(-τ)
    
    # initial guess
    A = TensorMap(randn, ComplexF64, space(mpo_inv[2]))
    B = TensorMap(randn, ComplexF64, space(mpo_inv[1]))
    T1 = t*A + mpo_inv[2]
    T2 = t*B + mpo_inv[1]

    maxiter = 200
    inverse_dim = dim(space(mpo_inv[1])[1])
    alg = VOMPS_Inversion(inverse_dim; tol=1e-8, maxiter=maxiter, verbosity=1)
    
    Oinv, ϵ = invert_mpo(mpo, alg; init_guess = InfiniteMPO([T1, T2]))
    @test ϵ < 1e-8
    
    O_times_Oinv = mpo * Oinv
    mps = DisorderKit.transform_to_mps(mpo * Oinv)
    ϵ_acc = test_identity(O_times_Oinv)
    @show τ, t
    @test ϵ_acc < 1e-8
end