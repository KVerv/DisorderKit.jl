@testset "test time evolution operator at the clean limit" for _ in 1:10, steps in 0:2
    Omat = rand(ComplexF64, 2, 2)
    Omat = Omat + Omat'
    O = TensorMap(Omat, ℂ^2, ℂ^2)

    Δτ = 0.1

    distributions = [([3.0], [1.0]), 
                     ([2.9, 3.0, 3.1], [0.0, 1.0, 0.0]),
                     ([2.8, 2.9, 3.0, 3.1, 3.2], [0.0, 0.0, 1.0, 0.0, 0.0]),
                     ]

    trunc_method_Zmpo = StandardTruncation(trunc_method = truncerr(1e-10))
    inversion_alg = VOMPS_Inversion(1; tol = 1e-8, maxiter = 250, verbosity = 2)

    res = map(distributions) do (gs, ps)
        D_disorder = length(gs)
        T = RTFIM_time_evolution_Trotter(Δτ, gs, [1.])
        ρ = deepcopy(T)
        for _ in 1:steps
            ρ = ρ * T
        end
        ρn, Zinv = normalize_each_disorder_sector(ρ, trunc_method_Zmpo, inversion_alg)
        return measure(ρn, ps, O, 1)
    end

    @test all(imag.(res) .< 1e-12)
    @test all(res .≈ res[1])
end

@testset "test time evolution operator at the clean limit" for _ in 1:10, steps in 0:2
    Omat = rand(ComplexF64, 2, 2)
    Omat = Omat + Omat'
    O = TensorMap(Omat, ℂ^2, ℂ^2)

    Δτ = 0.1

    distributions = [([3.0], [1.0]), 
                     ([2.9, 3.0, 3.1], [0.0, 1.0, 0.0]),
                     ([2.8, 2.9, 3.0, 3.1, 3.2], [0.0, 0.0, 1.0, 0.0, 0.0]),
                     ]
    
    trunc_method_Zmpo = StandardTruncation(trunc_method = truncerr(1e-10))
    inversion_alg = VOMPS_Inversion(1; tol = 1e-8, maxiter = 250, verbosity = 2)

    res = map(distributions) do (gs, ps)
        D_disorder = length(gs)
        ρ =  RTFIM_time_evolution_Trotter(Δτ, gs, [1.])
        ρn, _ = normalize_each_disorder_sector(ρ, trunc_method_Zmpo, inversion_alg)
        return ρn_weighted = disorder_average(ρn, ps)
    end

    @test all(mpo_fidelity.(res, Ref(res[1])) .≈ 1)
end

@testset "test time evolution operator at the clean limit" begin
    Δτ = 0.1

    trunc_method_Zmpo = StandardTruncation(trunc_method = truncerr(1e-10))
    inversion_alg = VOMPS_Inversion(1; tol = 1e-8, maxiter = 250, verbosity = 2)

    Ta = let gs=[0.0], ps=[1.0]
        D_disorder = length(gs)
        T = RTFIM_time_evolution_Trotter(Δτ, gs, [1.])
        T = T * T
        ρn, _ = normalize_each_disorder_sector(T, trunc_method_Zmpo, inversion_alg)
        ρn_weighted = disorder_average(ρn, ps)
    end
    
    Ta2 = let gs=[0.0], ps=[1.0]
        D_disorder = length(gs)
        T = RTFIM_time_evolution_Trotter(2*Δτ, gs, [1.])
        ρn, _ = normalize_each_disorder_sector(T, trunc_method_Zmpo, inversion_alg)
        ρn_weighted = disorder_average(ρn, ps)
    end
    
    Tb = let gs=[-0.1, 0.0, 0.1], ps=[0.0, 1.0, 0.0]
        D_disorder = length(gs)
        T = RTFIM_time_evolution_Trotter(Δτ, gs, [1.])
        T = T * T
        ρn, _ = normalize_each_disorder_sector(T, trunc_method_Zmpo, inversion_alg)
        ρn_weighted = disorder_average(ρn, ps)
    end;
    
    Tb2 = let gs=[-0.1, 0.0, 0.1], ps=[0.0, 1.0, 0.0]
        D_disorder = length(gs)
        T = RTFIM_time_evolution_Trotter(2*Δτ, gs, [1.])
        ρn, _ = normalize_each_disorder_sector(T, trunc_method_Zmpo, inversion_alg)
        ρn_weighted = disorder_average(ρn, ps)
    end;

    @test mpo_fidelity(Ta, Ta2) ≈ 1
    @test mpo_fidelity(Tb, Tb2) ≈ 1
    @test mpo_fidelity(Ta2, Tb2) ≈ 1
    @test mpo_fidelity(Ta, Tb) ≈ 1
end

@testset "test iTEBD at the clean limit: compare disorder averaged density matrix at each step" begin
    Δτ = 0.1 

    trunc_method_Zmpo = StandardTruncation(trunc_method = truncerr(1e-10))
    inversion_alg = VOMPS_Inversion(1; tol = 1e-8, maxiter = 10, verbosity = 2)
    trunc_method_disordermpo =  DisorderTracedTruncation(trunc_method = truncerr(1e-10))

    ρns1 = let gs = [3.0], ps = [1.0]
        T = RTFIM_time_evolution_Trotter(Δτ, gs, [1.])
        ρ = T
        ρns = map(1:100) do ix
            ρ = evolve_one_time_step(ρ, T)
            ρ, _ = normalize_each_disorder_sector(ρ, trunc_method_Zmpo, inversion_alg)
            ρ = truncate_mpo(ρ, ps, trunc_method_disordermpo)

            ρn = disorder_average(ρ, ps)
        end
    end;

    ρns2 = let gs = [2.9, 3.0, 3.1], ps = [0.0, 1.0, 0.0]
        T = RTFIM_time_evolution_Trotter(Δτ, gs, [1.])
        ρ = T
        ρns = map(1:100) do ix
            ρ = evolve_one_time_step(ρ, T)
            ρ, _ = normalize_each_disorder_sector(ρ, trunc_method_Zmpo, inversion_alg)
            ρ = truncate_mpo(ρ, ps, trunc_method_disordermpo)

            ρn = disorder_average(ρ, ps)
        end
    end;

    @test all(mpo_fidelity.(ρns1, ρns2) .> 1 - 1e-12)
    
    get_bonddim(x) = dim(space(x[1], 1))
    dims1 = get_bonddim.(ρns1)
    dims2 = get_bonddim.(ρns2)
    @test all(dims1 .≈ dims2)
end
