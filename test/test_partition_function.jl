@testset "test partition function at the clean limit" for Δβ in 0.1:0.1:1.0
    distributions = [([3.0], [1.0]), 
                     ([2.9, 3.0, 3.1], [0.0, 1.0, 0.0]),
                     ([2.8, 2.9, 3.0, 3.1, 3.2], [0.0, 0.0, 1.0, 0.0, 0.0]),
                     ]

    res = map(distributions) do (hs, ps)
        D_disorder = length(hs)
        Hs = DisorderKit.random_transverse_field_ising([1.0], hs)
        Us = DisorderKit.time_evolution_MPO(Hs, Δβ/2)
        ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix(Us.opp, ps)
        ρ = DisorderKit.gauge!(ρ₀)
        ρ_product = DisorderKit.renormalize(ρ; χ=1, N=1)
        es = DisorderKit.entanglement_spectrum_norm(ρ_product)
        ϵ = sum(es[1:end-1])
        return ϵ
    end

    @test all(imag.(res) .< 1e-12)
    @test all(res .< 1e-8)
end

@testset "test partition function at the weak limit" for Δβ in 0.1:0.1:1.0, p in 0.1:0.1:0.5
    distributions = [([2.8, 2.9], [p, 1-p]),([2.9, 3.0, 3.1], [p/2, 1-p, p/2]),
                     ]

    res = map(distributions) do (hs, ps)
        D_disorder = length(hs)
        Hs = DisorderKit.random_transverse_field_ising([1.0], hs)
        Us = DisorderKit.time_evolution_MPO(Hs, Δβ/2)
        ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix(Us.opp, ps)
        ρ = DisorderKit.gauge!(ρ₀)
        ρ_product = DisorderKit.renormalize(ρ; χ=1, N=1)
        es = DisorderKit.entanglement_spectrum_norm(ρ_product)
        ϵ = sum(es[1:end-1])
        return ϵ
    end

    @test all(imag.(res) .< 1e-12)
    @test all(res .< 1e-8)
end

@testset "test partition function at the strong limit: $(Δβ)" for Δβ in 0.1:0.1:1.0
    distributions = [[0.8, 0.9],[0.9, 1.0, 1.1], [1.0, 1.3, 1.7]
                     ]

    res, res0 = map(distributions) do (hs)
        D_disorder = length(hs)^2
        ps = ones(D_disorder)./D_disorder
        Hs = DisorderKit.random_transverse_field_ising(hs, hs)
        Us = DisorderKit.time_evolution_MPO(Hs, Δβ/2)
        ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix(Us.opp, ps)
        ρ = DisorderKit.gauge!(ρ₀)
        ρ_product = DisorderKit.renormalize(ρ; χ=1, N=1)
        es = DisorderKit.entanglement_spectrum_norm(ρ_product)
        ϵ = sum(es[1:end-1])
        es0 = DisorderKit.entanglement_spectrum_norm(ρ)
        ϵ_0 = sum(es0[1:end-1])
        @show (ϵ, ϵ_0)
        return ϵ, ϵ_0
    end

    @test all(imag.(res) .< 1e-12)
    @test all(res .< 1e-8)
    @test all(res .< res0)
end