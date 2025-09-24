@testset "DisorderMPS Left Gauge" for ix in 1:10
    D = 3
    D_dis = 5
    D_phys = 2
    L = 4

    As = [[TensorMap(rand, ComplexF64, ℂ^D⊗ℂ^D_phys,ℂ^D) for i in 1:D_dis] for j in 1:L]
    ρ = FiniteDisorderMPS(As)
    ρ = DisorderKit.left_gauge(ρ)

    for A in ρ
        for Ap in A
            @tensor V[-1; -2] := Ap[1 2; -1] * conj(Ap[1 2; -2])
            Id = id(ComplexF64, space(Ap,3))
            @test norm(V - Id) < 1e-6
        end
    end
    @test (overlap(ρ) - D*D_dis^L) <1e-12
end
