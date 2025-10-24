@testset "DisorderMPS Left Gauge" for ix in 1:10
    D = 5
    D_dis = 3
    D_phys = 2
    L = 6

    # As = Vector{Vector{AbstractMPSTensor}}(undef, L)
    # As[1] = [TensorMap(rand, ComplexF64, ℂ^1⊗ℂ^D_phys,ℂ^D) for i in 1:D_dis]
    # As[2] = [TensorMap(rand, ComplexF64, ℂ^D_phys⊗ℂ^D_phys,ℂ^D) for i in 1:D_dis]
    # As[3:L-1] = [[TensorMap(rand, ComplexF64, ℂ^D⊗ℂ^D_phys,ℂ^D) for i in 1:D_dis] for j in 3:L-1]
    # As[L] = [TensorMap(rand, ComplexF64, ℂ^D⊗ℂ^D_phys,ℂ^1) for i in 1:D_dis]
    ρ = FiniteDisorderMPS(L, D_dis, D_phys, D; T=ComplexF64)
    ρ = DisorderKit.left_gauge(ρ)

    for A in ρ
        for Ap in A
            @tensor V[-1; -2] := Ap[1 2; -1] * conj(Ap[1 2; -2])
            Id = id(ComplexF64, space(Ap,3))
            @test norm(V - Id) < 1e-6
        end
    end
    @test abs(overlap(ρ) - 1) <1e-12
end
