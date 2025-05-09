@testset "ρ_transfer" for ix in 1:10

    function ρtransfer(O,A)
        iso = isomorphism(fuse(space(O)[2],space(O)[3]),space(O)[2] ⊗ space(O)[3])
        @tensor tl[-1 -2; -3 -4] := conj(O[-3 2; 1 -1])*conj(A[-4 3; -2])*iso[3; 1 2]
        @tensor tr[-1 -2; -3 -4] := conj(O[-1 2; 1 -3])*conj(A[-2 3; -4])*iso[3; 1 2]
        return tl, tr 
    end

    n = 5
    d = 2
    D = 3
    O = TensorMap(rand,ComplexF64,ℂ^D⊗ℂ^n,ℂ^n⊗ℂ^D)
    A = TensorMap(rand,ComplexF64,ℂ^d⊗ℂ^(n^2),ℂ^d)
    
    vl = Tensor(rand,ComplexF64,ℂ^D⊗ℂ^d)
    vr = Tensor(rand,ComplexF64,(ℂ^D)'⊗(ℂ^d)')

    tl , tr = ρtransfer(O,A)

    @show space(tl*vl)
    @show space(DisorderKit.ρ_transfer_left(A, O)(vl))
    @test norm(tl*vl - DisorderKit.ρ_transfer_left(A, O)(vl)) < 1e-8
    @test norm(tr*vr - DisorderKit.ρ_transfer_right(A, O)(vr)) < 1e-8
end

@testset "E_transfer" for ix in 1:10

    function Etransfer(O,A)
        iso = isomorphism(fuse(space(O)[2],space(O)[3]),space(O)[2] ⊗ space(O)[3])
        @tensor tl[-1 -2 -3 -4; -5 -6 -7 -8] := A[-5 1; -1] * conj(iso[1; 2 3]) * O[-6 4; 2 -2] * conj(O[-7 4; 5 -3])*conj(A[-8 6; -4])*iso[6; 5 3]
        @tensor tr[-1 -2 -3 -4; -5 -6 -7 -8] := A[-1 1; -5] * conj(iso[1; 2 3]) * O[-2 4; 2 -6] * conj(O[-3 4; 5 -7])*conj(A[-4 6; -8])*iso[6; 5 3]
        return tl, tr 
    end

    n = 5
    d = 2
    D = 3
    O = TensorMap(rand,ComplexF64,ℂ^D⊗ℂ^n,ℂ^n⊗ℂ^D)
    A = TensorMap(rand,ComplexF64,ℂ^d⊗ℂ^(n^2),ℂ^d)
    
    vl = Tensor(rand,ComplexF64,(ℂ^d)'⊗(ℂ^D)'⊗ℂ^D⊗ℂ^d)
    vr = Tensor(rand,ComplexF64,ℂ^d⊗ℂ^D⊗(ℂ^D)'⊗(ℂ^d)')

    tl , tr = Etransfer(O,A)


    @test norm(tl*vl - DisorderKit.E_transfer_left(A, O)(vl)) < 1e-8
    @test norm(tr*vr - DisorderKit.E_transfer_right(A, O)(vr)) < 1e-8
end

@testset "AC_system" for ix in 1:10
    d = 3
    D = 4
    p = 2

    El = Tensor(rand, ComplexF64, (ℂ^d)'⊗(ℂ^D)'⊗ℂ^D⊗ℂ^d)
    Er = Tensor(rand, ComplexF64, ℂ^d⊗ℂ^D⊗(ℂ^D)'⊗(ℂ^d)')
    ρl = Tensor(rand, ComplexF64, ℂ^D⊗ℂ^d)
    ρr = Tensor(rand, ComplexF64, (ℂ^D)'⊗(ℂ^d)')
    O = TensorMap(rand,  ComplexF64, ℂ^D⊗ℂ^p,ℂ^p⊗ℂ^D)
    A = TensorMap(rand, ComplexF64,ℂ^d⊗ℂ^(p^2),ℂ^d)

    λρ = ρl[1]
    λE = El[1]
    b, fmap = DisorderKit.AC_system(A, O, λρ, ρl, ρr, λE, El, Er)

    iso = isomorphism(fuse(space(O)[2],space(O)[3]),space(O)[2] ⊗ space(O)[3])
    @tensor f[-1 -2 -3; -4 -5 -6] := El[-4 1 2 -1]*conj(O[2 7; 8 4])*O[1 7; 5 3]*Er[-6 3 4 -3]*iso[-2; 8 9]*conj(iso[-5; 5 9])
    @tensor NE[] := El[1 2 3 4]*DisorderKit.E_transfer_right(A,O)(Er)[1 2 3 4]
    @tensor Nρ[] := ρl[1 2] * DisorderKit.ρ_transfer_right(A,O)(ρr)[1 2]
    @tensor y[-1 -2; -3] := f[-1 -2 -3; 1 2 3]*A[1 2; 3] 

    @test norm(fmap(A)-y*Nρ[1]/NE[1]) < 1e-8
end