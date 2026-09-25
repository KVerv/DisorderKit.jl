@testset "VOMPS: Diagonal MPO" for ix in 1:1, D in 1:1, N in 2:2
    vspace = BlockTensorKit.boxplus(ℂ^D)
    pspace = BlockTensorKit.boxplus(fill(ℂ^1, N)...)

    O = spzeros(ComplexF64, vspace ⊗ pspace, pspace ⊗ vspace)

    for i in 1:N
        O[1,i,i,1] = rand(ComplexF64, vspace ⊗ ℂ^1, ℂ^1 ⊗ vspace)
    end
    
    Oinv, ϵ = DisorderKit.invert_mpo(O, DisorderKit.VOMPS_Inversion(1; maxiter = 100, tol = 1e-12, verbosity = 5))
    @tensor Id[-1 -2; -3 -4] := id(vspace)[-1; -4] * id(pspace)[-2; -3]
    

    iso = isomorphism(fuse(space(O,1)⊗space(Oinv,1)), space(O,1)⊗space(Oinv,1))
    @tensor Omult[-1 -2; -3 -4] := iso[-1; 1 3]*O[1 2; -3 4] * Oinv[3 -2; 2 5] * conj(iso[-4; 4 5])
    fid = DisorderKit.mpo_fidelity(Omult, Id)
    @show fid
    @show abs(O[1,1,1,1].data[1])
    @show abs(Oinv[1,1,1,1].data[1])
end

