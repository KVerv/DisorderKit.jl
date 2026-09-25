# function construct_renormalisation(ρ::InfiniteDisorderDensityMatrix, H::DisorderMPOHam, dβ::Real, alg::Groundstate_iDTEBD)
#     λ, l, r = DisorderKit.environments(ρ)
#     iso = isomorphism(fuse(space(ρ[1], 1)'⊗space(ρ[1], 1)), space(ρ[1], 1)'⊗space(ρ[1], 1))
#     isoχ = isomorphism(fuse(space(iso, 1), space(H.R, 1)), space(iso, 1) ⊗ space(H.R, 1))

#     @timeit alg.timer_output "Compute operators in Renormalization" begin
#         @tensor Ql[-1;-2 -3] := l[1; 5] * ρ[1][5 2 -1; 3 4 7] * conj(ρ[1][1 2 -2; 3 4 6]) * conj(iso[-3; 6 7])
#         @tensor Qr[-1 -2; -3] := iso[-1; 1 5] * ρ[1][5 2 -2; 3 4 7] * conj(ρ[1][1 2 -3; 3 4 6]) * r[7; 6]
#         @tensor D[-1 -2; -3 -4] := iso[-1; 1 5] * ρ[1][5 8 9; 3 4 7] * conj(ρ[1][1 2 -3; 3 4 6]) * conj(iso[-4; 6 7]) * H.D[2 -2; 8 9]
#         @tensor L[-1 -2; -3 -4] := iso[-1; 1 5] * ρ[1][5 8 9; 3 4 7] * conj(ρ[1][1 2 -3; 3 4 6]) * conj(iso[10; 6 7]) * H.L[2 -2; 8 9 11] * conj(isoχ[-4; 10 11])
#         @tensor R[-1 -2; -3 -4] := isoχ[-1; 10 11] * iso[10; 1 5] * ρ[1][5 8 9; 3 4 7] * conj(ρ[1][1 2 -3; 3 4 6]) * conj(iso[-4; 6 7]) * H.R[11 2 -2; 8 9]
#     end

#     vspace = space(Qr, 1)
#     vχspace = space(R, 1)

#     dspace = space(D, 2)
#     Wcodomain = BlockTensorKit.boxplus([ℂ^1, vspace, vχspace, vspace]...) ⊗ BlockTensorKit.boxplus(dspace)
#     Wdomain = BlockTensorKit.boxplus(dspace) ⊗  BlockTensorKit.boxplus([ℂ^1, vspace, vχspace, vspace]...)
#     W = SparseBlockTensorMap{AbstractTensorMap{ComplexF64, ComplexSpace, 2, 2}}(undef, Wcodomain, Wdomain)

#     @timeit alg.timer_output "Allocate W" begin
#         for i in 1:length(dspace)
#             W[1, i, i, 1] = BraidingTensor{ComplexF64, ComplexSpace}(dspace[1], ℂ^1)
#             W[1, i, i, 2] = TensorMap(Ql[i, i, 1].data, ℂ^1 ⊗ dspace[1], dspace[1] ⊗ vspace[1])
#             W[2, i, i, 4] = TensorMap(D[1, i, i, 1].data, vspace[1] ⊗ dspace[1], dspace[1] ⊗ vspace[1])
#             W[2, i, i, 3] = TensorMap(L[1, i, i, 1].data, vspace[1] ⊗ dspace[1], dspace[1] ⊗ vχspace[1])
#             W[3, i, i, 4] = TensorMap(R[1, i, i, 1].data, vχspace[1] ⊗ dspace[1], dspace[1] ⊗ vspace[1])
#             W[4, i, i, 1] = TensorMap(dβ*Qr[1, i, i].data, vspace[1] ⊗ dspace[1], dspace[1] ⊗ ℂ^1)
#         end
#     end


#     @timeit alg.timer_output "Truncate Renormalization" Wtrunc, ϵ, _ = truncate_mpo(W, alg.trunc_method_norm; timer=alg.timer_output)
#     Idp = id(ComplexF64, space(ρ[1],2))
#     @tensor U[-1 -2 -3; -4 -5 -6] := Idp[-2; -4] * Wtrunc[-1 -3; -5 -6]

#     @timeit alg.timer_output "Equation Check" begin
#         @tensor EqA1[-1; -2] := l[1; 5] * ρ[1][5 2 -1; 3 4 7] * conj(ρ[1][1 2 -2; 3 4 6]) * r[7; 6]
#         ϵA1 = norm(EqA1 - id(ComplexF64, dspace))
#         @tensor EqA2[-1 -2; -3 -4] := l[1; 5] * ρ[1][5 2 -1; 3 4 7] * conj(ρ[1][1 2 -3; 3 4 6]) * ρ[1][7 8 -2; 9 10 11] * conj(ρ[1][6 8 -4; 9 10 12]) * r[11; 12]
#         Idd = id(ComplexF64, dspace ⊗ dspace)
#         ϵA2 = norm(EqA2 - Idd)
#     end
#     return InfiniteDisorderMPO([U]), ϵ, ϵA1, ϵA2
# end

# function construct_renormalisation(ρ::InfiniteDisorderMPS, H::DisorderMPOHam, dβ::Real, alg::Groundstate_iDTEBD)
#     λ, l, r = DisorderKit.environments(ρ)
#     A = ρ[1]
#     iso = isomorphism(fuse(space(A, 1)⊗space(A, 1)'), space(A, 1)⊗space(A, 1)')
#     isoχ = isomorphism(fuse(space(A, 1)⊗space(H.R,1)⊗space(A, 1)'), (space(A, 1)⊗space(H.R,1)⊗space(A, 1)'))

#     @timeit alg.timer_output "Compute operators in Renormalization" begin
#         @tensor Ql[-1;-2 -3] := l[4; 1] * A[1 2 3; -2 5] * conj(A[4 2 3; -1 6]) * conj(iso[-3; 5 6])
#         @tensor Qr[-1 -2; -3] := iso[-1; 5 6] * A[5 2 3; -3 1] * conj(A[6 2 3; -2 4]) * r[1; 4]
#         # @tensor Q[-1 -2; -3 -4] := iso[-1; 1 6] * A[1 2 3; -3 7] * conj(A[6 2 3; -2 8]) * conj(iso[-4; 7 8])
#         @tensor D[-1 -2; -3 -4] := iso[-1; 1 6] * A[1 2 3; -3 7] * conj(A[6 4 5; -2 8]) * conj(iso[-4; 7 8]) * H.D[4 5; 2 3]
#         @tensor L[-1 -2; -3 -4] := iso[-1; 1 6] * A[1 2 3; -3 7] * conj(A[6 4 5; -2 9]) * conj(isoχ[-4; 7 8 9]) * H.L[4 5; 2 3 8]
#         @tensor R[-1 -2; -3 -4] := isoχ[-1; 1 4 7] * A[1 2 3; -3 8] * conj(A[7 5 6; -2 9]) * conj(iso[-4; 8 9]) * H.R[4 5 6; 2 3]
#     end

#     vspace = space(Qr, 1)
#     vχspace = space(R, 1)

#     dspace = space(D, 2)
#     Wcodomain = BlockTensorKit.boxplus([ℂ^1, vspace, vχspace, vspace]...) ⊗ BlockTensorKit.boxplus(dspace)
#     Wdomain = BlockTensorKit.boxplus(dspace) ⊗  BlockTensorKit.boxplus([ℂ^1, vspace, vχspace, vspace]...)
#     W = SparseBlockTensorMap{AbstractTensorMap{ComplexF64, ComplexSpace, 2, 2}}(undef, Wcodomain, Wdomain)

#     @timeit alg.timer_output "Allocate W" begin
#         for i in 1:length(dspace)
#             W[1, i, i, 1] = BraidingTensor{ComplexF64, ComplexSpace}(dspace[1], ℂ^1)
#             W[1, i, i, 2] = TensorMap(Ql[i, i, 1].data, ℂ^1 ⊗ dspace[1], dspace[1] ⊗ vspace[1])
#             # W[2, i, i, 2] = TensorMap(Q[1, i, i, 1].data, vspace[1] ⊗ dspace[1], dspace[1] ⊗ vspace[1])
#             W[2, i, i, 4] = TensorMap(D[1, i, i, 1].data, vspace[1] ⊗ dspace[1], dspace[1] ⊗ vspace[1])
#             W[2, i, i, 3] = TensorMap(L[1, i, i, 1].data, vspace[1] ⊗ dspace[1], dspace[1] ⊗ vχspace[1])
#             W[3, i, i, 4] = TensorMap(R[1, i, i, 1].data, vχspace[1] ⊗ dspace[1], dspace[1] ⊗ vspace[1])
#             W[4, i, i, 1] = TensorMap(dβ*Qr[1, i, i].data, vspace[1] ⊗ dspace[1], dspace[1] ⊗ ℂ^1)
#             # W[4, i, i, 4] = TensorMap(Q[1, i, i, 1].data, vspace[1] ⊗ dspace[1], dspace[1] ⊗ vspace[1])
#         end
#     end


#     @timeit alg.timer_output "Truncate Renormalization" Wtrunc, ϵ, _ = truncate_mpo(W, alg.trunc_method_norm; timer=alg.timer_output)
#     Idp = id(ComplexF64, space(ρ[1],2))
#     @tensor U[-1 -2 -3; -4 -5 -6] := Idp[-2; -4] * Wtrunc[-1 -3; -5 -6]

#     return InfiniteDisorderMPO([U]), ϵ
# end

function construct_renormalisation(ρ::InfiniteDisorderMPS, H::DisorderMPOHam, dτ::Real, alg::Groundstate_iDTEBD)
    λ, l, r = DisorderKit.environments(ρ)
    A = ρ[1]
    iso = isomorphism(fuse(space(A, 1)⊗space(A, 1)'), space(A, 1)⊗space(A, 1)')
    # @tensor D[-1; -2] := l[2; 1] * A[1 3 4; -2 5] * conj(A[2 3 4; -1 6]) * r[5; 6]
    # @tensor T[-1 -2; -3 -4] := iso[-1; 1 2] * r[1; 2] * D[-2; -3] * l[3; 4] * conj(iso[-4; 4 3])
    @tensor Z[-1 -2; -3 -4] := iso[-1; 1 2] * A[1 3 4; -3 5] * conj(A[2 3 4; -2 6]) * conj(iso[-4; 5 6])
    vspace = space(Z, 1)
    dspace = space(Z, 2)

    # @tensor XD[-1; -2] := l[3; 2] * conj(iso[1; 2 3]) * X[1 -1; -2 4] * iso[4; 5 6] * r[5; 6]
    @tensor M[-1; -2] := Z[-1 1; 1 -2]
    M /= dim(dspace)
    @tensor T[-1 -2; -3 -4] := M[-1; -4] * id(ComplexF64, dspace)[-2; -3]
    # @tensor Tinv[-1 -2; -3 -4] := inv(M)[-1; -4] * id(ComplexF64, dspace)[-2; -3]

    X = Z - T

    @tensor XD[-1; -2] := X[-1 1; 1 -2]
    @show norm(XD)
    # @show norm(D)
    @show ε = norm(X)

    # Dinv = inv(D)
    # Dsqrtinv = sqrt(Dinv)


    # @tensor R[-1 -2; -3 -4] := X[-1 1; 2 -4] * Dinv[-2; 1] * Dsqrtinv[2; -3]
    # @tensor Q[-1 -2; -3 -4] := iso[-1; 1 2] * r[1; 2] * Dsqrtinv[-2; -3] * l[3; 4] * conj(iso[-4; 4 3])

    Wcodomain = BlockTensorKit.boxplus([vspace, vspace]...) ⊗ BlockTensorKit.boxplus(dspace)
    Wdomain = BlockTensorKit.boxplus(dspace) ⊗  BlockTensorKit.boxplus([vspace, vspace]...)
    W = SparseBlockTensorMap{AbstractTensorMap{ComplexF64, ComplexSpace, 2, 2}}(undef, Wcodomain, Wdomain)

    @timeit alg.timer_output "Allocate W" begin
        for i in 1:length(dspace)
            W[1, i, i, 1] = T[1,i,i,1] #TensorMap(T[1].data, vspace[1] ⊗ dspace[1], dspace[1] ⊗ vspace[1])
            W[1, i, i, 2] = -1/2*X[1, i, i, 1] #TensorMap(X.data, vspace[1] ⊗ dspace[1], dspace[1] ⊗ vspace[1])
            W[2, i, i, 2] = T[1, i, i, 1]
        end
    end

    @timeit alg.timer_output "Truncate Renormalization" Wtrunc, _, _ = truncate_mpo(W, alg.trunc_method_norm; timer=alg.timer_output)

    return Wtrunc, ε
end