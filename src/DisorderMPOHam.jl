# One site unit-cell DisorderMPOHam
struct DisorderMPOHam{T <: Number, S <: IndexSpace}
    A::AbstractTensorMap{T, S, 3, 3}
    L::AbstractTensorMap{T, S, 2, 3}
    R::AbstractTensorMap{T, S, 3, 2}
    D::AbstractTensorMap{T, S, 2, 2}
end

function DisorderMPOHam(A::AbstractTensorMap{T, S, 3, 3}, L::AbstractTensorMap{T, S, 2, 3}, R::AbstractTensorMap{T, S, 2, 3}, D::AbstractTensorMap{T, S, 2, 2}) where {T, S}
    #FIXME add some checks
    return DisorderMPOHam(A, L, R, D)
end

function random_transverse_field_ising(Js::Vector{Float64}, hs::Vector{Float64})
    N_disorder = length(Js)*length(hs)
    dspace = BlockTensorKit.boxplus(fill(ℂ^1, N_disorder)...)
    A = zeros(ComplexF64, BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(ℂ^2) ⊗ dspace, BlockTensorKit.boxplus(ℂ^2) ⊗ dspace ⊗ BlockTensorKit.boxplus(ℂ^1))
    L = zeros(ComplexF64, BlockTensorKit.boxplus(ℂ^2) ⊗ dspace, BlockTensorKit.boxplus(ℂ^2) ⊗ dspace ⊗ BlockTensorKit.boxplus(ℂ^1))
    R = zeros(ComplexF64, BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(ℂ^2) ⊗ dspace, BlockTensorKit.boxplus(ℂ^2) ⊗ dspace)
    D = zeros(ComplexF64, BlockTensorKit.boxplus(ℂ^2) ⊗ dspace, BlockTensorKit.boxplus(ℂ^2) ⊗ dspace)


    for (i, (h, J)) in enumerate(Iterators.product(hs, Js))
        @show (h,J)
        L[1, i, 1, i, 1] = TensorMap([(1. +0im) 0.; 0. -1.], ℂ^2 ⊗ ℂ^1, ℂ^2 ⊗ ℂ^1 ⊗ ℂ^1)
        R[1, 1, i, 1, i] = -J*TensorMap([(1. +0im) 0.; 0. -1.], ℂ^1 ⊗ ℂ^2 ⊗ ℂ^1, ℂ^2 ⊗ ℂ^1)
        D[1, i, 1, i] = -h*TensorMap([(0. +0im) 1.; 1. 0.], ℂ^2 ⊗ ℂ^1, ℂ^2 ⊗ ℂ^1)
    end

    return DisorderMPOHam(A, L, R, D)
end

function time_evolution_MPO(H::DisorderMPOHam, dt::Real; N::Int = 2)
    pspace = space(H.D, 1)
    N_disorder = dim(space(H.D, 2))
    vspace = space(H.A, 1)
    
    #FIXME UVspace hardcoded for now
    if N == 1
        Uvspace = BlockTensorKit.boxplus([ℂ^1, ℂ^1]...)
    elseif N == 2
        Uvspace = BlockTensorKit.boxplus([ℂ^1, ℂ^1, ℂ^1]...)
    end

    Ucodomain = BlockTensorKit.boxplus(Uvspace) ⊗ BlockTensorKit.boxplus(pspace) ⊗ space(H.D, 2)
    Udomain = BlockTensorKit.boxplus(pspace) ⊗ space(H.D, 2) ⊗ BlockTensorKit.boxplus(Uvspace)

    U = zeros(ComplexF64, Ucodomain, Udomain)


    Wcodomain = BlockTensorKit.boxplus([ℂ^1, vspace, ℂ^1]...) ⊗ BlockTensorKit.boxplus(pspace)
    Wdomain = BlockTensorKit.boxplus(pspace) ⊗ BlockTensorKit.boxplus([ℂ^1, vspace, ℂ^1]...)

    for i in 1:N_disorder
        W = SparseBlockTensorMap{AbstractTensorMap{ComplexF64, ComplexSpace, 2, 2}}(undef, Wcodomain, Wdomain)

        W[1, 1, 1, 1] = BraidingTensor{ComplexF64, ComplexSpace}(pspace[1], ℂ^1)
        W[1, 1, 1, 2] = TensorMap(H.L[1, i, 1, i, 1].data, ℂ^1 ⊗ pspace[1], pspace[1] ⊗ vspace[1])
        W[1, 1, 1, 3] = TensorMap(H.D[1, i, 1, i].data, ℂ^1 ⊗ pspace[1], pspace[1] ⊗ ℂ^1)
        W[2, 1, 1, 2] = TensorMap(H.A[1, 1, i, 1, i, 1].data, vspace[1] ⊗ pspace[1], pspace[1] ⊗ vspace[1])
        W[2, 1, 1, 3] = TensorMap(H.R[1, 1, i, 1, i].data, vspace[1] ⊗ pspace[1], pspace[1] ⊗ ℂ^1)
        W[3, 1, 1, 3] = BraidingTensor{ComplexF64, ComplexSpace}(pspace[1], ℂ^1)
        MPOham = MPOHamiltonian(PeriodicArray([MPSKit.JordanMPOTensor(W)]))
        Ui = MPSKit.make_time_mpo(MPOham, dt, TaylorCluster(;N=N, compression = true, extension = true); imaginary_evolution=true)
        
        for (q1, s1) in enumerate(space(Ui[1],1))
            for (q2, s2) in enumerate(space(Ui[1],4))
                U[q1,1,i,1,i,q2] = TensorMap(Ui[1][q1,1,1,q2].data, ℂ^1 ⊗ pspace[1] ⊗ ℂ^1, pspace[1] ⊗ ℂ^1 ⊗ ℂ^1)
            end
        end
    end

    return InfiniteDisorderMPO([U])
end