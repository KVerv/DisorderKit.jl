# One site unit-cell DisorderMPOHam
struct DisorderMPOHam{T <: Number, S <: IndexSpace}
    As::Vector{<:AbstractTensorMap{T, S, 2, 2}}
    Bs::Vector{<:AbstractTensorMap{T, S, 2, 1}}
    Cs::Vector{<:AbstractTensorMap{T, S, 1, 2}}
    Ds::Vector{<:AbstractTensorMap{T, S, 1, 1}}
end

function DisorderMPOHam(As::Vector{<:AbstractTensorMap{T, S, 2, 2}},
                     Bs::Vector{<:AbstractTensorMap{T, S, 2, 1}},
                     Cs::Vector{<:AbstractTensorMap{T, S, 1, 2}},
                     Ds::Vector{<:AbstractTensorMap{T, S, 1, 1}}) where {T, S}
    length(As) == length(Bs) == length(Cs) == length(Ds) || error("All input vectors must have the same length")
    return DisorderMPOHam(As, Bs, Cs, Ds)
end

function random_transverse_field_ising(Js::Vector{Float64}, hs::Vector{Float64}; ϵ::Real=0.0)
    A = zeros(ComplexF64, ℂ^1⊗ℂ^2, ℂ^2⊗ℂ^1)
    B = TensorMap([(1. +0im) 0.; 0. -1.], ℂ^1⊗ℂ^2, ℂ^2)
    C =  TensorMap([(1. +0im) 0.; 0. -1.], ℂ^2, ℂ^2⊗ℂ^1)
    D =  TensorMap([0. (1. +0im); 1. 0.], ℂ^2, ℂ^2)
    Z = TensorMap([(1. +0im) 0.; 0. -1.], ℂ^2, ℂ^2)

    As = typeof(A)[]
    Bs = typeof(B)[]
    Cs = typeof(C)[]
    Ds = typeof(D)[]

    for (i, (h, J)) in enumerate(Iterators.product(hs, Js))
        @show (h,J)
        push!(As, A)
        push!(Bs, -J*B)
        push!(Cs, C)
        push!(Ds, -h*D - ϵ*Z)
    end
    return DisorderMPOHam(As, Bs, Cs, Ds)
end

function random_transverse_field_isingZ2(Js::Vector{Float64}, hs::Vector{Float64})
    V = Z2Space(0 => 1, 1 => 1)
    Vtriv = Z2Space(0 => 1)
    ZZ = TensorMap(zeros, ComplexF64, V ⊗ V ← V ⊗ V)
    flip_charge(charge::Z2Irrep) = only(charge ⊗ Z2Irrep(1))
    for (s, f) in fusiontrees(ZZ)
        if s.uncoupled == map(flip_charge, f.uncoupled)
            ZZ[s, f] .= 1
        end
    end
    X = TensorMap(zeros, ComplexF64, V ← V)
    for (s, f) in fusiontrees(X)
        if only(f.uncoupled) == Z2Irrep(0)
            X[s, f] .= 1
        else
            X[s, f] .= -1
        end
    end

    U, S, VV = tsvd(ZZ, ((1,3),(2,4)))
    C = U*S
    C = permute(C, ((1,),(2,3)))
    B = VV
    B = permute(B, ((1,2),(3,)))
    A = TensorMap(zeros, ComplexF64, Vtriv ⊗ V ← V ⊗ Vtriv)
    D = X

    As = typeof(A)[]
    Bs = typeof(B)[]
    Cs = typeof(C)[]
    Ds = typeof(D)[]

    for (i, (h, J)) in enumerate(Iterators.product(hs, Js))
        @show (h,J)
        push!(As, A)
        push!(Bs, B)
        push!(Cs, -J*C)
        push!(Ds, -h*X)
    end
    return DisorderMPOHam(As, Bs, Cs, Ds)
end

function random_heisenberg(Js::Vector{Float64})
    X = TensorMap([0. 1.; 1. 0.], ℂ^2, ℂ^2)
    Y = TensorMap([0. -1.0im; 1.0im 0.], ℂ^2, ℂ^2)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)

    A = zeros(ComplexF64, ℂ^1⊗ℂ^2, ℂ^2⊗ℂ^1)

    BC = X⊗X + Y⊗Y + Z⊗Z

    U, S, V = tsvd(BC, ((1,3),(2,4)))
    C = U*S
    C = permute(C, ((1,),(2,3)))
    B = V
    B = permute(B, ((1,2),(3,)))

    D =  zeros(ComplexF64, ℂ^2, ℂ^2)

    As = typeof(A)[]
    Bs = typeof(B)[]
    Cs = typeof(C)[]
    Ds = typeof(D)[]

    for (i, J) in enumerate(Js)
        @show (J)
        push!(As, A)
        push!(Bs, -J*B)
        push!(Cs, C)
        push!(Ds, D)
    end

    return DisorderMPOHam(As, Bs, Cs, Ds)
end


function time_evolution_MPO(Hs::DisorderMPOHam, dt::Real; N::Int = 1)
    Us = Vector{typeof(Hs.As[1])}(undef, length(Hs.Bs))
    MPOHAM = []
    for (p, B) in enumerate(Hs.Bs)
        A = Hs.As[p]
        C = Hs.Cs[p]
        D = Hs.Ds[p]
        Wcodomain = BlockTensorKit.boxplus(ℂ^2) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, 3)...)
        Wdomain = BlockTensorKit.boxplus(fill(ℂ^1, 3)...) ⊗ BlockTensorKit.boxplus(ℂ^2)
        W = SparseBlockTensorMap{AbstractTensorMap{ComplexF64, ComplexSpace, 2, 2}}(undef, Wdomain, Wcodomain)

        W[1, 1, 1, 1] = BraidingTensor{ComplexF64, ComplexSpace}(ℂ^2, ℂ^1)
        W[1, 1, 1, 2] = BlockTensorKit.insertrightunit(C, 0)
        W[1, 1, 1, 3] = MPSKit.add_util_leg(D)
        W[2, 1, 1, 2] = A
        W[2, 1, 1, 3] = BlockTensorKit.insertrightunit(B, 3)
        W[3, 1, 1, 3] = BraidingTensor{ComplexF64, ComplexSpace}(ℂ^2, ℂ^1)
        MPOham = MPOHamiltonian(PeriodicArray([MPSKit.JordanMPOTensor(W)]))
        Up = MPSKit.make_time_mpo(MPOham, dt, TaylorCluster(;N=N, compression = true, extension = true); imaginary_evolution=true)
        Us[p] = Up[1]
        push!(MPOHAM, Up)
    end
    return DisorderMPO(Us)#, MPOHAM
end