function random_transverse_field_ising_evolution(Js::Vector{Float64}, hs::Vector{Float64}, dβ::Float64; order::Int = 1)
    alg = TaylorCluster(; N=order, extension=true, compression=true)
    D_disorder = length(Js) * length(hs)
    expHs = 0
    for (i, (h, J)) in enumerate(Iterators.product(hs, Js))
        @show i, h, J
        H = transverse_field_ising(; J = J, g = h)
        expH = make_time_mpo(H, -1im*dβ, alg)

        Us = map(expH) do U
            disordermap = DiagonalTensorMap(zeros(ComplexF64, D_disorder),ℂ^D_disorder)
            disordermap[i,i] = 1.0
            @tensor U_full[-1 -2 -3; -4 -5 -6] := U[-1 -2; -4 -6]*disordermap[-3; -5]
            return U_full
        end
        if i == 1
            expHs = Us
        else
            expHs += Us
        end
    end
    return DisorderMPO(TensorMap.(expHs))
end

function TFIM_time_evolution_with_disorder(Δτ::Real, gs::Vector{<:Real}, Js::Vector{<:Real}=[1.0])
    X, Z, Id = zeros(ComplexF64, 2, 2), zeros(ComplexF64, 2, 2), zeros(ComplexF64, 2, 2)
    X[1, 2], X[2, 1] = 1, 1
    Z[1, 1], Z[2, 2] = 1, -1
    Id[1, 1], Id[2, 2] = 1, 1

    D_disorder = length(Js) * length(gs)
    expHs = zeros(ComplexF64, D_disorder, 4, D_disorder, 4)
    for (i, (g, J)) in enumerate(Iterators.product(gs, Js))
        @show i, g, J
        expHs[i, :, i, :] = exp(-Δτ * (-J*kron(Z, Z) - g*kron(X, Id)))
    end

    expHs = reshape(expHs, D_disorder, 2,2, D_disorder, 2,2)
    expHs = TensorMap(expHs, ℂ^D_disorder*ℂ^2*ℂ^2, ℂ^D_disorder*ℂ^2*ℂ^2)

    L, S, R = tsvd(expHs, (1, 2, 4, 5), (3, 6), trunc=truncerr(1e-9))
    @show space(L), space(S), space(R)
    L = permute(L * sqrt(S), (1, 2), (3, 4, 5))
    R = permute(sqrt(S) * R, (1, 2), (3,) )
    
    @tensor T1[-1 -3 -2; -5 -4 -6] := L[-2 -3; -4 1 -6] * R[-1 1; -5]
    @tensor T2[-1 -3 -2; -5 -4 -6] := L[-2 1; -4 -5 -6] * R[-1 -3; 1]
    return DisorderMPO([T1, T2])
end
