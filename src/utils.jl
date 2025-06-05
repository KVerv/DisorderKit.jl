function transfer_left_mpo(O::AbstractMPOTensor)
    function ftransfer(vl)
        @tensor vl[-1; -2] := O[2 4; 3 -2] * conj(O[1 4; 3 -1]) * vl[1; 2]
    end
    return ftransfer
end

function transfer_right_mpo(O::AbstractMPOTensor)
    function ftransfer(vr)
        @tensor vr[-1; -2] := O[-1 4; 3 1] * conj(O[-2 4; 3 2]) * vr[1; 2]
        return vr
    end
    return ftransfer
end

# Entanglement spectrum of MPO
function entanglement_spectrum(Os::InfiniteMPO, i::Int)
    unit_cell = length(Os)
    transfer_l = transfer_left_mpo(Os[i+1])
    transfer_r = transfer_right_mpo(Os[i])
    for j = i+2:i+unit_cell
        transfer_l = transfer_left_mpo(Os[j]) ∘ transfer_l
    end
    for j = i-1:-1:i-unit_cell+1
        transfer_r = transfer_right_mpo(Os[j]) ∘ transfer_r
    end

    Dl = space(Os[i+1], 1)
    Dr = space(Os[i+1], 1)

    ρl0 = TensorMap(rand, ComplexF64, Dl, Dl)
    ρr0 = TensorMap(rand, ComplexF64, Dr, Dr)  
    
    _, ρls, infol = eigsolve(transfer_l, ρl0, 1, :LM)
    _, ρrs, infor = eigsolve(transfer_r, ρr0, 1, :LM)

    _, S, _ = tsvd((ρls[1] * ρrs[1]))
    es = S.data
    es /= sum(es)
    return es
end

# Test if MPO is equal to the identity MPO
function test_identity(Os::InfiniteMPO)
    ϵs = zeros(Float64, length(Os))
    for i in eachindex(Os)
        es = entanglement_spectrum(Os, i)
        es = sort(es)
        ϵs[i] = abs(sum(es[1:end-1]))
    end
    return maximum(ϵs)
end

# Test if MPO is equal to the identity MPO
function test_identity_random(Os::InfiniteMPO; n_samples::Int = 5)
    ϵs = zeros(Float64, n_samples)
    for i in 1:n_samples
        pspaces = [space(Os[i])[2] for i in 1:length(Os)]
        vspaces = [ℂ^3 for i in 1:length(Os)]
        ψ = InfiniteMPS(rand, ComplexF64, pspaces, vspaces)
        ϵs[i] = abs(1-norm(dot(ψ, Os*ψ)))
    end
    return maximum(ϵs)
end

# MPO environments used in truncation
function env_left(Os::InfiniteMPO, ix::Int)
    v1 = TensorMap(rand, ComplexF64, space(Os[ix+1], 1), space(Os[ix+1], 1))
    transfer_left = transfer_left_mpo(Os[ix+1])
    for jx in ix+2:1:ix+length(Os)
        transfer_left = transfer_left_mpo(Os[jx]) ∘ transfer_left
    end
    _, ls = eigsolve(v -> transfer_left(v), v1, 1, :LM);
    return ls[1]
end

function env_right(Os::InfiniteMPO, ix::Int)
    v1 = TensorMap(rand, ComplexF64, space(Os[ix + 1], 1), space(Os[ix + 1], 1))
    transfer_right = transfer_right_mpo(Os[ix])
    for jx in ix-1:-1:ix-length(Os)+1
        transfer_right = transfer_right_mpo(Os[jx]) ∘ transfer_right
    end
    _, ls = eigsolve(v -> transfer_right(v), v1, 1, :LM);
    return ls[1]
end

function mpo_ovlp(A1::InfiniteMPO, A2::InfiniteMPO)
    V1 = space(A1[1], 1)
    V2 = space(A2[1], 1)

    function mpo_transf(v)
        for (M1, M2) in zip(A1, A2)
            @tensor Tv[-1; -2] := M1[1 3; 4 -2] * conj(M2[2 3; 4 -1]) * v[2; 1]
            v = Tv
        end
        return v
    end

    v0 = TensorMap(rand, ComplexF64, V2, V1)
    λs, _ = eigsolve(mpo_transf, v0, 1, :LM)
    return λs[1]
end

function mpo_fidelity(A1::InfiniteMPO, A2::InfiniteMPO)
    return norm(mpo_ovlp(A1, A2) / sqrt(mpo_ovlp(A1, A1) * mpo_ovlp(A2, A2)))
end

function mpo_ovlp(ρ1::DisorderMPO, ρ2::DisorderMPO)
    V1 = space(ρ1[1], 1)
    V2 = space(ρ2[1], 1)

    function mpo_transf(v)
        for (M1, M2) in zip(ρ1, ρ2)
            @tensor Tv[-1; -2] := M1[1 3 5; 4 6 -2] * conj(M2[2 3 5; 4 6 -1]) * v[2; 1]
            v = Tv
        end
        return v
    end

    v0 = TensorMap(rand, ComplexF64, V2, V1)
    λs, _ = eigsolve(mpo_transf, v0, 1, :LM)
    return λs[1]
end

function mpo_fidelity(A1::DisorderMPO, A2::DisorderMPO)
    return norm(mpo_ovlp(A1, A2) * mpo_ovlp(A2, A1) / mpo_ovlp(A2, A2))
end