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
    for j = 1:unit_cell-1
        transfer_l = transfer_left_mpo(Os[i+1+j]) ∘ transfer_l
        transfer_r = transfer_right_mpo(Os[i-j]) ∘ transfer_r
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