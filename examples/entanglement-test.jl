using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit
using BlockTensorKit, MatrixAlgebraKit, OptimKit, StatsBase, KrylovKit

const AbstractMPOTensor = AbstractTensorMap{T, S, 2, 2} where {T, S}

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

    ρl0 = rand(ComplexF64, Dl, Dl)
    ρr0 = rand(ComplexF64, Dr, Dr)

    _, ρls, infol = eigsolve(transfer_l, ρl0, 1, :LM)
    _, ρrs, infor = eigsolve(transfer_r, ρr0, 1, :LM)

    S = svd_vals((ρls[1] * ρrs[1]))
    es = S.data
    es /= sum(es)
    return es
end

N = 2
a = 0.7
b = 1.3

Js = Vector(a:(b-a)/(N-1):b)
hs = Vector(a:(b-a)/(N-1):b)
ps = ones(N^2)./N^2

β = 1.1

Eh = mean(hs.^2)
qbar = 1 + β^2/2*Eh
qs = 1 .+ β^2/2*hs.^2
Fh = hs.^2 .- Eh
FJ = Js .- mean(Js)
# ds = β^2*Fh/qbar
ds = qs/qbar .+ 0 * 1im
D = TensorMap(DiagonalTensorMap(vcat([ds; ds]...), ℂ^(length(ps))))
as = β * Js/qbar .+ 0 * 1im
A = TensorMap(DiagonalTensorMap(vcat([as; reverse(as)]...), ℂ^(length(ps))))

Wcodomain = BlockTensorKit.boxplus([ℂ^1, ℂ^1]...) ⊗ BlockTensorKit.boxplus( ℂ^(length(ps)))
Wdomain = BlockTensorKit.boxplus( ℂ^(length(ps))) ⊗ BlockTensorKit.boxplus([ℂ^1, ℂ^1]...)

W = SparseBlockTensorMap{AbstractTensorMap{ComplexF64, ComplexSpace, 2, 2}}(undef, Wcodomain, Wdomain)
W[1,1,1,1] = TensorMap(D.data, ℂ^1⊗ ℂ^(length(ps)),  ℂ^(length(ps))⊗ℂ^1)
W[2,1,1,2] = TensorMap(A.data, ℂ^1⊗ ℂ^(length(ps)),  ℂ^(length(ps))⊗ℂ^1)

Os = InfiniteMPO([W])
entanglement_spectrum(Os, 1)


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

    v0 = rand(ComplexF64, V2, V1)
    λs, _ = eigsolve(mpo_transf, v0, 1, :LM)
    return λs[1]
end

function mpo_fidelity(A1::InfiniteMPO, A2::InfiniteMPO)
    return norm(mpo_ovlp(A1, A2) / sqrt(mpo_ovlp(A1, A1) * mpo_ovlp(A2, A2)))
end