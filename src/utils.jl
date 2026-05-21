function fidelity_left_transfer(ρs::InfiniteDisorderDensityMatrix, σs::InfiniteDisorderDensityMatrix)
    function ftransfer(vl)
        v = zeros(ComplexF64,space(ρs[1],1)⊗space(ρs[1],1)',space(σs[1],1)'⊗space(σs[1],1))
        for (p, ρ) in enumerate(ρs)
            @tensor vp[-1 -2; -3 -4] := conj(ρ[1 8; 2 -1]) * ρ[3 4; 2 -2] * σs[p][7 8; 6 -4] * conj(σs[p][5 4; 6 -3]) * vl[1 3; 5 7]
            v += ρs.ps[p]*vp
        end
        return v
    end
    return ftransfer
end

function fidelity_right_transfer(ρs::InfiniteDisorderDensityMatrix, σs::InfiniteDisorderDensityMatrix)
    function ftransfer(vr)
        v = zeros(ComplexF64,space(σs[1],1)'⊗space(σs[1],1),space(ρs[1],1)⊗space(ρs[1],1)')
        for (p, ρ) in enumerate(ρs)
            @tensor vp[-3 -4; -1 -2] := conj(ρ[-1 8; 2 1]) * ρ[-2 4; 2 3] * σs[p][-4 8; 6 7] * conj(σs[p][-3 4; 6 5]) * vr[5 7; 1 3]
            v += ρs.ps[p]*vp
        end
        return v
    end
    return ftransfer
end

function fidelity(ρ::InfiniteDisorderDensityMatrix, σ::InfiniteDisorderDensityMatrix)
    vspace1 = space(ρ[1],1)
    vspace2 = space(σ[1],1)

    ftABl = fidelity_left_transfer(ρ, σ)
    ftABr = fidelity_right_transfer(ρ, σ)
    vl0 = rand(ComplexF64, vspace1 ⊗ vspace1', vspace2' ⊗ vspace2)
    vr0 = rand(ComplexF64, vspace2' ⊗ vspace2, vspace1 ⊗ vspace1')
    vals, vrs = eigsolve(x->ftABr(x), vr0, 1, :LM)
    vals, vls = eigsolve(x->ftABl(x), vl0, 1, :LM)

    r = vrs[1]
    l = vls[1]

    ftAAl = fidelity_left_transfer(ρ, ρ)
    ftAAr = fidelity_right_transfer(ρ, ρ)
    vl0 = rand(ComplexF64, vspace1 ⊗ vspace1', vspace1' ⊗ vspace1)
    vr0 = rand(ComplexF64, vspace1' ⊗ vspace1, vspace1 ⊗ vspace1')
    vals, vrAs = eigsolve(x->ftAAr(x), vr0, 1, :LM)
    vals, vlAs = eigsolve(x->ftAAl(x), vl0, 1, :LM)

    rA = vrAs[1]
    lA = vlAs[1]


    ftBBl = fidelity_left_transfer(σ, σ)
    ftBBr = fidelity_right_transfer(σ, σ)
    vl0 = rand(ComplexF64, vspace2 ⊗ vspace2', vspace2' ⊗ vspace2)
    vr0 = rand(ComplexF64, vspace2' ⊗ vspace2, vspace2 ⊗ vspace2')
    vals, vrBs = eigsolve(x->ftBBr(x), vr0, 1, :LM)
    vals, vlBs = eigsolve(x->ftBBl(x), vl0, 1, :LM)

    rB = vrBs[1]
    lB = vlBs[1]

    @tensor trAB = l[1 2; 3 4] * ftABr(r)[3 4; 1 2]
    @tensor NAB = l[1 2; 3 4] * r[3 4; 1 2]
    @tensor trAA = lA[1 2; 3 4] * ftAAr(rA)[3 4; 1 2]
    @tensor NAA = lA[1 2; 3 4] * rA[3 4; 1 2]
    @tensor trBB = lB[1 2; 3 4] * ftBBr(rB)[3 4; 1 2]
    @tensor NBB = lB[1 2; 3 4] * rB[3 4; 1 2]

    F = (trAB / NAB) * conj(trAB / NAB)/ ((trAA/NAA) * (trBB/NBB))
    return F
end
