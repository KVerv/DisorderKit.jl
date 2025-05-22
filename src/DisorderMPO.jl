# Convention virtual, physical, disorder ← physical, disorder, virtual

struct DisorderMPO{T<:AbstractDisorderMPOTensor}
    opp::PeriodicVector{T}
end

function DisorderMPO(opp::Vector{<:AbstractDisorderMPOTensor})
    return DisorderMPO(PeriodicVector(opp))
end

Base.getindex(T::DisorderMPO, ix::Int) = T.opp[ix]
Base.size(T::DisorderMPO) = size(T.opp)
Base.length(T::DisorderMPO) = length(T.opp)
Base.iterate(t::DisorderMPO, i=1) = (i > length(t.opp)) ? nothing : (t[i], i + 1)

# Rescale the disorder MPO by a vector of scalars
function rescale(ρs::DisorderMPO, αs::Vector{<:Number})
    ρ1 = deepcopy(ρs)
    for ix in 1:length(αs)
        ρ1.opp[ix] *= αs[ix]
    end
    return ρ1
end

# Multiply a disorder MPO with a DenseMPO on the disorder leg, used in the normalization of each disorder sector
function Base.:*(ρs::DisorderMPO, Os::InfiniteMPO)

    unit_cell = length(ρs)
    disordermpo = Vector{AbstractDisorderMPOTensor}(undef, length(ρs))
    for i = 1:unit_cell
        space(ρs[i], 3) == space(Os[i], 2) || throw(DimensionMismatch("Physical space of MPO should be match the disorder space of DisorderMPO."))
        iso1 = isomorphism(fuse(space(ρs[i], 1), space(Os[i], 1)), space(ρs[i], 1) ⊗ space(Os[i], 1))
        iso2 = isomorphism(space(ρs[i], 6)' ⊗ space(Os[i], 4)', fuse(space(ρs[i], 6), space(Os[i], 4)))
        @tensor ρs_product[-1 -2 -3; -4 -5 -6] := ρs[i][1 -2 2; -4 -5 4] * Os[i][3 -3; 2 5] * iso1[-1; 1 3] * iso2[4 5; -6]
        disordermpo[i] = ρs_product
    end

    return DisorderMPO(disordermpo)
end

# Multiply two DisorderMPOs, used in time evolution
function Base.:*(T1::DisorderMPO, T2::DisorderMPO)

    (length(T1) == length(T2)) || throw(ArgumentError("T1 T2 should have the same length"))

    Tprods = map(1:length(T1)) do ix
        spL1, spR1 = space(T1[ix], 1), space(T1[ix], 6)'
        spL2, spR2 = space(T2[ix], 1), space(T2[ix], 6)'
        isoL = isomorphism(fuse(spL1, spL2), spL1*spL2)
        isoR = isomorphism(spR1*spR2, fuse(spR1, spR2))
        @tensor Tprod[-1 -2 -3; -4 -5 -6] := isoL[-1; 2 1] * T1[ix][2 5 6; -4 -5 4] * T2[ix][1 -2 -3; 5 6 3] * isoR[4 3; -6]; 
        return Tprod
    end

    return DisorderMPO(Tprods)
end

# trace out the physical indices, represent the partition functions for different disorder sectors as an MPO
function partition_functions(ρs::DisorderMPO)

    Zs = map(ρs) do ρ
        @tensor Z[-1 -2; -3 -4] := ρ[-1 1 -2; 1 -3 -4]
        return Z
    end
    return InfiniteMPO(Zs)
end

# trace out the disorder indices with corresponding probabilities
function disorder_average(ρs::DisorderMPO, ps::Vector{<:Real})
    D_disorder = length(ps)
    P = TensorMap(diagm(ps), ℂ^D_disorder, ℂ^D_disorder)

    # weighted density matrix 
    ρ_weighted = map(ρs) do ρx
       @tensor ρx_weighted[-1 -2; -3 -4] := ρx[-1 -2 1;-3 2 -4] * P[2; 1];
       return ρx_weighted
    end
    return InfiniteMPO(ρ_weighted)
end

# measure the expectation value of a local operator on site i
function measure(ρ_weighted::InfiniteMPO, O::AbstractBondTensor, i::Int)
    TMs = map(ρ_weighted) do ρx
        @tensor TM[-1; -2] := ρx[-1 1; 1 -2]
        return TM
    end
    TMs = PeriodicArray(TMs)

    M = length(ρ_weighted)
    TMl = prod(map(ix->TMs[ix], i-M:i-1))
    TMr = prod(map(ix->TMs[ix], i+1:i+M))

    vl = Tensor(rand, ComplexF64, space(ρ_weighted[i], 1)')
    vr = Tensor(rand, ComplexF64, space(ρ_weighted[i], 4)')
    vl = permute(vl, ((), (1,)))

    _, vls = eigsolve(x->x*TMl, vl, 1, :LM)
    _, vrs = eigsolve(x->TMr*x, vr, 1, :LM)
    vl = vls[1]
    vr = vrs[1]

    @tensor O_res = vl[1] * ρ_weighted[i][1 3; 4 2] * O[4; 3] * vr[2]
    return O_res / (vl * TMs[i] * vr)[1]
end

# measure the correlator of a local operator on site i and site i+Δ    
function measure(ρ_weighted::InfiniteMPO, O1::AbstractBondTensor, O2::AbstractBondTensor, i1::Int, Δ::Int)
    TMs = map(ρ_weighted) do ρx
        @tensor TM[-1; -2] := ρx[-1 1; 1 -2]
        return TM
    end
    TMs = PeriodicArray(TMs)

    M = length(ρ_weighted)
    i2 = mod(i1 + Δ, M)

    TMl = prod(map(ix->TMs[ix], i1-M:i1-1))
    TMr = prod(map(ix->TMs[ix], i2+1:i2+M))

    vl = Tensor(rand, ComplexF64, space(ρ_weighted[i1], 1)')
    vr = Tensor(rand, ComplexF64, space(ρ_weighted[i2], 4)')
    vl = permute(vl, ((), (1,)))

    λs, vls = eigsolve(x->x*TMl, vl, 1, :LM)
    _, vrs = eigsolve(x->TMr*x, vr, 1, :LM)
    λ = λs[1]
    vl = vls[1]
    vr = vrs[1]

    TMc = prod(map(ix->TMs[ix], i1+1:i1+M))
    Λc, Vc = eigen(TMc)
    Uc = inv(Vc)
    center_range = i1+1:i1+mod(Δ-1, M)
    if length(center_range) > 0
        centers = prod(map(ix->TMs[ix], center_range))
        interval = Vc * (Λc / λ)^((Δ-1) ÷ M) * Uc * centers
    else
        interval = Vc * (Λc / λ)^((Δ-1) ÷ M) * Uc
    end

    @tensor TM_O1[-1; -2] := ρ_weighted[i1][-1 1; 2 -2] * O1[2; 1]
    @tensor TM_O2[-1; -2] := ρ_weighted[i2][-1 1; 2 -2] * O2[2; 1]

    O_res = (vl * TM_O1 * interval * TM_O2 * vr)[1] / (vl * TMs[i1] * interval * TMs[i2] * vr)[1]
    return O_res
end


# measure the disorder average of a local operator on site i
function measure(ρ::DisorderMPO, ps::Vector{<:Real}, O::AbstractBondTensor, i::Int)
    ρ_weighted = disorder_average(ρ, ps)
    O_res = measure(ρ_weighted, O, i)
    return O_res
end

# measure the disorder average of a correlator between a local operator on site i and site i+Δ
function measure(ρ::DisorderMPO, ps::Vector{<:Real}, O1::AbstractBondTensor, O2::AbstractBondTensor, i1::Int, Δ::Int)
    ρ_weighted = disorder_average(ρ, ps)
    O_res = measure(ρ_weighted, O1, O2, i1, Δ)
    return O_res
end

# measure the disorder average of a disorder operator on site i
function measure(ρ::DisorderMPO, ps::Vector{<:Real}, Os::DisorderMPO, i::Int)
    ρO_weighted = disorder_average(ρ*Os, ps)
    TMs = map(ρO_weighted) do ρx
        @tensor TM[-1; -2] := ρx[-1 1; 1 -2]
        return TM
    end
    TMs = PeriodicArray(TMs)

    M = length(ρO_weighted)
    TMl = prod(map(ix->TMs[ix], i-M:i-1))
    TMr = prod(map(ix->TMs[ix], i+1:i+M))

    vl = Tensor(rand, ComplexF64, space(ρO_weighted[i], 1)')
    vr = Tensor(rand, ComplexF64, space(ρO_weighted[i], 4)')
    vl = permute(vl, ((), (1,)))

    _, vls = eigsolve(x->x*TMl, vl, 1, :LM)
    _, vrs = eigsolve(x->TMr*x, vr, 1, :LM)
    vl = vls[1]
    vr = vrs[1]

    @tensor O_res = vl[1] * ρO_weighted[i][1 3; 3 2] * vr[2]
    @show O_res
    @show (vl * vr)
    return O_res/ (vl*vr)[1]
end

# Fix phase of the disorder MPO after multiplying with inverse partition function
function fix_phase(ρs::DisorderMPO)
    Z = partition_functions(ρs)
    @tensor A[-1; -2] := Z[1][-1 1;1 2]*Z[2][2 3;3 -2] 

    vl = Tensor(rand, ComplexF64, space(A, 1)')
    vr = Tensor(rand, ComplexF64, space(A, 2)')
    vl = permute(vl, ((), (1,)))

    λs, vls = eigsolve(x->x*A, vl, 1, :LM)
    _, vrs = eigsolve(x->A*x, vr, 1, :LM)
    λ = λs[1]
    vl = vls[1]
    vr = vrs[1]

    d = dim(space(ρs[1],3))
    ρ_normalized = rescale(ρs, [d/sqrt(λ) for ix in 1:length(ρs)])

    return ρ_normalized
end

# Normalize the density matrix in each disorder sector
function normalize_each_disorder_sector(ρ::DisorderMPO, trunc_alg::AbstractTruncationAlgorithm, inversion_alg::AbstractInversionAlgorithm; init_guess::Union{InfiniteMPO,Nothing} = nothing, verbosity::Int = 0, invtol::Float64 = 1e-8)
    (verbosity > 0) && (@info(crayon"yellow"("Normalizing Each Disorder sector")))

    # Compute partition function
    mpoZ = partition_functions(ρ)
    (verbosity > 0) && (@info(crayon"yellow"("Truncate Partition Function")))
    (verbosity > 1) && (@info(crayon"yellow"("Before truncation: Bonddimension of Z = $(dim(codomain(mpoZ[1])[1]))")))
    mpoZ = truncate_mpo(mpoZ, trunc_alg)
    (verbosity > 1) && (@info(crayon"yellow"("After truncation: Bonddimension of Z = $(dim(codomain(mpoZ[1])[1]))")))

    # Compute inverse of partition function
    (verbosity > 0) && (@info(crayon"cyan"("Invert Partition Function")))
    mpoZinv, _ = invert_mpo(mpoZ, inversion_alg; init_guess = init_guess)

    # Check accuracy of inversion
    (verbosity > 0) && (@info(crayon"yellow"("Accuracy check")))
    ϵ_acc = test_identity(mpoZ*mpoZinv)
    ϵ_acc > invtol || ((verbosity > 0) && (@info(crayon"green"("accuracy for MPO inversion: ϵ_acc = $ϵ_acc"))))
    ϵ_acc < invtol || @warn(crayon"red"("Inverse not accurate: ϵ_acc = $ϵ_acc"))  

    # Normalize each disorder sector by multiplying with inverse of partition function
    (verbosity > 0) && (@info(crayon"yellow"("Multiply Partition Function with Density Matrix")))
    (verbosity > 1) && (@info(crayon"yellow"("Before multiplication: Bonddimension of ρ = $(dim(codomain(ρ[1])[1]))")))
    ρ_product = ρ * mpoZinv
    (verbosity > 1) && (@info(crayon"yellow"("After multiplication: Bonddimension of ρ = $(dim(codomain(ρ_product[1])[1]))")))

    # Fix phase ambiguity
    (verbosity > 0) && (@info(crayon"yellow"("Fix Phase")))
    ρ_normalized = fix_phase(ρ_product)

    return ρ_normalized, ϵ_acc, mpoZinv
end

# measure average correlation length
function average_correlation_length(ρs::DisorderMPO, ps::Vector{<:Real})
    unit_cell = length(ρs)
    D_disorder = length(ps)
    P = DiagonalTensorMap(ps, ℂ^D_disorder)

    Zs = partition_functions(ρs)

    A = id(ComplexF64,space(Zs[1])[1])
    for i in 1:unit_cell
        @tensor Z_traced[-1; -2] := Zs[i][-1 1;3 -2]*P[3;1]
        A = A*Z_traced
    end

    vr = Tensor(rand, ComplexF64, space(A, 2)')

    λs, vrs = eigsolve(x->A*x, vr, 2, :LM)
    λ1 = λs[1]
    λ2 = λs[2]
    
    ξ = real(unit_cell/log(λ1/λ2))
    return ξ
end