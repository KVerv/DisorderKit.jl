struct InfiniteDisorderDensityMatrix{T<:AbstractMPOTensor}
    opp::Vector{T}
    ps::Vector{<:Real}
end

function InfiniteDisorderDensityMatrix(ps::Vector{Float64}, pspace::ElementarySpace, tspace::ElementarySpace, vspace::ElementarySpace; T=ComplexF64)
    D_dis = length(ps)
    opp = [rand(T, vspace ⊗ pspace, tspace ⊗ vspace) for i in 1:D_dis]
    return InfiniteDisorderDensityMatrix(opp, ps)
end

Base.getindex(T::InfiniteDisorderDensityMatrix, ix::Int) = T.opp[ix]
Base.size(T::InfiniteDisorderDensityMatrix) = size(T.opp)
Base.length(T::InfiniteDisorderDensityMatrix) = length(T.opp)
Base.iterate(t::InfiniteDisorderDensityMatrix, i=1) = (i > length(t.opp)) ? nothing : (t[i], i + 1)

# Rescale the tensors of the Density Matrix by a scalar
function rescale(ρs::InfiniteDisorderDensityMatrix, αs::Vector{<:Number})
    ρ1 = deepcopy(ρs)
    for ix in 1:length(αs)
        ρ1.opp[ix] *= αs[ix]
    end
    return ρ1
end

# Multiply DensityMatrix with DisorderMPO
function Base.:*(ρs::InfiniteDisorderDensityMatrix, O::DisorderMPO)

    (length(ρs.opp) == length(O)) || throw(ArgumentError("O should have the same amount of disorder sectors as ρ"))

    opp = Vector{typeof(ρs.opp[1])}(undef, length(ρs.opp))
    for (p, ρ) in  enumerate(ρs)
        iso = isomorphism(ComplexF64, fuse(space(ρ, 1), space(O[p],1)), space(ρ, 1) ⊗ space(O[p],1))
        @tensor ρnew[-1 -2; -3 -4] := iso[-1; 1 2]*ρ[1 3;-3 4] *O[p][2 -2;3 5] * conj(iso[-4; 4 5])
        opp[p] = ρnew
    end

    return InfiniteDisorderDensityMatrix(opp, ρs.ps)
end

# Right transfer matrix
function right_transfer_matrix(ρs::InfiniteDisorderDensityMatrix)
    function ftransfer(vr)
        v = zeros(ComplexF64,space(ρs[1],4)',space(ρs[1],4)')
        for (p, A) in enumerate(ρs)
            @tensor vp[-1; -2] := A[-1 4; 3 2] * conj(A[-2 4;3 1]) * vr[2; 1]
            v += ρs.ps[p]*vp
        end
        return v
    end
end

# Left transfer matrix
function left_transfer_matrix(ρs::InfiniteDisorderDensityMatrix)
    function ftransfer(vl)
        v = zeros(ComplexF64,space(ρs[1],1),space(ρs[1],1))
        for (p, A) in enumerate(ρs)
            @tensor vp[-1; -2] := A[2 4;3 -2] * conj(A[1 4;3 -1]) * vl[1; 2]
            v += ρs.ps[p]*vp
        end
        return v
    end
end

# Compute environments
function right_environment(ρs::InfiniteDisorderDensityMatrix)
    ftransfer = right_transfer_matrix(ρs)
    vr = rand(ComplexF64, space(ρs[1],1), space(ρs[1],1))
    vals, vrs = eigsolve(x->ftransfer(x), vr, 1, :LM)
    return vals[1], vrs[1]
end

function left_environment(ρs::InfiniteDisorderDensityMatrix)
    ftransfer = left_transfer_matrix(ρs)
    vl = rand(ComplexF64, space(ρs[1],1), space(ρs[1],1))
    vals, vls = eigsolve(x->ftransfer(x), vl, 1, :LM)
    return vals[1], vls[1]
end

# Measure local operator
function expectation_value(ρs::InfiniteDisorderDensityMatrix, O::AbstractBondTensor)
    Os = [O for i in 1:length(ρs.opp)]

    return expectation_value(ρs, Os)
end

function expectation_value(ρs::InfiniteDisorderDensityMatrix, Os::Vector{<:AbstractBondTensor})
    λr, vr = right_environment(ρs)
    λl, vl = left_environment(ρs)

    E = 0
    for (p, ρ) in enumerate(ρs)
        @tensor Ep = vl[1; 2] * ρ[2 4; 3 6] * Os[p][5;4] * conj(ρ[1 5; 3 7]) * vr[6; 7]
        @tensor Np = vl[1; 2] * ρ[2 4; 3 6] * conj(ρ[1 4; 3 7]) * vr[6; 7]
        E += Ep * ρs.ps[p]/Np
    end

    return E
end

# measure average correlation length
function average_correlation_length(ρs::InfiniteDisorderDensityMatrix)
    f_t = left_transfer_matrix(ρs)

    vl = rand(ComplexF64, space(ρs[1], 1), space(ρs[1], 1))

    λs, vls = eigsolve(x->f_t(x), vl, 3, :LM)
    λ1 = λs[1]
    λ2 = λs[2]

    
    ξ = real(1/log(λ1/λ2))

    return ξ
end

function energy_density(ρs::InfiniteDisorderDensityMatrix, Hs::DisorderMPOHam)
    l = left_environment(ρs)[2]
    r = right_environment(ρs)[2]
    E = 0
    for (p, ρ) in enumerate(ρs)
        D = Hs.Ds[p]
        C = Hs.Cs[p]

        @tensor ND = l[1; 2] * ρ[2 4; 3 6] * conj(ρ[1 4; 3 7]) * r[6; 7]
        @tensor ED = l[1; 2] * ρ[2 4; 3 6] * D[5; 4] * conj(ρ[1 5; 3 7]) * r[6; 7]
        E += ED/ND*ρs.ps[p]
        for (q, R) in enumerate(ρs)
            B = Hs.Bs[p]
            @tensor NCB = l[1; 2] * ρ[2 4; 3 6] * conj(ρ[1 4; 3 7]) * R[6 10; 9 12] * conj(R[7 10; 9 13]) * r[12; 13]
            @tensor ECB = l[1; 2] * ρ[2 4; 3 6] * C[5; 4 8] * conj(ρ[1 5; 3 7]) * R[6 10; 9 12] * conj(R[7 11; 9 13]) * B[8 11; 10] * r[12; 13]
            E += ECB/NCB*ρs.ps[q]*ρs.ps[p]
        end
    end
    return real(E)
end


# Normalize the density matrix in each disorder sector
function normalize_each_disorder_sector(ρ::InfiniteDisorderDensityMatrix, Z_trunc_method::MatrixAlgebraKit.TruncationStrategy, init_guess::AbstractMPOTensor, inversion_method::InversionStrategy; verbosity::Int = 0, invtol::Float64 = 1e-8)
    (verbosity > 0) && (@info(crayon"yellow"("Normalizing Each Disorder sector")))

    # Compute partition function
    (verbosity > 0) && (@info(crayon"yellow"("Compute Partition Function: Bonddimension ρ : $(space(ρ.opp[1], 1))")))
    # Z = partition_function_mpo(ρ, Z_trunc_method)
    ZL, ZR, ZC, C = partition_function_svd(ρ, Z_trunc_method)
    Z = ZL
    # @show space(Z)
    (verbosity > 0) && (@info(crayon"yellow"("Partition Function Virtual Space: Z : $(space(Z, 1))")))
    
    # Compute inverse of partition function
    (verbosity > 0) && (@info(crayon"yellow"("Invert Partition Function")))
    Zinv = inv_sqrt(Z, init_guess, inversion_method)

    # Check accuracy of inversion
    (verbosity > 0) && (@info(crayon"yellow"("Accuracy check")))
    ϵ_acc = entanglement_error(Zinv, Z)

    ϵ_acc > invtol || ((verbosity > 0) && (@info(crayon"green"("accuracy for MPO inversion: ϵ_acc = $ϵ_acc"))))
    ϵ_acc < invtol || @warn(crayon"red"("Inverse not accurate: ϵ_acc = $ϵ_acc"))  

    # Normalize each disorder sector by multiplying with inverse of partition function
    (verbosity > 0) && (@info(crayon"yellow"("Multiply Partition Function with Density Matrix")))
    ZinvDMPO = DisorderMPO(Zinv, space(ρ.opp[1],2))
    ρ_product = ρ * ZinvDMPO

    # Fix phase ambiguity
    (verbosity > 0) && (@info(crayon"yellow"("Fix Phase")))
    ρ_normalized = fix_phase(ρ_product)
    # ρ_normalized = ρ_product
    return ρ_normalized, ϵ_acc, Zinv
end

function entanglement_spectrum(ρs::InfiniteDisorderDensityMatrix)
    vspace = space(ρs[1],1)

    function ftl(vl)
        v = zeros(ComplexF64, vspace ⊗ vspace', vspace ⊗ vspace')
        for (p, A) in enumerate(ρs)
            @tensor vp[-1 -2; -3 -4] := A[1 5;8 -3] * conj(A[2 5;6 -4]) * conj(A[3 6;7 -1]) * A[4 8;7 -2] * vl[3 4; 1 2]
            v += ρs.ps[p]*vp
        end
        return v
    end

    function ftr(vr)
        v = zeros(ComplexF64, vspace ⊗ vspace', vspace ⊗ vspace')
        for (p, A) in enumerate(ρs)
            @tensor vp[-1 -2; -3 -4] := A[-1 5;8 1] * conj(A[-2 5;8 2]) * conj(A[-3 6;7 3]) * A[-4 6;7 4] * vr[1 2; 3 4]
            v += ρs.ps[p]*vp
        end
        return v
    end
    
    v0 = rand(ComplexF64, vspace ⊗ vspace', vspace ⊗ vspace')
    vals, vrs = eigsolve(x->ftr(x), v0, 1, :LM)
    vals, vls = eigsolve(x->ftl(x), v0, 1, :LM)

    r = vrs[1]
    l = vls[1]

    S = svd_vals((l * r))
    es = S.data
    es /= sum(es)
    return es
end

# Fix phase of the disorder MPO after multiplying with inverse partition function
function fix_phase(ρs::InfiniteDisorderDensityMatrix)
    λl, _ = left_environment(ρs)
    # λr, r = right_environment(ρs)

    d = length(ρs)
    ρ_normalized = rescale(ρs, [1/sqrt(λl) for ix in 1:length(ρs)])
    λl, _ = left_environment(ρ_normalized)
    return ρ_normalized
end
