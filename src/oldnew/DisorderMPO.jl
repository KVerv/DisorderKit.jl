struct DisorderMPO{T<:AbstractMPOTensor}
    opp::Vector{T}
end

Base.getindex(T::DisorderMPO, ix::Int) = T.opp[ix]
Base.size(T::DisorderMPO) = size(T.opp)
Base.length(T::DisorderMPO) = length(T.opp)
Base.iterate(t::DisorderMPO, i=1) = (i > length(t.opp)) ? nothing : (t[i], i + 1)

# Convert an MPO on disorderleg to a DisorderMPO
function DisorderMPO(O::AbstractMPOTensor, pspace::ElementarySpace)
    D_disorder = dim(space(O)[2])
    vspace = space(O)[1]
    As = Vector{AbstractTensorMap{ComplexF64, ComplexSpace, 2, 2}}(undef, D_disorder)
    for p in 1:D_disorder
        diagel = zeros(ComplexF64, D_disorder)
        diagel[p] = 1.
        P = DiagonalTensorMap(diagel, ℂ^D_disorder)
        Id = id(ComplexF64, pspace)
 
        @tensor A[-1 -2; -3 -4] := O[-1 1; 2 -4] * P[2; 1] * Id[-2; -3]
        As[p] = A
    end


    return DisorderMPO(As)
end

# Make MPO tensor (with physical & Disorder leg fused) from DisorderMPO
function get_MPOTensor(ρ::DisorderMPO)
    D_disorder = length(ρ.opp)
    vspace = space(ρ[1])[1]
    pspace1 = space(ρ[1],2)
    pspace2 = space(ρ[1],3)'
    dspace = ℂ^D_disorder
    O = zeros(ComplexF64, vspace⊗pspace1⊗dspace, pspace2⊗dspace⊗vspace)
    for p in 1:D_disorder
        diagel = zeros(ComplexF64, D_disorder)
        diagel[p] = 1.
        P = DiagonalTensorMap(diagel, ℂ^D_disorder)

        @tensor Op[-1 -2 -3; -4 -5 -6] := ρ[p][-1 -2; -4 -6] * P[-3; -5]
        O += Op
    end
    iso1 = isomorphism(fuse(pspace1, dspace), pspace1*dspace)
    iso2 = isomorphism(pspace2*dspace, fuse(pspace2, dspace))
    @tensor O_fused[-1 -2; -3 -4] := iso1[-2; 1 2] * O[-1 1 2; 3 4 -4] * iso2[3 4; -3]

    return O_fused
end