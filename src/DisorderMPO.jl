struct InfiniteDisorderMPO{T<:AbstractDisorderMPOTensor}
    opp::PeriodicVector{T}
end

Base.getindex(T::InfiniteDisorderMPO, ix::Int) = T.opp[ix]
Base.size(T::InfiniteDisorderMPO) = size(T.opp)
Base.length(T::InfiniteDisorderMPO) = length(T.opp)
Base.iterate(t::InfiniteDisorderMPO, i=1) = (i > length(t.opp)) ? nothing : (t[i], i + 1)

function InfiniteDisorderMPO(opp::Vector{<:AbstractDisorderMPOTensor})
    return InfiniteDisorderMPO(PeriodicVector(opp))
end

# Multiply DensityMatrix with DisorderMPO
function Base.:*(ρ::InfiniteDisorderDensityMatrix, O::InfiniteDisorderMPO)
    vspace_ρ = space(ρ.opp[1],1)
    vspace_O = space(O.opp[1],1)

    #FIXME Add consistency checks


    iso = isomorphism(ComplexF64, fuse(vspace_ρ ⊗ vspace_O), vspace_ρ ⊗ vspace_O)
    @tensor W[-1 -2 -3; -4 -5 -6] := iso[-1; 1 2] * ρ[1][1 3 4; -4 -5 5] * O[1][2 -2 -3; 3 4 6] * conj(iso[-6; 5 6])

    return InfiniteDisorderDensityMatrix([W], ρ.ps)
end

# Multiply DensityMatrix with MPO on disorder space
function Base.:*(ρ::InfiniteDisorderDensityMatrix, O::InfiniteMPO)
    vspace_ρ = space(ρ[1],1)
    vspace_O = space(O[1],1)

    #FIXME Add consistency checks


    iso = isomorphism(ComplexF64, fuse(vspace_ρ ⊗ vspace_O), vspace_ρ ⊗ vspace_O)
    @tensor W[-1 -2 -3; -4 -5 -6] := iso[-1; 1 2] * ρ[1][1 -2 4; -4 -5 5] * O[1][2 -3; 4 6] * conj(iso[-6; 5 6])

    return InfiniteDisorderDensityMatrix([W], ρ.ps)
end

# Multiply DensityMatrix with MPO on disorder space
function Base.:*(ρ::InfiniteDisorderDensityMatrix, O::AbstractMPOTensor)
    vspace_ρ = space(ρ[1],1)
    vspace_O = space(O,1)

    #FIXME Add consistency checks


    iso = isomorphism(ComplexF64, fuse(vspace_ρ ⊗ vspace_O), vspace_ρ ⊗ vspace_O)
    @tensor W[-1 -2 -3; -4 -5 -6] := iso[-1; 1 2] * ρ[1][1 -2 4; -4 -5 5] * O[2 -3; 4 6] * conj(iso[-6; 5 6])

    return InfiniteDisorderDensityMatrix([W], ρ.ps)
end

