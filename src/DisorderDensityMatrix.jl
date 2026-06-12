# Convention virtual, physical, disorder ← physical, disorder, virtual
struct InfiniteDisorderDensityMatrix{T<:AbstractDisorderMPOTensor}
    opp::PeriodicVector{T}
    ps::Vector{<:Real}
end

function InfiniteDisorderDensityMatrix(opp::Vector{<:AbstractDisorderMPOTensor}, ps::Vector{<:Real})
    return InfiniteDisorderDensityMatrix(PeriodicVector(opp), ps)
end

function InfiniteDisorderDensityMatrix(ps::Vector{Float64}, pspace::ElementarySpace, tspace::ElementarySpace, vspace::ElementarySpace; T=ComplexF64)
    D_dis = length(ps)
    Wdomain = BlockTensorKit.boxplus([tspace]...) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, D_dis)...) ⊗  BlockTensorKit.boxplus([vspace]...)
    Wcodomain = BlockTensorKit.boxplus([vspace]...) ⊗ BlockTensorKit.boxplus([pspace]...) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, D_dis)...)

    W = spzeros(ComplexF64, Wcodomain, Wdomain)
    for i in 1:D_dis
        W[1,1,i,1,i,1] = rand(ComplexF64, vspace ⊗ pspace ⊗ ℂ^1, tspace ⊗ ℂ^1 ⊗ vspace)
    end
    return InfiniteDisorderDensityMatrix([W], ps)
end

Base.getindex(T::InfiniteDisorderDensityMatrix, ix::Int) = T.opp[ix]
Base.size(T::InfiniteDisorderDensityMatrix) = size(T.opp)
Base.length(T::InfiniteDisorderDensityMatrix) = length(T.opp)
Base.iterate(t::InfiniteDisorderDensityMatrix, i=1) = (i > length(t.opp)) ? nothing : (t[i], i + 1)

# Rescale the tensors of the Density Matrix by a scalar
function rescale(ρ::InfiniteDisorderDensityMatrix, α::Number)
    opp = ρ.opp*α
    return InfiniteDisorderDensityMatrix(opp, ρ.ps)
end

# Right transfer matrix
function right_transfer_matrix(ρ::InfiniteDisorderDensityMatrix)
    A = ρ[1]
    P = make_DiagonalBlockTensorMap(ρ.ps)
    function ftransfer(vr)
        @tensor v[-1; -2] := A[-1 7 3; 4 5 1] * conj(A[-2 7 6; 4 5 2]) * P[6; 3] * vr[1; 2]
        return v
    end
    return ftransfer
end

# Left transfer matrix
function left_transfer_matrix(ρ::InfiniteDisorderDensityMatrix)
    A = ρ[1]
    P = make_DiagonalBlockTensorMap(ρ.ps)
    function ftransfer(vl)
        @tensor v[-1; -2] := A[1 3 6; 4 5 -2] * conj(A[2 3 7; 4 5 -1]) * P[7; 6] * vl[2; 1]
        return v
    end
    return ftransfer
end

# Compute environments
function right_environment(ρ::InfiniteDisorderDensityMatrix)
    ftransfer = right_transfer_matrix(ρ)
    vr = rand(ComplexF64, space(ρ[1],1), space(ρ[1],1))
    vals, vrs = eigsolve(x->ftransfer(x), vr, 1, :LM)
    return vals[1], vrs[1]
end

function left_environment(ρ::InfiniteDisorderDensityMatrix)
    ftransfer = left_transfer_matrix(ρ)
    vl = rand(ComplexF64, space(ρ[1],1), space(ρ[1],1))
    vals, vls = eigsolve(x->ftransfer(x), vl, 1, :LM)
    return vals[1], vls[1]
end

function environments(ρ::InfiniteDisorderDensityMatrix)
    λr, vr = right_environment(ρ)
    λl, vl = left_environment(ρ)
    l = vl/sqrt(tr(vl*vr))
    r = vr/sqrt(tr(vl*vr))

    return λl, l, r
end

# Right second moment transfer matrix
function right_second_moment_transfer_matrix(ρ::InfiniteDisorderDensityMatrix)
    A = ρ[1]
    P = make_DiagonalBlockTensorMap(ρ.ps) 
    function ftransfer(vr)
        @tensor v[-1 -2; -3 -4] := A[-2 2 6; 1 3 5] * conj(A[-1 2 7; 1 3 4]) * A[-4 9 7; 10 11 8] * conj(A[-3 9 13; 10 11 12]) * P[13; 6] * vr[4 5; 12 8]
        return v
    end
end

# Left second moment transfer matrix
function left_second_moment_transfer_matrix(ρ::InfiniteDisorderDensityMatrix)
    A = ρ[1]
    P = make_DiagonalBlockTensorMap(ρ.ps) 
    function ftransfer(vl)
        @tensor v[-1 -2; -3 -4] := A[8 2 6; 1 3 -4] * conj(A[12 2 7; 1 3 -3]) * A[5 9 7; 10 11 -2] * conj(A[4 9 13; 10 11 -1]) * P[13; 6] * vl[4 5; 12 8]
        return v
    end
end

# Gauge density matrix such that dominant eigenvalue is 1
function gauge(ρ::InfiniteDisorderDensityMatrix)
    λ, _ = left_environment(ρ)
    return rescale(ρ, 1/sqrt(λ))
end

# Measure local operator
function expectation_value(ρ::InfiniteDisorderDensityMatrix, O::AbstractBondTensor)
    λ, vl, vr = DisorderKit.environments(ρ)

    P = make_DiagonalBlockTensorMap(ρ.ps)
    @tensor E = vl[9; 10] * O[8; 7] * ρ[1][10 7 3; 4 5 1] * conj(ρ[1][9 8 6; 4 5 2]) * P[6; 3] * vr[1; 2]

    return E/λ
end

# Measure two-point correlation function
function two_point_correlator(ρ::InfiniteDisorderDensityMatrix, O1::AbstractBondTensor, O2::AbstractBondTensor, r::Int)
    d = max(1, r)
    d = min(d, 100)
    λ, vl, vr = DisorderKit.environments(ρ)
    ft = left_transfer_matrix(ρ)
    P = make_DiagonalBlockTensorMap(ρ.ps)

    Cs = zeros(ComplexF64, d)
    @tensor lO[-1; -2] := vl[8; 1] * ρ[1][1 2 3; 4 5 -2] * P[6; 3] * O1[7; 2] * conj(ρ[1][8 7 6; 4 5 -1])
    @tensor rO[-1; -2] := vr[1; 8] * ρ[1][-1 2 3; 4 5 1] * P[6; 3] * O2[7; 2] * conj(ρ[1][-2 7 6; 4 5 8])
    lO /= λ
    rO /= λ

    @tensor C = lO[1; 2] * rO[2; 1]
    Cs[1] = C
    for i in 2:d
        lO = ft(lO)/λ
        @tensor C = lO[1; 2] * rO[2; 1]
        Cs[i] = C
    end
    return Cs
end

# Measure energy density
function energy_density(ρ::InfiniteDisorderDensityMatrix, Hs::DisorderMPOHam)
    λ, l, r = DisorderKit.environments(ρ)
    P = make_DiagonalBlockTensorMap(ρ.ps)
    E = 0.
    @tensor ED =  l[6; 1] * ρ[1][1 2 3; 7 8 11] * Hs.D[4 5; 2 3] * conj(ρ[1][6 4 9; 7 8 10]) * r[11; 10] * P[9;5]
    ED /= λ
    #FIXME Include long range interactions
    @tensor vL[-1; -2 -3] := l[6; 1] * ρ[1][1 2 3; 7 8 -3] * Hs.L[4 5; 2 3 -2] * conj(ρ[1][6 4 9; 7 8 -1]) * P[9;5]
    @tensor vR[-1 -2; -3] := r[6; 1] * ρ[1][-2 2 3; 7 8 6] * Hs.R[-1 4 5; 2 3] * conj(ρ[1][-3 4 9; 7 8 1]) * P[9;5]

    @tensor ECB = vL[1; 2 3] * vR[2 3; 1]
    ECB /= λ^2

    E = real(ED + ECB)
    return E

end

# Measure average correlation length
function average_correlation_length(ρ::InfiniteDisorderDensityMatrix)
    f_t = left_transfer_matrix(ρ)

    vl = rand(ComplexF64, space(ρ[1], 1), space(ρ[1], 1))

    λs, _ = eigsolve(x->f_t(x), vl, 3, :LM)
    λ1 = λs[1]
    if length(λs) < 2
        return ξ = 1e-16
    end
    λ2 = λs[2]

    ξ = real(-1/log(abs(λ2)))

    return ξ
end


function entanglement_spectrum_norm(ρ::InfiniteDisorderDensityMatrix)
    ft_l = left_second_moment_transfer_matrix(ρ)
    ft_r = right_second_moment_transfer_matrix(ρ)

    l = rand(ComplexF64,space(ρ[1],1)⊗space(ρ[1],1)',space(ρ[1],1)'⊗space(ρ[1],1))
    r = rand(ComplexF64,space(ρ[1],1)'⊗space(ρ[1],1),space(ρ[1],1)⊗space(ρ[1],1)')


    λ, ρls, infol = eigsolve(ft_l, l, 1, :LM)
    _, ρrs, infor = eigsolve(ft_r, r, 1, :LM)

    S = svd_vals((ρls[1] * ρrs[1]))
    es = S.data
    es /= sum(es)
    return sort(es), λ[1]
end