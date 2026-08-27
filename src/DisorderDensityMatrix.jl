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
        @tensor v[-1; -2] := A[-1 4 3; 5 6 2] * conj(A[-2 4 1; 5 6 7]) * P[1; 3] * vr[2; 7]
        return v
    end
    return ftransfer
end

# Left transfer matrix
function left_transfer_matrix(ρ::InfiniteDisorderDensityMatrix)
    A = ρ[1]
    P = make_DiagonalBlockTensorMap(ρ.ps)
    function ftransfer(vl)
        @tensor v[-1; -2] := A[2 5 3; 6 7 -2] * conj(A[4 5 1; 6 7 -1]) * P[1; 3] * vl[4; 2]
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
        @tensor v[-1 -2; -3 -4] := A[-2 10 1; 11 12 13] * conj(A[-1 10 7; 11 12 8]) * A[-4 3 7; 4 5 2] * conj(A[-3 3 9; 4 5 6]) * P[9; 1] * vr[8 13; 6 2]
        return v
    end
end

# Left second moment transfer matrix
function left_second_moment_transfer_matrix(ρ::InfiniteDisorderDensityMatrix)
    A = ρ[1]
    P = make_DiagonalBlockTensorMap(ρ.ps) 
    function ftransfer(vl)
        @tensor v[-1 -2; -3 -4] := A[10 11 1; 12 13 -4] * conj(A[7 11 8; 12 13 -3]) * A[2 4 8; 5 6 -2] * conj(A[3 4 9; 5 6 -1]) * P[9; 1] * vl[3 2; 7 10]
 
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
    λ, l, r = DisorderKit.environments(ρ)

    ft_l = left_second_moment_transfer_matrix(ρ)
    ft_r = right_second_moment_transfer_matrix(ρ)

    l0 = rand(ComplexF64,space(ρ[1],1)⊗space(ρ[1],1)',space(ρ[1],1)'⊗space(ρ[1],1))
    r0 = rand(ComplexF64,space(ρ[1],1)'⊗space(ρ[1],1),space(ρ[1],1)⊗space(ρ[1],1)')

    λs, ρls, infol = eigsolve(ft_l, l0, 1, :LM)
    _, ρrs, infor = eigsolve(ft_r, r0, 1, :LM)
    λ2 = λs[1]

    S = svd_vals((ρls[1] * ρrs[1]))
    es = S.data
    es /= sum(es)
    return sort(es), real(log(λ2)-2*log(λ))
end

function lyapunovexp(ρ::InfiniteDisorderDensityMatrix; L::Int = 100, n_samples::Int = 20)
    λs = Float64[]
    As = []
    Ps = []
    for i in 1:length(ρ.ps)
        ρi = ρ[1][1,1,i,1,i,1]
        @tensor A[-1 -2; -3 -4] := ρi[-2 2 3; 4 5 -4] * conj(ρi[-1 2 3; 4 5 -3])
        push!(As, A)
    end
    for _ in ProgressBar(1:n_samples)
        P = 1
        u = rand(ComplexF64,space(ρ[1],1)'⊗space(ρ[1],1))
        for n in 1:L
            sample = rand(1:length(ρ.ps), 1)
            u = As[sample[1]]*u
            P *= ρ.ps[sample[1]]
        end
        push!(λs, 1/L*log(TensorKit.norm(u)))
        push!(Ps, P)
    end
    @show λs
    average_λ = sum(λs.*Ps)/sum(Ps)
    var_λ = sum(λs.^2 .*Ps)/sum(Ps) - average_λ^2
    return average_λ, var_λ, λs
end

# Right transfer matrix
function right_mixed_transfer_matrix(ρ1::InfiniteDisorderDensityMatrix, ρ2::InfiniteDisorderDensityMatrix)
    A = ρ1[1]
    B = ρ2[1]
    P = make_DiagonalBlockTensorMap(ρ1.ps)
    function ftransfer(vr)
        @tensor v[-1; -2] := A[-1 7 3; 4 5 1] * conj(B[-2 7 6; 4 5 2]) * P[6; 3] * vr[1; 2]
        return v
    end
    return ftransfer
end

function average_trace_distance(ρ1::InfiniteDisorderDensityMatrix, ρ2::InfiniteDisorderDensityMatrix)
    @assert ρ1.ps == ρ2.ps "Disorder sectors must match"
    vr = rand(ComplexF64, space(ρ1[1],1), space(ρ2[1],1))
    vals, vrs = eigsolve(x->right_mixed_transfer_matrix(ρ1, ρ2)(x), vr, 1, :LM)

    # ε = sqrt(1. - abs(vals[1])^2)
    ε = 1. - abs(vals[1])
    return ε
end

#FIXME: Is this the correct way to compute the entanglement entropy?
function average_entanglement_entropy(ρ::InfiniteDisorderDensityMatrix)
    _, r = right_environment(ρ)
    r /= tr(r)

    D, _ = eig_full(r)
    S = D.data
    Se = real.(-sum(S .*log.(S .+ 1e-16)))
    @show S
    @show sum(S)
    return Se, real.(S)
end