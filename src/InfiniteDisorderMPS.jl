# Convention virtual, physical, disorder ← disorder, virtual
struct InfiniteDisorderMPS{T<:AbstractDisorderMPSTensor}
    opp::PeriodicVector{T}
    ps::Vector{<:Real}
end

function InfiniteDisorderMPS(opp::Vector{<:AbstractDisorderMPSTensor}, ps::Vector{<:Real})
    return InfiniteDisorderMPS(PeriodicVector(opp), ps)
end

function InfiniteDisorderMPS(ps::Vector{Float64}, pspace::ElementarySpace, vspace::ElementarySpace; T=ComplexF64)
    D_dis = length(ps)
    Wdomain = BlockTensorKit.boxplus(fill(ℂ^1, D_dis)...) ⊗  BlockTensorKit.boxplus([vspace]...)
    Wcodomain = BlockTensorKit.boxplus([vspace]...) ⊗ BlockTensorKit.boxplus([pspace]...) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, D_dis)...)

    W = spzeros(ComplexF64, Wcodomain, Wdomain)
    for i in 1:D_dis
        W[1,1,i,1,i,1] = rand(ComplexF64, vspace ⊗ pspace ⊗ ℂ^1, ℂ^1 ⊗ vspace)
    end
    return InfiniteDisorderMPS([W], ps)
end

Base.getindex(T::InfiniteDisorderMPS, ix::Int) = T.opp[ix]
Base.size(T::InfiniteDisorderMPS) = size(T.opp)
Base.length(T::InfiniteDisorderMPS) = length(T.opp)
Base.iterate(t::InfiniteDisorderMPS, i=1) = (i > length(t.opp)) ? nothing : (t[i], i + 1)

# Rescale the tensors of the MPS by a scalar
function rescale(ρ::InfiniteDisorderMPS, α::Number)
    opp = ρ.opp*α
    return InfiniteDisorderMPS(opp, ρ.ps)
end

# Right transfer matrix
function right_transfer_matrix(ρ::InfiniteDisorderMPS)
    A = ρ[1]
    P = make_DiagonalBlockTensorMap(ρ.ps)
    function ftransfer(vr)
        @tensor v[-1; -2] := A[-1 3 4; 2 1] * conj(A[-2 3 4; 5 6]) * P[2; 5] * vr[1; 6]
        return v
    end
    return ftransfer
end

# Left transfer matrix
function left_transfer_matrix(ρ::InfiniteDisorderMPS)
    A = ρ[1]
    P = make_DiagonalBlockTensorMap(ρ.ps)
    function ftransfer(vl)
        @tensor v[-1; -2] := A[1 3 4; 2 -2] * conj(A[6 3 4; 5 -1]) * P[2; 5] * vl[6; 1]
        return v
    end
    return ftransfer
end

# Compute environments
function right_environment(ρ::InfiniteDisorderMPS)
    ftransfer = right_transfer_matrix(ρ)
    vr = rand(ComplexF64, space(ρ[1],1), space(ρ[1],1))
    vals, vrs = eigsolve(x->ftransfer(x), vr, 1, :LM)
    return vals[1], vrs[1]
end

function left_environment(ρ::InfiniteDisorderMPS)
    ftransfer = left_transfer_matrix(ρ)
    vl = rand(ComplexF64, space(ρ[1],1), space(ρ[1],1))
    vals, vls = eigsolve(x->ftransfer(x), vl, 1, :LM)
    return vals[1], vls[1]
end

function environments(ρ::InfiniteDisorderMPS)
    λr, vr = right_environment(ρ)
    λl, vl = left_environment(ρ)

    l = vl/sqrt(tr(vl*vr))
    r = vr/sqrt(tr(vl*vr))

    return λl, l, r
end

# Gauge density matrix such that dominant eigenvalue is 1
function gauge(ρ::InfiniteDisorderMPS)
    λ, _ = left_environment(ρ)

    return rescale(ρ, 1/sqrt(λ))
end

# Measure local operator
function expectation_value(ρ::InfiniteDisorderMPS, O::AbstractBondTensor)
    λ, l, r = DisorderKit.environments(ρ)

    P = make_DiagonalBlockTensorMap(ρ.ps)
    A = ρ[1]
    @tensor E = l[6; 1] * O[4; 3] * A[1 3 5; 2 8] * conj(A[6 4 5; 7 9]) * P[2; 7] * r[8; 9]

    return E/λ
end

# Measure two-point correlation function
function two_point_correlator(ρ::InfiniteDisorderMPS, O1::AbstractBondTensor, O2::AbstractBondTensor, r::Int)
    d = max(1, r)
    d = min(d, 100)
    λ, l, r = DisorderKit.environments(ρ)
    ft = left_transfer_matrix(ρ)
    P = make_DiagonalBlockTensorMap(ρ.ps)
    A = ρ[1]

    Cs = zeros(ComplexF64, d)
    @tensor lO[-1; -2] := l[6; 1] * O1[4; 3] * A[1 3 5; 2 -2] * conj(A[6 4 5; 7 -1]) * P[2; 7]
    @tensor rO[-1; -2] := r[1; 6] * O2[4; 3] * A[-1 3 5; 2 1] * conj(A[-2 4 5; 7 6]) * P[2; 7]
    lO /= λ
    rO /= λ

    @tensor C = tr(lO*rO)
    Cs[1] = C
    for i in 2:d
        lO = ft(lO)/λ
        @tensor C = tr(lO*rO)
        Cs[i] = C
    end
    return Cs
end

# Measure energy density
function energy_density(ρ::InfiniteDisorderMPS, H::DisorderMPOHam)
    λ, l, r = DisorderKit.environments(ρ)
    P = make_DiagonalBlockTensorMap(ρ.ps)
    A = ρ[1]
    E = 0.
    @tensor ED =  l[7; 1] * H.D[5 6; 3 4] * A[1 3 4; 2 9] * conj(A[7 5 6; 8 10]) * P[2; 8] * r[9; 10]
    ED /= λ
    #FIXME Include long range interactions
    @tensor vL[-1; -2 -3] := l[7; 1] * H.L[5 6; 3 4 -3] * A[1 3 4; 2 -2] * conj(A[7 5 6; 8 -1]) * P[2; 8]
    @tensor vR[-1 -2; -3] := r[1; 7] * H.R[-2 5 6; 3 4] * A[-1 3 4; 2 1] * conj(A[-3 5 6; 8 7]) * P[2; 8]

    @tensor ECB = vL[1; 3 2] * vR[3 2; 1]
    ECB /= λ^2

    E = ED + ECB
    imag(E) > 1e-12 && @warn("Energy density has a large imaginary part: $E")
    return real(E)
end

# Measure average correlation length
function average_correlation_length(ρ::InfiniteDisorderMPS)
    f_t = left_transfer_matrix(ρ)

    vl = rand(ComplexF64, space(ρ[1], 1), space(ρ[1], 1))

    λs, _ = eigsolve(x->f_t(x), vl, 3, :LM)
    λ1 = λs[1]
    if length(λs) < 2
        return ξ = 1e-16
    end
    λ2 = λs[2]

    ξ = real(-1/log(abs(λ2/λ1)))

    return ξ
end

# Right transfer matrix
function right_mixed_transfer_matrix(ρ1::InfiniteDisorderMPS, ρ2::InfiniteDisorderMPS)
    A = ρ1[1]
    B = ρ2[1]
    P = make_DiagonalBlockTensorMap(ρ1.ps)
    function ftransfer(vr)
        @tensor v[-1; -2] := A[-1 3 4; 2 1] * conj(B[-2 3 4; 5 6]) * P[2; 5] * vr[1; 6]
        return v
    end
    return ftransfer
end

function average_trace_distance(ρ1::InfiniteDisorderMPS, ρ2::InfiniteDisorderMPS)
    @assert ρ1.ps == ρ2.ps "Disorder sectors must match"
    ρ1 = gauge(ρ1)
    ρ2 = gauge(ρ2)
    vr = rand(ComplexF64, space(ρ1[1],1), space(ρ2[1],1))
    vals, vrs = eigsolve(x->right_mixed_transfer_matrix(ρ1, ρ2)(x), vr, 1, :LM)


    λmixed = vals[1]
 
    ε = 1 - abs(λmixed)
    return ε
end

#FIXME: Is this the correct way to compute the entanglement entropy?
function average_entanglement_entropy(ρ::InfiniteDisorderMPS)
    @info("Computing entanglement entropy...")
    pspace = space(ρ[1], 2)
    dspace = space(ρ[1], 3)
    isop = isomorphism(ComplexF64, fuse(pspace ⊗ dspace), pspace ⊗ dspace)
    @tensor O[-1 -2; -3 -4] :=  isop[-2; 1 2] * ρ[1][-1 1 2; -3 -4]

    L₀ = rand(ComplexF64, space(O, 1), space(O, 1))
    OL, L, λ, ϵL = left_orthonormalize_mpo(O, L₀; conv_tol=1e-9)
    C₀ = rand(ComplexF64, space(OL, 4)', space(OL, 4)')
    OR, C, λ, ϵR = right_orthonormalize_mpo(OL, C₀; conv_tol=1e-9)
    U, S, V, ϵC = svd_trunc(C; trunc=truncerror(atol=1e-12))

    S = S.data
    @show sum(S.^2)
    Se = real.(-sum(S.^2 .* log.(S.^2 .+ 1e-16)))
    @info("Entanglement entropy: $Se")
    return Se, real.(S)
end