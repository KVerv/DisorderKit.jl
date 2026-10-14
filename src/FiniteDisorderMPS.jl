# Convention virtual, physical ← virtual

# Finite (open boundary) DisorderMPS of length L
# opp[i][p] is the MPS tensor at site i for disorder sector p, same disorder distribution ps on every site
struct FiniteDisorderMPS{T<:AbstractMPSTensor}
    opp::Vector{Vector{T}}
    ps::Vector{<:Real}
end

function FiniteDisorderMPS(ps::Vector{Float64}, L::Int, D_dis::Int, D_phys::Int, D::Int; T=ComplexF64, zero_variance::Bool=false)
    # Bond dimensions, capped by the maximal dimension allowed by the open boundaries
    Ds = ones(Int, L+1)
    for i in 2:L
        Ds[i] = min(D, Ds[i-1]*D_phys)
    end
    for i in L:-1:2
        Ds[i] = min(Ds[i], Ds[i+1]*D_phys)
    end

    opp = map(1:L) do i
        if zero_variance
            Q, _ = qr_compact(rand(T, ℂ^Ds[i]⊗ℂ^D_phys, ℂ^Ds[i+1]))
            [Q for p in 1:D_dis]
        else
            [first(qr_compact(rand(T, ℂ^Ds[i]⊗ℂ^D_phys, ℂ^Ds[i+1]))) for p in 1:D_dis]
        end
    end
    return FiniteDisorderMPS(opp, ps)
end

Base.getindex(ρ::FiniteDisorderMPS, i::Int) = ρ.opp[i]
Base.getindex(ρ::FiniteDisorderMPS, i::Int, p::Int) = ρ.opp[i][p]
Base.size(ρ::FiniteDisorderMPS) = size(ρ.opp)
Base.length(ρ::FiniteDisorderMPS) = length(ρ.opp)
Base.eachindex(ρ::FiniteDisorderMPS) = 1:length(ρ.opp)
Base.iterate(ρ::FiniteDisorderMPS, i=1) = (i > length(ρ.opp)) ? nothing : (ρ[i], i + 1)

function rescale(ρ::FiniteDisorderMPS, α::Number)
    opp = [Ws*α for Ws in ρ.opp]
    return FiniteDisorderMPS(opp, ρ.ps)
end

# Construct application left transfer matrix at site i: v*T -> v
function transfer_left(ρ::FiniteDisorderMPS, i::Int)
    function ftransfer(vl)
        v = zeros(ComplexF64, space(ρ[i,1],3)', space(ρ[i,1],3)')
        for (p,W) in enumerate(ρ[i])
            @tensor vp[-2; -1] := W[1 3; -1] * conj(W[2 3; -2]) * vl[2; 1]
            v += ρ.ps[p]*vp
        end
        return v
    end
    return ftransfer
end

# Construct application right transfer matrix at site i: T*v -> v
function transfer_right(ρ::FiniteDisorderMPS, i::Int)
    function ftransfer(vr)
        v = zeros(ComplexF64, space(ρ[i,1],1), space(ρ[i,1],1))
        for (p,W) in enumerate(ρ[i])
            @tensor vp[-1; -2] := W[-1 3; 1] * conj(W[-2 3; 2]) * vr[1; 2]
            v += ρ.ps[p]*vp
        end
        return v
    end
    return ftransfer
end

# Left environments on every bond: Els[k+1] lives on the bond between site k and k+1 (k = 0..L)
function left_environments(ρ::FiniteDisorderMPS)
    # Built without mutation so that Zygote can differentiate through it
    Els = [id(ComplexF64, space(ρ[1,1], 1))]
    for i in 1:length(ρ)
        Els = vcat(Els, [transfer_left(ρ, i)(Els[end])])
    end
    return Els
end

# Right environments on every bond: Ers[k+1] lives on the bond between site k and k+1 (k = 0..L)
function right_environments(ρ::FiniteDisorderMPS)
    # Built without mutation so that Zygote can differentiate through it
    L = length(ρ)
    Ers = [id(ComplexF64, space(ρ[L,1], 3)')]
    for i in L:-1:1
        Ers = vcat([transfer_right(ρ, i)(Ers[1])], Ers)
    end
    return Ers
end

function environments(ρ::FiniteDisorderMPS)
    return left_environments(ρ), right_environments(ρ)
end

# Compute the norm of a FiniteDisorderMPS
function norm(ρ::FiniteDisorderMPS)
    Ers = right_environments(ρ)

    N = tr(Ers[1])
    imag(N) < 1e-4 || @warn("Norm has imaginary part: N = $N")
    N = real.(N)
    return N
end

# Compute the total energy of a FiniteDisorderMPS with respect to a DisorderMPOHam
function energy(ρ::FiniteDisorderMPS, Hs::DisorderMPOHam)
    L = length(ρ)
    ls, rs = environments(ρ)
    N = tr(rs[1])

    E = 0
    for i in 1:L
        l = ls[i]
        r = rs[i+1]
        for (p, W) in enumerate(ρ[i])
            @tensor ED = l[4; 1] * W[1 2; 5] * Hs.Ds[p][3; 2] * conj(W[4 3; 6]) * r[5; 6]
            E += ρ.ps[p] * ED
        end
    end
    for i in 1:L-1
        l = ls[i]
        r = rs[i+2]
        for (p, W) in enumerate(ρ[i])
            for (q, V) in enumerate(ρ[i+1])
                @tensor ECB = l[11; 1] * W[1 2; 4] * Hs.Cs[p][3; 2 5] * conj(W[11 3; 6]) * V[4 7; 9] * Hs.Bs[p][5 8; 7] * conj(V[6 8; 10]) * r[9; 10]
                E += ρ.ps[p] * ρ.ps[q] * ECB
            end
        end
        #FIXME : currently only nearest-neighbor interactions
    end
    E = E/N

    imag(E) < 1e-4 || @warn("Energy has imaginary part: E = $E")

    return real.(E)
end

function energy_density(ρ::FiniteDisorderMPS, Hs::DisorderMPOHam)
    return energy(ρ, Hs)/length(ρ)
end

# Expectation value of a local operator O at site i
function expectation_value(ρ::FiniteDisorderMPS, O::AbstractBondTensor, i::Int)
    Os = [O for p in 1:length(ρ[i])]

    return expectation_value(ρ, Os, i)
end

# Expectation value of a local disorder operator O at site i
function expectation_value(ρ::FiniteDisorderMPS, Os::Vector{<:AbstractBondTensor}, i::Int)
    ls, rs = environments(ρ)
    return _expectation_value(ρ, Os, i, ls[i], rs[i+1]) / tr(rs[1])
end

# Expectation value of a local (disorder) operator on every site
function expectation_value(ρ::FiniteDisorderMPS, O::AbstractBondTensor)
    Os = [O for p in 1:length(ρ[1])]

    return expectation_value(ρ, Os)
end

function expectation_value(ρ::FiniteDisorderMPS, Os::Vector{<:AbstractBondTensor})
    ls, rs = environments(ρ)
    N = tr(rs[1])
    return [_expectation_value(ρ, Os, i, ls[i], rs[i+1]) / N for i in 1:length(ρ)]
end

function _expectation_value(ρ::FiniteDisorderMPS, Os::Vector{<:AbstractBondTensor}, i::Int, l, r)
    O = 0
    for (p, W) in enumerate(ρ[i])
        @tensor Op = l[4; 1] * W[1 3; 5] * Os[p][2; 3] * conj(W[4 2; 6]) * r[5; 6]
        O += ρ.ps[p]*Op
    end
    return O
end

# Compute correlation function of two local operators at site i and site j
function correlator(ρ::FiniteDisorderMPS, O1::AbstractBondTensor, O2::AbstractBondTensor, i::Int, j::Int)
    O1s = [O1 for p in 1:length(ρ[i])]
    O2s = [O2 for p in 1:length(ρ[i])]

    return correlator(ρ, O1s, O2s, i, j)
end

# Compute correlation function between O1 at site i and O2 at sites k = i+1..j
function correlator(ρ::FiniteDisorderMPS, O1s::Vector{<:AbstractBondTensor}, O2s::Vector{<:AbstractBondTensor}, i::Int, j::Int)
    (1 <= i < j <= length(ρ)) || throw(ArgumentError("Need 1 ≤ i < j ≤ L"))

    Cs = Vector{ComplexF64}(undef, j-i)
    ls, rs = environments(ρ)
    N = tr(rs[1])

    l = ls[i]
    vl = zeros(ComplexF64, space(ρ[i,1],3)', space(ρ[i,1],3)')
    for (p, W) in enumerate(ρ[i])
        @tensor vlO1[-2; -1] := l[2; 1] * W[1 3; -1] * O1s[p][4; 3] * conj(W[2 4; -2])
        vl += ρ.ps[p]*vlO1
    end
    for k in i+1:j
        Cs[k-i] = _expectation_value(ρ, O2s, k, vl, rs[k+1]) / N
        vl = transfer_left(ρ, k)(vl)
    end
    return real.(Cs)
end

function transfer_mixed_right(ρ1::FiniteDisorderMPS, ρ2::FiniteDisorderMPS, i::Int)
    v1space = space(ρ1[i,1],1)
    v2space = space(ρ2[i,1],1)
    function ftransfer(vr)
        v = zeros(ComplexF64, v2space, v1space)
        for (p,X) in enumerate(ρ1[i])
            Y = ρ2[i,p]
            @tensor vp[-1; -2] := Y[-1 3; 1] * conj(X[-2 3; 2]) * vr[1; 2]
            v += ρ1.ps[p]*vp
        end
        return v
    end
    return ftransfer
end

function transfer_mixed2_right(ρ1::FiniteDisorderMPS, ρ2::FiniteDisorderMPS, i::Int)
    v1space = space(ρ1[i,1],1)
    v2space = space(ρ2[i,1],1)
    function ftransfer(vr)
        v = zeros(ComplexF64, v2space ⊗ v1space', v1space' ⊗ v2space)
        for (p,X) in enumerate(ρ1[i])
            Y = ρ2[i,p]
            @tensor vp[-1 -2; -3 -4] := Y[-1 2; 1] * conj(X[-2 2; 3]) * X[-3 5; 4] * conj(Y[-4 5; 6]) * vr[1 3; 4 6]
            v += ρ1.ps[p]*vp
        end
        return v
    end
    return ftransfer
end

function transfer_mixed2_left(ρ1::FiniteDisorderMPS, ρ2::FiniteDisorderMPS, i::Int)
    v1space = space(ρ1[i,1],3)'
    v2space = space(ρ2[i,1],3)'
    function ftransfer(vl)
        v = zeros(ComplexF64, v1space' ⊗ v2space, v2space ⊗ v1space')
        for (p,X) in enumerate(ρ1[i])
            Y = ρ2[i,p]
            @tensor vp[-1 -2; -3 -4] := Y[1 2; -3] * conj(X[3 2; -4]) * X[4 5; -1] * conj(Y[6 5; -2]) * vl[4 6; 1 3]
            v += ρ1.ps[p]*vp
        end
        return v
    end
    return ftransfer
end

function overlap(ρ1::FiniteDisorderMPS, ρ2::FiniteDisorderMPS)
    (length(ρ1) == length(ρ2)) || throw(ArgumentError("ρ1 and ρ2 should have the same length"))
    (length(ρ1[1]) == length(ρ2[1])) || throw(ArgumentError("ρ1 and ρ2 should have the same amount of disorder sectors"))

    L = length(ρ1)
    v1space = space(ρ1[L,1],3)'
    v2space = space(ρ2[L,1],3)'
    v = ones(ComplexF64, v2space, v1space)
    for i in L:-1:1
        v = transfer_mixed_right(ρ1, ρ2, i)(v)
    end

    return tr(v)
end

function overlap_squared(ρ1::FiniteDisorderMPS, ρ2::FiniteDisorderMPS)
    (length(ρ1) == length(ρ2)) || throw(ArgumentError("ρ1 and ρ2 should have the same length"))
    (length(ρ1[1]) == length(ρ2[1])) || throw(ArgumentError("ρ1 and ρ2 should have the same amount of disorder sectors"))

    L = length(ρ1)
    v1space = space(ρ1[L,1],3)'
    v2space = space(ρ2[L,1],3)'
    v = ones(ComplexF64, v2space ⊗ v1space', v1space' ⊗ v2space)
    for i in L:-1:1
        v = transfer_mixed2_right(ρ1, ρ2, i)(v)
    end

    return only(convert(Array, v))
end

function fidelity(ρ1::FiniteDisorderMPS, ρ2::FiniteDisorderMPS)
    # ov11 = overlap(ρ1, ρ1)
    ov22 = overlap(ρ2, ρ2)
    ov12 = overlap_squared(ρ1, ρ2)

    F = ov12/ov22
    return abs(F)
end

# Entanglement entropy S_w(k) of the conditional state ψ_w of word w on every bond k = 1..L-1, from the dense tensors
# A = _dense(ρ). The scaled environments of _advance (FiniteDisorderFilter.jl) are Gram matrices [bra, ket] of the
# left (sites 1..k) and right (sites k+1..L) Schmidt vectors; the Schmidt spectrum is eig(transpose(r) * l) normalized
# to one, so ψ_w does not have to be normalized
function conditional_entanglement_entropy(A::Vector{<:Vector{<:Array{ComplexF64, 3}}}, word::AbstractVector{Int})
    L = length(word)
    lefts = Vector{Tuple{Matrix{ComplexF64}, Float64}}(undef, L)
    rights = Vector{Tuple{Matrix{ComplexF64}, Float64}}(undef, L+1)
    lefts[1] = _boundary_environment
    for i in 1:L-1
        lefts[i+1] = _advance(lefts[i], A[i][word[i]])
    end
    rights[L+1] = _boundary_environment
    for i in L:-1:2
        rights[i] = _advance(rights[i+1], A[i][word[i]]; right = true)
    end
    return map(1:L-1) do k
        λ = max.(real.(eigvals(transpose(rights[k+1][1]) * lefts[k+1][1])), 0.0)
        λ ./= sum(λ)
        -sum(x -> x > 0 ? x * log(x) : 0.0, λ)
    end
end

# Disorder average 𝔼_w[S_w(k)] of the entanglement entropy on every bond k = 1..L-1 and its standard error, over words
# w = (p₁…p_L) with pᵢ ~ ρ.ps iid. If there are at most n_samples words they are all enumerated (exact, zero error),
# otherwise n_samples words are sampled
function average_entanglement_entropy(ρ::FiniteDisorderMPS; n_samples::Int = 1000, rng::AbstractRNG = Random.default_rng())
    A = _dense(ρ)
    L, d = length(ρ), length(ρ.ps)
    if float(d)^L <= n_samples
        S_avg = zeros(Float64, L-1)
        for word in Iterators.product(fill(1:d, L)...)
            w = prod(ρ.ps[p] for p in word)
            iszero(w) && continue
            S_avg .+= w .* conditional_entanglement_entropy(A, collect(word))
        end
        return S_avg, zeros(Float64, L-1)
    end

    Ss = reduce(hcat, [conditional_entanglement_entropy(A, sample(rng, 1:d, Weights(ρ.ps), L)) for _ in 1:n_samples])
    return vec(mean(Ss; dims=2)), vec(std(Ss; dims=2)) ./ sqrt(n_samples)
end

# Disorder averaged entanglement entropy on the bond between site k and k+1 and its standard error
function average_entanglement_entropy(ρ::FiniteDisorderMPS, k::Int; kwargs...)
    Ss, Ss_err = average_entanglement_entropy(ρ; kwargs...)
    return Ss[k], Ss_err[k]
end
