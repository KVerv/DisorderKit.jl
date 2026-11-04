# Convention virtual, physical ← virtual

# Finite (open boundary) DisorderMPS of length L
# opp[i][p] is the MPS tensor at site i for disorder sector p, with probability ps[i][p]. Every site has its own sectors:
# sectors[i][p] is the Hamiltonian sector (index into Hs.Ds, Hs.Ls, Hs.Rs) of sector p on site i
struct FiniteDisorderMPS{T<:AbstractMPSTensor}
    opp::Vector{Vector{T}}
    ps::Vector{Vector{Float64}}
    sectors::Vector{Vector{Int}}
    function FiniteDisorderMPS(opp::Vector{Vector{T}}, ps::Vector{<:Vector{<:Real}}, sectors::Vector{<:Vector{<:Integer}}) where {T<:AbstractMPSTensor}
        (length(opp) == length(ps) == length(sectors)) || throw(ArgumentError("opp, ps and sectors should have the same length"))
        all(length(opp[i]) == length(ps[i]) == length(sectors[i]) for i in eachindex(opp)) || throw(ArgumentError("opp, ps and sectors should have the same number of sectors on every site"))
        return new{T}(opp, ps, sectors)
    end
end

# Disorder sectors of every site of a chain of length L for the Hamiltonian sectors of Hs with probabilities ps.
# The coupling of bond (i, i+1) is set by the sector of site i, so the last site has no coupling: the sectors with the same
# on-site term are merged there into one sector with the summed (marginal) probability. Returns (site ps, site sectors)
function site_sectors(Hs::DisorderMPOHam, ps::Vector{<:Real}, L::Int)
    length(ps) == length(Hs.Ds) || throw(ArgumentError("ps should have one probability per sector of Hs"))
    reps, ps_last = Int[], Float64[]
    for p in eachindex(ps)
        k = findfirst(r -> Hs.Ds[r] ≈ Hs.Ds[p], reps)
        if isnothing(k)
            push!(reps, p)
            push!(ps_last, ps[p])
        else
            ps_last[k] += ps[p]
        end
    end
    site_ps = [i < L ? collect(Float64, ps) : ps_last for i in 1:L]
    sectors = [i < L ? collect(eachindex(ps)) : reps for i in 1:L]
    return site_ps, sectors
end

function FiniteDisorderMPS(Hs::DisorderMPOHam, ps::Vector{<:Real}, L::Int, D_phys::Int, D::Int; kwargs...)
    return FiniteDisorderMPS(site_sectors(Hs, ps, L)..., D_phys, D; kwargs...)
end

function FiniteDisorderMPS(ps::Vector{<:Vector{<:Real}}, sectors::Vector{<:Vector{<:Integer}}, D_phys::Int, D::Int; T=ComplexF64, zero_variance::Bool=false)
    L = length(ps)
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
            [Q for _ in ps[i]]
        else
            [first(qr_compact(rand(T, ℂ^Ds[i]⊗ℂ^D_phys, ℂ^Ds[i+1]))) for _ in ps[i]]
        end
    end
    return FiniteDisorderMPS(opp, ps, sectors)
end

# Same disorder sectors (and probabilities) on every site
same_sectors(ρ1::FiniteDisorderMPS, ρ2::FiniteDisorderMPS) = (ρ1.sectors == ρ2.sectors) && (ρ1.ps == ρ2.ps)

# Number of Hamiltonian sectors the site sectors of ρ refer to
num_hamiltonian_sectors(ρ::FiniteDisorderMPS) = maximum(maximum, ρ.sectors)

Base.getindex(ρ::FiniteDisorderMPS, i::Int) = ρ.opp[i]
Base.getindex(ρ::FiniteDisorderMPS, i::Int, p::Int) = ρ.opp[i][p]
Base.size(ρ::FiniteDisorderMPS) = size(ρ.opp)
Base.length(ρ::FiniteDisorderMPS) = length(ρ.opp)
Base.eachindex(ρ::FiniteDisorderMPS) = 1:length(ρ.opp)
Base.iterate(ρ::FiniteDisorderMPS, i=1) = (i > length(ρ.opp)) ? nothing : (ρ[i], i + 1)

function rescale(ρ::FiniteDisorderMPS, α::Number)
    opp = [Ws*α for Ws in ρ.opp]
    return FiniteDisorderMPS(opp, ρ.ps, ρ.sectors)
end

# Construct application left transfer matrix at site i: v*T -> v
function transfer_left(ρ::FiniteDisorderMPS, i::Int)
    function ftransfer(vl)
        v = zeros(ComplexF64, space(ρ[i,1],3)', space(ρ[i,1],3)')
        for (p,W) in enumerate(ρ[i])
            @tensor vp[-2; -1] := W[1 3; -1] * conj(W[2 3; -2]) * vl[2; 1]
            v += ρ.ps[i][p]*vp
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
            v += ρ.ps[i][p]*vp
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
            @tensor ED = l[4; 1] * W[1 2; 5] * Hs.Ds[ρ.sectors[i][p]][3; 2] * conj(W[4 3; 6]) * r[5; 6]
            E += ρ.ps[i][p] * ED
        end
    end
    for i in 1:L-1
        l = ls[i]
        r = rs[i+2]
        for (p, W) in enumerate(ρ[i])
            s = ρ.sectors[i][p]     # the coupling of bond (i, i+1) is set by the sector of site i
            for (q, V) in enumerate(ρ[i+1])
                @tensor ELR = l[11; 1] * W[1 2; 4] * Hs.Ls[s][3; 2 5] * conj(W[11 3; 6]) * V[4 7; 9] * Hs.Rs[s][5 8; 7] * conj(V[6 8; 10]) * r[9; 10]
                E += ρ.ps[i][p] * ρ.ps[i+1][q] * ELR
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
    Os = fill(O, num_hamiltonian_sectors(ρ))

    return expectation_value(ρ, Os, i)
end

# Expectation value of a local disorder operator at site i, Os[s] is the operator of Hamiltonian sector s
function expectation_value(ρ::FiniteDisorderMPS, Os::Vector{<:AbstractBondTensor}, i::Int)
    ls, rs = environments(ρ)
    O = _expectation_value(ρ, Os, i, ls[i], rs[i+1])
    N = tr(ls[i+1] * rs[i+1])
    return real(O / N)
end

# Unnormalized disorder average of Os at site i between the environments l and r
function _expectation_value(ρ::FiniteDisorderMPS, Os::Vector{<:AbstractBondTensor}, i::Int, l, r)
    O = 0
    for (p, W) in enumerate(ρ[i])
        @tensor Op = l[4; 1] * W[1 3; 5] * Os[ρ.sectors[i][p]][2; 3] * conj(W[4 2; 6]) * r[5; 6]
        O += ρ.ps[i][p]*Op
    end
    return O
end

# Expectation value of a local (disorder) operator on every site
function expectation_value(ρ::FiniteDisorderMPS, O::AbstractBondTensor)
    Os = fill(O, num_hamiltonian_sectors(ρ))

    return expectation_value(ρ, Os)
end

function expectation_value(ρ::FiniteDisorderMPS, Os::Vector{<:AbstractBondTensor})
    return [expectation_value(ρ, Os, i) for i in 1:length(ρ)]
end


# Compute correlation function of two local operators at site i and site j
function correlator(ρ::FiniteDisorderMPS, O1::AbstractBondTensor, O2::AbstractBondTensor, i::Int, j::Int)
    O1s = fill(O1, num_hamiltonian_sectors(ρ))
    O2s = fill(O2, num_hamiltonian_sectors(ρ))

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
        @tensor vlO1[-2; -1] := l[2; 1] * W[1 3; -1] * O1s[ρ.sectors[i][p]][4; 3] * conj(W[2 4; -2])
        vl += ρ.ps[i][p]*vlO1
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
            v += ρ1.ps[i][p]*vp
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
            v += ρ1.ps[i][p]*vp
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
            v += ρ1.ps[i][p]*vp
        end
        return v
    end
    return ftransfer
end

function overlap(ρ1::FiniteDisorderMPS, ρ2::FiniteDisorderMPS)
    (length(ρ1) == length(ρ2)) || throw(ArgumentError("ρ1 and ρ2 should have the same length"))
    same_sectors(ρ1, ρ2) || throw(ArgumentError("ρ1 and ρ2 should have the same disorder sectors on every site"))

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
    same_sectors(ρ1, ρ2) || throw(ArgumentError("ρ1 and ρ2 should have the same disorder sectors on every site"))

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

# Entanglement entropy S_w(k) of the conditional state ψ_w of word w on every bond k = 1..L-1. The scaled norm
# environments of _advance (FiniteDisorderFilter.jl) are the Gram matrices l[bra; ket] and r[ket; bra] of the left
# (sites 1..k) and right (sites k+1..L) Schmidt vectors; the Schmidt spectrum is eig(r * l) normalized to one, so ψ_w
# does not have to be normalized
#
# With charges (the U(1) charge of every physical basis state, e.g. 0:nmax bosons) ψ_w is first projected onto its most
# likely total charge N*. A conditional state of a charge conserving H can be a superposition of charges (ITE from a
# state that contains every N); the charge fluctuations then no longer entangle the two halves, which lowers S most in
# the bulk. The projected state is block diagonal in the charge c left of the cut, so its Schmidt spectrum is the union
# over c of the spectra of the charge-resolved Gram matrices r[N* - c] * l[c]
function conditional_entanglement_entropy(ρ::FiniteDisorderMPS, word::AbstractVector{Int}; charges = nothing)
    isnothing(charges) || return _projected_entanglement_entropy(ρ, word, charges)
    L = length(word)
    lefts = Vector{Any}(undef, L)
    rights = Vector{Any}(undef, L+1)
    lefts[1] = _boundary_environment(space(ρ[1, 1], 1))
    for i in 1:L-1
        lefts[i+1] = _advance(lefts[i], ρ[i, word[i]])
    end
    rights[L+1] = _boundary_environment(domain(ρ[L, 1])[1])
    for i in L:-1:2
        rights[i] = _advance(rights[i+1], ρ[i, word[i]]; right = true)
    end
    return map(1:L-1) do k
        λ = max.(real.(eigvals(convert(Array, rights[k+1][1] * lefts[k+1][1]))), 0.0)
        λ ./= sum(λ)
        -sum(x -> x > 0 ? x * log(x) : 0.0, λ)
    end
end

# Charge-resolved Gram matrices of the conditional state of word w: lefts[i][c] of sites 1..i-1 with total charge c
# (G_L[b̄, b] = Σ conj(X_b̄) X_b) and rights[i][c] of sites i..L (G_R[a, ā] = Σ Y_a conj(Y_ā)). All charge blocks of a
# site are divided by one common factor, which cancels once the spectrum is normalized
function _charge_environments(ρ::FiniteDisorderMPS, word::AbstractVector{Int}, charges)
    L = length(word)
    As = [convert(Array, ρ[i, p]) for (i, p) in enumerate(word)]   # A[a, s, b]
    all(size(A, 2) == length(charges) for A in As) || throw(ArgumentError("charges should have one entry per physical basis state"))
    rescale!(E) = (m = maximum(e -> LinearAlgebra.norm(e, Inf), values(E)); m > 0 && foreach(c -> E[c] ./= m, keys(E)); E)
    lefts = Vector{Dict{Int, Matrix{ComplexF64}}}(undef, L+1)
    rights = Vector{Dict{Int, Matrix{ComplexF64}}}(undef, L+1)
    lefts[1] = Dict(0 => ones(ComplexF64, 1, 1))
    for i in 1:L
        E = Dict{Int, Matrix{ComplexF64}}()
        for (c, G) in lefts[i], (s, n) in enumerate(charges)
            As_ = As[i][:, s, :]
            E[c+n] = get(E, c+n, zeros(ComplexF64, size(As_, 2), size(As_, 2))) + As_' * G * As_
        end
        lefts[i+1] = rescale!(E)
    end
    rights[L+1] = Dict(0 => ones(ComplexF64, 1, 1))
    for i in L:-1:1
        E = Dict{Int, Matrix{ComplexF64}}()
        for (c, G) in rights[i+1], (s, n) in enumerate(charges)
            As_ = As[i][:, s, :]
            E[c+n] = get(E, c+n, zeros(ComplexF64, size(As_, 1), size(As_, 1))) + As_ * G * As_'
        end
        rights[i] = rescale!(E)
    end
    return lefts, rights
end

# Distribution p_N of the total charge in the conditional state of word w (charges as in conditional_entanglement_entropy)
function conditional_number_distribution(ρ::FiniteDisorderMPS, word::AbstractVector{Int}, charges)
    lefts, _ = _charge_environments(ρ, word, charges)
    p = Dict(N => max(real(tr(G)), 0.0) for (N, G) in lefts[end])
    Z = sum(values(p))
    return Dict(N => x / Z for (N, x) in p)
end

function _projected_entanglement_entropy(ρ::FiniteDisorderMPS, word::AbstractVector{Int}, charges)
    L = length(word)
    lefts, rights = _charge_environments(ρ, word, charges)
    N = argmax(N -> real(tr(lefts[end][N])), collect(keys(lefts[end])))
    return map(1:L-1) do k
        λ = Float64[]
        for (c, G) in lefts[k+1]
            haskey(rights[k+1], N - c) || continue
            append!(λ, max.(real.(eigvals(rights[k+1][N - c] * G)), 0.0))
        end
        λ ./= sum(λ)
        -sum(x -> x > 0 ? x * log(x) : 0.0, λ)
    end
end

# Disorder average 𝔼_w[S_w(k)] of the entanglement entropy on every bond k = 1..L-1, its standard error and the disorder
# variance Var_w[S_w(k)] = 𝔼_w[S_w(k)²] - 𝔼_w[S_w(k)]², over words w = (p₁…p_L) with pᵢ ~ ρ.ps[i] independent. If there are at
# most n_samples words they are all enumerated (exact, zero error), otherwise n_samples words are sampled. With charges
# the entropies of the conditional states projected onto their most likely total charge (conditional_entanglement_entropy)
function average_entanglement_entropy(ρ::FiniteDisorderMPS; n_samples::Int = 1000, rng::AbstractRNG = Random.default_rng(),
                                     charges = nothing)
    L = length(ρ)
    if prod(float.(length.(ρ.ps))) <= n_samples
        S_avg = zeros(Float64, L-1)
        S2_avg = zeros(Float64, L-1)
        for word in ProgressBar(Iterators.product(eachindex.(ρ.ps)...))
            w = prod(ρ.ps[i][p] for (i, p) in enumerate(word))
            iszero(w) && continue
            S_w = conditional_entanglement_entropy(ρ, collect(word); charges)
            S_avg .+= w .* S_w
            S2_avg .+= w .* S_w.^2
        end
        return S_avg, zeros(Float64, L-1), max.(S2_avg .- S_avg.^2, 0.0)
    end

    Ss = reduce(hcat, [conditional_entanglement_entropy(ρ, sample_word(rng, ρ.ps); charges) for _ in ProgressBar(1:n_samples)])
    return vec(mean(Ss; dims=2)), vec(std(Ss; dims=2)) ./ sqrt(n_samples), vec(var(Ss; dims=2))
end

# Disorder word with the sector of site i drawn from ps[i]
sample_word(rng::AbstractRNG, ps::Vector{<:Vector{<:Real}}) = [sample(rng, eachindex(p), Weights(p)) for p in ps]

# Disorder averaged entanglement entropy on the bond between site k and k+1, its standard error and its disorder variance
function average_entanglement_entropy(ρ::FiniteDisorderMPS, k::Int; kwargs...)
    Ss, Ss_err, Ss_var = average_entanglement_entropy(ρ; kwargs...)
    return Ss[k], Ss_err[k], Ss_var[k]
end

# Per sample w: the mean S̄_w and standard deviation σ_w of S_w(k) across the bonds k = 1..L-1. Returns the averages over
# n_samples sampled words of S̄_w and σ_w, with their standard errors: (𝔼[S̄_w], err, 𝔼[σ_w], err). As in
# average_entanglement_entropy, if there are at most n_samples words they are all enumerated (exact, zero errors)
function bond_averaged_entanglement_entropy(ρ::FiniteDisorderMPS; n_samples::Int = 1000,
                                            rng::AbstractRNG = Random.default_rng(), charges = nothing)
    if prod(float.(length.(ρ.ps))) <= n_samples
        S_mean, S_std = 0.0, 0.0
        for word in ProgressBar(Iterators.product(eachindex.(ρ.ps)...))
            w = prod(ρ.ps[i][p] for (i, p) in enumerate(word))
            iszero(w) && continue
            S_w = conditional_entanglement_entropy(ρ, collect(word); charges)
            S_mean += w * mean(S_w)
            S_std += w * std(S_w)
        end
        return S_mean, 0.0, S_std, 0.0
    end

    S_means = zeros(Float64, n_samples)
    S_stds = zeros(Float64, n_samples)
    for n in ProgressBar(1:n_samples)
        S_w = conditional_entanglement_entropy(ρ, sample_word(rng, ρ.ps); charges)
        S_means[n] = mean(S_w)
        S_stds[n] = std(S_w)
    end
    return mean(S_means), std(S_means) / sqrt(n_samples), mean(S_stds), std(S_stds) / sqrt(n_samples)
end
