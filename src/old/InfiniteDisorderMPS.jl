# Convention virtual, physical ← virtual

# One site unit-cell isometric infiniteDMPS
struct InfiniteDisorderMPS{T<:AbstractMPSTensor}
    opp::Vector{T}
    ps::Vector{<:Real}
end

function InfiniteDisorderMPSC(ps::Vector{Float64}, D_dis::Int, D_phys::Int, D::Int; T=ComplexF64)
    As = [rand(T, ℂ^D⊗ℂ^D_phys,ℂ^D) for i in 1:D_dis]
    for (p,A) in enumerate(As)
        Q, _ = qr_compact(A)
        As[p] = Q
    end
    return InfiniteDisorderMPS{typeof(As[1])}(As, ps)
end

function InfiniteDisorderMPS(ps::Vector{Float64}, D_dis::Int, V_phys::GradedSpace, V_virt::GradedSpace; T=ComplexF64)
    As = [TensorMap(rand, T, V_virt⊗V_phys,V_virt) for i in 1:D_dis]
    for (p,A) in enumerate(As)
        Q, _ = qr_compact(A, ((1,2),(3,)))
        As[p] = Q
    end
    return InfiniteDisorderMPS{typeof(As[1])}(As, ps)
end

function expand(ρ::InfiniteDisorderMPS, d::Int)
    D = dim(space(ρ.opp[1], 1))
    A = rand(ComplexF64, ℂ^(D*d)⊗ℂ^2, ℂ^(D*d))
    ϵ = 0.0001
    # A = id(ComplexF64, ℂ^d)
    iso = isomorphism(ComplexF64, ℂ^(D*d), ℂ^D ⊗ ℂ^d)
    opp = Vector{typeof(ρ.opp[1])}(undef, length(ρ.opp))
    for i in 1:length(ρ.opp)
        @tensor W[-1 -2; -3] := iso[-1; 1 2] * ρ.opp[i][1 -2; 3] * conj(iso[-3; 3 2])
        W = W + ϵ*A
        Q, _ = leftorth(W, ((1,2),(3,)))
        opp[i] = Q
    end

    return InfiniteDisorderMPS(opp, ρ.ps)
end

Base.getindex(T::InfiniteDisorderMPS, ix::Int) = T.opp[ix]
Base.size(T::InfiniteDisorderMPS) = size(T.opp)
Base.length(T::InfiniteDisorderMPS) = length(T.opp)
Base.eachindex(T::InfiniteDisorderMPS) = 1:length(T.opp)
Base.iterate(t::InfiniteDisorderMPS, i=1) = (i > length(t.opp)) ? nothing : (t[i], i + 1)


# Construct application left transfer matrix: v*T -> v
function transfer_left(ρ::InfiniteDisorderMPS)
    function ftransfer(vl)
        v = zeros(ComplexF64,space(ρ.opp[1],3)',space(ρ.opp[1],3)')
        for (p,W) in enumerate(ρ.opp)
            @tensor vp[-2; -1] := W[1 3; -1] * conj(W[2 3; -2]) * vl[2; 1]
            v += ρ.ps[p]*vp
        end
        return v
    end
    return ftransfer
end

# Construct application right transfer matrix: T*v -> v
function transfer_right(ρ::InfiniteDisorderMPS)
    function ftransfer(vr)
        v = zeros(ComplexF64,space(ρ.opp[1],1),space(ρ.opp[1],1))
        for (p,W) in enumerate(ρ.opp)
            @tensor vp[-1; -2] := W[-1 3; 1] * conj(W[-2 3; 2]) * vr[1; 2]
            v += ρ.ps[p]*vp
        end
        return v
    end
    return ftransfer
end

# Compute right environment of InfiniteDisorderMPS
function right_environment(ρ::InfiniteDisorderMPS)
    v0 = id(ComplexF64, space(ρ.opp[length(ρ.opp)], 3)')
    f_t = transfer_right(ρ)
    λ, Er = eigsolve(f_t, v0, 2, :LM)

    Er = Er[1]/tr(Er[1])
    return λ[1], Er
end

# Compute left environment of InfiniteDisorderMPS
function left_environment(ρ::InfiniteDisorderMPS)
    v0 = id(ComplexF64, space(ρ.opp[1], 1))
    f_t = transfer_left(ρ)
    λ, El = eigsolve(f_t, v0, 1, :LM)

    # El = El[1]/tr(El[1])
    El = El[1]/El[1][1]
    return λ[1], El
end

# Compute the norm of a InfiniteDisorderMPS
function norm(ρ::InfiniteDisorderMPS)
    _, Er = right_environment(ρ)

    N = tr(Er)
    imag(N) < 1e-4 || @warn("Norm has imaginary part: N = $N")
    N = real.(N)
    return N
end

function effective_couplings(ρ::InfiniteDisorderMPS, Hs::DisorderMPOHam)
    r = right_environment(ρ)[2]

    Js = Float64[]
    hs = Float64[]
    for (p, W) in enumerate(ρ.opp)
        @tensor ED = W[1 2; 4] * Hs.Ds[p][3; 2] * conj(W[1 3; 5]) * r[4;5]
        push!(hs, abs.(ED))
        for (q, V) in enumerate(ρ.opp)
            @tensor ECB = W[1 2; 4] * Hs.Cs[p][3; 2 5] * conj(W[1 3; 6]) * V[4 7; 9] * Hs.Bs[p][5 8; 7] * conj(V[6 8; 10]) * r[9;10]
            push!(Js, abs.(ECB))
        end
        #FIXME : currently only nearest-neighbor interactions
    end

    return hs, Js
end

# Compute the energy density of a InfiniteDisorderMPS with respect to a DisorderMPOHam
function energy_density(ρ::InfiniteDisorderMPS, Hs::DisorderMPOHam)
    r = right_environment(ρ)[2]

    E = 0
    for (p, W) in enumerate(ρ.opp)
        @tensor ED = W[1 2; 4] * Hs.Ds[p][3; 2] * conj(W[1 3; 5]) * r[4;5]
        E += ρ.ps[p] * ED
        for (q, V) in enumerate(ρ.opp)
            # @tensor EDW = W[1 2; 4] * Hs.Ds[p][3; 2] * conj(W[1 3; 6]) * V[4 7; 9] * conj(V[6 7; 10]) * r[9;10]
            # E += ρ.ps[p] * ρ.ps[q] * EDW / 2
            # @tensor EDV = W[1 2; 4] * Hs.Ds[q][7; 8] * conj(W[1 2; 6]) * V[4 8; 9] * conj(V[6 7; 10]) * r[9;10]
            # E += ρ.ps[p] * ρ.ps[q] * EDV / 2
            @tensor ECB = W[1 2; 4] * Hs.Cs[p][3; 2 5] * conj(W[1 3; 6]) * V[4 7; 9] * Hs.Bs[p][5 8; 7] * conj(V[6 8; 10]) * r[9;10]
            E += ρ.ps[p] * ρ.ps[q] * ECB
        end
        #FIXME : currently only nearest-neighbor interactions
    end

    imag(E) < 1e-4 || @warn("Energy density has imaginary part: E = $E")

    return real.(E)
end

function median_energy_density(ρ::InfiniteDisorderMPS, Hs::DisorderMPOHam; λ = 1.)
    r = right_environment(ρ)[2]

    E = 0
    N = 0
    for (p, W) in enumerate(ρ.opp)
        @tensor ED = W[1 2; 4] * Hs.Ds[p][3; 2] * conj(W[1 3; 5]) * r[4;5]
        for (q, V) in enumerate(ρ.opp)
            @tensor ECB = W[1 2; 4] * Hs.Cs[p][3; 2 5] * conj(W[1 3; 6]) * V[4 7; 9] * Hs.Bs[p][5 8; 7] * conj(V[6 8; 10]) * r[9;10]
            E += ρ.ps[p] * ρ.ps[q] *(ECB + ED) * exp(λ * (ECB + ED))
            N += ρ.ps[p] * ρ.ps[q] * exp(λ * (ECB + ED))
        end
        #FIXME : currently only nearest-neighbor interactions
    end

    E = E/N
    imag(E) < 1e-4 || @warn("Energy density has imaginary part: E = $E")

    return real.(E)
end

function energy_density_dist(ρ::InfiniteDisorderMPS, Hs::DisorderMPOHam)
    r = right_environment(ρ)[2]

    Es = Float64[]
    for (p, W) in enumerate(ρ.opp)
        @tensor ED = W[1 2; 4] * Hs.Ds[p][3; 2] * conj(W[1 3; 5]) * r[4;5]
        for (q, V) in enumerate(ρ.opp)
            # @tensor EDW = W[1 2; 4] * Hs.Ds[p][3; 2] * conj(W[1 3; 6]) * V[4 7; 9] * conj(V[6 7; 10]) * r[9;10]
            # E += ρ.ps[p] * ρ.ps[q] * EDW / 2
            # @tensor EDV = W[1 2; 4] * Hs.Ds[q][7; 8] * conj(W[1 2; 6]) * V[4 8; 9] * conj(V[6 7; 10]) * r[9;10]
            # E += ρ.ps[p] * ρ.ps[q] * EDV / 2
            @tensor ECB = W[1 2; 4] * Hs.Cs[p][3; 2 5] * conj(W[1 3; 6]) * V[4 7; 9] * Hs.Bs[p][5 8; 7] * conj(V[6 8; 10]) * r[9;10]
            push!(Es, real(ECB + ED))
        end
        #FIXME : currently only nearest-neighbor interactions
    end

    return Es
end


# Expectation value of a local operator O at site p
function expectation_value(ρ::InfiniteDisorderMPS, O::AbstractBondTensor)
    Os = [O for i in 1:length(ρ.opp)]

    return expectation_value(ρ, Os)
end

# Expectation value of a local disorder operator O at site p
function expectation_value(ρ::InfiniteDisorderMPS, Os::Vector{<:AbstractBondTensor})
    _, vr = right_environment(ρ)    
    vl = zeros(ComplexF64, space(ρ.opp[1],3)',space(ρ.opp[1],3)')

    @show space(vl)
    for (p, W) in enumerate(ρ.opp)
        @tensor vlO1[-1; -2] := W[1 3; -2] * Os[p][2; 3] * conj(W[1 2; -1]) 
        vl += ρ.ps[p]*vlO1
    end

    return tr(vl * vr)
end


# Compute correlation function of two local operators at site i and site j
function correlator(ρ::InfiniteDisorderMPS, O1::AbstractBondTensor, O2::AbstractBondTensor, i::Int, j::Int)
    O1s = [O1 for i in 1:length(ρ.opp)]
    O2s = [O2 for i in 1:length(ρ.opp)]

    return correlator(ρ, O1s, O2s, i, j)
end

# Compute correlation function of two local operators at distance r
function correlator(ρ::InfiniteDisorderMPS, O1s::Vector{<:AbstractBondTensor}, O2s::Vector{<:AbstractBondTensor}, i::Int, j::Int)
    Cs = Vector{ComplexF64}(undef, j-i)
    _, vr = right_environment(ρ)
    f_l = transfer_left(ρ)
    f_r = transfer_right(ρ)
    vl = zeros(ComplexF64, space(ρ.opp[1],3)',space(ρ.opp[1],3)')
    vrt = zeros(ComplexF64, space(vr))
    for (p, W) in enumerate(ρ.opp)
        @tensor vlO1[-1; -2] := W[1 3; -2] * O1s[p][2; 3] * conj(W[1 2; -1])
        vl += ρ.ps[p]*vlO1
    end
    for (q, W) in enumerate(ρ.opp)
        @tensor vrO2[-1; -2] := W[-1 3; 1] * O2s[q][2; 3] * conj(W[-2 2; 4]) * vr[1; 4]
        vrt += ρ.ps[q]*vrO2
    end
    Cs[1] = tr(vl * vrt)
    for k in 1:j-i
        vl = f_l(vl)
        C = tr(vl * vrt)
        Cs[k] = C
    end
    return real.(Cs)
end

function average_correlation_length(ρ::InfiniteDisorderMPS)
    f_l = transfer_left(ρ)

    v0 = rand(ComplexF64, space(ρ.opp[1],1), space(ρ.opp[1],1))
    λl, _ = eigsolve(f_l, v0, 3, :LM)

    ξ = -1/log.(abs(λl[2]))
    return ξ
end

function typical_correlation_length(ρ::InfiniteDisorderMPS; L::Int = 100, Nsamples::Int = 1000)
    D_disorder = length(ρ.ps)


    λfs1 = Float64[]
    λfs2 = Float64[]

    for _ in ProgressBar(1:Nsamples)

        λs1 = Float64[]
        λs2 = Float64[]

        u1 = rand(ComplexF64,space(ρ.opp[1],1)⊗space(ρ.opp[1],1)')
        u1 /= TensorKit.norm(u1)
        u2 = rand(ComplexF64,space(ρ.opp[1],1)⊗space(ρ.opp[1],1)')
        u2 /= TensorKit.norm(u2)

        for n in 1:L
            sample = rand(1:D_disorder, 1)
            W = ρ.opp[sample[1]]
            
            @tensor Ap[-1 -2; -3 -4] := W[-1 1; -3] * conj(W[-2 1; -4])

            u1 = Ap*u1
            u2 = Ap*u2
            push!(λs1, 1/n*log(TensorKit.norm(u1)))
            push!(λs2, 1/n*log(TensorKit.norm(u2)))
            if n==L
                push!(λfs1, 1/n*log(TensorKit.norm(u1)))
                push!(λfs2, 1/n*log(TensorKit.norm(u2)))
            end
            u2 = u2 - u1*u1'*u2
        end
    end
    ξs = 1 ./(λfs1 .- λfs2)

    filter!(x -> x<0, λfs2)
    filter!(x -> x>0, ξs)


    ξt = StatsBase.geomean(ξs)
    return ξt
end

function average_entanglement_entropy(ρ::InfiniteDisorderMPS)
    _, r = right_environment(ρ)
    # U, S, V = tsvd(r)
    # S = S.data

    # Se = -sum(S.^2 .*log.(S.^2 .+ 1e-16))

    D, V = eig_full(r)
    S = D.data
    Se = real.(-sum(S .*log.(S .+ 1e-16)))
    return Se, real.(S)
end

