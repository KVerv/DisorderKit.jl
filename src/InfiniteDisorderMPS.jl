# Convention virtual, physical ← virtual

# One site unit-cell isometric InfiniteDMPS
struct InfiniteDisorderMPS{T<:AbstractMPSTensor}
    opp::Vector{T}
    ps::Vector{<:Real}
end

function InfiniteDisorderMPS(ps::Vector{Float64}, D_dis::Int, D_phys::Int, D::Int; T=ComplexF64)
    As = [rand(T, ℂ^D⊗ℂ^D_phys,ℂ^D) for i in 1:D_dis]
    for (p,A) in enumerate(As)
        Q, _ = qr_compact(A)
        As[p] = Q
    end
    return InfiniteDisorderMPS{typeof(As[1])}(As, ps)
end

Base.getindex(T::InfiniteDisorderMPS, ix::Int) = T.opp[ix]
Base.size(T::InfiniteDisorderMPS) = size(T.opp)
Base.length(T::InfiniteDisorderMPS) = length(T.opp)
Base.eachindex(T::InfiniteDisorderMPS) = 1:length(T.opp)
Base.iterate(t::InfiniteDisorderMPS, i=1) = (i > length(t.opp)) ? nothing : (t[i], i + 1)

function rescale(ρ::InfiniteDisorderMPS, α::Number)
    opp = ρ.opp*α
    return InfiniteDisorderMPS(opp, ρ.ps)
end

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

    El = El[1]/El[1][1]
    return λ[1], El
end

function environments(ρ::InfiniteDisorderMPS)
    λr, Er = right_environment(ρ)
    λl, El = left_environment(ρ)

    return λl, El, Er
end

# Compute the norm of a InfiniteDisorderMPS
function norm(ρ::InfiniteDisorderMPS)
    _, Er = right_environment(ρ)

    N = tr(Er)
    imag(N) < 1e-4 || @warn("Norm has imaginary part: N = $N")
    N = real.(N)
    return N
end

# Compute the energy density of a InfiniteDisorderMPS with respect to a DisorderMPOHam
function energy_density(ρ::InfiniteDisorderMPS, Hs::DisorderMPOHam)
    λ, l, r = environments(ρ)

    E = 0
    for (p, W) in enumerate(ρ.opp)
        @tensor ED = l[4; 1] * W[1 2; 5] * Hs.Ds[p][3; 2] * conj(W[4 3; 6]) * r[5; 6]
        E += ρ.ps[p] * ED/λ
        for (q, V) in enumerate(ρ.opp)
            @tensor ECB = W[1 2; 4] * Hs.Cs[p][3; 2 5] * conj(W[1 3; 6]) * V[4 7; 9] * Hs.Bs[p][5 8; 7] * conj(V[6 8; 10]) * r[9;10]
            E += ρ.ps[p] * ρ.ps[q] * ECB/λ^2
        end
        #FIXME : currently only nearest-neighbor interactions
    end

    imag(E) < 1e-4 || @warn("Energy density has imaginary part: E = $E")

    return real.(E)
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

    if length(λl) < 2
        @warn("Only one eigenvalue found for the left transfer matrix. Cannot compute correlation length.")
        ξ = 0
    else
        ξ = -1/log.(abs(λl[2]))
    end
    return ξ
end

function transfer_mixed_right(ρ1::InfiniteDisorderMPS, ρ2::InfiniteDisorderMPS)
    v1space = space(ρ1.opp[1],1)
    v2space = space(ρ2.opp[1],1)
    function ftransfer(vr)
        v = zeros(ComplexF64,v2space, v1space)
        for (p,X) in enumerate(ρ1.opp)
            Y = ρ2.opp[p]
            @tensor vp[-1; -2] := Y[-1 3; 1] * conj(X[-2 3; 2]) * vr[1; 2]
            v += ρ1.ps[p]*vp
        end
        return v
    end
    return ftransfer
end


function transfer_mixed2_right(ρ1::InfiniteDisorderMPS, ρ2::InfiniteDisorderMPS)
    v1space = space(ρ1.opp[1],1)
    v2space = space(ρ2.opp[1],1)
    function ftransfer(vr)
        v = zeros(ComplexF64,v2space ⊗ v1space', v1space' ⊗ v2space)
        for (p,X) in enumerate(ρ1.opp)
            Y = ρ2.opp[p]
            @tensor vp[-1 -2; -3 -4] := Y[-1 2; 1] * conj(X[-2 2; 3]) * X[-3 5; 4] * conj(Y[-4 5; 6]) * vr[1 3; 4 6]
            v += ρ1.ps[p]*vp
        end
        return v
    end
    return ftransfer
end

function overlap(ρ1::InfiniteDisorderMPS, ρ2::InfiniteDisorderMPS)
    (length(ρ1.opp) == length(ρ2.opp)) || throw(ArgumentError("ρ1 and ρ2 should have the same amount of disorder sectors"))
    
    v1space = space(ρ1.opp[1],1)
    v2space = space(ρ2.opp[1],1)
    v0 = ones(ComplexF64,v2space, v1space)
    f_t = transfer_mixed_right(ρ1, ρ2)
    λ, _ = eigsolve(f_t, v0, 1, :LM)

    return λ[1]
end


function overlap_squared(ρ1::InfiniteDisorderMPS, ρ2::InfiniteDisorderMPS)
    (length(ρ1.opp) == length(ρ2.opp)) || throw(ArgumentError("ρ1 and ρ2 should have the same amount of disorder sectors"))
    
    v1space = space(ρ1.opp[1],1)
    v2space = space(ρ2.opp[1],1)
    v0 = ones(ComplexF64,v2space ⊗ v1space', v1space' ⊗ v2space)
    f_t = transfer_mixed2_right(ρ1, ρ2)
    λ, _ = eigsolve(f_t, v0, 1, :LM)

    return λ[1]
end

function fidelity(ρ1::InfiniteDisorderMPS, ρ2::InfiniteDisorderMPS)
    # ov11 = overlap(ρ1, ρ1)
    ov22 = overlap(ρ2, ρ2)
    ov12 = overlap_squared(ρ1, ρ2)

    F = ov12/ov22
    return abs(F)
end
