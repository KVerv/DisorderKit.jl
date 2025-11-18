# Convention virtual, physical, disorder ← virtual

struct FiniteDisorderMPS{T<:AbstractMPSTensor}
    opp::Vector{Vector{T}}
end

function FiniteDisorderMPS(opp::Vector{<:Vector{<:AbstractMPSTensor}})
    return FiniteDisorderMPS(opp)
end

function FiniteDisorderMPS(L::Int, D_dis::Int, D_phys::Int, D::Int; T=ComplexF64)
    As = Vector{Vector{AbstractMPSTensor}}(undef, L)
    As[1] = [TensorMap(rand, T, ℂ^1⊗ℂ^D_phys,ℂ^D_phys) for i in 1:D_dis]
    for j in 2:L-1
        Dd = dim(domain(As[j-1][1]))
        Dc = Dd * D_phys
        if Dc < D
            As[j] = [TensorMap(rand, T, ℂ^Dd⊗ℂ^D_phys,ℂ^Dc) for i in 1:D_dis]
        else
            As[j] = [TensorMap(rand, T, ℂ^Dd⊗ℂ^D_phys,ℂ^D) for i in 1:D_dis]
        end
    end
    D_L = dim(domain(As[L-1][1]))
    As[L] = [TensorMap(rand, T, ℂ^D_L⊗ℂ^D_phys,ℂ^1) for i in 1:D_dis]
    return FiniteDisorderMPS(As)
end

Base.getindex(T::FiniteDisorderMPS, ix::Int) = T.opp[ix]
Base.size(T::FiniteDisorderMPS) = size(T.opp)
Base.length(T::FiniteDisorderMPS) = length(T.opp)
Base.eachindex(T::FiniteDisorderMPS) = 1:length(T.opp)
Base.iterate(t::FiniteDisorderMPS, i=1) = (i > length(t.opp)) ? nothing : (t[i], i + 1)


# Construct application left transfer matrix: v*T -> v
function transfer_left(As::Vector{<:AbstractMPSTensor})
    D = length(As)
    function ftransfer(vl)
        v = zeros(ComplexF64,space(As[1],3)',space(As[1],3)')
        for A in As
            @tensor vp[-2; -1] := A[1 3; -1] * conj(A[2 3; -2]) * vl[2; 1]
            v += vp
        end
        return v/D
    end
    return ftransfer
end

# Construct application right transfer matrix: T*v -> v
function transfer_right(As::Vector{<:AbstractMPSTensor})
    D = length(As)
    function ftransfer(vr)
        v = zeros(ComplexF64,space(As[1],1),space(As[1],1))
        for A in As
            @tensor vp[-1; -2] := A[-1 3; 1] * conj(A[-2 3; 2]) * vr[1; 2]
            v += vp
        end
        return v/D
    end
    return ftransfer
end

# Construct application left transfer matrix: v*T -> v
function Otransfer_left(As::Vector{<:AbstractMPSTensor}, Os::Vector{<:AbstractMPOTensor})
    D = length(As)
    function ftransfer(vl)
        v = zeros(ComplexF64,space(Os[1], 1)⊗space(As[1],3)',space(As[1],3)'⊗space(Os[1], 1))
        for (p,A) in enumerate(As)
            @tensor vp[-1 -2; -3 -4] := A[1 4; -3] *Os[p][2 5; 4 -4] * conj(A[3 5; -2]) * vl[-1 3; 1 2]
            v += vp
        end
        return v/D
    end
    return ftransfer
end

# Construct application left transfer matrix: v*T -> v
function Otransfer_left(As::Vector{<:AbstractMPSTensor}, O::AbstractBondTensor)
    D = length(As)
    function ftransfer(vl)
        v = zeros(ComplexF64,space(As[1],3)',space(As[1],3)')
        for A in As
            @tensor vp[-2; -1] := A[1 3; -1] * O[4;3] * conj(A[2 4; -2]) * vl[2; 1]
            v += vp
        end
        return v/D
    end
    return ftransfer
end

# Right environment at site i
function right_env(ρs::FiniteDisorderMPS, i::Int)
    vr = id(ComplexF64, space(ρs[length(ρs)][1], 3)')
    for j in length(ρs):-1:i+1
        f_t = transfer_right(ρs[j])
        vr = f_t(vr)
    end
    return vr
end

# Bring a DisorderMPS to left gauge
function left_gauge(As::Vector{<:Vector{<:AbstractMPSTensor}})
    ALs = Vector{AbstractMPSTensor}[]
    for A in As 
        AL = AbstractMPSTensor[]
        for Ap in A
            Q, _ = leftorth(Ap, ((1,2),(3,)))
            push!(AL, Q)
        end
        push!(ALs, AL)
    end
    return ALs
end

function left_gauge(As::FiniteDisorderMPS)
    return FiniteDisorderMPS(left_gauge(As.opp))
end

# Compute the norm of a DisorderMPS
function overlap(ρs::FiniteDisorderMPS)
    vl = id(ComplexF64, space(ρs[1][1], 1))
    for As in ρs
        vl = transfer_left(As)(vl)
    end
    @tensor N = vl[1 ;1]
    imag(N) < 1e-4 || @warn("Norm has imaginary part: N = $N")
    N = real.(N)
    return N
end

# Compute the expectation value of a MPO (Open boundary conditions) for a fixed disorder value p on site j
function measure(Ap::AbstractMPSTensor, j::Int, p::Int, ρs::FiniteDisorderMPS, Os::Vector{<:AbstractMPOTensor})
    vl1 = id(ComplexF64, space(Os[1], 1))
    vl2 = id(ComplexF64, space(ρs[1][1], 1))
    @tensor vl[-1 -2; -3 -4] := vl1[-1; -4] * vl2[-2; -3] 
    for (i, As) in enumerate(ρs)
        if i == j
            vl = Otransfer_left([Ap], [Os[p]])(vl)
        else
            vl = Otransfer_left(As, Os)(vl)
        end
    end
    P = TensorMap([0. 0. 0.; 0. 0. 0.; 1. 0. 0.], ℂ^3, ℂ^3)
    @tensor E = vl[3 1 ;1 2] * P[2; 3]
    E = real.(E)/overlap(ρs)
    imag(E) < 1e-4 || @warn("Energy has imaginary part: E = $E")
    return E
end

# Compute the expectation value of a MPO (Open boundary conditions)
function measure(ρs::FiniteDisorderMPS, Os::Vector{<:AbstractMPOTensor})
    vl1 = id(ComplexF64, space(Os[1], 1))
    vl2 = id(ComplexF64, space(ρs[1][1], 1))
    @tensor vl[-1 -2; -3 -4] := vl1[-1; -4] * vl2[-2; -3] 
    for (i, As) in enumerate(ρs)
        vl = Otransfer_left(As, Os)(vl)
    end
    P = TensorMap([0. 0. 0.; 0. 0. 0.; 1. 0. 0.], ℂ^3, ℂ^3)
    @tensor E = vl[3 1 ;1 2] * P[2; 3]
    E = real.(E)/overlap(ρs)
    imag(E) < 1e-4 || @warn("Energy has imaginary part: E = $E")
    return E
end

# Compute the correlator of a local operator O₁ on site i and  O₂ on site j
function measure(ρs::FiniteDisorderMPS, O₁::AbstractBondTensor, O₂::AbstractBondTensor, i::Int, j::Int)
    vl = id(ComplexF64, space(ρs[1][1], 1))
    for (k, As) in enumerate(ρs)
        if k == i
            vl = Otransfer_left(As, O₁)(vl)
        elseif k == j
            vl = Otransfer_left(As, O₂)(vl)
        else
            vl = transfer_left(As)(vl)
        end
    end
    @tensor E = vl[1 ;1]
    E = real.(E)/overlap(ρs)
    imag(E) < 1e-4 || @warn("Energy has imaginary part: E = $E")
    return E
end

# Compute the correlator of local operators O₁ and  O₂ seperated by distance r
function measure(ρs::FiniteDisorderMPS, O₁::AbstractBondTensor, O₂::AbstractBondTensor, r::Int)
    E = 0
    n = 0
    for i in 1:(length(ρs)-r)
        Ep = measure(ρs, O₁, O₂, i, i+r)
        @show Ep
        E += Ep
        n += 1
    end
    @show E, n
    @show "hello there"
    return E/(n)
end

# Compute the expectation value of a local operator O₁ on site i
function measure(ρs::FiniteDisorderMPS, O₁::AbstractBondTensor, i::Int)
    vl = id(ComplexF64, space(ρs[1][1], 1))
    for (k, As) in enumerate(ρs)
        if k == i
            vl = Otransfer_left(As, O₁)(vl)
        else
            vl = transfer_left(As)(vl)
        end
    end
    @tensor E = vl[1 ;1]
    E = real.(E)/overlap(ρs)
    imag(E) < 1e-4 || @warn("Energy has imaginary part: E = $E")
    return E
end

# Compute the expectation value of a local operator O₁
function measure(ρs::FiniteDisorderMPS, O₁::AbstractBondTensor)
    E = 0
    L = length(ρs)
    for i in 1:L
        E += measure(ρs, O₁, i)
    end
    return E/L
end