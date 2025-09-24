# Convention virtual, physical, disorder ← virtual

struct FiniteDisorderMPS{T<:AbstractMPSTensor}
    opp::Vector{Vector{T}}
end

function FiniteDisorderMPS(opp::Vector{<:Vector{<:AbstractMPSTensor}})
    return FiniteDisorderMPS(opp)
end


Base.getindex(T::FiniteDisorderMPS, ix::Int) = T.opp[ix]
Base.size(T::FiniteDisorderMPS) = size(T.opp)
Base.length(T::FiniteDisorderMPS) = length(T.opp)
Base.eachindex(T::FiniteDisorderMPS) = 1:length(T.opp)
Base.iterate(t::FiniteDisorderMPS, i=1) = (i > length(t.opp)) ? nothing : (t[i], i + 1)


# Construct application left transfer matrix: v*T -> v
function transfer_left(As::Vector{<:AbstractMPSTensor})
    function ftransfer(vl)
        v = zeros(ComplexF64,space(vl))
        for A in As
            @tensor vp[-2; -1] := A[1 3; -1] * conj(A[2 3; -2]) * vl[2; 1]
            v += vp
        end
        return v
    end
    return ftransfer
end

# Construct application rightt transfer matrix: T*v -> v
function transfer_right(A::AbstractMPSTensor)
    function ftransfer(vr)
        @tensor vr[-1; -2] := A[-1 3; 1] * conj(A[-2 3; 2]) * vr[1; 2]
        return vr
    end
    return ftransfer
end

# Construct application left transfer matrix: v*T -> v
function Otransfer_left(As::Vector{<:AbstractMPSTensor}, Os::Vector{<:AbstractMPOTensor})
    function ftransfer(vl)
        v = zeros(ComplexF64,space(vl))
        for (p,A) in enumerate(As)
            @tensor vp[-1 -2; -3 -4] := A[1 4; -3] *Os[p][2 5; 4 -4] * conj(A[3 5; -2]) * vl[-1 3; 1 2]
            v += vp
        end
        return v
    end
    return ftransfer
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
    imag(E) < 1e-4 || @warn("Energy has imaginary part: E = $E")
    E = real.(E)/overlap(ρs)
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
    imag(E) < 1e-4 || @warn("Energy has imaginary part: E = $E")
    E = real.(E)/overlap(ρs)
    return E
end
