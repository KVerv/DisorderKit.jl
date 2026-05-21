# Convention virtual, physical ← virtual

# One site unit-cell isometric infiniteDMPS
struct InfiniteDisorderMPS2{T<:AbstractMPSTensor}
    Ws::Vector{T}
    Vs::Vector{T}
    ps::Vector{<:Real}
end

function InfiniteDisorderMPS2(ps::Vector{Float64}, D_dis::Int, D_phys::Int, D::Int; T=ComplexF64)
    As = [rand(T, ℂ^D⊗ℂ^D_phys,ℂ^D) for i in 1:D_dis]
    Bs = [rand(T, ℂ^D⊗ℂ^D_phys,ℂ^D) for i in 1:D_dis]
    for p in eachindex(As)
        Q1, _ = qr_compact(As[p])
        As[p] = Q1
        Q2, _ = qr_compact(Bs[p])
        Bs[p] = Q2
    end
    return InfiniteDisorderMPS2{typeof(As[1])}(As, Bs, ps)
end


# Construct application left transfer matrix: v*T -> v
function transfer_leftW(ρ::InfiniteDisorderMPS2)
    function ftransfer(vl)
        v = zeros(ComplexF64,space(ρ.Ws[1],3)',space(ρ.Ws[1],3)')
        for (p,W) in enumerate(ρ.Ws)
            for (q, V) in enumerate(ρ.Vs)
                @tensor vp[-2; -1] := W[1 3; 4] * conj(W[2 3; 6]) * V[4 5; -1] * conj(V[6 5; -2]) * vl[2; 1]
                v += ρ.ps[p]*ρ.ps[q]*vp
            end
        end
        return v
    end
    return ftransfer
end

function transfer_leftV(ρ::InfiniteDisorderMPS2)
    function ftransfer(vl)
        v = zeros(ComplexF64,space(ρ.Vs[1],3)',space(ρ.Vs[1],3)')
        for (p,W) in enumerate(ρ.Ws)
            for (q, V) in enumerate(ρ.Vs)
                @tensor vp[-2; -1] := V[1 3; 4] * conj(V[2 3; 6]) * W[4 5; -1] * conj(W[6 5; -2]) * vl[2; 1]
                v += ρ.ps[p]*ρ.ps[q]*vp
            end
        end
        return v
    end
    return ftransfer
end

# Construct application right transfer matrix: T*v -> v
function transfer_rightW(ρ::InfiniteDisorderMPS2)
    function ftransfer(vr)
        v = zeros(ComplexF64,space(ρ.Ws[1],1),space(ρ.Ws[1],1))
        for (p,W) in enumerate(ρ.Ws)
            for (q, V) in enumerate(ρ.Vs)
                @tensor vp[-1; -2] := W[-1 6; 4] * conj(W[-2 6; 5]) * V[4 3; 1] * conj(V[5 3; 2]) * vr[1; 2]
                v += ρ.ps[p]*ρ.ps[q]*vp
            end
        end
        return v
    end
    return ftransfer
end

function transfer_rightV(ρ::InfiniteDisorderMPS2)
    function ftransfer(vr)
        v = zeros(ComplexF64,space(ρ.Vs[1],1),space(ρ.Vs[1],1))
        for (p,W) in enumerate(ρ.Ws)
            for (q, V) in enumerate(ρ.Vs)
                @tensor vp[-1; -2] := V[-1 6; 4] * conj(V[-2 6; 5]) * W[4 3; 1] * conj(W[5 3; 2]) * vr[1; 2]
                v += ρ.ps[p]*ρ.ps[q]*vp
            end
        end
        return v
    end
    return ftransfer
end

# Compute right environment of InfiniteDisorderMPS
function right_environment(ρ::InfiniteDisorderMPS2)
    v0W = id(ComplexF64, space(ρ.Ws[1], 3)')
    f_tW = transfer_rightW(ρ)
    λW, rWs = eigsolve(f_tW, v0W, 2, :LM)
    rW = rWs[1]/tr(rWs[1])

    v0V = id(ComplexF64, space(ρ.Vs[1], 3)')
    f_tV = transfer_rightV(ρ)
    λV, rVs = eigsolve(f_tV, v0V, 2, :LM)
    rV = rVs[1]/tr(rVs[1])
    return λW[1], rW, λV[1], rV
end

# Compute left environment of InfiniteDisorderMPS
function left_environment(ρ::InfiniteDisorderMPS2)
    v0W = id(ComplexF64, space(ρ.Ws[1], 1))
    f_t = transfer_leftW(ρ)
    λW, lW = eigsolve(f_t, v0W, 1, :LM)
    lW = lW[1]/tr(lW[1])

    v0V = id(ComplexF64, space(ρ.Vs[1], 1))
    f_t = transfer_leftV(ρ)
    λV, lV = eigsolve(f_t, v0V, 1, :LM)
    lV = lV[1]/tr(lV[1])
    return λW[1], lW, λV[1], lV
end

# Compute the norm of a InfiniteDisorderMPS
function norm(ρ::InfiniteDisorderMPS2)
    _, rW, _, rV = right_environment(ρ)

    N = tr(rW)
    imag(N) < 1e-4 || @warn("Norm has imaginary part: N = $N")
    N = real.(N)
    return N
end


# # Compute the energy density of a InfiniteDisorderMPS with respect to a DisorderMPOHam
function energy_density(ρ::InfiniteDisorderMPS2, Hs::DisorderMPOHam)
    _, rW, _, rV = right_environment(ρ)

    E = 0
    for (p, W) in enumerate(ρ.Ws)
        for (q, V) in enumerate(ρ.Vs)
            @tensor ED = W[1 2; 4] * Hs.Ds[p][3; 2] * conj(W[1 3; 5]) * rV[4;5]
            E += ρ.ps[p]*ρ.ps[q] * ED / 2
            @tensor ED = V[1 2; 4] * Hs.Ds[q][3; 2] * conj(V[1 3; 5]) * rW[4;5]
            E += ρ.ps[q]*ρ.ps[p] * ED / 2
            @tensor ECBW = W[1 2; 4] * Hs.Cs[p][3; 2 5] * conj(W[1 3; 6]) * V[4 7; 9] * Hs.Bs[q][5 8; 7] * conj(V[6 8; 10]) * rW[9;10]
            E += ρ.ps[p] * ρ.ps[q] * ECBW / 2
            @tensor ECBV = V[1 2; 4] * Hs.Cs[q][3; 2 5] * conj(V[1 3; 6]) * W[4 7; 9] * Hs.Bs[p][5 8; 7] * conj(W[6 8; 10]) * rV[9;10]
            E += ρ.ps[p] * ρ.ps[q] * ECBV/2
        #FIXME : currently only nearest-neighbor interactions
        end
    end

    imag(E) < 1e-4 || @warn("Energy density has imaginary part: E = $E")

    return real.(E)
end

# function energy_density_dist(ρ::InfiniteDisorderMPS, Hs::DisorderMPOHam)
#     r = right_environment(ρ)[2]

#     Es = Float64[]
#     for (p, W) in enumerate(ρ.opp)
#         @tensor ED = W[1 2; 4] * Hs.Ds[p][3; 2] * conj(W[1 3; 5]) * r[4;5]
#         for (q, V) in enumerate(ρ.opp)
#             # @tensor EDW = W[1 2; 4] * Hs.Ds[p][3; 2] * conj(W[1 3; 6]) * V[4 7; 9] * conj(V[6 7; 10]) * r[9;10]
#             # E += ρ.ps[p] * ρ.ps[q] * EDW / 2
#             # @tensor EDV = W[1 2; 4] * Hs.Ds[q][7; 8] * conj(W[1 2; 6]) * V[4 8; 9] * conj(V[6 7; 10]) * r[9;10]
#             # E += ρ.ps[p] * ρ.ps[q] * EDV / 2
#             @tensor ECB = W[1 2; 4] * Hs.Cs[p][3; 2 5] * conj(W[1 3; 6]) * V[4 7; 9] * Hs.Bs[p][5 8; 7] * conj(V[6 8; 10]) * r[9;10]
#             push!(Es, real(ECB + ED))
#         end
#         #FIXME : currently only nearest-neighbor interactions
#     end

#     return Es
# end


# Expectation value of a local operator O at site p
function expectation_value(ρ::InfiniteDisorderMPS2, O::AbstractBondTensor)
    Os = [O for i in 1:length(ρ.Ws)]

    return expectation_value(ρ, Os)
end

# Expectation value of a local disorder operator O at site p
function expectation_value(ρ::InfiniteDisorderMPS2, Os::Vector{<:AbstractBondTensor})
    _, rW, _, rV = right_environment(ρ)  

    EW = 0.
    EV = 0.
    for (p, W) in enumerate(ρ.Ws)
        for (q, V) in enumerate(ρ.Vs)
            @tensor MW = W[5 3; 1] * Os[p][2; 3] * conj(W[5 2; 4]) * rV[1; 4]
            EW += ρ.ps[p]*ρ.ps[q]*MW
            @tensor MV = V[5 3; 1] * Os[q][2; 3] * conj(V[5 2; 4]) * rW[1; 4]
            EV += ρ.ps[p]*ρ.ps[q]*MV
        end
    end

    return real.(EW), real.(EV)
end


# Compute correlation function of two local operators at site i and site j
function correlator(ρ::InfiniteDisorderMPS2, O1::AbstractBondTensor, O2::AbstractBondTensor, r::Int)
    O1s = [O1 for i in 1:length(ρ.Ws)]
    O2s = [O2 for i in 1:length(ρ.Ws)]

    return correlator(ρ, O1s, O2s, r)
end

# Compute correlation function of two local operators at distance r
function correlator(ρ::InfiniteDisorderMPS2, O1s::Vector{<:AbstractBondTensor}, O2s::Vector{<:AbstractBondTensor}, r::Int)
    Cs = Vector{ComplexF64}(undef, r)
    _, rW, _, rV = right_environment(ρ)  

    vlW = zeros(ComplexF64, space(ρ.Ws[1],3)',space(ρ.Ws[1],3)')
    vlV = zeros(ComplexF64, space(ρ.Vs[1],3)',space(ρ.Vs[1],3)')
    vrW = zeros(ComplexF64, space(rW))
    vrV = zeros(ComplexF64, space(rV))

    f_lW = transfer_leftW(ρ)
    f_lV = transfer_leftV(ρ)

    for p in eachindex(ρ.Ws)
        @tensor vlVO1[-1; -2] := ρ.Vs[p][1 3; -2] * O1s[p][2; 3] * conj(ρ.Vs[p][1 2; -1])
        vlV += ρ.ps[p]*vlVO1
        @tensor vlWO1[-1; -2] := ρ.Ws[p][1 3; -2] * O1s[p][2; 3] * conj(ρ.Ws[p][1 2; -1])
        vlW += ρ.ps[p]*vlWO1

        @tensor vrWO2[-1; -2] := ρ.Ws[p][-1 3; 1] * O2s[p][2; 3] * conj(ρ.Ws[p][-2 2; 4]) * rV[1; 4]
        vrW += ρ.ps[p]*vrWO2
        @tensor vrVO2[-1; -2] := ρ.Vs[p][-1 3; 1] * O2s[p][2; 3] * conj(ρ.Vs[p][-2 2; 4]) * rW[1; 4]
        vrV += ρ.ps[p]*vrVO2
    end


    # Cs[1] = tr(vlV * vrW)/2 + tr(vlW * vrV)/2
    # Cs[1] = tr(vlV * vrW)
    Cs[1] = tr(vlW * vrV)
    for k in 2:r
        vlV = f_lW(vlV)
        vlW = f_lV(vlW)
        # C = tr(vlV * vrW)/2 + tr(vlW * vrV)/2
        C = tr(vlW * vrV)
        # C = tr(vlV * vrW)
        Cs[k] = C
    end
    return real.(Cs)
end

function average_correlation_length(ρ::InfiniteDisorderMPS2)
    f_l = transfer_leftV(ρ)

    v0 = rand(ComplexF64, space(ρ.Ws[1],1), space(ρ.Ws[1],1))
    λl, _ = eigsolve(f_l, v0, 3, :LM)

    @show λl
    ξ = -1/log.(abs(λl[2]))
    return ξ
end

# function typical_correlation_length(ρ::InfiniteDisorderMPS; L::Int = 100, Nsamples::Int = 1000)
#     D_disorder = length(ρ.ps)


#     λfs1 = Float64[]
#     λfs2 = Float64[]

#     for _ in ProgressBar(1:Nsamples)

#         λs1 = Float64[]
#         λs2 = Float64[]

#         u1 = rand(ComplexF64,space(ρ.opp[1],1)⊗space(ρ.opp[1],1)')
#         u1 /= TensorKit.norm(u1)
#         u2 = rand(ComplexF64,space(ρ.opp[1],1)⊗space(ρ.opp[1],1)')
#         u2 /= TensorKit.norm(u2)

#         for n in 1:L
#             sample = rand(1:D_disorder, 1)
#             W = ρ.opp[sample[1]]
            
#             @tensor Ap[-1 -2; -3 -4] := W[-1 1; -3] * conj(W[-2 1; -4])

#             u1 = Ap*u1
#             u2 = Ap*u2
#             push!(λs1, 1/n*log(TensorKit.norm(u1)))
#             push!(λs2, 1/n*log(TensorKit.norm(u2)))
#             if n==L
#                 push!(λfs1, 1/n*log(TensorKit.norm(u1)))
#                 push!(λfs2, 1/n*log(TensorKit.norm(u2)))
#             end
#             u2 = u2 - u1*u1'*u2
#         end
#     end
#     ξs = 1 ./(λfs1 .- λfs2)

#     filter!(x -> x<0, λfs2)
#     filter!(x -> x>0, ξs)


#     ξt = StatsBase.geomean(ξs)
#     return ξt
# end

function average_entanglement_entropy(ρ::InfiniteDisorderMPS2)
    _, rW, _, rV = right_environment(ρ)  
    # U, S, V = tsvd(r)
    # S = S.data

    # Se = -sum(S.^2 .*log.(S.^2 .+ 1e-16))

    Dr, V = eig_full(rW)
    Sr = Dr.data
    Ser = real.(-sum(Sr .*log.(Sr .+ 1e-16)))

    Dl, V = eig_full(rV)
    Sl = Dl.data
    Sel = real.(-sum(Sl .*log.(Sl .+ 1e-16)))
    return Ser, Sel
end

