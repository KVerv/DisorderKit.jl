abstract type AbstractInversionAlgorithm end

struct  VOMPS_Inversion <: AbstractInversionAlgorithm
    inverse_dim::Int
    tol::Float64
    maxiter::Int
    verbose::Bool

    finalize!::Function
    function VOMPS_Inversion(inverse_dim::Int; tol::Float64 = 1e-8, maxiter::Int = 50, verbose::Bool = true, finalize=(finalize!))
        return new(inverse_dim, tol, maxiter, verbose, finalize)
    end
end

# Convention for MPO:
#       3
#       v
#     1<+<4
#       v
#       2

function transform_to_mps(Ts::InfiniteMPO)
    mps = map(Ts) do T
        iso = isomorphism(fuse(space(T)[2],space(T)[3]),space(T)[2]⊗space(T)[3])
        @tensor T_mps[-1 -2; -3] := T[-1 1; 2 -4]*iso[-2; 1 2]
        return T_mps
    end
    return InfiniteMPS(mps)
end

function ρ_transfer_left(AL::AbstractMPSTensor, O::AbstractMPOTensor)
    iso = isomorphism(space(AL)[2], space(O)[2]*space(O)[3])
    function ftransfer(vl)
        @tensor vl[-1 -2] := conj(AL[5 4; -2]) * conj(O[1 3; 2 -1]) * vl[1 5] * iso[4; 2 3]
        return vl
    end
    return ftransfer
end

function ρ_transfer_right(AR::AbstractMPSTensor, O::AbstractMPOTensor)
    iso = isomorphism(space(AR)[2], space(O)[2]*space(O)[3])
    function ftransfer(vr)
        @tensor vr[-1 -2] := conj(AR[-2 4; 5]) * conj(O[-1 3; 2 1]) * vr[1 5] * iso[4; 2 3]
        return vr
    end
    return ftransfer
end

# Compute the left and right ρ environment at site i
function ρ_environments(ALs::Vector, ARs::Vector, Os::InfiniteMPO, i::Int)
    unit_cell = length(ALs)
    
    transfer_l = map(0:unit_cell-1) do j
       transfer_func = ρ_transfer_left(ALs[i+j], O[i+j]) ∘ transfer_func
       return transfer_func
    end

    transfer_r = map(0:unit_cell-1) do j
        transfer_func = ρ_transfer_right(ARs[i-j], O[i-j]) ∘ transfer_func
        return transfer_func
    end
    
    xl = Tensor(rand, ComplexF64, space(Os[i])[1]⊗space(ALs[i])[1])
    xr = Tensor(rand, ComplexF64, space(Os[i])[1]⊗space(ARs[i])[1])

    valsl, envsl = eigsolve(transfer_l, xl, 1, :LM)
    valsr, envsr = eigsolve(transfer_r, xr, 1, :LM)

    if length(valsl) > 1
        degeneratel = valsl[1] ≈ valsl[2]
        (degeneratel) && (@warn "Left ρ-transfer matrix has degenerate eigenvalues. Try reducing the bond dimension.")
    end
    if length(valsr) > 1
        degenerater = valsr[1] ≈ valsr[2]
        (degenerater) && (@warn "Right ρ-transfer matrix has degenerate eigenvalues. Try reducing the bond dimension.")
    end

    ρl = envsl[1]
    ρr = envsr[1]

    return ρl, ρr
end

function E_transfer_left(AL::AbstractMPSTensor, O::AbstractMPOTensor)
    iso = isomorphism(space(AL)[2], space(O)[2]*space(O)[3])
    function ftransfer(vl)
        @tensor vl[-1 -2 -3 -4] := vl[1 4 6 10] * conj(AL[10 9; -4]) * conj(O[6 7; 5 -3]) * iso[9; 7 8] * O[4 5; 3 -2] * conj(iso[2; 3 8]) * AL[1 2; -1]
        return vl
    end
    return ftransfer
end

function E_transfer_right(AR::AbstractMPSTensor, O::AbstractMPOTensor)
    iso = isomorphism(space(AR)[2], space(O)[2]*space(O)[3])
    function ftransfer(vr)
        @tensor vr[-1 -2 -3 -4] := vr[1 4 6 10] * conj(AR[-4 9; 10]) * conj(O[-3 7; 5 6]) * iso[9; 7 8] * O[-2 5; 3 4] * conj(iso[2; 3 8]) * AR[-1 2; 1]
        return vr
    end
    return ftransfer
end

# Compute the left and right E environment at site i
function E_environments(ALs::Vector, ARs::Vector, Os::InfiniteMPO, i::Int)
    unit_cell = length(ALs)
    
    transfer_l = map(0:unit_cell-1) do j
       transfer_func = E_transfer_left(ALs[i+j], O[i+j]) ∘ transfer_func
       return transfer_func
    end

    transfer_r = map(0:unit_cell-1) do j
        transfer_func = E_transfer_right(ARs[i-j], O[i-j]) ∘ transfer_func
        return transfer_func
    end
    
    xl = Tensor(rand, ComplexF64, space(ALs[i])[1]'⊗space(Os[i])[1]'⊗space(Os[i])[1]⊗space(ALs[i])[1])
    xr = Tensor(rand, ComplexF64, space(ARs[i])[1]⊗space(Os[i])[1]⊗space(Os[i])[1]'⊗space(ARs[i])[1]')

    valsl, envsl = eigsolve(transfer_l, xl, 1, :LM)
    valsr, envsr = eigsolve(transfer_r, xr, 1, :LM)

    if length(valsl) > 1
        degeneratel = valsl[1] ≈ valsl[2]
        (degeneratel) && (@warn "Left E-transfer matrix has degenerate eigenvalues. Try reducing the bond dimension.")
    end
    if length(valsr) > 1
        degenerater = valsr[1] ≈ valsr[2]
        (degenerater) && (@warn "Right E-transfer matrix has degenerate eigenvalues. Try reducing the bond dimension.")
    end

    El = envsl[1]
    Er = envsr[1]

    return El, Er
end

# Inversion of MPO through VOMPS Algorihtm
function invert_mpo(Os::InfiniteMPO, alg::VOMPS_Inversion; init_guess::InfiniteMPO = nothing)
    unit_cell = length(t)
    # Make initial guess
    if isnothing(init_guess) 
        inits = [TensorMap(rand, ComplexF64, space(Os[i])) for i in 1:length(t)]
        init_guess = InfiniteMPO(inits)
    end
    # Convert initial guess to MPSKit
    As = transform_to_mps(init_guess)

    # Bring MPS in canonical forms
    ALs = As.AL
    ARs = As.AR
    ACs = As.AC
    Cs  = As.C

    i = 0
    ϵ = 1
    while ϵ > alg.tol && i < alg.maxiter

    end
end