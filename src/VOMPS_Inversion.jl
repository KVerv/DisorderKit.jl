abstract type AbstractInversionAlgorithm end

struct  VOMPS_Inversion <: AbstractInversionAlgorithm
    inverse_dim::Int
    tol::Float64
    maxiter::Int
    verbose::Bool

    function VOMPS_Inversion(inverse_dim::Int; tol::Float64 = 1e-8, maxiter::Int = 50, verbose::Bool = true)
        return new(inverse_dim, tol, maxiter, verbose)
    end
end

function transform_to_mps(Ts::InfiniteMPO)
    mps = map(Ts) do T
        iso = isomorphism(fuse(space(T)[2],space(T)[3]),space(T)[2] ⊗ space(T)[3])
        @tensor T_mps[-1 -2; -3] := T[-1 1; 2 -3]*iso[-2; 1 2]
        return T_mps
    end
    return InfiniteMPS(mps)
end

function transform_to_mpo(Ts::InfiniteMPS, pspaces::Vector{ComplexSpace})
    mpo = Vector{AbstractMPOTensor}(undef, length(Ts))
    for i in eachindex(Ts)
        T = Ts.AL[i]
        iso = isomorphism(pspaces[i]⊗pspaces[i]',space(T)[2])
        @tensor T_mpo[-1 -2; -3 -4] := T[-1 1; -4]*iso[-2 -3; 1]
        mpo[i] = T_mpo
    end
    return InfiniteMPO(mpo)
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
function ρ_environments(ALs::PeriodicVector, ARs::PeriodicVector, Os::InfiniteMPO, i::Int)
    unit_cell = length(ALs)
    
    transfer_l =  ρ_transfer_left(ALs[i], Os[i])
    for j = 1:unit_cell-1
        transfer_l = ρ_transfer_left(ALs[i+j], Os[i+j]) ∘ transfer_l
    end

    transfer_r = ρ_transfer_right(ARs[i], Os[i])
    for j = 1:unit_cell-1
        transfer_r = ρ_transfer_right(ARs[i-j], Os[i-j]) ∘ transfer_r
    end
    
    xl = Tensor(rand, ComplexF64, space(Os[i])[1]⊗space(ALs[i])[1])
    xr = Tensor(rand, ComplexF64, space(Os[i])[1]'⊗space(ARs[i])[1]')

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
    λρ = valsl[1]

    ρl2 = ρ_transfer_left(ALs[i], Os[i])(ρl)
    ρl2 = ρl2/norm(ρl2)

    return λρ, ρl, ρr, ρl2
end

function E_transfer_left(AL::AbstractMPSTensor, O::AbstractMPOTensor)
    iso = isomorphism(space(AL)[2], space(O)[2]*space(O)[3])
    function ftransfer(vl)
        @tensor vl[-1 -2 -3 -4] := vl[1 4 6 10] * conj(AL[10 9; -4]) * conj(O[6 5; 7 -3]) * iso[9; 7 8] * O[4 5; 3 -2] * conj(iso[2; 3 8]) * AL[1 2; -1]
        return vl
    end
    return ftransfer
end

function E_transfer_right(AR::AbstractMPSTensor, O::AbstractMPOTensor)
    iso = isomorphism(space(AR)[2], space(O)[2]*space(O)[3])
    function ftransfer(vr)
        @tensor vr[-1 -2 -3 -4] := vr[1 4 6 10] * conj(AR[-4 9; 10]) * conj(O[-3 5; 7 6]) * iso[9; 7 8] * O[-2 5; 3 4] * conj(iso[2; 3 8]) * AR[-1 2; 1]
        return vr
    end
    return ftransfer
end

# Compute the left and right E environment at site i
function E_environments(ALs::PeriodicVector, ARs::PeriodicVector, Os::InfiniteMPO, i::Int)
    unit_cell = length(ALs)
    
    transfer_l =  E_transfer_left(ALs[i], Os[i])
    for j = 1:unit_cell-1
        transfer_l = E_transfer_left(ALs[i+j], Os[i+j]) ∘ transfer_l
    end

    transfer_r = E_transfer_right(ARs[i], Os[i])
    for j = 1:unit_cell-1
        transfer_r = E_transfer_right(ARs[i-j], Os[i-j]) ∘ transfer_r
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
    λE = valsl[1]

    El2 = E_transfer_left(ALs[i], Os[i])(El)
    El2 = El2/norm(El2)
    return λE ,El, Er, El2
end

# Construct system for AC at site i
function AC_system(AC::AbstractMPSTensor, C::AbstractBondTensor, O::AbstractMPOTensor, λρ::ComplexF64, ρl::AbstractRhoEnv, ρr::AbstractRhoEnv, λE::ComplexF64, El::AbstractEEnv, Er::AbstractEEnv)
    iso = isomorphism(space(AC)[2], space(O)[2]*space(O)[3])
    @tensor NE[] := El[1 2 3 4] * E_transfer_right(AC,O)(Er)[1 2 3 4]
    @tensor Nρ[] := ρl[1 2] * ρ_transfer_right(AC,O)(ρr)[1 2]
    function f(AC)
        @tensor AC_new[-1 -2; -3] := El[1 4 6 -1] * conj(O[6 5; 7 11]) * iso[-2; 7 8] * O[4 5; 3 10] * conj(iso[2; 3 8]) * AC[1 2; 9] * Er[9 10 11 -3]
        return AC_new * Nρ[1] / (NE[1])
    end
    @tensor b[-1 -2; -3] := ρl[1 -1] * conj(O[1 3; 2 4]) * iso[-2; 2 3] * ρr[4 -3]

    return b, f
end

# Construct system for C at site i
function C_system(AC::AbstractMPSTensor, C::AbstractBondTensor, O::AbstractMPOTensor, ρl::AbstractRhoEnv, ρr::AbstractRhoEnv, El::AbstractEEnv, Er::AbstractEEnv,  λρ::ComplexF64,  λE::ComplexF64)
    @tensor NE[] := El[1 2 3 4] * C[1; 5] * conj(C[4; 8]) * Er[5 2 3 8]
    @tensor Nρ[] := ρl[1 2] * conj(C[2; 3]) *ρr[1 3]
    function f(C)
        @tensor C_new[-1; -2] := El[1 2 3 -1] * C[1; 4] * Er[4 2 3 -2]
        return C_new * Nρ[1] / NE[1]
    end
    @tensor b[-1; -2] := ρl[1 -1] * ρr[1 -2]

    return b, f
end

function get_AL(AC::AbstractMPSTensor, C::AbstractBondTensor)
    UAC_l, PAC_l = leftorth(AC; alg = Polar())
    UC_l, PC_l = leftorth(C; alg = Polar())
    AL = UAC_l * UC_l'

    # check AC - AL * C and AC - C * AR
    # FIXME. why do we still need a coefficient here
    # @show (PAC_l[1] / PC_l[1])
    ϵL = norm(PAC_l - PC_l * (PAC_l[1] / PC_l[1])) 
    return AL, ϵL
end

function get_AR(AC::AbstractMPSTensor, C::AbstractBondTensor)
    PAC_r, UAC_r = rightorth(permute(AC, ((1,), (2, 3))); alg = Polar())
    PC_r, UC_r = rightorth(C; alg=Polar())
    AR = permute(UC_r' * UAC_r, ((1, 2), (3,)))

    # check AC - AL * C and AC - C * AR
    # FIXME. why do we still need a coefficient here
    # @show (PAC_r[1] / PC_r[1])
    ϵR = norm(PAC_r - PC_r * (PAC_r[1] / PC_r[1]))
    return AR, ϵR
end

# Inversion of MPO through VOMPS Algorihtm
function invert_mpo(Os::InfiniteMPO, alg::VOMPS_Inversion; init_guess::Union{InfiniteMPO,Nothing} = nothing)
    unit_cell = length(Os)
    # Make initial guess
    if isnothing(init_guess) 
        inits = [TensorMap(rand, ComplexF64, ℂ^alg.inverse_dim ⊗ space(Os[i])[2], space(Os[i])[3] ⊗ ℂ^alg.inverse_dim) for i in 1:unit_cell]
        init_guess = InfiniteMPO(inits)
    end
    # Convert initial guess to MPS
    As = transform_to_mps(init_guess)

    # Bring MPS in canonical forms
    ALs = As.AL
    ARs = As.AR
    ACs = As.AC
    Cs  = As.C

    @show space(ALs[1])
    it = 0
    ϵ = 1
    AC1 = ACs[1]
    AC2 = AC1
    while ϵ > alg.tol && it < alg.maxiter
        ϵ_inv = 1e-4 *alg.tol

        it+=1
        ϵs = zeros(unit_cell)
        # Optimize each site sequentially
        for i in 1:unit_cell
            # Compute the left and right environments
            @info(crayon"cyan"("step $it) Computing ρ-environments for site $i"))
            λρ, ρl, ρr, ρl2 = ρ_environments(ALs, ARs, Os, i)
            @info(crayon"cyan"("step $it) Computing E-environments for site $i"))
            λE, El, Er, El2 = E_environments(ALs, ARs, Os, i)
            # Construct linear system for AC and C
            bAC, fAC = AC_system(ACs[i], Cs[i], Os[i], λρ, ρl, ρr, λE, El, Er)
            bC, fC = C_system(ACs[i], Cs[i], Os[i], ρl2, ρr, El2, Er, λρ, λE)
            # Solve linear systems
            @info(crayon"cyan"("step $it) Solving linear systems for site $i"))
            x₀ = TensorMap(rand,ComplexF64,space(ACs[i]))
            AC_new = linsolve(x -> fAC(x) + ϵ_inv * x,bAC,x₀;maxiter = 500)[1]
            x₀ = TensorMap(rand,ComplexF64,space(Cs[i]))
            C_new = linsolve(x -> fC(x) + ϵ_inv * x,bC,x₀;maxiter = 500)[1]
            # Update the AC tensors
            Cs[i] = C_new
            ACs[i] = AC_new
        end
        for i in 1:unit_cell
            @info(crayon"cyan"("step $it) Updating AL and AR tensors for site $i"))
            ALs[i], ϵL = get_AL(ACs[i], Cs[i])
            ARs[i], ϵR = get_AR(ACs[i], Cs[i-1])
            ϵs[i] = max(ϵL, ϵR) 
        end
        
        ϵ = maximum(ϵs)
        (alg.verbose) && (@info(crayon"cyan"("step $it) Convergence error = $(ϵ)")))
    end
    # Convert MPS back to MPO
    pspaces = map(1:unit_cell) do i
        pspace = space(Os[i])[2]
        return pspace
    end
    inverse_mpo = transform_to_mpo(InfiniteMPS(ALs, ARs, Cs, ACs), pspaces)
    return inverse_mpo, ϵ
end