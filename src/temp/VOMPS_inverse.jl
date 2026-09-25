abstract type AbstractInversionAlgorithm end

struct VOMPS_Inversion <: AbstractInversionAlgorithm
    inverse_dim::Int
    tol::Float64
    maxiter::Int
    verbosity::Int #0 is no message, 1 shows convergence of each step, 2 shows all steps

    function VOMPS_Inversion(inverse_dim::Int; tol::Float64 = 1e-8, maxiter::Int = 50, verbosity::Int = 0)
        return new(inverse_dim, tol, maxiter, verbosity)
    end
end

function ρ_transfer_left(AL::AbstractMPSTensor, O::AbstractMPSTensor)
    function ftransfer(vl)
        @tensor vl[-1 -2] := conj(AL[3 2; -2]) * conj(O[1 2; -1]) * vl[1 3]
        return vl
    end
    return ftransfer
end

function ρ_transfer_right(AR::AbstractMPSTensor, O::AbstractMPSTensor)
    function ftransfer(vr)
        @tensor vr[-1 -2] := conj(AR[-2 2; 3]) * conj(O[-1 2; 1]) * vr[1 3]
        return vr
    end
    return ftransfer
end

# Compute the left and right ρ environment at site i
function ρ_environments(AL::AbstractMPSTensor, AR::AbstractMPSTensor, O::AbstractMPSTensor)
    transfer_l =  ρ_transfer_left(AL, O)
    transfer_r = ρ_transfer_right(AR, O)

    xl = rand(ComplexF64, space(O, 1)⊗space(AL, 1))
    xr = rand(ComplexF64, space(O, 1)'⊗space(AR, 1)')

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

    return λρ, ρl, ρr
end

#FIXME
function E_transfer_left(AL::AbstractMPSTensor, O::AbstractMPOTensor)
    function ftransfer(vl)
        @tensor vl[-1 -2; -3 -4] := AL[1 2; -3] * O[3 4; 2 -4] * conj(O[5 4; 6 -1]) * conj(AL[7 6; -2]) * vl[5 7; 1 3] 
    end
    return ftransfer
end

function E_transfer_right(AR::AbstractMPSTensor, O::AbstractMPOTensor)
    function ftransfer(vr)
        @tensor vr[-1 -2; -3 -4] := AR[-1 2; 1] * O[-2 4; 2 3] * conj(O[-3 4; 6 5]) * conj(AR[-4 6; 7]) * vr[1 3; 5 7] 
    end
    return ftransfer
end

# Compute the left and right E environment at site i
function E_environments(AL::AbstractMPSTensor, AR::AbstractMPSTensor, O::AbstractMPOTensor)
    transfer_l =  E_transfer_left(AL, O)
    transfer_r = E_transfer_right(AR, O)

    xl = rand(ComplexF64, space(O, 1) ⊗ space(AL, 1), space(AL, 1) ⊗ space(O, 1))
    xr = rand(ComplexF64, space(AR, 1) ⊗ space(O, 1), space(O, 1) ⊗ space(AR, 1))

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

    return λE ,El, Er
end

# Construct system for AC at site i
function AC_system(AC::AbstractMPSTensor, O::AbstractMPOTensor, Omps::AbstractMPSTensor, ρl::AbstractRhoEnv, ρr::AbstractRhoEnv, El::AbstractEEnv, Er::AbstractEEnv)
    @tensor NE = El[3 4; 1 2] * E_transfer_right(AC,O)(Er)[1 2; 3 4]
    @tensor Nρ = ρl[1 2] * ρ_transfer_right(AC,Omps)(ρr)[1 2]
    function f(AC)
        @tensor AC_new[-1 -2; -3] := El[5 -1; 1 3] * AC[1 2; 6] * O[3 4; 2 7] * conj(O[5 4; -2 8]) * Er[6 7; 8 -3]
        return AC_new * Nρ / NE
    end
    @tensor b[-1 -2; -3] := ρl[1 -1] * conj(Omps[1 -2; 2]) * ρr[2 -3] 

    return b, f
end

# Construct system for C at site i
function C_system(C::AbstractBondTensor, ρl::AbstractRhoEnv, ρr::AbstractRhoEnv, El::AbstractEEnv, Er::AbstractEEnv)
    @tensor NE = El[5 2; 1 4] * C[1; 3] * conj(C[2; 6]) * Er[3 4; 5 6]
    @tensor Nρ = ρl[1 2] * conj(C[2; 3]) * ρr[1 3]
    
    function f(C)
        @tensor C_new[-1; -2] := El[4 -1; 1 3] * C[1; 2] * Er[2 3; 4 -2]
        @show norm(C_new * Nρ / NE)
        return C_new
    end
    @tensor b[-1; -2] := ρl[1 -1] * ρr[1 -2]
    # b *= NE / Nρ
    return b, f
end

function get_AL(AC::AbstractMPSTensor, C::AbstractBondTensor)
    UAC_l, PAC_l = left_polar(AC)
    UC_l, PC_l = left_polar(C)
    AL = UAC_l * UC_l'

    # check AC - AL * C and AC - C * AR
    # FIXME. why do we still need a coefficient here
    # @show (PAC_l[1] / PC_l[1])
    # ϵL = norm(PAC_l - PC_l * (PAC_l[1] / PC_l[1])) 
    ϵL = norm(AL *C  - AC)
    return AL, ϵL
end

function get_AR(AC::AbstractMPSTensor, C::AbstractBondTensor)
    PAC_r, UAC_r = right_polar(permute(AC, ((1,), (2, 3))))
    PC_r, UC_r = right_polar(C)
    AR = permute(UC_r' * UAC_r, ((1, 2), (3,)))

    # check AC - AL * C and AC - C * AR
    # FIXME. why do we still need a coefficient here
    # @show (PAC_r[1] / PC_r[1])
    ϵR = norm(PAC_r - PC_r * (PAC_r[1] / PC_r[1]))
    # @tensor CR[-1 -2; -3] := C[-1; 1] * AR[1 -2; -3]
    # ϵR = norm(CR - AC)
    return AR, ϵR
end

# Inversion of MPO through VOMPS Algorithm
function invert_mpo(O::AbstractMPOTensor, alg::VOMPS_Inversion; init_guess::Union{AbstractMPOTensor,Nothing} = nothing)
    vspace = space(O, 1)
    dspace = space(O, 2)

    #Construct initial guess
    A = spzeros(ComplexF64, BlockTensorKit.boxplus(ℂ^alg.inverse_dim) ⊗ BlockTensorKit.boxplus(dspace), BlockTensorKit.boxplus(ℂ^alg.inverse_dim))
    Omps = spzeros(ComplexF64, BlockTensorKit.boxplus(vspace) ⊗ BlockTensorKit.boxplus(dspace)', BlockTensorKit.boxplus(vspace))

    (alg.verbosity > 1) && @info(crayon"yellow"("step 0) Generating initial guess"))    
    for i in 1:length(dspace)
        if isnothing(init_guess) 
            A[1,i,1] = rand(ComplexF64, ℂ^alg.inverse_dim ⊗ ℂ^1, ℂ^alg.inverse_dim)
        else
            A[1,i,1] = TensorMap(init_guess.data, ℂ^alg.inverse_dim ⊗ ℂ^1, ℂ^alg.inverse_dim)
        end
    end

    Amps = InfiniteMPS([A])
    # Bring MPS in canonical forms
    AL = Amps.AL[1]
    AR = Amps.AR[1]
    AC = Amps.AC[1]
    C  = Amps.C[1]

    @show norm(AL)
    @show norm(C)

    it = 0
    ε = 1
    while ε > alg.tol && it < alg.maxiter
        ε_inv = 1e-3 * alg.tol

        it+=1
        # Compute the left and right environments
        (alg.verbosity > 1) && @info(crayon"cyan"("step $it) Computing ρ-environments"))
        λρ, ρl, ρr = ρ_environments(AL, AR, Omps)
        (alg.verbosity > 1) && @info(crayon"cyan"("step $it) Computing E-environments"))
        λE, El, Er = E_environments(AL, AR, O)
        # Construct linear system for AC and C
        bAC, fAC = AC_system(AC, O, Omps, ρl, ρr, El, Er)
        bC, fC = C_system(C, ρl, ρr, El, Er)
        # Solve linear systems
        (alg.verbosity > 1) && @info(crayon"cyan"("step $it) Solving linear systems"))
        x₀ = rand(ComplexF64,space(AC))
        AC_new = linsolve(x -> fAC(x) + ε_inv * x,bAC, x₀;maxiter = 500)[1]
        x₀ = rand(ComplexF64,space(C))
        C_new = linsolve(x -> fC(x) + ε_inv * x,bC,x₀;maxiter = 500)[1]
        # Update the AC tensors
        C = C_new
        AC = AC_new
        (alg.verbosity > 1) && @info(crayon"cyan"("step $it) Updating AL and AR tensors"))
        AL, ϵL = get_AL(AC, C)
        AR, ϵR = get_AR(AC, C)
        @show ϵL, ϵR
        ε = max(ϵL, ϵR)

        (alg.verbosity > 0) && (@info(crayon"cyan"("step $it) Convergence error = $(ε)")))
    end
    ε < alg.tol || @warn("Inverse not converged: ε = $(ε)")

    Oinv = spzeros(ComplexF64, BlockTensorKit.boxplus(ℂ^alg.inverse_dim) ⊗ BlockTensorKit.boxplus(dspace), BlockTensorKit.boxplus(dspace) ⊗ BlockTensorKit.boxplus(ℂ^alg.inverse_dim))

    for i in 1:length(dspace)
        Oinv[1,i,i,1] = TensorMap(AL[1,i,1].data, ℂ^alg.inverse_dim ⊗ ℂ^1, ℂ^1 ⊗ ℂ^alg.inverse_dim)
    end
    return Oinv, ε
end