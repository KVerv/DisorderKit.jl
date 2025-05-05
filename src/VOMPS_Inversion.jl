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