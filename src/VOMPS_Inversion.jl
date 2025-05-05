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