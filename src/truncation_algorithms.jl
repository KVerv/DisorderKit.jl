abstract type AbstractTruncationAlgorithm end

# Standard truncation algorithm for ordinary MPOs
struct StandardTruncation <: AbstractTruncationAlgorithm
    trunc_method::MatrixAlgebraKit.TruncationStrategy
end

StandardTruncation(; trunc_method::MatrixAlgebraKit.TruncationStrategy = truncerr(1e-6)) = StandardTruncation(trunc_method)

# Truncation algorithm for the disorder MPO by using succesive SVD
struct SuccessiveSVD <: AbstractTruncationAlgorithm
    trunc_method::MatrixAlgebraKit.TruncationStrategy
    conv_tol::Float64
end

SuccessiveSVD(; trunc_method::MatrixAlgebraKit.TruncationStrategy = truncerr(1e-6), conv_tol::Real = 1e-8) = SuccessiveSVD(trunc_method, conv_tol)