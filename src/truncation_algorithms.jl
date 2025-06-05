abstract type AbstractTruncationAlgorithm end

# Standard truncation algorithm for ordinary MPOs
struct StandardTruncation <: AbstractTruncationAlgorithm
    trunc_method::TruncationScheme
    verbosity::Int
end

StandardTruncation(; trunc_method::TruncationScheme = truncerr(1e-6), verbosity::Int = 0) = StandardTruncation(trunc_method, verbosity)

# Truncation algorithm for the disorder MPO by tracing out disorder sectors
struct DisorderTracedTruncation <: AbstractTruncationAlgorithm
    trunc_method::TruncationScheme # Method for truncating ordinary mpo
    verbosity::Int
end

DisorderTracedTruncation(; trunc_method::TruncationScheme = truncerr(1e-6), verbosity::Int = 0) = DisorderTracedTruncation(trunc_method, verbosity)

# Truncation algorithm for the disorder MPO with open disorder sectors
struct DisorderOpenTruncation <: AbstractTruncationAlgorithm
    trunc_method::TruncationScheme # Method for truncating ordinary mpo
    verbosity::Int
end

DisorderOpenTruncation(; trunc_method::TruncationScheme = truncerr(1e-6), verbosity::Int = 0) = DisorderOpenTruncation(trunc_method, verbosity)


# Truncation algorithm for the disorder MPO by using SVD optimization for isometries
struct SVDUpdateTruncation <: AbstractTruncationAlgorithm
    D_max::Int
    tol::Float64
    maxit::Int
    verbosity::Int
end

SVDUpdateTruncation(D_max::Int; tol::Float64 = 1e-8, maxit::Int = 10, verbosity::Int = 0) = SVDUpdateTruncation(D_max, tol, maxit, verbosity)

