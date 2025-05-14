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
