
function all_combinations(A::AbstractVector, L::Integer)
    @assert L ≥ 0 "L must be non-negative"
    combos = Vector{Vector{eltype(A)}}()
    for t in Base.Iterators.product(ntuple(_ -> A, L)...)
        push!(combos, collect(t))
    end
    return combos
end

function skew_canonical(A::Matrix{<:Real})
    N = size(A)[1] ÷ 2

    Q, H = hessenberg(A)
    P = zeros(Float64, 2*N, 2*N)
    for ix in 1:N
        P[2*ix-1, ix] = 1
        P[2*ix, N+ix] = 1
    end
    J = -(P' * H * P)[N+1:2*N, 1:N]
    V, Σ, W = svd(J)
    WV = zeros(Float64, 2*N, 2*N)
    WV[1:N, 1:N] = W 
    WV[N+1:2*N, N+1:2*N] = V
    U = Q * P * WV * P'
    Λ = U' * A * U

    return Λ, U, Σ
end

function covariance_matrix(M::Matrix{<:Real}, β::Real)
    _, U, Σ = skew_canonical(M)
    N = size(M)[1] ÷ 2
    ReC̃ = zeros(Float64, size(M))
    for ix in 1:N
        if β == Inf
            ReC̃[2*ix-1, 2*ix] = sign(Σ[ix])
            ReC̃[2*ix, 2*ix-1] = -sign(Σ[ix])
        else
            ReC̃[2*ix-1, 2*ix] = tanh(β*Σ[ix]/2)
            ReC̃[2*ix, 2*ix-1] = -tanh(β*Σ[ix]/2)
        end
    end
    ReC = U * ReC̃ * U' 
    return ReC
end


function majorana_ED(L::Int, Js::Vector{Float64}, hs::Vector{Float64}; n_samples::Int=10)
    Jconfigs = all_combinations(Js, L-1)
    hconfigs = all_combinations(hs, L)
    GEs = []
    Ms = []
    if n_samples < length(Jconfigs)*length(hconfigs)
        sampled_indices = rand(1:length(Jconfigs)*length(hconfigs), n_samples)
        sampled_Jh_configs = []
        for index in sampled_indices
            j_index = div(index-1, length(hconfigs)) + 1
            h_index = mod(index-1, length(hconfigs)) + 1
            push!(sampled_Jh_configs, (Jconfigs[j_index], hconfigs[h_index]))
        end
    else
        sampled_Jh_configs = Iterators.product(Jconfigs, hconfigs)
    end
    for (i,(Jconf,hconf)) in ProgressBar(enumerate(sampled_Jh_configs))
        H = TensorMap(zeros, Float64, ℂ^(2L), ℂ^(2L))
        H[2L-1,2L] = -2 * hconf[end]
        H[2L,2L-1] = 2 * hconf[end]
        for n in 1:L-1
            H[2*n,2*n+1] = -2 * Jconf[n] 
            H[2*n+1,2*n] = 2 * Jconf[n] 
            H[2*n-1,2*n] = -2 * hconf[n] 
            H[2*n,2*n-1] = 2 * hconf[n] 
        end
        Λ, U, Σ = skew_canonical(reshape(H.data,(2L,2L)))

        push!(GEs, -sum(Σ.* 0.5))

        # j = round(Int, L/2)
        # C = covariance_matrix(reshape(H.data,(2L,2L)), 30.0)
        # M = C[2j-1, 2j]
        M = 0
        push!(Ms, M)
        # @show length(Σ)
    end

    return sum(GEs)/length(GEs), sum(Ms)/length(Ms)
end

