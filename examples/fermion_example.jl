using Revise, TensorKit, MPSKit, MPSKitModels, KrylovKit
using OptimKit, TensorKitManifolds, Zygote, LinearAlgebra
using DisorderKit, TimerOutputs, CairoMakie, LsqFit, ProgressBars
using StatsBase

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
        ReC̃[2*ix-1, 2*ix] = tanh(β*Σ[ix]/2)
        ReC̃[2*ix, 2*ix-1] = -tanh(β*Σ[ix]/2)
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
        H = zeros(Float64, ℂ^(2L), ℂ^(2L))
        H[2L-1,2L] = -2 * hconf[end]
        H[2L,2L-1] = 2 * hconf[end]
        for n in 1:L-1
            H[2*n,2*n+1] = -2 * Jconf[n] 
            H[2*n+1,2*n] = 2 * Jconf[n] 
            H[2*n-1,2*n] = -2 * hconf[n] 
            H[2*n,2*n-1] = 2 * hconf[n] 
        end
        Λ, U, Σ = skew_canonical(reshape(H.data,(2L,2L)))
        # push!(GEs, -sum(filter(x -> x>=0, Σ.*0.5)))
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







# N = 2
# a = 0.99
# b = 1.01
# w = 0.0
# # hs = Vector(a:(b-a)/(N-1):b)
# # Js = hs
# # Js = [0.1]

# hs = [a, b]
# V = var(log.(hs);corrected = false)
# Js = [a, b]
# hs = [a, b]*exp(w*V)
N = 2
a = 0.7
b = 1.3

Js = Vector(a:(b-a)/(N-1):b)
hs = Vector(a:(b-a)/(N-1):b)
ps = ones(N^2)./N^2

# hs = [0.2577952029653816, 0.7383152846434098, 0.21154225131386992, 0.6319157844800344]
# Js = [0.3806987884083633, 0.2994504233053557, 0.2461909416843507, 0.32056795601359583, 0.27171366282084075, 0.6271996534235563, 0.8399309623164335, 0.7092336979417615, 0.5634881368851596, 0.744255627246975]
# Jmax = maximum(Js)
# Js./= Jmax
# hs./= Jmax
# Js = [1.0]
# a = 0.7
# b = 1.3
# hs = Vector(a:(b-a)/(N-1):b)
# Js = [b]
# hs = [a]
Ls = [4, 6, 8, 10, 12, 14]
# Ls = [3, 5, 7, 9, 11, 13, 15]
# Ls = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# Ls = [6]

result = majorana_ED.(Ls, Ref(Js), Ref(hs); n_samples=1000000)
FFs = first.(result)
Ms = last.(result)
# tF = 1 .- Es[1]./FFs

set_theme!(theme_latexfonts())
fig = Figure(backgroundcolor=:white, fontsize=40, size=(1000, 600))
ax1 = Axis(fig[1, 1], 
        xlabel = L"1/L",
        ylabel = L"$E/L$",
        # xscale = log10,
        # yscale = log10
        )
# ax2 = Axis(fig[1, 2], 
# xlabel = L"1/L",
# ylabel = L"$ϵ$",
# # xscale = log10,
# # yscale = log10
# )

a = 1
scatter!(ax1, 1 ./ (Ls).^(a), FFs./Ls, label=L"$FF$", markersize=20)
maxfit = length(Ls)
p0q = [1., 1.]
linmodel(t, p) = p[1] .+ p[2] * t
linfit = curve_fit(linmodel, 1 ./ (Ls).^(a), FFs./Ls, p0q)
xs = [0, 1 ./ ((Ls).^(a))...]
lines!(ax1, xs, linmodel(xs, linfit.param), color=:black, linewidth=2)
fig

Edens= FFs[end]/(Ls[end])

EF = linfit.param[1]
# t = E/EF
# scatter!(ax1,Ls,Es, label=L"$MPS$",markersize = 20)
# scatter!(ax2,1 ./Ls,abs.((FFs.-Es)./FFs), label=L"$data$",markersize = 20)
# fig

# D | EF | EMPS D = 8 | EMPS/EF
# -- | ------- | ---------- | -------
# 0.2 | -1.27462 | -1.27493 | 100.03%
# 0.4 | -1.28481 | -1.27955 | 99.60%
# 0.6 | -1.28896 | -1.28305 | 99.04%| 99.23%| 99.23%