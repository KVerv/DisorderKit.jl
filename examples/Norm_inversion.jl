using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit
using BlockTensorKit, MatrixAlgebraKit, KrylovKit, StatsBase


function FS_norm(ρ, L::Int)
    λ, l ,r = DisorderKit.environments(ρ)
    A = [1, 2, 3, 4]
    # A =[1]
    combos = Vector{Vector{eltype(A)}}()
    for t in Base.Iterators.product(ntuple(_ -> A, L)...)
        push!(combos, collect(t))
    end

    Ns = zeros(ComplexF64, length(combos))
    for (x, config) in enumerate(combos)
        T = id(ComplexF64, space(ρ[1], 1)'⊗space(ρ[1], 1))
        for i in 1:L
            @tensor Tζ[-1 -2; -3 -4] := ρ[config[i]][-2 4; 3 -4] * conj( ρ[config[i]][-1 4; 3 -3])
            T = Tζ*T
        end
        @tensor N = l[1; 2] * T[1 2; 4 3] * r[3; 4]
        Ns[x] = N^(1/L)
    end
    return Ns
end

N = 2
a = 0.7
b = 1.3

Js = Vector(a:(b-a)/(N-1):b)
hs = Vector(a:(b-a)/(N-1):b)
ps = ones(N^2)./N^2

# Js = [1.]
# hs = [1.]
# ps = [1.]

Hs = DisorderKit.random_transverse_field_ising(Js, hs)

Δβ = 0.2
χ = 4

Us = DisorderKit.time_evolution_MPO(Hs, Δβ; N=2)
ρ = DisorderKit.InfiniteDisorderDensityMatrix(Us.opp, ps)
ρ = DisorderKit.gauge(ρ)
# ρ = DisorderKit.InfiniteDisorderDensityMatrix(ps, ℂ^2, ℂ^1, ℂ^2; σ=1e-8)
# ρ = ρ * Us
# ZL = DisorderKit.compute_ZL(ρ; maxiter = 10)
Z = DisorderKit.PartitionFunction(ρ)

function Tav(ρ)
    λ, l, r = DisorderKit.environments(ρ)
    iso = isomorphism(fuse(space(ρ[1], 1)'⊗space(ρ[1], 1)), space(ρ[1], 1)'⊗space(ρ[1], 1))
    @tensor Tζf[-1 -2; -3 -4] := iso[-1; 1 2] * ρ[1][2 5 -2; 6 7 4] * conj(ρ[1][1 5 -3; 6 7 3]) * conj(iso[-4; 3 4])
    P = DisorderKit.make_DiagonalBlockTensorMap(ρ.ps)
    

    @tensor Tbar_fused[-1; -2] := Tζf[-1 1; 2 -2] * P[2; 1]
    Id = id(ComplexF64, space(ρ[1], 3))
    @tensor Tbar_tensor[-1 -2; -3 -4] := Tbar_fused[-1; -4] * Id[-2; -3]
    δT = Tζf - Tbar_tensor
    
    vspace= space(Tζf, 1)
    Q = DisorderKit.make_DiagonalBlockTensorMap([0., .1, .0, .0])
    @tensor M[-1; -2] := Tζf[-1 1; 2 -2] * Q[2;1]
    vals = eig_vals(M)
    @show vals
    return Tbar_fused, Tζf, δT
end

Test, Tz, δT = Tav(ρ)
x=1

# N₀ = DisorderKit.norm_moments(ρ)
es, N = DisorderKit.entanglement_spectrum_norm(ρ)
ϵ0 = sum(es[1:end-1])
ρ_norm = DisorderKit.normalize(ρ)
es, N = DisorderKit.entanglement_spectrum_norm(ρ_norm)
@show ϵ0
@show sum(es[1:end-1])

Wcodomain = BlockTensorKit.boxplus([ℂ^1, ℂ^χ]...) ⊗ BlockTensorKit.boxplus( space(ρ[1], 3))
Wdomain = BlockTensorKit.boxplus(space(ρ[1], 3)) ⊗ BlockTensorKit.boxplus([ℂ^1, ℂ^χ]...)

W = SparseBlockTensorMap{AbstractTensorMap{ComplexF64, ComplexSpace, 2, 2}}(undef, Wcodomain, Wdomain)
for i in 1:length(ps)
    W[1,i,i,1] = TensorMap(1 .+Z.D[i,i].data, ℂ^1⊗ ℂ^1,  ℂ^1⊗ℂ^1)
    W[1,i,i,2] = TensorMap(Z.L[i,i,1].data, ℂ^1⊗ ℂ^1,  ℂ^1⊗ℂ^χ)
    W[2,i,i,1] = TensorMap(Z.R[1,i,i].data, ℂ^χ⊗ ℂ^1,  ℂ^1⊗ℂ^1)
    W[2,i,i,2] = TensorMap(Z.A[1,i,i,1].data, ℂ^χ ⊗ ℂ^1,  ℂ^1⊗ℂ^χ)
end

Os = InfiniteMPO([W])
entanglement_spectrum(Os, 1)

# p = 2
# diagel = zeros(ComplexF64, (N^2))
# diagel[p] = 1.
# P = DiagonalTensorMap(diagel, ℂ^(N^2))
# @tensor A[-1; -4] := Z.A[-1 1; 2 -4] * P[2; 1]
# A.data


# # ρ_product = DisorderKit.renormalize(ρ; χ=χ, N=1)
# ρ_product = DisorderKit.normalize_density_matrix(ρ)
# Z2 = DisorderKit.PartitionFunction(ρ_product; χ=χ)
# Z2.D

# ρ_product = DisorderKit.renormalize(ρ_product; χ=χ, N=1)
# Z3 = DisorderKit.PartitionFunction(ρ_product; χ=χ)
# Z3.D
# ρ_product = DisorderKit.renormalize(ρ_product; χ=χ, N=1)
# Z3 = DisorderKit.PartitionFunction(ρ_product; χ=χ)
# Z3.D
# ρ_product = DisorderKit.gauge!(ρ_product)
# N₁ = DisorderKit.norm_moments(ρ_product)
# # es = DisorderKit.entanglement_spectrum_norm(ρ_product)
# # ϵ = sum(es[1:end-1])
# # @show (ϵ0, ϵ)
# # @show N₁

# @show "Hello"
# L = 5
# Ns = FS_norm(ρ, L)

# Ns2 = FS_norm(ρ_product, L)
# μ1 = mean(Ns)
# μ2 = mean(Ns2)
# σ1 = var(Ns; corrected = false)
# σ2 = var(Ns2; corrected = false)
# @show (μ1, μ2)
# @show (σ1, σ2)

# @show (abs(1-N₀[1]), abs(1-N₁[1]))
# (abs(1-N₀[2]), abs(1-N₁[2]))