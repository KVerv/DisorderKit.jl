using Revise, TensorKit, MPSKit, MPSKitModels, DisorderKit, TimerOutputs, CairoMakie, LsqFit
using BlockTensorKit, MatrixAlgebraKit, KrylovKit

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

Δβ = 0.8


Us = DisorderKit.time_evolution_MPO(Hs, Δβ/2)
ρ = DisorderKit.InfiniteDisorderDensityMatrix(Us.opp, ps)
ρ = DisorderKit.gauge!(ρ)
# ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix(ps, ℂ^2, ℂ^1, ℂ^6)

Z = DisorderKit.PartitionFunction(ρ)
N₀ = DisorderKit.norm_moments(ρ)
@show N₀

ρ_product = DisorderKit.renormalize(ρ)
ρ_product = DisorderKit.gauge!(ρ_product)
N₁ = DisorderKit.norm_moments(ρ_product)