using Revise, TensorKit, DisorderKit
using BlockTensorKit, TimerOutputs, MatrixAlgebraKit

N = 2
a = 0.7
b = 1.3

Js = Vector(a:(b-a)/(N-1):b)
hs = Vector(a:(b-a)/(N-1):b)
ps = ones(N^2)./N^2

# Js = [1.]
# hs = [1.]
# ps = [1.]


ρ = DisorderKit.InfiniteDisorderDensityMatrix(ps, ℂ^2, ℂ^3, ℂ^6)
ρ = DisorderKit.rescale(ρ, 5)
ρ = DisorderKit.gauge(ρ)
# ρ = ρ * Us

ftr = DisorderKit.right_transfer_matrix(ρ)
ftl = DisorderKit.left_transfer_matrix(ρ)
v = rand(ComplexF64, BlockTensorKit.boxplus([ℂ^6]...),BlockTensorKit.boxplus([ℂ^6]...))
vr = ftr(v)
λ,r = DisorderKit.right_environment(ρ)
λ, l, r = DisorderKit.environments(ρ)

Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
M = DisorderKit.expectation_value(ρ, Z)
ξ = DisorderKit.average_correlation_length(ρ)

H = DisorderKit.random_transverse_field_ising(Js, hs)
U = DisorderKit.time_evolution_MPO(H, 0.1)


Cs = DisorderKit.two_point_correlator(ρ, Z, Z, 10)

E = DisorderKit.energy_density(ρ, H)

ρ_trunc = DisorderKit.truncate(ρ, MatrixAlgebraKit.truncrank(4))

Z = DisorderKit.PartitionFunction(ρ_trunc; χ=1)
O = DisorderKit.inv_sqrt_MPO(Z)
