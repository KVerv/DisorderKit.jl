using Revise, TensorKit, MPSKit, MPSKitModels, KrylovKit
using OptimKit, TensorKitManifolds, Zygote
using DisorderKit, TimerOutputs, CairoMakie, LsqFit, ProgressBars
const AbstractMPOTensor = AbstractTensorMap{T, S, 2, 2} where {T, S}
const AbstractMPSTensor = AbstractTensorMap{T, S, 2, 1} where {T, S}

# Define model
N = 2
a = 0.7
b = 1.3
hs = Vector(a:(b-a)/(N-1):b)
Js = hs
L = 3
D_dis = length(hs)*length(Js)

# Make DisorderMPS
D = 5
As = [[TensorMap(rand, ComplexF64, ℂ^D⊗ℂ^2,ℂ^D) for i in 1:D_dis] for j in 1:L]
# As = push!(As,[TensorMap(rand, ComplexF64, ℂ^D⊗ℂ^2,ℂ^D) for i in 1:2])
ρ = FiniteDisorderMPS(As)
ρ = DisorderKit.left_gauge(ρ)

function hams_Ising(Js::Vector{Float64}, hs::Vector{Float64})
    X, Z, Id = zeros(ComplexF64, ℂ^2, ℂ^2), zeros(ComplexF64, ℂ^2, ℂ^2), zeros(ComplexF64, ℂ^2, ℂ^2)
    X[1, 2], X[2, 1] = 1, 1
    Z[1, 1], Z[2, 2] = 1, -1
    Id[1, 1], Id[2, 2] = 1, 1
    Hs = AbstractMPOTensor[]
    for (i, (h, J)) in enumerate(Iterators.product(hs, Js))
        H = zeros(ComplexF64, ℂ^3⊗ℂ^2, ℂ^2⊗ℂ^3)
        H[1,:, :, 1] = Id.data
        H[2,:, :, 1] = zeros(ComplexF64, 2, 2)
        H[3,:, :, 1] = zeros(ComplexF64, 2, 2)
        H[1,:, :, 2] = Z.data
        H[2,:, :, 2] = zeros(ComplexF64, 2, 2)
        H[3,:, :, 2] = zeros(ComplexF64, 2, 2)
        H[1,:, :, 3] = -h*X.data
        H[2,:, :, 3] = -J*Z.data
        H[3,:, :, 3] = Id.data
        push!(Hs,convert(TensorMap,H))
    end
    return Hs
end

Hs = hams_Ising(Js, hs)

# alg = StiefelOptim(2, 1e-4, 5)
# fg = DisorderKit.target_func(Hs)
# E, g = fg(ρ)
# ρs = groundstate!(ρ, Hs, alg)
# E = measure(ρs, Hs)

function compute_Es(Ls::Vector{Int}, Hs::Vector{<:AbstractMPOTensor})
    Es = Float64[]
    D = 10
    ρs = []
    alg = StiefelOptim(2, 5e-4, 1)
    for (iL,L) in enumerate(Ls)
        As = [[TensorMap(rand, ComplexF64, ℂ^D⊗ℂ^2,ℂ^D) for i in 1:D_dis] for j in 1:L]
        ρ = FiniteDisorderMPS(As)
        ρ = DisorderKit.left_gauge(ρ)
        ρL = groundstate!(ρ, Hs, alg)
        EL = measure(ρL, Hs)
        push!(Es, EL)
        push!(ρs, ρL)
    end
    return Es, ρs
end


Ls = [3,6,9,12]
Es, ρs = compute_Es(Ls, Hs)
Es = Es
