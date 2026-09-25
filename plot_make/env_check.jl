using TensorKit, KrylovKit
using DelimitedFiles, LinearAlgebra, CairoMakie, LsqFit

using Revise, JLD2
using DisorderKit, Statistics, ProgressBars, StatsBase


N = 3
a = 0.7
b = 1.3
# a = 0.5
# b = 1.5
hs = Vector(a:(b-a)/(N-1):b)
Js = hs
ps = ones(N^2)./N^2

β = 40
Dmax = 100
τ = 0.05
invtol = 1e-6
truncfrequency = 1
δs = Float64[]

folder_name ="ρ_a$(a)_b$(b)_N$(N)"


folder_name1 = folder_name*"_e$(Dmax)_Δτ$(τ)_invtol$(invtol)_Z4"


isdir("data/$folder_name1") || @error("Folder $folder_name1 does not exist.")

@show Dmax, β, τ, invtol

β = convert(Int, round(β))

ρ = load_object("data/$folder_name1/ρ_β$β.jld2")
ρ = DisorderKit.fix_phase(ρ,ps)

D_disorder = length(hs)*length(Js)
sample = 4
W = zeros(ComplexF64, ℂ^(D_disorder))
W[sample] = 1.0
            
@tensor Ap[-1; -2] := ρ.opp[1][-1 4 1;4 2 -2] * conj(W[1]) *W[2]


v0V = rand(ComplexF64, space(Ap, 1))
λs, rs = eigsolve(x->Ap*x, v0V, 1, :LM)
r = rs[1]
λ = λs[1]
iso = isomorphism(ℂ^10⊗(ℂ^10)', space(Ap, 1))

@tensor ρr[-1; -2] := iso[-1 -2; 1] * r[1]

data = reshape(ρr.data, (10,10))
# data ./= data[1]

fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
ax3 = Axis(fig[1, 3])
centers_x = 1:10
centers_y = 1:10
# data = reshape(id(ComplexF64, ℂ^10).data, (10,10))

heatmap!(ax1, centers_x, centers_y, real.(data))
heatmap!(ax2, centers_x, centers_y, imag.(data))
heatmap!(ax3, centers_x, centers_y, abs.(data))
Colorbar(fig[:, end+1])
fig

U, S, V = svd_full(ρr)
sort(S.data)