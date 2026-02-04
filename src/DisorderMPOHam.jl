# One site unit-cell DisorderMPOHam
struct DisorderMPOHam{T <: Number, S <: IndexSpace}
    As::Vector{<:AbstractTensorMap{T, S, 2, 2}}
    Bs::Vector{<:AbstractTensorMap{T, S, 2, 1}}
    Cs::Vector{<:AbstractTensorMap{T, S, 1, 2}}
    Ds::Vector{<:AbstractTensorMap{T, S, 1, 1}}
end

function DisorderMPOHam(As::Vector{<:AbstractTensorMap{T, S, 2, 2}},
                     Bs::Vector{<:AbstractTensorMap{T, S, 2, 1}},
                     Cs::Vector{<:AbstractTensorMap{T, S, 1, 2}},
                     Ds::Vector{<:AbstractTensorMap{T, S, 1, 1}}) where {T, S}
    length(As) == length(Bs) == length(Cs) == length(Ds) || error("All input vectors must have the same length")
    return DisorderMPOHam(As, Bs, Cs, Ds)
end

function random_transverse_field_ising(Js::Vector{Float64}, hs::Vector{Float64}; ϵ::Real=0.0)
    A = zeros(Float64, ℂ^1⊗ℂ^2, ℂ^2⊗ℂ^1)
    B = TensorMap([1. 0.; 0. -1.], ℂ^1⊗ℂ^2, ℂ^2)
    C =  TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2⊗ℂ^1)
    D =  TensorMap([0. 1.; 1. 0.], ℂ^2, ℂ^2)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)

    As = typeof(A)[]
    Bs = typeof(B)[]
    Cs = typeof(C)[]
    Ds = typeof(D)[]

    ϵ = 0
    for (i, (h, J)) in enumerate(Iterators.product(hs, Js))
        @show (h,J)
        push!(As, A)
        push!(Bs, sqrt(J)B)
        push!(Cs, -sqrt(J)*C)
        push!(Ds, -h*D - ϵ*Z)
    end
    return DisorderMPOHam(As, Bs, Cs, Ds)
end
