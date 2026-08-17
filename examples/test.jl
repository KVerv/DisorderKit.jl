using Revise, TensorKit, TimerOutputs, BlockTensorKit

timer = TimerOutput()

D = 50
N = 10
ps = ones(N)/N
function construct_tensor()
    A = spzeros(ComplexF64, BlockTensorKit.boxplus(ℂ^D) ⊗ BlockTensorKit.boxplus(ℂ^2) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...), BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...) ⊗ BlockTensorKit.boxplus(ℂ^D))
    Aᵢ = rand(ComplexF64, ℂ^D ⊗ ℂ^2 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1 ⊗ ℂ^D) 
    for i in eachindex(ps)
            A[1, 1, i, 1, i, 1] = Aᵢ # Same tensor for each disorder sector
    end
    return A
end

function construct_MPStensor()
    A = spzeros(ComplexF64, BlockTensorKit.boxplus(ℂ^D) ⊗ BlockTensorKit.boxplus(ℂ^2) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...), BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...) ⊗ BlockTensorKit.boxplus(ℂ^D))
    Aᵢ = rand(ComplexF64, ℂ^D ⊗ ℂ^2 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^D) 
    for i in eachindex(ps)
            A[1, 1, i, i, 1] = Aᵢ # Same tensor for each disorder sector
    end
    return A
end

for i in 1:50
    @timeit timer "construct_initial_state" construct_tensor()
    @timeit timer "construct_initial_MPStensor" construct_MPStensor()
end
