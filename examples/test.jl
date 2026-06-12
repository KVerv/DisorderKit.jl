using Revise, TensorKit, TimerOutputs, BlockTensorKit

timer = TimerOutput()

d = 50


@timeit timer "prepare normal tensor" begin
    T = rand(ComplexF64,ℂ^2⊗ℂ^(d),ℂ^(d)⊗ ℂ^2)
end

Wcodomain = BlockTensorKit.boxplus(fill(ℂ^1, d)...) ⊗  BlockTensorKit.boxplus([ℂ^2]...)
Wdomain = BlockTensorKit.boxplus([ℂ^2]...) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, d)...)

@timeit timer "prepare zeros block tensor" begin
    Z = spzeros(ComplexF64, Wdomain, Wcodomain)
end

@timeit timer "prepare diagonal block tensor" begin
    W = spzeros(ComplexF64, Wdomain, Wcodomain)
    for i in 1:d
        @timeit timer "fill" W[1,i,i,1] = rand(ComplexF64, ℂ^2 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^2)
    end
    @timeit timer "W" W = W
end

V = [rand(ComplexF64, ℂ^2⊗ℂ^1, ℂ^1⊗ℂ^2) for i in 1:d]

# @timeit timer " add normal" begin
#     Q = T + 3*T
# end

# @timeit timer " add block" begin
#     P = W + 3*W
# end

