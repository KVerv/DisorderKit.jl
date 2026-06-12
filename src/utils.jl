function make_DiagonalBlockTensorMap(ps::Vector{<:Number})
    P = spzeros(ComplexF64, BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...), BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...))
    for (i,p) in enumerate(ps)
        P[i,i] = TensorMap([p],ℂ^1,ℂ^1)
    end
    return P
end