module DisorderKit

__precompile__(true)

using TensorKit, MPSKit, KrylovKit
using LinearAlgebra
using Crayons, TimerOutputs

export VOMPS_Inversion, invert_mpo 
export test_identity

const AbstractMPSTensor = AbstractTensorMap{T, S, 2, 1} where {T, S}
const AbstractMPOTensor = AbstractTensorMap{T, S, 2, 2} where {T, S}
const AbstractDisorderMPOTensor = AbstractTensorMap{T, S, 3, 3} where {T, S}
const AbstractRhoEnv = AbstractTensor{T, S, 2} where {T, S}
const AbstractEEnv = AbstractTensor{T, S, 4} where {T, S}
const AbstractBondTensor = AbstractTensorMap{T, S, 1, 1} where {T, S}

include("VOMPS_Inversion.jl")
include("utils.jl")

end # module DisorderKit
