module DisorderKit

__precompile__(true)

using TensorKit, MPSKit, MPSKitModels, KrylovKit
using LinearAlgebra
using Crayons, TimerOutputs

export VOMPS_Inversion, invert_mpo 
export test_identity, mpo_fidelity
export DisorderMPO, measure, partition_functions, disorder_average, average_correlation_length
export StandardTruncation, DisorderTracedTruncation, truncate_mpo
export iDTEBD, evolve_densitymatrix
export random_transverse_field_ising_evolution, RTFIM_time_evolution_Trotter, RBH_time_evolution_Trotter, RTFIM_hamiltonian

const AbstractMPSTensor = AbstractTensorMap{T, S, 2, 1} where {T, S}
const AbstractMPOTensor = AbstractTensorMap{T, S, 2, 2} where {T, S}
const AbstractDisorderMPOTensor = AbstractTensorMap{T, S, 3, 3} where {T, S}
const AbstractRhoEnv = AbstractTensor{T, S, 2} where {T, S}
const AbstractEEnv = AbstractTensor{T, S, 4} where {T, S}
const AbstractBondTensor = AbstractTensorMap{T, S, 1, 1} where {T, S}

include("VOMPS_Inversion.jl")
include("utils.jl")
include("truncation_algorithms.jl")
include("DisorderMPO.jl")
include("mpo_truncation.jl")
include("iDTEBD.jl")
include("models.jl")

end # module DisorderKit
