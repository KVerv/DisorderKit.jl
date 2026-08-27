module DisorderKit

__precompile__(true)

using TensorKit, MPSKit, MPSKitModels, KrylovKit, BlockTensorKit, MatrixAlgebraKit
using LinearAlgebra, StatsBase, Zygote, OptimKit, TensorKitManifolds
using Crayons, TimerOutputs, ProgressBars

# export VOMPS_Inversion, invert_mpo 
# export test_identity, mpo_fidelity
# export DisorderMPO, measure, partition_functions, disorder_average, average_correlation_length, normalize_each_disorder_sector
# export StandardTruncation, DisorderTracedTruncation, DisorderOpenTruncation,  SVDUpdateTruncation, truncate_mpo
# export iDTEBD, evolve_densitymatrix, evolve_one_time_step
# export random_transverse_field_ising_evolution, RTFIM_time_evolution_Trotter, RBH_time_evolution_Trotter, RTFIM_hamiltonian
# export average_renyi_entropy2, renyi_entropy2
# export InfiniteDisorderMPS, expectation_value, correlator
# export DisorderMPOHam, random_transverse_field_ising

const AbstractMPSTensor = AbstractTensorMap{T, S, 2, 1} where {T, S}
const AbstractMPOTensor = AbstractTensorMap{T, S, 2, 2} where {T, S}
const AbstractDisorderMPOTensor = AbstractTensorMap{T, S, 3, 3} where {T, S}
const AbstractRhoEnv = AbstractTensor{T, S, 2} where {T, S}
const AbstractEEnv = AbstractTensor{T, S, 4} where {T, S}
const AbstractBondTensor = AbstractTensorMap{T, S, 1, 1} where {T, S}
const AbstractDisorderMPSTensor = AbstractTensorMap{T, S, 2, 2} where {T, S}


# include("VOMPS_Inversion.jl")
# include("truncation_algorithms.jl")
# include("inversion.jl")
include("utils.jl")
include("DisorderMPOHam.jl")
include("DisorderDensityMatrix.jl")
include("DisorderMPO.jl")
include("finalizer.jl")
# include("partition_function.jl")
# include("PartitionFunction.jl")
# include("inv_sqrt_mpo.jl")
# include("svd_optimization.jl")
# include("mpo_truncation.jl")
include("truncation_algorithms.jl")
include("algorithms.jl")
include("renorm_op.jl")
include("iDTEBD_groundstate.jl")
# include("iDTEBD.jl")
include("models.jl")
# include("InfiniteDisorderMPS.jl")
# include("InfiniteDisorderMPS2.jl")
# include("InfiniteDisorderTangent.jl")
# include("InfiniteDisorderTangent2.jl")
include("SDRG.jl")
include("ordinary_truncation.jl")
include("SVD_truncation.jl")


end # module DisorderKit
