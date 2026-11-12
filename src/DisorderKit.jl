# DisorderKit: ground states of disordered chains as a FiniteDisorderMPS (one MPS per disorder realization, sharing
# tensors per site and disorder sector), found with disorder-normalized imaginary time evolution
# (Groundstate_FiniteNormalizedITE), and their exact or sampled disorder-averaged observables
module DisorderKit
__precompile__(true)

using TensorKit, MPSKit, BlockTensorKit, MatrixAlgebraKit
using LinearAlgebra, StatsBase, Random
using Crayons, TimerOutputs, ProgressBars
using BenchmarkFreeFermions

const AbstractMPSTensor = AbstractTensorMap{T, S, 2, 1} where {T, S}
const AbstractMPOTensor = AbstractTensorMap{T, S, 2, 2} where {T, S}
const AbstractBondTensor = AbstractTensorMap{T, S, 1, 1} where {T, S}

include("utils.jl")                    # AbstractAlgorithm, Finalizer, robust SVD, space helpers
include("DisorderMPOHam.jl")           # disorder Hamiltonians (RTFIM, Heisenberg, Bose-Hubbard) and their time evolution MPO
include("FiniteDisorderMPS.jl")        # the state, environments, energy and exact disorder averages
include("FiniteDisorderMPO.jl")        # finite time evolution MPO
include("ConditionalObservables.jl")   # per-realization observables and their (enumerated or sampled) disorder average
include("FiniteDisorderFilter.jl")     # normalization filter
include("FiniteSRC.jl")                # streaming randomized compression
include("FiniteSRCGraded.jl")          # its graded (U₁) variant
include("FiniteNormalizedITE.jl")      # the ground-state algorithm
include("FreeFermionEd.jl")            # exact free-fermion reference of the RTFIM

end # module DisorderKit
