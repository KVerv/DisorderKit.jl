include("dep_helper.jl")

using TensorKit, KrylovKit, BlockTensorKit, MatrixAlgebraKit

using JLD2, TimerOutputs
using DisorderKit

D = parse(Int64, ARGS[1])
D_R = parse(Int64, ARGS[2])
W = parse(Float64, ARGS[3])
δ = parse(Float64, ARGS[4])

# Construct Finalizer for computing observables after each iteration of the groundstate algorithm
function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    ξ = DisorderKit.average_correlation_length(ρ)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
    M = real.(DisorderKit.expectation_value(ρ, Z))
    S, _ = DisorderKit.average_entanglement_entropy(ρ)
    return (E, ξ, M, S)
end

function job(D, D_R, W, δ)
    Δτ = 0.05 # Step size for imaginary time evolution
    maxiter = 20 # Maximum number of iterations for the groundstate algorithm

    hs = exp(δ) * [1., exp(-2W)]
    Js = [1., exp(-2W)]
    w1 = 0.5
    w2 = 1-w1
    ps = [w1*w1, w1*w2, w2*w1, w2*w2]

    Hs = DisorderKit.random_transverse_field_ising(Js, hs)

    myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64, Float64, Float64}, my_finalize!)

    # Construct initial random state
    A = spzeros(ComplexF64, BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(ℂ^2) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...), BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(fill(ℂ^1, length(ps))...) ⊗ BlockTensorKit.boxplus(ℂ^1))
    Aᵢ = rand(ComplexF64, ℂ^1 ⊗ ℂ^2 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1 ⊗ ℂ^1) 
    for i in eachindex(ps)
            A[1, 1, i, 1, i, 1] = Aᵢ # Same tensor for each disorder sector
    end

    ρ₀ = DisorderKit.InfiniteDisorderDensityMatrix([A], ps)
    ρ₀ = DisorderKit.gauge(ρ₀)

    isdir("/data/gent/487/vsc48713/data") || mkdir("/data/gent/487/vsc48713/data")

    folder_name1 = "ρ_δ$(δ)_W$(W)_D$(D)_Δτ$(Δτ)_DR$(D_R)"

    println(folder_name1)
    isdir("/data/gent/487/vsc48713/data/$folder_name1") || mkdir("/data/gent/487/vsc48713/data/$folder_name1")

# Generate algorithm object with corresponding parameters
    # alg_trunc_norm = DisorderKit.StandardTruncation(; trunc_method = MatrixAlgebraKit.truncrank(D_R))
    alg_trunc_norm = DisorderKit.SuccessiveSVD(; trunc_method = MatrixAlgebraKit.truncrank(D_R))

    alg_trunc_state = DisorderKit.StandardTruncation(; trunc_method = MatrixAlgebraKit.truncrank(D))
    alg_evo = DisorderKit.Groundstate_iDTEBD(alg_trunc_state, alg_trunc_norm; convtol = 1e-9, maxiter = maxiter, verbosity = 2, timer_output = TimerOutput(), finalizer = myFinalizer)

    # Compute the groundstate
    @timeit alg_evo.timer_output "compute_groundstate" ρs, data, info = DisorderKit.groundstate(ρ₀, Hs, Δτ, alg_evo)

    τ = maxiter * Δτ
    save_object("/data/gent/487/vsc48713/data/$folder_name1/ρs_$(τ).jld2", ρs)
    save_object("/data/gent/487/vsc48713/data/$folder_name1/data_$(τ).jld2", data)
    save_object("/data/gent/487/vsc48713/data/$folder_name1/info_$(τ).jld2", info)

    @show alg_evo.timer_output
end

job(D, D_R, W, δ)