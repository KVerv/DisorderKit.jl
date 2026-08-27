include("dep_helper.jl")

using TensorKit, KrylovKit, BlockTensorKit, MatrixAlgebraKit

using JLD2, TimerOutputs
using DisorderKit

D = parse(Int64, ARGS[1])
D_R = parse(Int64, ARGS[2])
W = parse(Float64, ARGS[3])

# Construct Finalizer for computing observables after each iteration of the groundstate algorithm
function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    ξ = DisorderKit.average_correlation_length(ρ)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
    M = real.(DisorderKit.expectation_value(ρ, Z))
    S, _ = DisorderKit.average_entanglement_entropy(ρ)
    return (E, ξ, M, S)
end

function job(D, D_R, W)
    Δτ = 0.05 # Step size for imaginary time evolution
    maxiter = 2000 # Maximum number of iterations for the groundstate algorithm

    a = 0.7
    b = 1.3
    hs = [a, b]*exp(W)
    Js = [a, b]
    p1 = 0.5
    p2 = 1-p1
    ps = [p1*p1, p1*p2, p2*p1, p2*p2]

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

    folder_name1 = "ρ_a$(a)_b$(b)_D$(D)_Δτ$(Δτ)_DR$(D_R)_W$(W)"

    println(folder_name1)
    isdir("/data/gent/487/vsc48713/data/$folder_name1") || mkdir("/data/gent/487/vsc48713/data/$folder_name1")

    # Generate algorithm object with corresponding parameters
    alg_evo = DisorderKit.Groundstate_iDTEBD(MatrixAlgebraKit.truncrank(D), MatrixAlgebraKit.truncrank(D_R); convtol = 1e-9, maxiter = maxiter, verbosity = 2, timer_output = TimerOutput(), finalizer = myFinalizer)

    # Compute the groundstate
    @timeit alg_evo.timer_output "compute_groundstate" ρs, data, info = DisorderKit.groundstate(ρ₀, Hs, Δτ, alg_evo)

    τ = maxiter * Δτ
    save_object("/data/gent/487/vsc48713/data/$folder_name1/ρs_$(τ).jld2", ρs)
    save_object("/data/gent/487/vsc48713/data/$folder_name1/data_$(τ).jld2", data)
    save_object("/data/gent/487/vsc48713/data/$folder_name1/info_$(τ).jld2", info)

    # acc = 1e-16
    # for (i,β) in enumerate(βs)
    #     @show (i,β,acc)
    #     if acc < 1e-2
    #         # if β > 0
    #         #     dβ = βs[i+1] - βs[i]
    #         #     nsteps = round(Int, dβ/dτ)
    #         #     ρ0 = load_object("/data/gent/487/vsc48713/data/$folder_name1/ρ_β$(βs[i]).jld2")
    #         #     ϵs = load_object("/data/gent/487/vsc48713/data/$folder_name1/ϵs.jld2")
    #         # else
    #         #     nsteps = round(Int, βs[2]/dτ)
    #         #     ρ0 = nothing
    #         #     ϵs = []
    #         # end
    #         inversion_frequency = 1
    #         alg_evolution = iDTEBD(alg_inversion, alg_trunc_Z, alg_trunc_disordermpo; invtol = invtol, nsteps = nsteps, verbosity = 2, truncfrequency = 1, inversion_frequency = inversion_frequency, timer_output = TimerOutput(), max_inverse_dim = D_z)
    #         ρs, ϵ = evolve_densitymatrix(Us, ps, alg_evolution; ρ0 = ρ0)

    #         acc = maximum(ϵ)
    #         save_object("/data/gent/487/vsc48713/data/$folder_name1/ρ_β$(βs[i+1]).jld2", ρs)
    #         push!(ϵs, ϵ...)
    #         save_object("/data/gent/487/vsc48713/data/$folder_name1/ϵs.jld2", ϵs)
    #     end
    # end
    @show alg_evo.timer_output
end

job(D, D_R, W)