# Job script: random transverse field Ising model, scan h0 for fixed Dmax and store the final observables and states.
# The couplings are Js = [exp(-2), 1] and the fields hs = [h0*exp(-2), h0], so ⟨ln h⟩ - ⟨ln J⟩ = ln h0
# and var(ln h) + var(ln J) = 2; δ = ln h0/2 is stored alongside.
# Usage (from repo root): julia --project=. hpc/EMPS_hpc.jl <Dmax>
# The h0 grid is range(H0_MIN, H0_MAX, length = NH0), set through the environment.
using TensorKit, DisorderKit, TimerOutputs, JLD2

Dmax = parse(Int, ARGS[1])

quick = get(ENV, "EMPS_QUICK", "0") == "1" # short run for smoke tests

Δτ = 0.05
maxiter = quick ? 20 : 1000
trunc_tol = 1e-6
convtol = 1e-7
h0min = parse(Float64, get(ENV, "H0_MIN", "0.8"))
h0max = parse(Float64, get(ENV, "H0_MAX", "1.2"))
nh0 = parse(Int, get(ENV, "NH0", quick ? "2" : "11"))
h0min > 0 || error("H0_MIN must be positive, since δ = ln h0/2")
h0s = collect(range(h0min, h0max, length = nh0))
lnh0s = log.(h0s)
δs = lnh0s ./ 2

function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    ξ = DisorderKit.average_correlation_length(ρ)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
    M = real.(DisorderKit.expectation_value(ρ, Z))
    return (E, M, ξ)
end

function scan_phasespace(h0s, lnh0s, δs, Δτ::Float64, alg, Dmax::Int, filename::String, params)
    Es = fill(NaN, length(h0s))
    Ms = fill(NaN, length(h0s))
    ξs = fill(NaN, length(h0s))
    Ss = fill(NaN, length(h0s))
    ϵs = fill(NaN, length(h0s))
    ψs = Vector{Union{Nothing, DisorderKit.InfiniteDisorderMPS}}(nothing, length(h0s))

    # Every h0 starts from the same random state (groundstate deep-copies it, so it is never modified)
    D_disorder = 2*2 # length(Js)*length(hs)
    ps = ones(D_disorder)./D_disorder
    ψ = DisorderKit.InfiniteDisorderMPS(ps, D_disorder, 2, 1)

    for (i, h0) in enumerate(h0s)
        println("Dmax = $Dmax, h0 = $h0, ln h0 = $(lnh0s[i]), δ = $(δs[i]) ($i/$(length(h0s)))")
        # Define model
        hs = [h0*exp(-2), h0]
        Js = [exp(-2), 1.]

        Hs = DisorderKit.random_transverse_field_ising(Js, hs)

        ψgs, data, info = DisorderKit.groundstate(ψ, Hs, Δτ, alg, maxD = Dmax)

        Es[i] = getindex.(data, 1)[end]
        Ms[i] = getindex.(data, 2)[end]
        ξs[i] = getindex.(data, 3)[end]
        Ss[i] = first(DisorderKit.average_entanglement_entropy(ψgs))
        ϵs[i] = info.ϵsconv[end]
        ψs[i] = ψgs

        # Save after every h0 so a walltime kill only loses the current point
        jldsave(filename; Es, Ms, ξs, Ss, ϵs, ψs, h0s, lnh0s, δs, params...)
    end
    return Es, Ms, ξs, Ss, ϵs
end

# Organized storage: <root>/RTFIM_phase_line/dtau<Δτ>_maxiter<maxiter>/Dmax<Dmax>.jld2
data_root = get(ENV, "DATA_DIR", joinpath(get(ENV, "VSC_DATA", "."), "data"))
folder = joinpath(data_root, "RTFIM_phase_line", "dtau$(Δτ)_maxiter$(maxiter)")
mkpath(folder)
filename = joinpath(folder, "Dmax$(Dmax).jld2")
println("Saving results to $filename")

params = (; Dmax, Δτ, maxiter, trunc_tol, convtol)

myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64, Float64}, my_finalize!)
alg = DisorderKit.Groundstate_iDTEBD(trunc_tol, convtol, maxiter, 5, TimerOutput(), myFinalizer)

@timeit alg.timer_output "scan_phasespace" scan_phasespace(h0s, lnh0s, δs, Δτ, alg, Dmax, filename, params)

@show alg.timer_output
