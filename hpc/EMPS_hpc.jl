# Job script: scan δ = μ/W² for fixed W and Dmax and store the final observables and states.
# The fields are hs = exp(μ)*[1, exp(-2W)], so every W scans the same δ interval.
# Usage (from repo root): julia --project=. hpc/EMPS_hpc.jl <W> <Dmax>
# The δ grid is range(DELTA_MIN, DELTA_MAX, length = NDELTA), set through the environment.
using TensorKit, DisorderKit, TimerOutputs, JLD2

W = parse(Float64, ARGS[1])
Dmax = parse(Int, ARGS[2])
W > 0 || error("W must be positive, since δ = μ/W²")

quick = get(ENV, "EMPS_QUICK", "0") == "1" # short run for smoke tests

Δτ = 0.05
maxiter = quick ? 20 : 1000
trunc_tol = 1e-6
convtol = 1e-7
δmin = parse(Float64, get(ENV, "DELTA_MIN", "-0.5"))
δmax = parse(Float64, get(ENV, "DELTA_MAX", "0.5"))
nδ = quick ? 2 : parse(Int, get(ENV, "NDELTA", "11"))
δs = collect(range(δmin, δmax, length = nδ))
μs = δs .* W^2

function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    ξ = DisorderKit.average_correlation_length(ρ)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
    M = real.(DisorderKit.expectation_value(ρ, Z))
    return (E, M, ξ)
end

function scan_phasespace(μs, δs, W, Δτ::Float64, alg, Dmax::Int, filename::String, params)
    Es = fill(NaN, length(μs))
    Ms = fill(NaN, length(μs))
    ξs = fill(NaN, length(μs))
    Ss = fill(NaN, length(μs))
    ϵs = fill(NaN, length(μs))
    ψs = Vector{Union{Nothing, DisorderKit.InfiniteDisorderMPS}}(nothing, length(μs))

    for (iμ, μ) in enumerate(μs)
        println("W = $W, Dmax = $Dmax, δ = $(δs[iμ]), μ = $μ ($iμ/$(length(μs)))")
        # Define model
        J₀ = 1.0
        hs = exp(μ) * [1., exp(-2W)]
        Js = J₀*[1., exp(-2W)]

        Hs = DisorderKit.random_transverse_field_ising(Js, hs)
        D_disorder = length(Js)*length(hs)
        ps = ones(D_disorder)./D_disorder

        ψ = DisorderKit.InfiniteDisorderMPS(ps, D_disorder, 2, 1)

        ψgs, data, info = DisorderKit.groundstate(ψ, Hs, Δτ, alg, maxD = Dmax)

        Es[iμ] = getindex.(data, 1)[end]
        Ms[iμ] = getindex.(data, 2)[end]
        ξs[iμ] = getindex.(data, 3)[end]
        Ss[iμ] = first(DisorderKit.average_entanglement_entropy(ψgs))
        ϵs[iμ] = info.ϵsconv[end]
        ψs[iμ] = ψgs

        # Save after every μ so a walltime kill only loses the current point
        jldsave(filename; Es, Ms, ξs, Ss, ϵs, ψs, μs, δs, params...)
    end
    return Es, Ms, ξs, Ss, ϵs
end

# Organized storage: <root>/EMPS_phase_line/dtau<Δτ>_maxiter<maxiter>/W<W>_Dmax<Dmax>.jld2
data_root = get(ENV, "DATA_DIR", joinpath(get(ENV, "VSC_DATA", "."), "data"))
folder = joinpath(data_root, "EMPS_phase_line", "dtau$(Δτ)_maxiter$(maxiter)")
mkpath(folder)
filename = joinpath(folder, "W$(W)_Dmax$(Dmax).jld2")
println("Saving results to $filename")

params = (; W, Dmax, Δτ, maxiter, trunc_tol, convtol)

myFinalizer = DisorderKit.Finalizer(Tuple{Float64, Float64, Float64}, my_finalize!)
alg = DisorderKit.Groundstate_iDTEBD(trunc_tol, convtol, maxiter, 5, TimerOutput(), myFinalizer)

@timeit alg.timer_output "scan_phasespace" scan_phasespace(μs, δs, W, Δτ, alg, Dmax, filename, params)

@show alg.timer_output
