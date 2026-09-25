# Job script: scan δ for fixed W and Dmax and store the final observables.
# Usage (from repo root): julia --project=. hpc/EMPS_hpc.jl <W> <Dmax>
using TensorKit, DisorderKit, TimerOutputs, JLD2

W = parse(Float64, ARGS[1])
Dmax = parse(Int, ARGS[2])

quick = get(ENV, "EMPS_QUICK", "0") == "1" # short run for smoke tests

Δτ = 0.05
maxiter = quick ? 20 : 1000
trunc_tol = 1e-6
convtol = 1e-7
δs = quick ? (-0.1:0.2:0.1) : (-0.1:0.05:0.3)

function my_finalize!(ρ, Hs)
    E = DisorderKit.energy_density(ρ, Hs)
    ξ = DisorderKit.average_correlation_length(ρ)
    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
    M = real.(DisorderKit.expectation_value(ρ, Z))
    return (E, M, ξ)
end

function scan_phasespace(δs, W, Δτ::Float64, alg, Dmax::Int, filename::String, params)
    Es = fill(NaN, length(δs))
    Ms = fill(NaN, length(δs))
    ξs = fill(NaN, length(δs))
    ϵs = fill(NaN, length(δs))

    for (iδ, δ) in enumerate(δs)
        println("W = $W, Dmax = $Dmax, δ = $δ ($iδ/$(length(δs)))")
        # Define model
        J₀ = 1.0
        hs = exp(δ) * [1., exp(-2W)]
        Js = J₀*[1., exp(-2W)]

        Hs = DisorderKit.random_transverse_field_ising(Js, hs)
        D_disorder = length(Js)*length(hs)
        ps = ones(D_disorder)./D_disorder

        ψ = DisorderKit.InfiniteDisorderMPS(ps, D_disorder, 2, 1)

        ψgs, data, info = DisorderKit.groundstate(ψ, Hs, Δτ, alg, maxD = Dmax)

        Es[iδ] = getindex.(data, 1)[end]
        Ms[iδ] = getindex.(data, 2)[end]
        ξs[iδ] = getindex.(data, 3)[end]
        ϵs[iδ] = info.ϵsconv[end]

        # Save after every δ so a walltime kill only loses the current point
        jldsave(filename; Es, Ms, ξs, ϵs, δs = collect(δs), params...)
    end
    return Es, Ms, ξs, ϵs
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

@timeit alg.timer_output "scan_phasespace" scan_phasespace(δs, W, Δτ, alg, Dmax, filename, params)

@show alg.timer_output
